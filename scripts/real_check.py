"""Quality check of a splat built from real photos: draw the object at the camera of every photo and compare.

Photos held out from training (every 8th training photo) tell how well the splat copies photos it has never seen.
The drawings of the test photos are saved for the defect search.

Usage (inside the WSL venv):
    python scripts/real_check.py --work ~/data/real/Bear --classname Bear_real
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--work", required=True)
parser.add_argument("--classname", required=True)
parser.add_argument("--iterations", type=int, default=30000)
parser.add_argument("--seg", action="store_true", help="object-only splat (photos cut out by segment_object.py, white background)")
parser.add_argument("--repo", default=str(Path.home() / "work" / "SplatPose"))
parser.add_argument("--results", default=str(Path(__file__).resolve().parents[1] / "results"))
cli = parser.parse_args()

sys.path.insert(0, cli.repo)
os.chdir(cli.repo)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pycolmap  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from gaussian_splatting.train import ArgumentParser, ModelParams, PipelineParams  # noqa: E402
from gaussian_splatting.scene import Scene, GaussianModel  # noqa: E402
from gaussian_splatting.scene.cameras import Camera  # noqa: E402
from gaussian_splatting.gaussian_renderer import render  # noqa: E402

work = Path(cli.work).expanduser()
p = ArgumentParser()
lp, pp = ModelParams(p), PipelineParams(p)
scene, model, photos_dir = (("seg_scene", "splat_seg", "seg_images") if cli.seg else ("train_scene", "splat", "sfm/images"))
args = p.parse_args(["--eval", "--sh_degree", "3", "-s", str(work / scene), "-m", str(work / model)] + (["-w"] if cli.seg else []))
gaussians = GaussianModel(3)
Scene(lp.extract(args), gaussians, load_iteration=cli.iterations, shuffle=False)
pipe = pp.extract(args)
background = (torch.ones if cli.seg else torch.zeros)(3, device="cuda")  # same as training

# the photos held out by the training code: every 8th training photo, in name order
train_names = sorted(f"train_{i:03d}.jpg" for i in range(1000) if (work / "images_all" / f"train_{i:03d}.jpg").exists())
held_out = set(train_names[::8])

rec = pycolmap.Reconstruction(str(work / "sfm" / "sparse" / "0"))
rows, renders, psnrs = {}, {}, {}
for image in sorted(rec.images.values(), key=lambda im: im.name):
    pose = image.cam_from_world() if callable(image.cam_from_world) else image.cam_from_world
    R_w2c, t = pose.rotation.matrix(), pose.translation
    cam_model = image.camera
    fx, fy = cam_model.params[0], cam_model.params[1]
    photo = np.asarray(Image.open(work / photos_dir / image.name).convert("RGB"), dtype=np.float32) / 255
    cam = Camera(0, R_w2c.T, np.asarray(t), 2 * math.atan(cam_model.width / (2 * fx)),
                 2 * math.atan(cam_model.height / (2 * fy)), torch.from_numpy(photo).permute(2, 0, 1), None, image.name, 0)
    with torch.no_grad():
        out = render(cam, gaussians, pipe, background)["render"].clamp(0, 1)
    mse = ((out - cam.original_image) ** 2).mean().item()
    psnrs[image.name] = -10 * math.log10(max(mse, 1e-10))
    renders[image.name] = (out.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    rows[image.name] = photo

def group(names):
    return float(np.mean([psnrs[n] for n in names])) if names else None

kinds = lambda k: [n for n in psnrs if n.startswith(k)]
summary = {
    "classname": cli.classname,
    "psnr_db": {
        "held_out_training_photos": group(sorted(held_out)),
        "training_photos": group([n for n in kinds("train_") if n not in held_out]),
        **{f"test_{k}": group(kinds(k + "_")) for k in sorted({n.split("_")[0] for n in psnrs} - {"train"})},
    },
    "n_held_out": len(held_out),
}
out = Path(cli.results)
out.mkdir(parents=True, exist_ok=True)
(out / f"real_check_{cli.classname}.json").write_text(json.dumps(summary, indent=2))
np.savez(work / ("renders_seg.npz" if cli.seg else "renders.npz"), names=np.array(sorted(renders)), renders=np.stack([renders[n] for n in sorted(renders)]))
print(json.dumps(summary, indent=2))

# figure: held-out training photos, then test photos
show = sorted(held_out)[:3] + [n for n in sorted(psnrs) if not n.startswith("train_")][::9][:3]
fig, ax = plt.subplots(len(show), 3, figsize=(9, 3 * len(show)))
for r, name in enumerate(show):
    photo, drawing = rows[name], renders[name].astype(np.float32) / 255
    for c, (img, title) in enumerate([(photo, "photo"), (drawing, "3D copy"), (np.abs(photo - drawing).sum(2), "difference")]):
        ax[r, c].imshow(img, cmap="inferno" if c == 2 else None, vmin=0 if c == 2 else None, vmax=1 if c == 2 else None)
        ax[r, c].set_xticks([]), ax[r, c].set_yticks([])
        if r == 0:
            ax[r, c].set_title(title)
    ax[r, 0].set_ylabel(f"{name}\n{psnrs[name]:.1f} dB", fontsize=8)
fig.tight_layout()
(out / "figures").mkdir(exist_ok=True)
fig.savefig(out / "figures" / f"real_check_{cli.classname}.png", dpi=90)
