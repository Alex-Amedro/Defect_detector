"""Does combining several views help on real photos? Same test as fusion_experiment.py, on the real bear.

Photos are put in sessions of a few photos (random, the same size for defect and healthy photos). Inside a session
the difference maps are combined in 3D, then read back in every photo and compared with the map of that photo alone.
Run real_check.py and real_detect.py first.

Usage (inside the WSL venv):
    python scripts/real_fusion.py --work ~/data/real/Bear --classname Bear_real
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
parser.add_argument("--seg", action="store_true", help="object-only splat: every pixel with a depth is on the object")
parser.add_argument("--session-size", type=int, default=6, help="photos per session")
parser.add_argument("--thr", type=float, default=0.3)
parser.add_argument("--repo", default=str(Path.home() / "work" / "SplatPose"))
parser.add_argument("--results", default=str(Path(__file__).resolve().parents[1] / "results"))
cli = parser.parse_args()

sys.path.insert(0, cli.repo)
sys.path.insert(1, str(Path(__file__).resolve().parent))
os.chdir(cli.repo)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pycolmap  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
import fusion  # noqa: E402
from gaussian_splatting.train import ArgumentParser, ModelParams, PipelineParams  # noqa: E402
from gaussian_splatting.scene import Scene, GaussianModel  # noqa: E402
from gaussian_splatting.scene.cameras import Camera  # noqa: E402

SIZE = 224
work = Path(cli.work).expanduser()
d = np.load(work / ("detect_seg.npz" if cli.seg else "detect.npz"))
names, maps, boxes, masks, is_bad = d["names"], d["maps"].astype(np.float32), d["boxes"], d["masks"], d["is_bad"]
center, radius = d["center"], float(d["radius"])
scene, model, photos_dir = (("seg_scene", "splat_seg", "seg_images") if cli.seg else ("train_scene", "splat", "sfm/images"))

p = ArgumentParser()
lp, pp = ModelParams(p), PipelineParams(p)
args = p.parse_args(["--eval", "--sh_degree", "3", "-s", str(work / scene), "-m", str(work / model)] + (["-w"] if cli.seg else []))
gaussians = GaussianModel(3)
Scene(lp.extract(args), gaussians, load_iteration=cli.iterations, shuffle=False)
pipe = pp.extract(args)
rec = pycolmap.Reconstruction(str(work / "sfm" / "sparse" / "0"))
by_name = {im.name: im for im in rec.images.values()}
value = lambda x: x() if callable(x) else x

# lift every pixel of every window to 3D, using the depth of the 3D copy seen from that photo
points, valid = [], []
for i, name in enumerate(names):
    image = by_name[str(name)]
    pose, camera = value(image.cam_from_world), image.camera
    fx, fy, cx, cy = camera.params
    photo = torch.ones(3, camera.height, camera.width)
    cam = Camera(0, pose.rotation.matrix().T, np.asarray(pose.translation), 2 * math.atan(camera.width / (2 * fx)),
                 2 * math.atan(camera.height / (2 * fy)), photo, None, str(name), 0)
    depth, alpha = fusion.render_depth(cam, gaussians, pipe)
    pts, ok = fusion.backproject_crop(depth, alpha, cam, (fx, fy, cx, cy), boxes[i], SIZE)
    points.append(pts)
    valid.append(ok)

if cli.seg:  # object-only splat: the grid is centred on the lifted points themselves
    lifted = torch.cat([points[i][valid[i]] for i in range(len(names))])[::20]
    middle = lifted.median(dim=0).values
    center = middle.cpu().numpy()
    radius = float(1.2 * (lifted - middle).norm(dim=1).quantile(0.98))

# sessions of the same size for defect photos and for healthy photos
rng = np.random.default_rng(0)
session_of = np.empty(len(names), dtype=object)
for label, group in [("defect", np.where(is_bad)[0]), ("healthy", np.where(~is_bad)[0])]:
    group = rng.permutation(group)
    for k, part in enumerate(np.array_split(group, max(1, round(len(group) / cli.session_size)))):
        session_of[part] = f"{label}#{k + 1}"

grid_args = dict(center=center, half_size=1.6 * radius, voxel=radius / 25)
fused = np.zeros_like(maps)
in_grid = np.zeros(maps.shape, dtype=bool)
for session in sorted(set(session_of)):
    idx = np.where(session_of == session)[0]
    grid = fusion.DefectGrid(**grid_args)
    for i in idx:
        grid.add_view(points[i][valid[i]], torch.as_tensor(maps[i], device="cuda")[valid[i]], cli.thr)
    field, seen = grid.mean_score(), grid.seen
    for i in idx:
        ok = valid[i]
        f, s = np.zeros((SIZE, SIZE), np.float32), np.zeros((SIZE, SIZE), bool)
        f[ok.cpu().numpy()] = grid.lookup(field, points[i][ok]).cpu().numpy()
        s[ok.cpu().numpy()] = (grid.lookup(seen, points[i][ok]) > 0).cpu().numpy()
        fused[i], in_grid[i] = f, s

# the object is what stands out of the table inside the sphere found in the point cloud: a pixel is "object"
# if its 3D point is inside the sphere and above the table plane
if cli.seg:  # the splat holds the object only, so a pixel with a depth is an object pixel
    near = np.stack([v.cpu().numpy() for v in valid])
else:
    centre_t = torch.as_tensor(center, dtype=torch.float32, device="cuda")
    plane_point, plane_normal = (torch.as_tensor(d[k], dtype=torch.float32, device="cuda") for k in ("plane_point", "plane_normal"))
    tol = float(d["plane_tol"])
    near = np.stack([(((points[i] - centre_t).norm(dim=-1) < radius) & ((points[i] - plane_point) @ plane_normal > 2 * tol)).cpu().numpy()
                     for i in range(len(names))])
good_i, bad_i = np.where(~is_bad)[0], np.where(is_bad)[0]
boot_rng = np.random.default_rng(1)
res = {"classname": cli.classname, "sessions": {s: int((session_of == s).sum()) for s in sorted(set(session_of))},
       "threshold": cli.thr, "voxel": grid_args["voxel"]}


def measure(keep):
    """Single view and combined views, on the same pixels `keep`."""
    scores = {name: np.array([field[i][keep[i]].max() if keep[i].any() else 0.0 for i in range(len(field))])
              for name, field in [("single_view", maps), ("fused", fused)]}
    out = {"pixels_kept": float(keep.mean()), "marked_pixels_kept": float(masks[keep].sum() / masks.sum())}
    for name, field in [("single_view", maps), ("fused", fused)]:
        out[name] = {"pixel_auroc": float(roc_auc_score(masks[keep], field[keep])),
                     "image_auroc": float(roc_auc_score(is_bad, scores[name]))}
    diffs = []
    for _ in range(1000):
        pick = np.concatenate([boot_rng.choice(good_i, len(good_i)), boot_rng.choice(bad_i, len(bad_i))])
        diffs.append(roc_auc_score(is_bad[pick], scores["fused"][pick]) - roc_auc_score(is_bad[pick], scores["single_view"][pick]))
    out["image_auroc_fused_minus_single"] = {"observed": out["fused"]["image_auroc"] - out["single_view"]["image_auroc"],
                                             "ci95_bootstrap": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]}
    return out, scores


res["whole_window"], _ = measure(in_grid)  # background included
res["object_only"], img_scores = measure(in_grid & near)  # background removed

# is the marked stain the same place on the object in every photo?
stain = torch.cat([points[i][torch.as_tensor(masks[i] & near[i] & valid[i].cpu().numpy(), device="cuda")] for i in bad_i])
dist = (stain - stain.median(dim=0).values).norm(dim=1)
res["marked_defect_3d_spread"] = {"median_distance": float(dist.median()), "p90": float(dist.quantile(0.9)),
                                  "object_radius": radius}
out = Path(cli.results)
(out / f"real_fusion_{cli.classname}.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))

rows = [i for i in bad_i[np.linspace(0, len(bad_i) - 1, 4).astype(int)]] + [good_i[np.argmax(img_scores["single_view"][good_i])]]
fig, ax = plt.subplots(len(rows), 4, figsize=(10, 2.5 * len(rows)))
photos = [np.asarray(Image.open(work / photos_dir / str(names[i])).convert("RGB").crop(tuple(boxes[i])).resize((SIZE, SIZE)))
          for i in rows]
for r, i in enumerate(rows):
    background = ~(in_grid[i] & near[i])  # greyed out: not part of the object
    show_maps = [np.where(background, np.nan, m) for m in (maps[i], fused[i])]
    for c, (img, title) in enumerate([(photos[r], "photo"), (show_maps[0], "one view, object only"),
                                      (show_maps[1], "views combined, object only"), (masks[i], "marked defect")]):
        ax[r, c].imshow(img, cmap=None if c == 0 else "inferno", vmin=0, vmax=1)
        ax[r, c].set_xticks([]), ax[r, c].set_yticks([])
        if r == 0:
            ax[r, c].set_title(title, fontsize=9)
    ax[r, 0].set_ylabel(f"{names[i]}\n{session_of[i]}", fontsize=8)
fig.tight_layout()
fig.savefig(out / "figures" / f"real_fusion_{cli.classname}.png", dpi=100)
