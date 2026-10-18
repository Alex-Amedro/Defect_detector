"""Visual sanity check of a trained splat: photos vs renders, and renders from unseen camera poses.

The "training views" row is NOT an evaluation (the splat was fitted on those images); it only shows
that the model reproduces the object. The "new poses" row shows what the model invents between them.

Usage (inside the WSL venv):
    python scripts/render_check.py --classname 01Gorilla --iterations 30000
"""
import argparse
import os
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--classname", default="01Gorilla")
parser.add_argument("--iterations", type=int, default=30000)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--repo", default=str(Path.home() / "work" / "SplatPose"))
parser.add_argument("--data", default=str(Path.home() / "data" / "MAD-Sim_3dgs"))
parser.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results" / "figures"))
parser.add_argument("--n", type=int, default=4, help="views per row")
cli = parser.parse_args()

sys.path.insert(0, cli.repo)
os.chdir(cli.repo)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from gaussian_splatting.train import ArgumentParser, ModelParams, PipelineParams  # noqa: E402
from gaussian_splatting.scene import Scene, GaussianModel  # noqa: E402
from gaussian_splatting.gaussian_renderer import render  # noqa: E402
from gaussian_splatting.utils.image_utils import psnr  # noqa: E402

model_dir = os.path.join(cli.data, f"results_step0_{cli.seed}", cli.classname)
p = ArgumentParser()
lp, pp = ModelParams(p), PipelineParams(p)
args = p.parse_args(["-w", "--eval", "-s", os.path.join(cli.data, cli.classname),
                     "-m", model_dir, "--sh_degree", "0"])

gaussians = GaussianModel(0)
scene = Scene(lp.extract(args), gaussians, load_iteration=cli.iterations, shuffle=False)
background = torch.ones(3, device="cuda")  # white background, as in training
pipe = pp.extract(args)


def render_view(cam):
    with torch.no_grad():
        return torch.clamp(render(cam, gaussians, pipe, background)["render"], 0, 1)


def to_img(t):
    return t.permute(1, 2, 0).cpu().numpy()


train_cams = scene.getTrainCameras()
step = max(1, len(train_cams) // cli.n)
train_views = train_cams[::step][: cli.n]
new_views = scene.getTestCameras()[: cli.n]

fig, axes = plt.subplots(3, cli.n, figsize=(3.2 * cli.n, 9.6))
for j, cam in enumerate(train_views):
    gt = torch.clamp(cam.original_image.cuda(), 0, 1)
    out = render_view(cam)
    axes[0, j].imshow(to_img(gt))
    axes[1, j].imshow(to_img(out))
    axes[1, j].set_xlabel(f"{psnr(out, gt).mean().item():.1f} dB (training view)")
for j, cam in enumerate(new_views):
    axes[2, j].imshow(to_img(render_view(cam)))
for ax in axes.ravel():
    ax.set_xticks([])
    ax.set_yticks([])
axes[0, 0].set_ylabel("photo")
axes[1, 0].set_ylabel("splat render")
axes[2, 0].set_ylabel("new poses")
fig.suptitle(f"{cli.classname}: {len(gaussians.get_xyz)} gaussians after {cli.iterations} iterations")
fig.tight_layout()

os.makedirs(cli.out, exist_ok=True)
path = Path(cli.out) / f"step0_{cli.classname}_renders.png"
fig.savefig(path, dpi=110)
print("saved", path)
