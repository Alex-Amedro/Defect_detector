"""Show a defect found by the multi-view fusion.

The anomaly maps of the photos that show one defect are fused in a 3D grid (as in fusion_experiment.py), the same is done
with the good photos, and the difference is the defect field: what the defect photos flag more than the good ones do.
Outputs, in results/media:
  <class>_defect_photo.png and <class>_healthy_model_same_angle.png: the defect photo with the defect outlined in red
      (ground-truth mask), and the healthy copy drawn at the same pose.
  <class>_photo_model_detection.png: one defect photo, the healthy copy drawn at the same pose (what is missing shows by
      comparison), and the photo with the fused map in red.
  <class>_defect_3d_turn.mp4: the copy turning, with the defect field glowing in orange.
Photos of the same defect are found by grouping the ground-truth defect positions in 3D, as in fusion_experiment.py.
The ground-truth masks, lifted to 3D, are only used for the numbers printed at the end.

Run evaluate.py first (it writes the poses and the maps). Usage inside the WSL venv:
    python scripts/defect_3d.py --classname 01Gorilla --tag missing_good --type Missing --photo 000.png
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--classname", default="01Gorilla")
parser.add_argument("--tag", default="missing_good", help="tag of the evaluate.py run")
parser.add_argument("--type", default="Missing", help="defect type to show")
parser.add_argument("--session", type=int, help="defect instance to show (default: the one seen by most photos)")
parser.add_argument("--instance-dist", type=float, default=0.15, help="3D distance below which two defects are the same")
parser.add_argument("--photo", help="name of the defect photo to show (default: the one where the defect covers the most pixels)")
parser.add_argument("--frames", type=int, default=180)
parser.add_argument("--distance", type=float, default=0.8, help="camera distance, as a share of the distance of the training photos")
parser.add_argument("--size", type=int, default=540, help="size of each panel in pixels")
parser.add_argument("--thr", type=float, default=0.3)
parser.add_argument("--glow-share", type=float, default=0.12, help="share of the gaussians that can glow")
parser.add_argument("--stills", action="store_true", help="also save four frames of each video as a PNG")
parser.add_argument("--repo", default=str(Path.home() / "work" / "SplatPose"))
parser.add_argument("--data", default=str(Path.home() / "data" / "MAD-Sim_3dgs"))
parser.add_argument("--cache", default=str(Path.home() / "data" / "cache"))
parser.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results" / "media"))
cli = parser.parse_args()

sys.path.insert(0, cli.repo)
sys.path.insert(1, str(Path(__file__).resolve().parent))
os.chdir(cli.repo)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from scipy.cluster.hierarchy import fcluster, linkage  # noqa: E402
from scipy.ndimage import binary_dilation, binary_erosion  # noqa: E402
import fusion  # noqa: E402
from gaussian_splatting.train import ArgumentParser, ModelParams, PipelineParams  # noqa: E402
from gaussian_splatting.scene import Scene, GaussianModel  # noqa: E402
from gaussian_splatting.scene.cameras import Camera  # noqa: E402
from gaussian_splatting.gaussian_renderer import render  # noqa: E402

SIZE = 224
cache = Path(cli.cache) / cli.classname / cli.tag
parts = [np.load(f) for f in sorted(cache.glob("chunk_*.npz"))]
cat = lambda key: np.concatenate([p[key] for p in parts])
mask, dtype, pose_R, pose_T = cat("mask"), cat("dtype"), cat("pose_R"), cat("pose_T")
photos, renders, names = cat("photo"), cat("render"), cat("name")
maps = np.load(cache / "maps.npy").astype(np.float32)
gt = np.stack([np.asarray(Image.fromarray(m * 255).resize((SIZE, SIZE), Image.NEAREST)) > 0 for m in mask])

source = os.path.join(cli.data, cli.classname)
p = ArgumentParser()
lp, pp = ModelParams(p), PipelineParams(p)
args = p.parse_args(["-w", "--eval", "-s", source, "-m", os.path.join(cli.data, "results_step0_0", cli.classname), "--sh_degree", "0"])
gaussians = GaussianModel(0)
Scene(lp.extract(args), gaussians, load_iteration=30000, shuffle=False)
pipe = pp.extract(args)
meta = json.load(open(os.path.join(source, "transforms_train.json")))
fov = meta["camera_angle_x"]
centres = np.array([np.array(f["transform_matrix"])[:3, 3] for f in meta["frames"]])

# lift every pixel of every photo to 3D
points, valid = [], []
for i in range(len(maps)):
    cam = Camera(0, pose_R[i], pose_T[i], fov, fov, torch.ones(3, 800, 800), None, "x", 0)
    depth, alpha = fusion.render_depth(cam, gaussians, pipe)
    pts, ok = fusion.backproject(depth, alpha, cam, fov, SIZE)
    points.append(pts)
    valid.append(ok)

# which photos show the same defect: group them by where the ground-truth defect sits in 3D
idx = [i for i in np.where(dtype == cli.type)[0] if (gt[i] & valid[i].cpu().numpy()).any()]
centroid = np.stack([points[i][torch.as_tensor(gt[i], device="cuda") & valid[i]].mean(dim=0).cpu().numpy() for i in idx])
labels = fcluster(linkage(centroid, "average"), cli.instance_dist, "distance")
counts = {int(k): int((labels == k).sum()) for k in np.unique(labels)}
print("defect instances (id: photos):", dict(sorted(counts.items(), key=lambda kv: -kv[1])))
chosen = cli.session if cli.session is not None else max(counts, key=counts.get)
defect = np.array([i for i, label in zip(idx, labels) if label == chosen])
good = np.where(dtype == "good")[0]
print(f"showing instance {chosen}: {len(defect)} photos")

surface = torch.cat([points[i][valid[i]] for i in range(len(points))])[::20]
target = surface.median(dim=0).values.cpu().numpy()  # centre of the object


def new_grid():  # same grid as fusion_experiment.py
    return fusion.DefectGrid(centres.mean(axis=0), 1.2, 0.03)


# one grid for the photos of the defect, one for the good photos; the defect field is the difference
grids, means = {}, {}
for label, selected in [("defect", defect), ("good", good)]:
    grids[label] = new_grid()
    for i in selected:
        grids[label].add_view(points[i][valid[i]], torch.as_tensor(maps[i], device="cuda")[valid[i]], cli.thr)
    means[label] = grids[label].mean_score()
field = (means["defect"] - means["good"]).clamp(min=0)
field = F.max_pool3d(field[None, None], 3, stride=1, padding=1)[0, 0]  # a little spread so the glow is visible
field = F.avg_pool3d(field[None, None], 5, stride=1, padding=2)[0, 0]

# glow of every gaussian = defect field at its position; only the strongest share of the gaussians glows
xyz = gaussians.get_xyz.detach()
glow = new_grid().lookup(field, xyz)
floor = torch.quantile(glow, 1 - cli.glow_share)
glow = ((glow - floor) / (glow.max() - floor).clamp_min(1e-6)).clamp(0, 1)
base = (0.5 + 0.28209479 * gaussians._features_dc[:, 0, :].detach()).clamp(0, 1)  # colour of each gaussian
strength = 0.85 * glow[:, None]
glowing = (1 - strength) * base + strength * torch.tensor([1.0, 0.3, 0.02], device="cuda")

# check against the ground truth: masks of the same photos lifted to 3D. A mask pixel can land on a part of the object
# behind the missing piece, but only in a few views, so a voxel counts only if most of the photos that see it mark it.
truth_grid = new_grid()
for i in defect:
    truth_grid.add_view(points[i][valid[i]], torch.as_tensor(gt[i], device="cuda").float()[valid[i]], 0.5)
solid = ((truth_grid.vote_ratio() > 0.5) & (truth_grid.hits >= 3)).float()
truth = F.max_pool3d(solid[None, None], 3, stride=1, padding=1)[0, 0]
on_defect = truth_grid.lookup(truth, xyz) > 0.5
lit = glow > 0.3
print(f"{int(lit.sum())} gaussians glow clearly, {int((lit & on_defect).sum())} of them on the true defect "
      f"({int(on_defect.sum())} gaussians lie on it)")

try:
    font = ImageFont.load_default(size=26)
except TypeError:  # older Pillow
    font = ImageFont.load_default()


def labelled(image, text, colour, background):
    """The image with a band on top that holds its label."""
    banner = Image.fromarray(np.full((48, image.shape[1], 3), background, dtype=np.uint8))
    ImageDraw.Draw(banner).text((16, 8), text, fill=colour, font=font)
    return np.concatenate([np.asarray(banner), image], axis=0)


out = Path(cli.out)
out.mkdir(parents=True, exist_ok=True)

# 1. still image: a defect photo, the healthy copy at the same pose, and the photo with the fused map in red
shown = (np.where((names == cli.photo) & (dtype == cli.type))[0][0] if cli.photo
         else max(defect, key=lambda i: int((mask[i] > 0).sum())))
print("photo shown:", names[shown])
shot = photos[shown].transpose(1, 2, 0)
drawn = (renders[shown].astype(np.float32).clip(0, 1) * 255).round().astype(np.uint8).transpose(1, 2, 0)
fused = np.zeros((SIZE, SIZE), np.float32)
fused[valid[shown].cpu().numpy()] = new_grid().lookup(means["defect"], points[shown][valid[shown]]).cpu().numpy()
fused = np.asarray(Image.fromarray(fused).resize(shot.shape[:2][::-1], Image.BILINEAR))
red = np.clip((fused / fused.max() - 0.45) / 0.35, 0, 1)[..., None] * 0.8
marked = ((1 - red) * shot + red * np.array([230, 20, 20])).round().astype(np.uint8)

ink = (np.abs(shot.astype(int) - 255).sum(2) > 30) | (np.abs(drawn.astype(int) - 255).sum(2) > 30)
rows, cols = np.nonzero(ink)
cy, cx, half = (rows.min() + rows.max()) // 2, (cols.min() + cols.max()) // 2, int(0.58 * max(np.ptp(rows), np.ptp(cols)))


def zoomed(image):  # the same square window around the object in every image, padded with white
    padded = np.pad(image, ((half, half), (half, half), (0, 0)), constant_values=255)
    return np.asarray(Image.fromarray(padded[cy:cy + 2 * half, cx:cx + 2 * half]).resize((cli.size, cli.size), Image.LANCZOS))


dark = (60, 60, 60)
still = np.concatenate([labelled(zoomed(shot), "photo with the defect", dark, 255),
                        labelled(zoomed(drawn), "healthy 3D model, same angle", dark, 255),
                        labelled(zoomed(marked), "detected (fused map)", dark, 255)], axis=1)
Image.fromarray(still).save(out / f"{cli.classname}_photo_model_detection.png", optimize=True)
print("saved", out / f"{cli.classname}_photo_model_detection.png")

# the same photo with the defect outlined (the ground-truth mask), and the healthy copy, as two separate images
defect_mask = mask[shown] > 0
outline = binary_dilation(defect_mask & ~binary_erosion(defect_mask, iterations=3), iterations=1)
outlined = shot.copy()
outlined[outline] = (230, 20, 20)
Image.fromarray(zoomed(outlined)).save(out / f"{cli.classname}_defect_photo.png", optimize=True)
Image.fromarray(zoomed(drawn)).save(out / f"{cli.classname}_healthy_model_same_angle.png", optimize=True)

# 2. and 3. a camera that goes around the object (z is up in MAD-Sim), at the average height of the training photos
up = np.array([0.0, 0.0, 1.0])
offsets = centres - target
distance = cli.distance * np.median(np.linalg.norm(offsets, axis=1))
elevation = float(np.mean(np.arcsin((offsets @ up) / np.linalg.norm(offsets, axis=1))))
a = np.array([1.0, 0.0, 0.0])
b = np.cross(up, a)


def camera_at(azimuth):
    position = target + distance * (math.cos(elevation) * (math.cos(azimuth) * a + math.sin(azimuth) * b) + math.sin(elevation) * up)
    forward = (target - position) / np.linalg.norm(target - position)
    right = np.cross(-up, forward)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    R_w2c = np.stack([right, down, forward])
    return Camera(0, R_w2c.T, -R_w2c @ position, fov, fov, torch.ones(3, cli.size, cli.size), None, "orbit", 0)


white = torch.ones(3, device="cuda")
turn_frames = []
for k in range(cli.frames):
    cam = camera_at(2 * math.pi * k / cli.frames)
    with torch.no_grad():
        view = render(cam, gaussians, pipe, white, override_color=glowing)["render"].clamp(0, 1)
    turn_frames.append((view.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8))


def save(images, name):
    path = out / f"{cli.classname}_{name}.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (images[0].shape[1], images[0].shape[0]))
    for image in images:
        writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    writer.release()
    print("saved", path)
    if cli.stills:
        Image.fromarray(np.vstack([images[k * len(images) // 4] for k in range(4)])).save(path.with_suffix(".png"))


save(turn_frames, "defect_3d_turn")
