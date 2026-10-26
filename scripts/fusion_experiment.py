"""Does fusing the anomaly maps of several views beat using one view? Measured on MAD-Sim test images.

A "session" is a set of photos of the object in one state (all the good images, or all the images with the same
kind of defect). Inside a session the maps are fused in 3D, then the fused score is read back in every view and
compared with the single-view map, on the same pixels (object pixels only, background excluded).

Limit: MAD does not say which test images show the same defect instance. Sessions are built from the folder name
(good / Burrs / ...), so one session may mix several instances. With a real capture a session is one video.

Run scripts/evaluate.py first (it writes the poses and the maps); usage inside the WSL venv:
    python scripts/fusion_experiment.py --tag burrs_good
"""
import argparse
import json
import os
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--classname", default="01Gorilla")
parser.add_argument("--tag", required=True, help="tag of the evaluate.py run that wrote poses and maps")
parser.add_argument("--voxel", type=float, default=0.03)
parser.add_argument("--half-size", type=float, default=1.2)
parser.add_argument("--thr", type=float, default=0.3, help="a view flags a voxel above this normalised score")
parser.add_argument("--instance-dist", type=float, default=0.15, help="3D distance below which two defects are the same")
parser.add_argument("--good-session-size", type=int, default=8, help="healthy images are split into sessions of about this size")
parser.add_argument("--repo", default=str(Path.home() / "work" / "SplatPose"))
parser.add_argument("--data", default=str(Path.home() / "data" / "MAD-Sim_3dgs"))
parser.add_argument("--cache", default=str(Path.home() / "data" / "cache"))
parser.add_argument("--results", default=str(Path(__file__).resolve().parents[1] / "results"))
cli = parser.parse_args()

sys.path.insert(0, cli.repo)
sys.path.insert(1, str(Path(__file__).resolve().parent))
os.chdir(cli.repo)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.cluster.hierarchy import fcluster, linkage  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
import fusion  # noqa: E402
from gaussian_splatting.train import ArgumentParser, ModelParams, PipelineParams  # noqa: E402
from gaussian_splatting.scene import Scene, GaussianModel  # noqa: E402
from gaussian_splatting.scene.cameras import Camera  # noqa: E402

SIZE = 224
cache = Path(cli.cache) / cli.classname / cli.tag
parts = [np.load(f) for f in sorted(cache.glob("chunk_*.npz"))]
cat = lambda key: np.concatenate([p[key] for p in parts])
photo, mask, dtype = cat("photo"), cat("mask"), cat("dtype")
pose_R, pose_T = cat("pose_R"), cat("pose_T")
maps = np.load(cache / "maps.npy").astype(np.float32)
gt = np.stack([np.asarray(Image.fromarray(m * 255).resize((SIZE, SIZE), Image.NEAREST)) > 0 for m in mask])

source = os.path.join(cli.data, cli.classname)
p = ArgumentParser()
lp, pp = ModelParams(p), PipelineParams(p)
args = p.parse_args(["-w", "--eval", "-s", source, "-m", os.path.join(cli.data, "results_step0_0", cli.classname),
                     "--sh_degree", "0"])
gaussians = GaussianModel(0)
Scene(lp.extract(args), gaussians, load_iteration=30000, shuffle=False)
pipe = pp.extract(args)
meta = json.load(open(os.path.join(source, "transforms_train.json")))
fov = meta["camera_angle_x"]
positions = np.array([np.array(f["transform_matrix"])[:3, 3] for f in meta["frames"]])
center = positions.mean(axis=0)

# lift every pixel of every view to 3D
points, valid = [], []
for i in range(len(maps)):
    cam = Camera(0, pose_R[i], pose_T[i], fov, fov, torch.ones(3, 800, 800), None, "x", 0)
    depth, alpha = fusion.render_depth(cam, gaussians, pipe)
    pts, ok = fusion.backproject(depth, alpha, cam, fov, SIZE)
    points.append(pts)
    valid.append(ok)
valid_np = torch.stack(valid).cpu().numpy()

# MAD does not say which images show the same defect, so defect images are grouped by where their ground-truth
# defect sits in 3D (stand-in for "one video = one object state"). The groups are used only to build sessions.
centroid = {}
for i in range(len(maps)):
    on_object = torch.as_tensor(gt[i] & valid_np[i], device="cuda")
    if on_object.any():
        centroid[i] = points[i][on_object].mean(dim=0).cpu().numpy()
session_of = dtype.astype(object).copy()
for t in sorted(set(dtype) - {"good"}):
    idx = [i for i in np.where(dtype == t)[0] if i in centroid]
    if len(idx) > 1:
        labels = fcluster(linkage(np.stack([centroid[i] for i in idx]), "average"), cli.instance_dist, "distance")
        for i, label in zip(idx, labels):
            session_of[i] = f"{t}#{label}"

# healthy images are split into sessions of similar size, otherwise one big healthy session would be averaged more
# than the small defect sessions and would look better for that reason alone
rng = np.random.default_rng(0)
good_idx = rng.permutation(np.where(dtype == "good")[0])
for k, group in enumerate(np.array_split(good_idx, max(1, round(len(good_idx) / cli.good_session_size)))):
    session_of[group] = f"good#{k + 1}"

# one grid per session; the fused map of a view is the grid's mean score read at the view's 3D points
fused = np.zeros_like(maps)
grids = {}
for session in sorted(set(session_of)):
    idx = np.where(session_of == session)[0]
    grid = fusion.DefectGrid(center, cli.half_size, cli.voxel)
    for i in idx:
        grid.add_view(points[i][valid[i]], torch.as_tensor(maps[i], device="cuda")[valid[i]], cli.thr)
    field = grid.mean_score()
    for i in idx:
        f = np.zeros((SIZE, SIZE), dtype=np.float32)
        f[valid_np[i]] = grid.lookup(field, points[i][valid[i]]).cpu().numpy()
        fused[i] = f
    grids[session] = grid

# metrics on object pixels only
is_bad = dtype != "good"
keep = valid_np
res = {"classname": cli.classname, "tag": cli.tag, "sessions": {s: int((session_of == s).sum()) for s in grids},
       "voxel": cli.voxel, "threshold": cli.thr, "instance_dist": cli.instance_dist,
       "gt_pixels_kept_after_object_mask": float(gt[keep].sum() / gt.sum())}
img_scores = {}
for name, field in [("single_view", maps), ("fused", fused)]:
    img_scores[name] = np.array([field[i][valid_np[i]].max() if valid_np[i].any() else 0.0 for i in range(len(field))])
    res[name] = {"pixel_auroc_object_pixels": float(roc_auc_score(gt[keep], field[keep])),
                 "image_auroc": float(roc_auc_score(is_bad, img_scores[name]))}

# with few images the difference may be luck: resample good and defect images (the same ones for both methods)
boot_rng = np.random.default_rng(1)
good_i, bad_i = np.where(~is_bad)[0], np.where(is_bad)[0]
diffs = []
for _ in range(1000):
    pick = np.concatenate([boot_rng.choice(good_i, len(good_i)), boot_rng.choice(bad_i, len(bad_i))])
    diffs.append(roc_auc_score(is_bad[pick], img_scores["fused"][pick]) - roc_auc_score(is_bad[pick], img_scores["single_view"][pick]))
res["image_auroc_fused_minus_single"] = {"observed": res["fused"]["image_auroc"] - res["single_view"]["image_auroc"],
                                         "ci95_bootstrap": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]}

# is the "defect" one place in 3D? (spread of the ground-truth pixels once lifted to 3D, per defect session)
for s in grids:
    if s == "good":
        continue
    pts = torch.cat([points[i][torch.as_tensor(gt[i] & valid_np[i], device="cuda")] for i in np.where(session_of == s)[0]])
    if len(pts) == 0:  # images whose defect mask is empty at this resolution
        continue
    d = (pts - pts.median(dim=0).values).norm(dim=1)
    res[f"gt_3d_spread_{s}"] = {"n_points": int(len(pts)), "median_dist_to_median_point": float(d.median()),
                                "p90": float(d.quantile(0.9))}
# scale reference: size of the object along each axis (1st to 99th percentile of all lifted surface points)
surface = torch.cat([points[i][valid[i]] for i in range(0, len(points), 4)])
res["object_extent_xyz"] = [float(surface[:, k].quantile(0.99) - surface[:, k].quantile(0.01)) for k in range(3)]
out = Path(cli.results)
(out / f"step3_fusion_{cli.tag}.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))

# figure: a few defect views and the healthy view with the highest false alarm
bad_idx = [i for i in np.where(is_bad)[0] if gt[i].any()]
rows = [bad_idx[k] for k in np.linspace(0, len(bad_idx) - 1, 4).astype(int)]
good_idx = np.where(~is_bad)[0]
if len(good_idx):
    rows.append(good_idx[np.argmax([maps[i][valid_np[i]].max() for i in good_idx])])
fig, ax = plt.subplots(len(rows), 4, figsize=(10, 2.5 * len(rows)))
for r, i in enumerate(rows):
    for c, (img, title) in enumerate([(photo[i].transpose(1, 2, 0), "photo"), (maps[i], "single view"),
                                      (fused[i], "fused"), (gt[i], "ground truth")]):
        ax[r, c].imshow(img, cmap=None if c == 0 else "inferno", vmin=0, vmax=1)
        ax[r, c].set_xticks([]), ax[r, c].set_yticks([])
        if r == 0:
            ax[r, c].set_title(title, fontsize=9)
    ax[r, 0].set_ylabel(f"{session_of[i]} (image {i})", fontsize=8)
fig.tight_layout()
(out / "figures").mkdir(exist_ok=True)
fig.savefig(out / "figures" / f"step3_fusion_{cli.tag}.png", dpi=100)
