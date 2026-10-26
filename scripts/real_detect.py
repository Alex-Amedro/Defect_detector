"""Find defects on real photos: compare each test photo with the 3D copy drawn at the same camera.

The photos are cropped around the object (found automatically in the SfM point cloud: it is what stands out of the
table), so the comparison is not drowned by the background. Run real_check.py first (it draws the 3D copy).

Usage (inside the WSL venv):
    python scripts/real_detect.py --work ~/data/real/Bear --classname Bear_real
"""
import argparse
import json
import os
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--work", required=True)
parser.add_argument("--classname", required=True)
parser.add_argument("--margin", type=float, default=1.3, help="crop size = this times the object diameter in the photo")
parser.add_argument("--seg", action="store_true", help="object-only splat: the crop comes from the masks of segment_object.py")
parser.add_argument("--repo", default=str(Path.home() / "work" / "SplatPose"))
parser.add_argument("--results", default=str(Path(__file__).resolve().parents[1] / "results"))
cli = parser.parse_args()

sys.path.insert(0, cli.repo)
sys.path.insert(1, str(Path(__file__).resolve().parent))
os.chdir(cli.repo)

import cv2  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pycolmap  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from scoring import SIZE, anomaly_maps  # noqa: E402

work = Path(cli.work).expanduser()
meta = json.loads((work / "meta.json").read_text())
rec = pycolmap.Reconstruction(str(work / "sfm" / "sparse" / "0"))
raw = pycolmap.Reconstruction(str(work / "sfm" / "raw" / "0"))
stored = np.load(work / ("renders_seg.npz" if cli.seg else "renders.npz"))
photos_dir = "seg_images" if cli.seg else "sfm/images"
render_of = dict(zip(stored["names"], stored["renders"]))


def value(x):
    return x() if callable(x) else x


def locate_object(rec):
    """Centre and radius of what stands out of the table, in SfM units."""
    pts = np.array([p.xyz for p in rec.points3D.values()])
    cams = np.array([value(im.projection_center) for im in rec.images.values()])
    extent = np.linalg.norm(np.percentile(pts, 95, axis=0) - np.percentile(pts, 5, axis=0))
    tol, rng, best = 0.01 * extent, np.random.default_rng(0), (0, None, None)
    for _ in range(600):  # RANSAC: the plane with the most points is the table
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        count = (np.abs((pts - a) @ n) < tol).sum()
        if count > best[0]:
            best = (count, a, n)
    _, a, n = best
    if ((cams - a) @ n).mean() < 0:
        n = -n  # cameras are above the table
    height = (pts - a) @ n
    above = pts[(height > 3 * tol) & (height < 0.5 * extent)]
    center = np.median(above, axis=0)
    for _ in range(5):  # keep the dense cluster, drop stray points
        d = np.linalg.norm(above - center, axis=1)
        above = above[d < 2 * np.median(d)]
        center = np.median(above, axis=0)
    radius = 1.2 * np.percentile(np.linalg.norm(above - center, axis=1), 90)
    return center, radius, int(best[0]), len(pts), len(above), (a, n, tol)


if cli.seg:  # the crop comes from the masks, no need to look for the object in the point cloud
    center, radius, plane = np.zeros(3), 0.0, (np.zeros(3), np.zeros(3), 0.0)
else:
    center, radius, n_table, n_all, n_obj, plane = locate_object(rec)
    print(f"table plane: {n_table} of {n_all} points; object: {n_obj} points, radius {radius:.3f}")

# the masks are drawn on the original photos: undistort them like the photos were
old = next(iter(raw.cameras.values()))
new = next(iter(rec.cameras.values()))
f, cx, cy, k = old.params
K_old = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
K_new = np.array([[new.params[0], 0, new.params[2]], [0, new.params[1], new.params[3]], [0, 0, 1]])
map_x, map_y = cv2.initUndistortRectifyMap(K_old, np.array([k, 0, 0, 0.0]), None, K_new, (new.width, new.height), cv2.CV_32FC1)

held_out = sorted(n for n in meta if n.startswith("train_"))[::8]
names = [n for n in sorted(meta) if not n.startswith("train_")] + held_out
photos, drawings, masks, boxes = [], [], [], []
for name in names:
    image = next(im for im in rec.images.values() if im.name == name)
    if cli.seg:  # square around what is drawn or cut out: the object in the photo, the healthy object in the drawing
        in_photo = np.asarray(Image.open(work / "masks" / (Path(name).stem + ".png"))) > 0
        drawn = np.abs(render_of[name].astype(int) - 255).sum(2) > 40
        rows, cols = np.nonzero(in_photo | drawn)
        u, v = (cols.min() + cols.max()) / 2, (rows.min() + rows.max()) / 2
        half = cli.margin * max(np.ptp(cols), np.ptp(rows)) / 2
    else:
        pose = value(image.cam_from_world)
        p_cam = pose.rotation.matrix() @ center + pose.translation
        u, v = new.params[0] * p_cam[0] / p_cam[2] + new.params[2], new.params[1] * p_cam[1] / p_cam[2] + new.params[3]
        half = cli.margin * new.params[0] * radius / p_cam[2]
    box = tuple(int(round(x)) for x in (u - half, v - half, u + half, v + half))
    photo = Image.open(work / photos_dir / name).convert("RGB")
    mask = np.zeros((new.height, new.width), np.uint8)
    if meta[name]["mask"]:
        m = np.asarray(Image.open(meta[name]["mask"]).resize((old.width, old.height), Image.NEAREST)) > 0
        mask = cv2.remap(m.astype(np.uint8), map_x, map_y, cv2.INTER_NEAREST)
    crop = lambda im: im.crop(box).resize((SIZE, SIZE), Image.BILINEAR)
    photos.append(np.asarray(crop(photo)).transpose(2, 0, 1).copy())
    drawings.append(np.asarray(crop(Image.fromarray(render_of[name]))).transpose(2, 0, 1).copy())
    masks.append(np.asarray(Image.fromarray(mask * 255).crop(box).resize((SIZE, SIZE), Image.NEAREST)) > 0)
    boxes.append(box)
photos, drawings, masks = np.stack(photos), np.stack(drawings), np.stack(masks)

maps = anomaly_maps(photos, drawings)
is_bad = np.array([meta[n]["kind"] != "good" and not n.startswith("train_") for n in names])
img_score = maps.reshape(len(maps), -1).max(axis=1)

sys.path.insert(2, cli.repo)
from aupro import calculate_au_pro_au_roc  # noqa: E402  (MVTec reference code shipped with SplatPose)

rng = np.random.default_rng(0)
good_i, bad_i = np.where(~is_bad)[0], np.where(is_bad)[0]
boot = [roc_auc_score(is_bad[s], img_score[s]) for s in
        (np.concatenate([rng.choice(good_i, len(good_i)), rng.choice(bad_i, len(bad_i))]) for _ in range(1000))]
res = {
    "classname": cli.classname,
    "n_defect_photos": int(is_bad.sum()),
    "n_healthy_photos": int((~is_bad).sum()),
    "healthy_photos": "test photos without defect + training photos held out from the splat",
    "defect_photos_with_empty_mask": int((is_bad & (masks.reshape(len(masks), -1).sum(1) == 0)).sum()),
    "image_auroc": float(roc_auc_score(is_bad, img_score)),
    "image_auroc_ci95_bootstrap": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
    "pixel_auroc": float(roc_auc_score(masks.ravel(), maps.ravel())),
    "aupro_fpr_0.3": float(calculate_au_pro_au_roc(masks.astype(np.uint8), maps)[0]),
    "object_radius_sfm_units": float(radius),
}
out = Path(cli.results)
out.mkdir(parents=True, exist_ok=True)
(out / f"real_detect_{cli.classname}.json").write_text(json.dumps(res, indent=2))
np.savez(work / ("detect_seg.npz" if cli.seg else "detect.npz"), names=np.array(names), maps=maps.astype(np.float16), boxes=np.array(boxes),
         masks=masks, is_bad=is_bad, center=center, radius=radius,
         plane_point=plane[0], plane_normal=plane[1], plane_tol=plane[2])
print(json.dumps(res, indent=2))

# figure: a few defect photos and the healthy photo with the highest false alarm
bad_with_mask = [i for i in bad_i if masks[i].any()]
rows = [bad_with_mask[j] for j in np.linspace(0, len(bad_with_mask) - 1, 4).astype(int)]
rows.append(good_i[np.argmax(img_score[good_i])])
fig, ax = plt.subplots(len(rows), 4, figsize=(10, 2.5 * len(rows)))
for r, i in enumerate(rows):
    for c, (img, title) in enumerate([(photos[i].transpose(1, 2, 0), "photo"), (drawings[i].transpose(1, 2, 0), "3D copy"),
                                      (maps[i], "difference map"), (masks[i], "marked defect")]):
        ax[r, c].imshow(img, cmap="inferno" if c >= 2 else None, vmin=0, vmax=1)
        ax[r, c].set_xticks([]), ax[r, c].set_yticks([])
        if r == 0:
            ax[r, c].set_title(title, fontsize=9)
    ax[r, 0].set_ylabel(names[i], fontsize=8)
fig.tight_layout()
(out / "figures").mkdir(exist_ok=True)
fig.savefig(out / "figures" / f"real_detect_{cli.classname}.png", dpi=100)
