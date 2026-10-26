"""Cut the object out of every photo, then build a training scene that contains the object only.

1. Where is the object? The photographer aims at it, so the optical axes of all the cameras cross near it:
   the point closest to all the axes is the object centre.
2. Two cut-outs of every photo, inside a box around that centre: SAM (Segment Anything, Meta, Apache 2.0) finds the
   big uniform pieces, U2-Net (via rembg, MIT) finds the dark and small ones. Their union is the raw mask.
3. The raw masks disagree from one photo to the next (each one forgets different pieces). So the object is carved in 3D
   from the raw masks of the training photos (visual hull): a point stays if it is inside the mask in most of the photos
   that see it. The hull drawn in every photo is complete and consistent across photos (shadows, which move from photo
   to photo, disappear) but always a bit fat. So it guides SAM once more: sure points inside the object and sure points
   on the table around it, and SAM draws the true outline, kept inside the silhouette. Test photos get their silhouette
   from the training hull only, because a test photo may show a missing part and would carve it away.
4. Photos are pasted on a white background (like the synthetic data), and the hull points start the splat.

Usage (inside the WSL venv, after prepare_real.py):
    python scripts/segment_object.py --work ~/data/real/Bird
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import pycolmap
from PIL import Image, ImageDraw
from plyfile import PlyData, PlyElement
from rembg import new_session, remove
from scipy.ndimage import binary_closing, binary_dilation, binary_erosion, binary_fill_holes
from segment_anything import SamPredictor, sam_model_registry

parser = argparse.ArgumentParser()
parser.add_argument("--work", required=True)
parser.add_argument("--sam", default=str(Path.home() / "data" / "models" / "sam_vit_b.pth"))
parser.add_argument("--box-scale", type=float, default=2.0, help="the whole object is this many times bigger than the piece SAM finds from one point")
parser.add_argument("--agreement", type=float, default=0.7, help="share of the photos that must put a 3D point inside their mask")
parser.add_argument("--grid", type=int, default=160, help="voxels per side of the carving grid")
parser.add_argument("--margin", type=int, default=6, help="pixels added around the 3D silhouette, so thin parts are not cut")
parser.add_argument("--refine-margin", type=int, default=2, help="pixels added around the outline SAM draws")
parser.add_argument("--sheet", help="where to save a contact sheet of the masks, to check them by eye")
args = parser.parse_args()

work = Path(args.work).expanduser()
rec = pycolmap.Reconstruction(str(work / "sfm" / "sparse" / "0"))
value = lambda x: x() if callable(x) else x
images = sorted(rec.images.values(), key=lambda im: im.name)

# 1. object centre: least-squares point closest to every optical axis (the worst fifth of the axes is dropped)
centres = np.array([value(im.projection_center) for im in images])
axes = np.array([value(im.viewing_direction) for im in images])
axes /= np.linalg.norm(axes, axis=1, keepdims=True)
keep = np.ones(len(images), bool)
for _ in range(2):
    A = sum(np.eye(3) - np.outer(d, d) for d, k in zip(axes, keep) if k)
    b = sum((np.eye(3) - np.outer(d, d)) @ c for c, d, k in zip(centres, axes, keep) if k)
    centre = np.linalg.solve(A, b)
    miss = np.linalg.norm((np.eye(3) - axes[:, :, None] * axes[:, None, :]) @ (centre - centres)[:, :, None], axis=(1, 2))
    keep = miss <= np.percentile(miss, 80)
print("object centre", centre.round(3), "| median distance to the optical axes", np.median(miss).round(3))


def camera_point(image, point):
    pose = value(image.cam_from_world)
    return point @ pose.rotation.matrix().T + pose.translation


def pixel_of(image, point):
    p = camera_point(image, point)
    fx, fy, cx, cy = image.camera.params
    return fx * p[..., 0] / p[..., 2] + cx, fy * p[..., 1] / p[..., 2] + cy


# 2. raw masks
sam = sam_model_registry["vit_b"](checkpoint=args.sam).to("cuda")
predictor = SamPredictor(sam)
rembg_session = new_session("u2net")


def load(image):
    photo = np.asarray(Image.open(work / "sfm" / "images" / image.name).convert("RGB"))
    predictor.set_image(photo)
    return pixel_of(image, centre)


# pass 1, on a few photos: one point gives a piece of the object (one colour), good enough to size the object in 3D
sizes = []
for image in images[:: max(1, len(images) // 20)]:
    u, v = load(image)
    candidates, scores, _ = predictor.predict(point_coords=np.array([[u, v]]), point_labels=np.array([1]), multimask_output=True)
    rows, cols = np.nonzero(candidates[int(np.argmax(scores))])
    sizes.append(max(np.ptp(rows), np.ptp(cols)) / 2 * camera_point(image, centre)[2] / image.camera.params[0])
half_size_3d = args.box_scale * np.median(sizes)
print(f"object half-size in 3D: {half_size_3d:.3f} (a piece measures {np.median(sizes):.3f})")

# pass 2, on every photo: SAM with a box, U2-Net on a wider crop, union
raw = {}
for image in images:
    u, v = load(image)
    r = half_size_3d * image.camera.params[0] / camera_point(image, centre)[2]
    candidates, _, _ = predictor.predict(box=np.array([u - r, v - r, u + r, v + r]), multimask_output=False)
    mask = candidates[0].copy()
    x0, y0, x1, y1 = (int(round(t)) for t in (u - 1.3 * r, v - 1.3 * r, u + 1.3 * r, v + 1.3 * r))
    photo = Image.open(work / "sfm" / "images" / image.name).convert("RGB")
    salient = np.asarray(remove(photo.crop((x0, y0, x1, y1)).resize((320, 320)), session=rembg_session, only_mask=True)) > 127
    salient = np.asarray(Image.fromarray(salient.astype(np.uint8) * 255).resize((x1 - x0, y1 - y0))) > 127
    h, w = mask.shape
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    mask[sy0:sy1, sx0:sx1] |= salient[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0]
    raw[image.name] = binary_fill_holes(binary_closing(mask, iterations=3))

# 3. carve the object in 3D from the raw masks of the training photos
train = [im for im in images if im.name.startswith("train_")]
half_px = [max(np.ptp(np.nonzero(raw[im.name])[0]), np.ptp(np.nonzero(raw[im.name])[1])) / 2 for im in train]
radius = 1.5 * np.median([h * camera_point(im, centre)[2] / im.camera.params[0] for h, im in zip(half_px, train)])
axis = np.linspace(-radius, radius, args.grid, dtype=np.float32)
voxels = (centre + np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)).astype(np.float32)
inside, seen = np.zeros(len(voxels), np.float32), np.zeros(len(voxels), np.float32)
colour = np.zeros((len(voxels), 3), np.float32)
for image in train:
    u, v = pixel_of(image, voxels)
    h, w = raw[image.name].shape
    visible = (camera_point(image, voxels)[:, 2] > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    idx = np.where(visible)[0]
    ui, vi = u[idx].astype(int), v[idx].astype(int)
    hit = raw[image.name][vi, ui]
    seen[idx] += 1
    inside[idx[hit]] += 1
    photo = np.asarray(Image.open(work / "sfm" / "images" / image.name).convert("RGB"), dtype=np.float32)
    colour[idx[hit]] += photo[vi[hit], ui[hit]]
hull = (seen >= 5) & (inside >= args.agreement * seen)
hull_points, hull_colour = voxels[hull], colour[hull] / inside[hull][:, None]
print(f"visual hull: {hull.sum()} voxels of {len(voxels)} (cube half-size {radius:.3f}, voxel {2 * radius / args.grid:.4f})")

# the hull drawn in every photo: a complete silhouette, but always a bit fat
fat, overlap = {}, []
for image in images:
    h, w = raw[image.name].shape
    u, v = pixel_of(image, hull_points)
    ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    m = np.zeros((h, w), bool)
    m[v[ok].astype(int), u[ok].astype(int)] = True
    fat[image.name] = binary_fill_holes(binary_closing(binary_dilation(m, iterations=args.margin), iterations=3))
    if image.name.startswith("train_"):
        overlap.append((fat[image.name] & raw[image.name]).sum() / max(1, raw[image.name].sum()))
print(f"share of the raw mask that the hull keeps (training photos): median {np.median(overlap):.2f}")


def sample(region, n):
    points = np.argwhere(region)  # (row, column)
    return points[rng.choice(len(points), n, replace=len(points) < n)][:, ::-1]  # (x, y), as SAM wants


# the fat silhouette guides SAM: sure points inside the object (the core of the silhouette) and sure points on the
# table (a ring just outside it). SAM draws the true outline, the silhouette stops it from spreading.
(work / "masks").mkdir(exist_ok=True)
(work / "seg_images").mkdir(exist_ok=True)
rng = np.random.default_rng(1)
masks, areas, fallbacks = {}, [], 0
for image in images:
    silhouette = fat[image.name]
    load(image)
    rows, cols = np.nonzero(silhouette)
    size = max(np.ptp(rows), np.ptp(cols))
    core = binary_erosion(silhouette, iterations=max(3, int(0.15 * size)))
    ring = binary_dilation(silhouette, iterations=max(10, int(0.3 * size))) & ~binary_dilation(silhouette, iterations=3)
    n = 12
    points = np.concatenate([sample(core if core.any() else silhouette, n), sample(ring, n)]).astype(float)
    # the silhouette itself is the starting mask (SAM wants 256 x 256 logits): SAM keeps what it covers and
    # moves the border onto the true outline
    start = np.asarray(Image.fromarray(silhouette.astype(np.uint8) * 255).resize((256, 256), Image.BILINEAR), np.float32) / 255
    out, _, _ = predictor.predict(point_coords=points, point_labels=np.array([1] * n + [0] * n),
                                  box=np.array([cols.min(), rows.min(), cols.max(), rows.max()]),
                                  mask_input=((start * 2 - 1) * 8)[None], multimask_output=False)
    m = binary_dilation(binary_fill_holes(binary_closing(out[0], iterations=2)), iterations=args.refine_margin) & silhouette
    if m.sum() < 0.4 * silhouette.sum():  # SAM lost the object: keep the silhouette
        m, fallbacks = silhouette, fallbacks + 1
    masks[image.name] = m
    areas.append(m.mean())
    Image.fromarray(m.astype(np.uint8) * 255).save(work / "masks" / (Path(image.name).stem + ".png"))
    photo = np.asarray(Image.open(work / "sfm" / "images" / image.name).convert("RGB")).copy()
    photo[~m] = 255
    Image.fromarray(photo).save(work / "seg_images" / image.name, quality=95)
print(f"{len(masks)} final masks ({fallbacks} fell back to the silhouette); share of the photo: "
      f"median {np.median(areas):.3f}, min {min(areas):.3f}, max {max(areas):.3f}; "
      f"share of the silhouette kept: median {np.median([masks[k].sum() / fat[k].sum() for k in masks]):.2f}")

# 4. training scene: same cameras, white-background photos, hull points as the starting point cloud
scene = work / "seg_scene"
shutil.rmtree(scene / "sparse", ignore_errors=True)
shutil.copytree(work / "train_scene" / "sparse" / "0", scene / "sparse" / "0")
if len(hull_points) > 150_000:
    pick = np.random.default_rng(0).choice(len(hull_points), 150_000, replace=False)
    hull_points, hull_colour = hull_points[pick], hull_colour[pick]
fields = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
          ("red", "u1"), ("green", "u1"), ("blue", "u1")]
vertices = np.empty(len(hull_points), dtype=fields)
vertices[:] = list(map(tuple, np.concatenate([hull_points, np.zeros_like(hull_points), hull_colour.clip(0, 255)], axis=1)))
PlyData([PlyElement.describe(vertices, "vertex")]).write(str(scene / "sparse" / "0" / "points3D.ply"))
if not (scene / "images").exists():
    (scene / "images").symlink_to(work / "seg_images")

if args.sheet:  # contact sheet to check the masks by eye: magenta filling
    shown = images[:: max(1, len(images) // 15)][:15]
    tiles = []
    for image in shown:
        photo = np.asarray(Image.open(work / "sfm" / "images" / image.name).convert("RGB")).astype(float)
        m = masks[image.name]
        photo[m] = 0.5 * photo[m] + 0.5 * np.array([255, 0, 255])
        t = Image.fromarray(photo.astype(np.uint8)).resize((256, 256))
        ImageDraw.Draw(t).text((4, 4), image.name, fill=(0, 0, 255))
        tiles.append(np.asarray(t))
    Image.fromarray(np.vstack([np.hstack(tiles[i:i + 5]) for i in range(0, 15, 5)])).save(args.sheet)
