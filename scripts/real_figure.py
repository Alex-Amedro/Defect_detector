"""Figure for the README: real photos with the defect outlined, next to the healthy 3D model drawn from the same camera.

The outline is the defect mask of the dataset at full resolution, put in the geometry of the undistorted photos the same
way real_detect.py does. The photos are shown whole, without cropping. With --map a third column shows the anomaly map
of that photo alone: it is computed on a window around the object and placed back at its position, black elsewhere.
Run real_check.py first (it writes the drawings of the 3D copy), and real_detect.py for --map. Usage inside the WSL venv:
    python scripts/real_figure.py Bird:Missing_006.jpg Bear:Stains_010.jpg
"""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib.cm as cm
import numpy as np
import pycolmap
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_dilation, binary_erosion

parser = argparse.ArgumentParser()
parser.add_argument("photos", nargs="+", help="one object:photo per row, e.g. Bird:Missing_006.jpg")
parser.add_argument("--data", default=str(Path.home() / "data" / "real"))
parser.add_argument("--size", type=int, default=400, help="size of each image in pixels")
parser.add_argument("--map", action="store_true", help="add a third column with the anomaly map of the photo")
parser.add_argument("--out", help="default: results/figures/real_photo_model.jpg (real_photo_model_map.jpg with --map)")
cli = parser.parse_args()
if cli.out is None:
    cli.out = str(Path(__file__).resolve().parents[1] / "results" / "figures" / ("real_photo_model_map.jpg" if cli.map else "real_photo_model.jpg"))

try:
    font = ImageFont.load_default(size=22)
except TypeError:  # older Pillow
    font = ImageFont.load_default()
RED = (230, 20, 20)
LABELS = {"Bird": "Bird: missing beak", "Bear": "Bear: stain"}


def dataset_mask(work, meta, name):
    """The defect mask of the dataset in the geometry of the undistorted photos."""
    rec = pycolmap.Reconstruction(str(work / "sfm" / "sparse" / "0"))
    raw = pycolmap.Reconstruction(str(work / "sfm" / "raw" / "0"))
    old, new = next(iter(raw.cameras.values())), next(iter(rec.cameras.values()))
    f, cx, cy, k = old.params
    K_old = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
    K_new = np.array([[new.params[0], 0, new.params[2]], [0, new.params[1], new.params[3]], [0, 0, 1]])
    map_x, map_y = cv2.initUndistortRectifyMap(K_old, np.array([k, 0, 0, 0.0]), None, K_new, (new.width, new.height), cv2.CV_32FC1)
    mask = np.asarray(Image.open(meta[name]["mask"]).resize((old.width, old.height), Image.NEAREST)) > 0
    return cv2.remap(mask.astype(np.uint8), map_x, map_y, cv2.INTER_NEAREST) > 0


def fit(image):
    return np.asarray(Image.fromarray(image).resize((cli.size, cli.size), Image.LANCZOS))


def map_in_photo(anomaly_map, box, shape):
    """The map of the window `box` (x0, y0, x1, y1) put back in a black image of the size of the photo."""
    x0, y0, x1, y1 = [int(v) for v in box]
    window = np.asarray(Image.fromarray(anomaly_map.astype(np.float32)).resize((x1 - x0, y1 - y0), Image.BILINEAR))
    full = np.zeros(shape, np.float32)
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x1, shape[1]), min(y1, shape[0])
    full[sy0:sy1, sx0:sx1] = window[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0]
    return (cm.inferno(np.clip(full, 0, 1))[..., :3] * 255).round().astype(np.uint8)


def images(obj, name):
    work = Path(cli.data) / obj
    meta = json.loads((work / "meta.json").read_text())
    stored = np.load(work / "renders.npz")
    drawn = dict(zip(stored["names"], stored["renders"]))[name]
    photo = np.asarray(Image.open(work / "sfm" / "images" / name).convert("RGB"))
    defect = dataset_mask(work, meta, name)
    outline = binary_dilation(defect & ~binary_erosion(defect, iterations=3), iterations=2)
    outlined = photo.copy()
    outlined[outline] = RED
    panels = [fit(outlined), fit(drawn)]
    if cli.map:
        found = np.load(work / "detect.npz")
        i = list(map(str, found["names"])).index(name)
        panels.append(fit(map_in_photo(found["maps"][i], found["boxes"][i], photo.shape[:2])))
    return panels


def side_label(text):
    strip = Image.new("RGB", (cli.size, 40), (255, 255, 255))
    ImageDraw.Draw(strip).text((10, 6), text, fill=(60, 60, 60), font=font)
    return np.asarray(strip.rotate(90, expand=True))


rows = []
for pick in cli.photos:
    obj, name = pick.split(":")
    rows.append(np.concatenate([side_label(LABELS.get(obj, obj))] + images(obj, name), axis=1))
header = Image.new("RGB", (rows[0].shape[1], 40), (255, 255, 255))
for k, title in enumerate(["photo, defect outlined", "healthy 3D model", "anomaly map (one photo)"][:3 if cli.map else 2]):
    ImageDraw.Draw(header).text((40 + k * cli.size + 10, 8), title, fill=(60, 60, 60), font=font)
out = Path(cli.out)
out.parent.mkdir(parents=True, exist_ok=True)
Image.fromarray(np.concatenate([np.asarray(header)] + rows, axis=0)).save(out, quality=90)
print("saved", out)
