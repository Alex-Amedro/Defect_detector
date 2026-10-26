"""Prepare a real-photo object (MAD-Real layout) for the splat: resize, find all camera poses with SfM, and build
a training scene that contains only the healthy training photos.

Test photos take part in the SfM only to get their camera pose; the splat never sees them.

Usage (inside the WSL venv):
    python scripts/prepare_real.py --src ~/data/MAD-Real/Bear --out ~/data/real/Bear
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import pycolmap
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--src", required=True, help="object folder with train/good, test/<type>, ground_truth/<type>")
parser.add_argument("--out", required=True)
parser.add_argument("--size", type=int, default=1024)
args = parser.parse_args()

src, out = Path(args.src).expanduser(), Path(args.out).expanduser()
images = out / "images_all"
images.mkdir(parents=True, exist_ok=True)

# 1. resize every photo and give it a name that says where it comes from
meta = {}
folders = [("train", src / "train" / "good")] + [(t.name, t) for t in sorted((src / "test").iterdir())]
for kind, folder in folders:
    for i, f in enumerate(sorted(folder.glob("*.jpg"), key=lambda p: int(p.stem))):
        name = f"{kind}_{i:03d}.jpg"
        im = Image.open(f).convert("RGB")
        im.thumbnail((args.size, args.size))
        im.save(images / name, quality=95)
        mask = src / "ground_truth" / kind / f"{f.stem}.png"
        meta[name] = {"kind": kind, "mask": str(mask) if mask.exists() else None}
(out / "meta.json").write_text(json.dumps(meta, indent=1))
print(len(meta), "photos resized to", args.size, "px")

# 2. camera poses of all photos together (skipped if already done)
if not (out / "sfm" / "sparse" / "0" / "images.bin").exists():
    subprocess.run([sys.executable, str(Path(__file__).with_name("run_sfm.py")), str(images), str(out / "sfm")], check=True)

# 3. training scene: same cameras and 3D points, but only the healthy training photos are kept
rec = pycolmap.Reconstruction(str(out / "sfm" / "sparse" / "0"))
for image in list(rec.images.values()):
    if not image.name.startswith("train_"):
        rec.deregister_frame(image.frame_id)  # pycolmap 4 removes a photo together with its frame
scene = out / "train_scene"
(scene / "sparse" / "0").mkdir(parents=True, exist_ok=True)
rec.write(str(scene / "sparse" / "0"))
if not (scene / "images").exists():
    (scene / "images").symlink_to(out / "sfm" / "images")
registered = {n for n in meta if (out / "sfm" / "images" / n).exists()}
print(f"training scene: {rec.num_reg_images()} photos; undistorted photos on disk: {len(registered)} of {len(meta)}")
