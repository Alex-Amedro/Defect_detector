"""Recover the camera poses of a set of photos with COLMAP (Structure from Motion), through pycolmap.

Output (in <work_dir>): sparse/0 (cameras, poses, 3D points) and images/ (distorted-free), the layout the
Gaussian Splatting code reads directly.

Usage (inside the WSL venv):
    python scripts/run_sfm.py data/object/images data/object/sfm
"""
import argparse
from pathlib import Path

import pycolmap

parser = argparse.ArgumentParser()
parser.add_argument("image_dir")
parser.add_argument("work_dir")
args = parser.parse_args()

work = Path(args.work_dir)
work.mkdir(parents=True, exist_ok=True)
db = work / "database.db"
raw = work / "raw"

# 1. find keypoints (SIFT) in every image; one camera shared by all photos (same phone, same zoom)
reader = pycolmap.ImageReaderOptions()
reader.camera_model = "SIMPLE_RADIAL"
pycolmap.extract_features(db, args.image_dir, camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader)
# 2. match keypoints between every pair of images (fine for a few hundred images)
pycolmap.match_exhaustive(db)
# 3. incremental reconstruction: register images one by one and triangulate 3D points
raw.mkdir(exist_ok=True)
models = pycolmap.incremental_mapping(db, args.image_dir, raw)
if not models:
    raise SystemExit("SfM failed: no model was reconstructed (object too plain? too few overlapping photos?)")
best_id = max(models, key=lambda k: models[k].num_reg_images())
best = models[best_id]
print(f"registered {best.num_reg_images()} images, {best.num_points3D()} 3D points, "
      f"mean reprojection error {best.compute_mean_reprojection_error():.2f} px")

# 4. remove lens distortion: the splatting code only understands ideal pinhole cameras
pycolmap.undistort_images(work, raw / str(best_id), args.image_dir)

# the splatting code reads sparse/0, pycolmap writes straight into sparse
sparse = work / "sparse"
(sparse / "0").mkdir(exist_ok=True)
for f in sparse.glob("*.bin"):
    f.rename(sparse / "0" / f.name)
print("undistorted images and sparse model written to", work)
