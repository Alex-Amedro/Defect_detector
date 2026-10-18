"""Pick the sharpest frames of a video: the video is cut into equal windows and each window keeps its sharpest frame.

Usage (inside the WSL venv):
    python scripts/extract_frames.py capture.mp4 data/object/images --count 150
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("video")
parser.add_argument("out_dir")
parser.add_argument("--count", type=int, default=150, help="number of frames to keep")
args = parser.parse_args()


def sharpness(frame):
    # variance of the Laplacian: high when edges are crisp, low when the frame is blurred
    return cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()


cap = cv2.VideoCapture(args.video)
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
count = min(args.count, n_frames)
edges = np.linspace(0, n_frames, count + 1).astype(int)

out = Path(args.out_dir)
out.mkdir(parents=True, exist_ok=True)
kept = []
for i in range(count):
    best, best_score = None, -1.0
    for _ in range(edges[i], edges[i + 1]):
        ok, frame = cap.read()
        if not ok:
            break
        score = sharpness(frame)
        if score > best_score:
            best, best_score = frame, score
    if best is not None:
        cv2.imwrite(str(out / f"frame_{i:04d}.jpg"), best, [cv2.IMWRITE_JPEG_QUALITY, 95])
        kept.append(best_score)

print(f"{len(kept)} frames kept out of {n_frames}; sharpness min {min(kept):.0f}, median {np.median(kept):.0f}")
