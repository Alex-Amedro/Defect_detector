"""Render a full turntable of a trained splat (mp4 + gif) for the README and the demo.

The camera orbits the object on a circle. We need to know which world axis points "up" for the object:
run once with --probe to get 6 views (one camera on each axis), look at them, then pass --up.

Usage (inside the WSL venv):
    python scripts/render_turntable.py --probe
    python scripts/render_turntable.py --up z
"""
import argparse
import json
import os
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--classname", default="01Gorilla")
parser.add_argument("--iterations", type=int, default=30000)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--repo", default=str(Path.home() / "work" / "SplatPose"))
parser.add_argument("--data", default=str(Path.home() / "data" / "MAD-Sim_3dgs"))
parser.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results" / "media"))
parser.add_argument("--probe", action="store_true", help="render 6 axis views to find the up axis, then stop")
parser.add_argument("--up", default="z", help="world up axis: x, y, z, optionally with a minus sign (-z)")
parser.add_argument("--frames", type=int, default=180)
parser.add_argument("--elevation", type=float, default=25.0, help="mean camera elevation in degrees")
parser.add_argument("--radius-scale", type=float, default=1.0, help="multiply the training camera distance")
cli = parser.parse_args()

sys.path.insert(0, cli.repo)
os.chdir(cli.repo)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from gaussian_splatting.train import ArgumentParser, ModelParams, PipelineParams  # noqa: E402
from gaussian_splatting.scene import Scene, GaussianModel  # noqa: E402
from gaussian_splatting.scene.cameras import Camera  # noqa: E402
from gaussian_splatting.gaussian_renderer import render  # noqa: E402

source = os.path.join(cli.data, cli.classname)
model_dir = os.path.join(cli.data, f"results_step0_{cli.seed}", cli.classname)
p = ArgumentParser()
lp, pp = ModelParams(p), PipelineParams(p)
args = p.parse_args(["-w", "--eval", "-s", source, "-m", model_dir, "--sh_degree", "0"])
gaussians = GaussianModel(0)
Scene(lp.extract(args), gaussians, load_iteration=cli.iterations, shuffle=False)
background = torch.ones(3, device="cuda")
pipe = pp.extract(args)

# Camera intrinsics and distance from the training cameras (the data are Blender-style cameras)
with open(os.path.join(source, "transforms_train.json")) as f:
    meta = json.load(f)
fov = meta["camera_angle_x"]
positions = np.array([np.array(fr["transform_matrix"])[:3, 3] for fr in meta["frames"]])
center = positions.mean(axis=0)
radius = np.linalg.norm(positions - center, axis=1).mean() * cli.radius_scale
SIZE = 800


def look_at_c2w(position, target, up):
    """Blender/OpenGL camera-to-world: the camera looks along its -z axis."""
    z = position - target
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    c2w = np.eye(4)
    c2w[:3, :3] = np.stack([x, y, z], axis=1)
    c2w[:3, 3] = position
    return c2w


def render_pose(c2w):
    c2w = c2w.copy()
    c2w[:3, 1:3] *= -1  # Blender/OpenGL axes -> COLMAP/OpenCV axes, as in the upstream data reader
    w2c = np.linalg.inv(c2w)
    cam = Camera(colmap_id=0, R=np.transpose(w2c[:3, :3]), T=w2c[:3, 3], FoVx=fov, FoVy=fov,
                 image=torch.ones(3, SIZE, SIZE), gt_alpha_mask=None, image_name="orbit", uid=0)
    with torch.no_grad():
        img = render(cam, gaussians, pipe, background)["render"].clamp(0, 1)
    return np.ascontiguousarray((img.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8))


AXES = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}


def axis_vector(name):
    sign = -1.0 if name.startswith("-") else 1.0
    return sign * np.array(AXES[name.lstrip("-")], dtype=float)


print(f"orbit center {np.round(center, 3)}, radius {radius:.3f}, fov {np.degrees(fov):.1f} deg")
os.makedirs(cli.out, exist_ok=True)

if cli.probe:
    tiles = []
    for name in ["x", "-x", "y", "-y", "z", "-z"]:
        pos = center + radius * axis_vector(name)
        up = AXES["y"] if name.lstrip("-") == "z" else AXES["z"]
        img = render_pose(look_at_c2w(pos, center, np.array(up, dtype=float)))
        cv2.putText(img, f"camera on {name}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (200, 0, 0), 3)
        tiles.append(cv2.resize(img, (400, 400)))
    sheet = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])])
    path = Path(cli.out) / "_probe_axes.png"
    Image.fromarray(sheet).save(path)
    print("saved", path)
    sys.exit(0)

up = axis_vector(cli.up)
frames = []
for i in range(cli.frames):
    azimuth = 2 * np.pi * i / cli.frames
    # the elevation oscillates a little so the top and the bottom of the object are both seen
    elevation = np.radians(cli.elevation + 12 * np.sin(azimuth))
    # build an orbit around the up axis: two axes perpendicular to it span the horizontal plane
    a = np.cross(up, [1.0, 0.0, 0.0] if abs(up[0]) < 0.9 else [0.0, 1.0, 0.0])
    a /= np.linalg.norm(a)
    b = np.cross(up, a)
    direction = np.cos(elevation) * (np.cos(azimuth) * a + np.sin(azimuth) * b) + np.sin(elevation) * up
    frames.append(render_pose(look_at_c2w(center + radius * direction, center, up)))

stem = Path(cli.out) / f"step1_{cli.classname}_turntable"
writer = cv2.VideoWriter(str(stem) + ".mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (SIZE, SIZE))
for fr in frames:
    writer.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
writer.release()

small = [Image.fromarray(fr).resize((400, 400), Image.LANCZOS) for fr in frames[::2]]
small[0].save(str(stem) + ".gif", save_all=True, append_images=small[1:], duration=66, loop=0, optimize=True)
print("saved", str(stem) + ".mp4 and .gif,", len(frames), "frames")
