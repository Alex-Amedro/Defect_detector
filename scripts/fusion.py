"""Multi-view fusion of 2D anomaly maps into a 3D voxel grid.

Idea: a real defect sits at a fixed place on the object, so it lights up at the same 3D position in every view
that sees it. A reflection or a shadow moves with the camera, so it lights up at different 3D positions.
Each pixel of an anomaly map is lifted to 3D using the depth of the splat, and the voxels vote.

Conventions: cameras are the Gaussian Splatting `Camera` objects (COLMAP axes: x right, y down, z forward).
cam.R is the camera-to-world rotation and cam.T the translation of the world-to-camera transform, so
    camera point = R^T @ world point + T        world point = R @ (camera point - T)
"""
import math

import torch
import torch.nn.functional as F
from gaussian_splatting.gaussian_renderer import render


def _pose(cam):
    return (torch.as_tensor(cam.R, dtype=torch.float32, device="cuda"),
            torch.as_tensor(cam.T, dtype=torch.float32, device="cuda"))


def render_depth(cam, gaussians, pipe):
    """Depth seen from `cam` (distance along the optical axis) and opacity (about 1 on the object, 0 outside).

    The rasterizer only renders colours, so each gaussian is painted with its own depth and the result is
    divided by the rendered opacity (a weighted average of depths along each ray).
    """
    R, T = _pose(cam)
    z = gaussians.get_xyz.detach() @ R[:, 2] + T[2]  # third coordinate of R^T x + T
    black = torch.zeros(3, device="cuda")
    with torch.no_grad():
        weighted_depth = render(cam, gaussians, pipe, black, override_color=z[:, None].expand(-1, 3).contiguous())["render"][0]
        alpha = render(cam, gaussians, pipe, black, override_color=torch.ones_like(z)[:, None].expand(-1, 3).contiguous())["render"][0]
    return weighted_depth / alpha.clamp_min(1e-6), alpha


def backproject(depth, alpha, cam, fov, size, min_alpha=0.9):
    """Lift every pixel of a size x size map to a 3D world point. Returns (size, size, 3) points and a validity mask."""
    depth = F.interpolate(depth[None, None], size=(size, size), mode="nearest")[0, 0]
    alpha = F.interpolate(alpha[None, None], size=(size, size), mode="nearest")[0, 0]
    f = size / (2 * math.tan(fov / 2))  # focal length in pixels
    v, u = torch.meshgrid(torch.arange(size, device="cuda") + 0.5, torch.arange(size, device="cuda") + 0.5, indexing="ij")
    cam_points = torch.stack([(u - size / 2) / f * depth, (v - size / 2) / f * depth, depth], dim=-1)
    R, T = _pose(cam)
    return (cam_points - T) @ R.T, alpha > min_alpha  # row-vector form of R @ (camera point - T)


def project(points, cam, fov, size):
    """World points (N, 3) to pixel coordinates (u, v) and depth z in `cam`."""
    R, T = _pose(cam)
    c = points @ R + T  # row-vector form of R^T @ point + T
    f = size / (2 * math.tan(fov / 2))
    return f * c[:, 0] / c[:, 2] + size / 2, f * c[:, 1] / c[:, 2] + size / 2, c[:, 2]


class DefectGrid:
    """Voxel grid where each view votes. A view votes once per voxel, so a view that sees a voxel from many pixels
    does not outweigh the others."""

    def __init__(self, center, half_size, voxel):
        self.origin = torch.as_tensor(center, dtype=torch.float32, device="cuda") - half_size
        self.voxel = voxel
        self.n = int(round(2 * half_size / voxel))
        shape = (self.n,) * 3
        self.seen = torch.zeros(shape, device="cuda")  # views that saw the surface inside this voxel
        self.hits = torch.zeros(shape, device="cuda")  # views whose anomaly score was above the threshold there

    def add_view(self, points, scores, threshold):
        """points (N, 3) visible surface points of one view, scores (N,) anomaly score of the matching pixels."""
        ijk = torch.floor((points - self.origin) / self.voxel).long()
        inside = ((ijk >= 0) & (ijk < self.n)).all(dim=1)
        ijk, scores = ijk[inside], scores[inside]
        flat = (ijk[:, 0] * self.n + ijk[:, 1]) * self.n + ijk[:, 2]
        best = torch.zeros(self.n ** 3, device="cuda").scatter_reduce(0, flat, scores, "amax", include_self=True)
        touched = torch.zeros(self.n ** 3, dtype=torch.bool, device="cuda")
        touched[flat] = True
        self.seen += touched.view_as(self.seen)
        self.hits += (touched & (best > threshold)).view_as(self.hits)

    def confirmed(self, min_views, min_ratio):
        """Voxels flagged by at least `min_views` views and by at least `min_ratio` of the views that saw them."""
        return (self.hits >= min_views) & (self.hits >= min_ratio * self.seen)

    def voxel_centers(self, mask):
        return (mask.nonzero().float() + 0.5) * self.voxel + self.origin
