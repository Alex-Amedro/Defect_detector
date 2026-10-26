"""Step 1: run SplatPose on the test images of one MAD-Sim class and measure it with our own metric code.

Pipeline, per test image (steps 1-2 are upstream SplatPose code, called unchanged):
  1. coarse pose: LoFTR matches the photo against the training photos, the best match gives a start pose
  2. pose refinement: 175 gradient steps that make the splat render match the photo
  3. anomaly map: EfficientNet features of the photo vs the render, squared difference, blurred
  4. metrics: image AUROC, pixel AUROC, AUPRO (MVTec reference code), plus a per-defect-type breakdown

Credits: step 3 follows the scoring recipe of SplatPose's train_and_render.py (itself taken from the PAD / MAD
evaluation code); AUPRO is the MVTec reference implementation shipped in SplatPose's aupro.py.

Poses are computed in chunks of images and cached on disk, so a run can be stopped and resumed.

Usage (inside the WSL venv):
    python scripts/evaluate.py --classname 01Gorilla --tag full
    python scripts/evaluate.py --classname 01Gorilla --raw ~/data/MAD-Sim_mini --tag mini
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--classname", default="01Gorilla")
parser.add_argument("--tag", default="full", help="name of this run (cache and result file names)")
parser.add_argument("--raw", default=str(Path.home() / "data" / "MAD-Sim"), help="raw MAD-Sim folder")
parser.add_argument("--data", default=str(Path.home() / "data" / "MAD-Sim_3dgs"))
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--k", type=int, default=175, help="pose refinement steps (SplatPose default)")
parser.add_argument("--chunk", type=int, default=32, help="max test images per cached chunk")
parser.add_argument("--types", default="", help="comma-separated test folders to keep, e.g. good,Burrs (default: all)")
parser.add_argument("--per-type", type=int, default=0, help="keep only the first N images of each type (quick tests)")
parser.add_argument("--repo", default=str(Path.home() / "work" / "SplatPose"))
parser.add_argument("--cache", default=str(Path.home() / "data" / "cache"))
parser.add_argument("--results", default=str(Path(__file__).resolve().parents[1] / "results"))
cli = parser.parse_args()

sys.path.insert(0, cli.repo)
os.chdir(cli.repo)  # upstream code uses paths relative to its repo root

sys.path.insert(1, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from scoring import SIZE as SCORE_SIZE, anomaly_maps  # noqa: E402

CLASS = cli.classname
RAW = Path(cli.raw).expanduser() / CLASS
MODEL_DIR = os.path.join(cli.data, f"results_step0_{cli.seed}", CLASS)
CACHE = Path(cli.cache) / CLASS / cli.tag
CACHE.mkdir(parents=True, exist_ok=True)


def list_test_images():
    """All test images as (defect_type, filename), in the order the upstream dataset class reads them."""
    keep = set(cli.types.split(",")) if cli.types else None
    items = []
    for dtype in sorted(os.listdir(RAW / "test")):
        if keep and dtype not in keep:
            continue
        names = sorted(os.listdir(RAW / "test" / dtype))
        items += [(dtype, n) for n in (names[:cli.per_type] if cli.per_type else names)]
    return items


def build_chunk_dataset(items, root):
    """A small MAD-Sim-shaped folder (symlinks) containing only the given test images."""
    shutil.rmtree(root, ignore_errors=True)
    cls = root / CLASS
    (cls / "test").mkdir(parents=True)
    shutil.copy(RAW / "transforms.json", cls / "transforms.json")
    os.symlink(RAW / "train", cls / "train")
    for dtype, name in items:
        (cls / "test" / dtype).mkdir(exist_ok=True)
        os.symlink(RAW / "test" / dtype / name, cls / "test" / dtype / name)
        if dtype != "good":
            (cls / "ground_truth" / dtype).mkdir(parents=True, exist_ok=True)
            mask = name[:-4] + "_mask.png"
            os.symlink(RAW / "ground_truth" / dtype / mask, cls / "ground_truth" / dtype / mask)
    return str(root) + "/"


def to_u8(x):
    return (x.clamp(0, 1) * 255).round().to(torch.uint8).numpy()


def record_poses():
    """Hook two upstream helpers so the final pose of every image can be recovered.

    Upstream keeps the camera at the coarse pose and moves the gaussians by a rigid motion (R_s, T_s).
    Seen from the camera: x_cam = R0^T (R_s^T x + T_s) + T0, i.e. an equivalent camera with
    R = R_s R0 and T = R0^T T_s + T0 on the untouched gaussians (Gaussian Splatting convention, see fusion.py).
    """
    import pose_estimation
    coarse, transforms_ = [], []
    retrieval = pose_estimation.pose_retrieval_loftr
    transf_class = pose_estimation.camera_transf

    def retrieval_hook(*args, **kwargs):
        c2w = retrieval(*args, **kwargs)
        coarse.append(np.array(c2w))
        return c2w

    class TransfHook(transf_class):
        def __init__(self):
            super().__init__()
            transforms_.append(self)

    pose_estimation.pose_retrieval_loftr = retrieval_hook
    pose_estimation.camera_transf = TransfHook

    def final_poses():
        R_out, T_out = [], []
        for c2w, transf in zip(coarse, transforms_):
            c2w = c2w.copy()
            c2w[:3, 1:3] *= -1  # Blender axes -> COLMAP axes, same two lines as upstream
            w2c = np.linalg.inv(c2w)
            R0, T0 = w2c[:3, :3].T, w2c[:3, 3]
            with torch.no_grad():
                Rs, Ts, _ = transf()
            Rs, Ts = Rs.cpu().numpy().astype(np.float64), Ts.cpu().numpy().astype(np.float64)
            R_out.append(Rs @ R0)
            T_out.append(R0.T @ Ts + T0)
        coarse.clear(), transforms_.clear()
        return np.stack(R_out), np.stack(T_out)

    return final_poses


def compute_poses():
    """Stage 1+2 on every chunk not yet cached. Returns the list of chunk files."""
    from pose_estimation import main_pose_estimation
    final_poses = record_poses()

    items = list_test_images()
    n_chunks = -(-len(items) // cli.chunk)
    chunks = [list(c) for c in np.array_split(np.arange(len(items)), n_chunks)]
    # upstream prints the first 5 images, so it needs at least 5 per chunk
    assert min(len(c) for c in chunks) >= 5, "chunks too small, raise --chunk"
    files = []
    for ci, idx in enumerate(chunks):
        path = CACHE / f"chunk_{ci:02d}.npz"
        files.append(path)
        if path.exists():
            print(f"chunk {ci + 1}/{n_chunks}: cached")
            continue
        chunk_items = [items[i] for i in idx]
        tmp = Path(cli.cache) / "_tmp_chunk"
        data_dir = build_chunk_dataset(chunk_items, tmp)
        print(f"chunk {ci + 1}/{n_chunks}: {len(chunk_items)} images", flush=True)
        t0 = time.perf_counter()
        photos, renders, _, masks, times = main_pose_estimation(
            cur_class=CLASS, model_dir_location=MODEL_DIR, k=cli.k, verbose=False, data_dir=data_dir)
        wall = time.perf_counter() - t0
        pose_R, pose_T = final_poses()
        np.savez(
            path,
            pose_R=pose_R,
            pose_T=pose_T,
            photo=np.stack([to_u8(p) for p in photos]),
            render=np.stack([r.clamp(0, 1).numpy().astype(np.float16) for r in renders]),
            raw_render_max=np.array([float(r.max()) for r in renders]),
            mask=np.stack([(m > 0).astype(np.uint8) for m in masks]),
            dtype=np.array([d for d, _ in chunk_items]),
            name=np.array([n for _, n in chunk_items]),
            refine_ms=np.array(times),
            wall_s=np.array([wall]),
        )
        shutil.rmtree(tmp, ignore_errors=True)
    return files


def load_chunks(files):
    parts = [dict(np.load(f)) for f in files]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0] if k != "wall_s"}, \
        float(sum(p["wall_s"][0] for p in parts))


def resize_masks(masks):
    return np.stack([np.asarray(Image.fromarray(m * 255).resize((SCORE_SIZE, SCORE_SIZE), Image.NEAREST)) > 0
                     for m in masks]).astype(np.uint8)


def metrics(maps, gt, dtype):
    from aupro import calculate_au_pro_au_roc  # MVTec reference code, shipped with SplatPose
    img_score = maps.reshape(len(maps), -1).max(axis=1)
    is_bad = dtype != "good"
    out = {
        "n_images": int(len(maps)),
        "n_good": int((~is_bad).sum()),
        "n_anomalous": int(is_bad.sum()),
        # tiny defects can vanish when the 800 px mask is shrunk to 224 px (nearest neighbour, as in the paper protocol)
        "n_anomalous_with_empty_mask_after_resize": int((is_bad & (gt.reshape(len(gt), -1).sum(axis=1) == 0)).sum()),
        "image_auroc":float(roc_auc_score(is_bad, img_score)),
        "pixel_auroc": float(roc_auc_score(gt.ravel(), maps.ravel())),
        "aupro_fpr_0.3": float(calculate_au_pro_au_roc(gt, maps)[0]),
        "image_auroc_by_type": {},
    }
    # With few good images the image AUROC is uncertain: resample good and anomalous images separately.
    rng = np.random.default_rng(0)
    good_idx, bad_idx = np.where(~is_bad)[0], np.where(is_bad)[0]
    boot = []
    for _ in range(1000):
        sel = np.concatenate([rng.choice(good_idx, len(good_idx)), rng.choice(bad_idx, len(bad_idx))])
        boot.append(roc_auc_score(is_bad[sel], img_score[sel]))
    out["image_auroc_ci95_bootstrap"] = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]

    good = ~is_bad
    for t in sorted(set(dtype) - {"good"}):
        sel = good | (dtype == t)
        out["image_auroc_by_type"][t] = float(roc_auc_score(dtype[sel] != "good", img_score[sel]))
    return out


def heatmap_figure(data, maps, gt, path):
    """Best and worst localisation per defect type, plus the good image with the highest false alarm."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dtype = data["dtype"]
    # images whose mask is empty after the resize cannot be ranked: put them last
    margin = np.array([maps[i][gt[i] > 0].mean() - maps[i][gt[i] == 0].mean() if gt[i].any() else -np.inf
                       for i in range(len(maps))])
    rows = []
    for t in sorted(set(dtype) - {"good"}):
        idx = np.where((dtype == t) & np.isfinite(margin))[0]  # only images that still have a mask
        order = idx[np.argsort(margin[idx])]
        rows += [(order[-1], f"{t}: best"), (order[0], f"{t}: worst")]
    good = np.where(dtype == "good")[0]
    if len(good):
        rows.append((good[np.argmax(maps[good].reshape(len(good), -1).max(axis=1))], "good: highest false alarm"))

    fig, axes = plt.subplots(len(rows), 4, figsize=(11, 2.7 * len(rows)))
    for r, (i, title) in enumerate(rows):
        show = [data["photo"][i].transpose(1, 2, 0), data["render"][i].astype(np.float32).transpose(1, 2, 0),
                maps[i], gt[i]]
        for c, (img, name) in enumerate(zip(show, ["photo", "splat render (healthy)", "anomaly map", "ground truth"])):
            axes[r, c].imshow(img, cmap="inferno" if c == 2 else "gray" if c == 3 else None, vmin=0 if c >= 2 else None,
                              vmax=1 if c >= 2 else None)
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(name, fontsize=9)
        axes[r, 0].set_ylabel(title, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)


if __name__ == "__main__":
    t_start = time.perf_counter()
    files = compute_poses()
    data, pose_wall_s = load_chunks(files)
    torch.cuda.empty_cache()
    maps = anomaly_maps(data["photo"], data["render"])
    np.save(CACHE / "maps.npy", maps.astype(np.float16))  # same order as the chunk files, used by fusion_experiment.py
    gt = resize_masks(data["mask"])
    res = metrics(maps, gt, data["dtype"])
    res.update({
        "classname": CLASS, "tag": cli.tag, "refine_steps_k": cli.k,
        "pose_seconds_per_image_total": round(pose_wall_s / len(maps), 2),
        "refine_ms_per_image_median": round(float(np.median(data["refine_ms"])), 1),
        "render_max_before_clamp_gt_1": int((data["raw_render_max"] > 1).sum()),
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
    })
    out = Path(cli.results)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"step1_{CLASS}_{cli.tag}.json").write_text(json.dumps(res, indent=2))
    heatmap_figure(data, maps, gt, out / "figures" / f"step1_{CLASS}_{cli.tag}_heatmaps.png")
    print(json.dumps(res, indent=2))
