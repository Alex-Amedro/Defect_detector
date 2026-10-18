"""Train one Gaussian splat on a MAD-Sim class and record time and peak VRAM.

The training loop itself is the upstream one (SplatPose / graphdeco 3DGS); this script only
sets the same arguments as SplatPose's train_and_render.py and adds measurements.

Usage (inside the WSL venv):
    python scripts/train_splat.py --classname 01Gorilla --iterations 30000
"""
import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--classname", default="01Gorilla")
parser.add_argument("--iterations", type=int, default=30000)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--repo", default=str(Path.home() / "work" / "SplatPose"))
parser.add_argument("--data", default=str(Path.home() / "data" / "MAD-Sim_3dgs"))
parser.add_argument("--source", help="scene folder (default: <data>/<classname>)")
parser.add_argument("--model-dir", help="where to write the splat (default: next to the data)")
parser.add_argument("--real", action="store_true",
                    help="real photos with COLMAP poses: colour depends on the viewing angle, every 8th photo held out")
parser.add_argument("--white", action="store_true", help="with --real: photos are on a white background (object only)")
parser.add_argument("--results", default=str(Path(__file__).resolve().parents[1] / "results"))
cli = parser.parse_args()

# The upstream code uses relative imports from its own repo root.
sys.path.insert(0, cli.repo)
os.chdir(cli.repo)

import torch  # noqa: E402
from gaussian_splatting.train import (  # noqa: E402
    ArgumentParser, ModelParams, OptimizationParams, PipelineParams, safe_state, training,
)

source = cli.source or os.path.join(cli.data, cli.classname)
model_dir = cli.model_dir or os.path.join(cli.data, f"results_step0_{cli.seed}", cli.classname)
os.makedirs(model_dir, exist_ok=True)

# MAD-Sim: same flags as SplatPose (white background, hold out test cameras, one colour per gaussian).
# Real photos: default Gaussian Splatting settings (view-dependent colour, every 8th photo held out).
flags = (["--eval", "--sh_degree", "3"] + (["-w"] if cli.white else [])) if cli.real else ["-w", "--eval", "--sh_degree", "0"]
argv = flags + ["-s", source, "-m", model_dir, "--iterations", str(cli.iterations),
                "--position_lr_max_steps", str(cli.iterations)]  # the position step must reach its end value too
p = ArgumentParser()
lp, op, pp = ModelParams(p), OptimizationParams(p), PipelineParams(p)
p.add_argument("--debug_from", type=int, default=-1)
args = p.parse_args(argv)


class VramMonitor(threading.Thread):
    """Samples the whole-device memory use once a second (includes the CUDA context)."""

    def __init__(self):
        super().__init__(daemon=True)
        self.min_free = torch.cuda.mem_get_info()[0]
        self.stop = threading.Event()

    def run(self):
        while not self.stop.is_set():
            self.min_free = min(self.min_free, torch.cuda.mem_get_info()[0])
            time.sleep(1.0)


total_bytes = torch.cuda.mem_get_info()[1]
gpu_name = torch.cuda.get_device_name(0)
torch.cuda.reset_peak_memory_stats()
monitor = VramMonitor()
monitor.start()

safe_state(False, cli.seed)
start = time.perf_counter()
training(lp.extract(args), op.extract(args), pp.extract(args),
         [cli.iterations], [cli.iterations], [], None, args.debug_from)
torch.cuda.synchronize()
seconds = time.perf_counter() - start
monitor.stop.set()

ply = Path(model_dir) / "point_cloud" / f"iteration_{cli.iterations}" / "point_cloud.ply"
with open(ply, "rb") as f:
    header = f.read(400).decode("latin-1")
n_gaussians = int(next(l.split()[-1] for l in header.splitlines() if l.startswith("element vertex")))

GB = 1024 ** 3
result = {
    "classname": cli.classname,
    "iterations": cli.iterations,
    "seed": cli.seed,
    "train_seconds": round(seconds, 1),
    "peak_vram_device_gb": round((total_bytes - monitor.min_free) / GB, 2),
    "peak_vram_torch_allocated_gb": round(torch.cuda.max_memory_allocated() / GB, 2),
    "peak_vram_torch_reserved_gb": round(torch.cuda.max_memory_reserved() / GB, 2),
    "n_gaussians": n_gaussians,
    "gpu": gpu_name,
    "torch": torch.__version__,
    "model_dir": model_dir,
}
os.makedirs(cli.results, exist_ok=True)
out = Path(cli.results) / f"step0_train_{cli.classname}_{cli.iterations}it.json"
out.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
