#!/usr/bin/env bash
# One-time setup of the WSL2 (Ubuntu 22.04) environment.
# Usage: scripts/setup_wsl.sh <base|cuda|python|all>
# Every stage can be interrupted (Ctrl+C) and rerun: finished work is skipped.
set -euo pipefail

CUDA_VERSION="12-8"
VENV="$HOME/venvs/splat"

# apt needs root; plain `sudo` is a no-op when we are already root.
as_root() { if [ "$(id -u)" -eq 0 ]; then "$@"; else sudo "$@"; fi; }

stage_base() {
  # Compiler and build tools: the Gaussian Splatting rasterizer is C++/CUDA code compiled on install.
  as_root apt-get update -qq
  as_root apt-get install -y --no-install-recommends \
    build-essential python3-pip python3-venv python3-dev cmake ninja-build unzip make
}

stage_cuda() {
  # The Windows driver already provides the GPU to WSL. Only the compiler (nvcc) and headers are missing.
  # Blackwell cards (RTX 50xx, sm_120) need CUDA >= 12.8, which is why the old CUDA 11 environment cannot be used.
  if ! dpkg -s cuda-keyring >/dev/null 2>&1; then
    wget -q -O /tmp/cuda-keyring.deb \
      https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
    as_root dpkg -i /tmp/cuda-keyring.deb
  fi
  as_root apt-get update -qq
  as_root apt-get install -y --no-install-recommends \
    cuda-nvcc-$CUDA_VERSION cuda-cudart-dev-$CUDA_VERSION cuda-nvrtc-dev-$CUDA_VERSION \
    libcusparse-dev-$CUDA_VERSION libcublas-dev-$CUDA_VERSION \
    libcusolver-dev-$CUDA_VERSION libcurand-dev-$CUDA_VERSION
}

stage_python() {
  # A venv is enough (no conda): smaller, and the Python version is fixed by the system.
  mkdir -p "$(dirname "$VENV")"
  [ -d "$VENV" ] || python3 -m venv "$VENV"
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  pip install --upgrade pip wheel setuptools
  # cu128 wheels bring their own CUDA runtime libraries (this is the biggest download).
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
  pip install numpy scipy scikit-learn matplotlib pillow opencv-python-headless \
    plyfile tqdm easydict einops kornia yacs gdown
}

case "${1:-}" in
  base)   stage_base ;;
  cuda)   stage_cuda ;;
  python) stage_python ;;
  all)    stage_base; stage_cuda; stage_python ;;
  *) echo "usage: $0 <base|cuda|python|all>"; exit 1 ;;
esac
