#!/usr/bin/env bash
# Compile the two CUDA extensions of SplatPose for the RTX 5070 (sm_120).
# Usage: scripts/build_extensions.sh   (run inside WSL)
set -euo pipefail

REPO="$HOME/work/SplatPose"
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
# Build only for our card: faster, and avoids PyTorch guessing a wrong architecture list.
export TORCH_CUDA_ARCH_LIST="12.0"
# Two parallel compile jobs keep the machine responsive.
export MAX_JOBS=2

source "$HOME/venvs/splat/bin/activate"

# Optional arguments select which extension to (re)build; default is both.
EXTENSIONS=("$@")
[ ${#EXTENSIONS[@]} -gt 0 ] || EXTENSIONS=(diff-gaussian-rasterization simple-knn)

# --no-build-isolation: build against the PyTorch already installed in the venv.
for ext in "${EXTENSIONS[@]}"; do
  echo "=== building $ext ==="
  time pip install --no-build-isolation "$REPO/submodules/$ext"
done

python -c "import diff_gaussian_rasterization, simple_knn; print('both extensions import OK')"
