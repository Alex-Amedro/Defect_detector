#!/usr/bin/env bash
# Clone SplatPose, apply the two fixes it needs with CUDA 12.8 / recent Pillow, and fetch its pretrained weights.
# Usage: scripts/prepare_upstream.sh   (run inside WSL, after scripts/setup_wsl.sh). Safe to rerun.
set -euo pipefail

REPO="$HOME/work/SplatPose"
WEIGHTS_ID="16FOwaqQE0NGY-1EpfoNlU0cGlHjATV0V"   # model.zip from the SplatPose README (OmniposeAD checkpoints)

mkdir -p "$HOME/work"
[ -d "$REPO" ] || git clone --recursive https://github.com/m-kruse98/SplatPose.git "$REPO"

# Fix 1: FLT_MAX is not defined without <cfloat> on the CUDA 12.8 toolchain.
grep -q "<cfloat>" "$REPO/submodules/simple-knn/simple_knn.cu" ||
  sed -i 's/#include <vector>/#include <vector>\n#include <cfloat>/' "$REPO/submodules/simple-knn/simple_knn.cu"

# Fix 2: recent Pillow refuses a signed 8-bit image; pixels are 0..255, so uint8 is the right type.
sed -i 's/dtype=np.byte/dtype=np.uint8/' "$REPO/gaussian_splatting/scene/dataset_readers.py"

# Pretrained weights: LoFTR (pose retrieval) and EfficientNet-b4 (anomaly scoring).
if [ ! -f "$REPO/PAD_utils/model/indoor_ds_new.ckpt" ]; then
  mkdir -p "$HOME/data"
  "$HOME/venvs/splat/bin/gdown" "$WEIGHTS_ID" -O "$HOME/data/pad_model.zip"
  unzip -q -o "$HOME/data/pad_model.zip" -d "$REPO/PAD_utils"
fi
# The EfficientNet loader looks for its weights under retrieval/model.
mkdir -p "$REPO/retrieval"
ln -sfn ../PAD_utils/model "$REPO/retrieval/model"
echo "upstream ready: $REPO"
