#!/usr/bin/env bash
# Extract and convert ONE MAD-Sim class for the Gaussian Splatting code (run inside WSL).
# Usage: scripts/prepare_mad_sim.sh [class_name]     e.g. 01Gorilla
# The upstream script converts all 20 classes at once; we keep only the class we need.
set -euo pipefail

CLASS="${1:-01Gorilla}"
DATA="$HOME/data"
REPO="$HOME/work/SplatPose"

cd "$DATA"
if [ ! -d "MAD-Sim/$CLASS" ]; then
  unzip -q -o MAD-Sim.zip "*/$CLASS/*" -d "$DATA/_unzip" || unzip -q -o MAD-Sim.zip "$CLASS/*" -d "$DATA/_unzip"
  # The archive may contain an extra top-level folder: find where the class really is.
  SRC="$(dirname "$(find "$DATA/_unzip" -maxdepth 3 -type d -name "$CLASS" | head -n 1)")"
  mkdir -p MAD-Sim
  mv "$SRC/$CLASS" MAD-Sim/
  rm -rf "$DATA/_unzip"
fi

# Work on a copy of the upstream script with its class list reduced to our class.
python3 - "$REPO/prepare_lego_ad.py" "$DATA/prepare_one_class.py" "$CLASS" <<'EOF'
import re, sys
src, dst, cls = sys.argv[1:4]
text = open(src).read()
text = re.sub(r"classnames = \[.*?\]", f'classnames = ["{cls}"]', text, flags=re.S)
open(dst, "w").write(text)
EOF

"$HOME/venvs/splat/bin/python" "$DATA/prepare_one_class.py"
echo "Prepared: $DATA/MAD-Sim_3dgs/$CLASS"
ls "$DATA/MAD-Sim_3dgs/$CLASS"
