#!/usr/bin/env bash
# Build and validate one official SOMA sidecar for each retained G1 clip.
set -euo pipefail
if [[ $# != 2 ]]; then
  echo "Usage: bash $0 MORPHMIMIC_CHECKOUT DATA_ROOT" >&2
  exit 2
fi
checkout=$(realpath "$1")
data=$(realpath "$2")
test "$(git -C "$checkout" rev-parse HEAD)" = 41c8382f127a6f5109bfe047445a9180f71f1fc0
robot="$data/seed_g1_npz50"
subset="$robot/subsets/all.yaml"
human="$robot/human50"
python="$checkout/.venv/bin/python"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$python" -m morphmimic.scripts.convert_human --npz-root "$robot" --subset-yaml "$subset" \
  --bvh-root "$data/seed/soma_uniform/bvh" --human-root "$human" --workers "${WORKERS:-16}"
"$python" -m morphmimic.scripts.check_human --subset-yaml "$subset" --human-root "$human" \
  --out "$robot/check_human.csv"
