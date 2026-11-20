#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
mountpoint -q /mnt/cfs || { echo 'CFS must be mounted before setup.' >&2; exit 1; }
source "$ROOT/deploy/aihc/cache.sh"
UV="${UV_BIN:-/root/miniforge/bin/uv}"
PYTHON="${BOOTSTRAP_PYTHON:-/root/miniforge/bin/python}"
export UV_LINK_MODE=copy
export PYTHONPATH="$ROOT:$ROOT/hskill/src:$ROOT/tasks/can_pnp/src:$ROOT/supermimic-reproduced/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ -f "$ROOT/deploy/aihc/environment/uv.lock" ]]; then
  "$UV" sync --project "$ROOT/deploy/aihc/environment" --python "$PYTHON" --locked
else
  "$UV" sync --project "$ROOT/deploy/aihc/environment" --python "$PYTHON"
fi
"$ROOT/deploy/aihc/environment/.venv/bin/python" -c \
  'import torch, mujoco, warp, mjlab, hskill, can_pnp, supermimic; print("torch", torch.__version__, "CUDA build", torch.version.cuda, "visible GPUs", torch.cuda.device_count()); print("mujoco", mujoco.__version__, "warp", warp.__version__)'
sha256sum "$ROOT/deploy/aihc/environment/uv.lock" | cut -d ' ' -f 1 > "$ROOT/deploy/aihc/environment/.ready"
