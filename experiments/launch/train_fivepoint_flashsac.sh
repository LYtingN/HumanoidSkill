#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
python_bin="${PYTHON_BIN:-$root/.venv/bin/python}"
if [[ -z "${PYTHON_BIN:-}" && -x "$root/can-pnp/.venv/bin/python" ]]; then
    python_bin="$root/can-pnp/.venv/bin/python"
fi
export PATH="$(dirname "$python_bin"):$PATH"
export PYTHONPATH="$root/supermimic-reproduced/src:$root/downloads/5points_FM/inference_source:$root/hskill/src:$root/tasks/can_pnp/src:$root/downloads/morphmimic-v0.1/source${PYTHONPATH:+:$PYTHONPATH}"
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-$root/.cache/warp-cu130}"
export PYTHONUNBUFFERED=1
if [[ "${NPROC_PER_NODE:-1}" != 1 ]]; then
    exec "$python_bin" -m torch.distributed.run --standalone --nproc-per-node "$NPROC_PER_NODE" -m experiments.train_fivepoint "$@"
fi
exec "$python_bin" -m experiments.train_fivepoint "$@"
