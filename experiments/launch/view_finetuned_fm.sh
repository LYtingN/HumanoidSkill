#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
python_bin="${PYTHON_BIN:-$root/.venv/bin/python}"
if [[ -z "${PYTHON_BIN:-}" && -x "$root/can-pnp/.venv/bin/python" ]]; then
    python_bin="$root/can-pnp/.venv/bin/python"
fi
export PATH="$(dirname "$python_bin"):$PATH"
export PYTHONPATH="$root/hskill/src:$root/tasks/can_pnp/src:$root/supermimic-reproduced/src${PYTHONPATH:+:$PYTHONPATH}"
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-$root/.cache/warp-cu130}"
export PYTHONUNBUFFERED=1
exec "$python_bin" -m experiments.view_policy \
    --source dsrl --noise-mode sample --device cuda:0 --num-envs 1 \
    --noise-checkpoint downloads/DSRL_finetune_20261116/best_cycle1885.pt \
    --diffusion-checkpoint downloads/supermimic-fm/can_pnp_supermimic_flow_matching_best_ema.pt \
    --diffusion-repo downloads/ODE/inference_source \
    --tracker-checkpoint downloads/supermimic/supermimic_0100000.pt \
    --start grasp --grasp-phase CLOSE --grasp-episodes 3 --can-mass 0.1 \
    --render-size 320 240 --no-randomize --seed 0 --seconds 0 --viewer --video-envs 0 \
    --out-dir logs/mjlab-gpu "$@"
