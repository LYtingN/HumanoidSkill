#!/usr/bin/env bash
# Full BONES-SEED training; parameters stay overridable for the current GPU budget.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
unset PYTHONPATH

sm_cache="$PWD/data/runtime"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS=1
export XDG_CACHE_HOME="$sm_cache/cache"
export MPLCONFIGDIR="$sm_cache/matplotlib"
export WARP_CACHE_PATH="$sm_cache/warp"
export TORCHINDUCTOR_CACHE_DIR="$sm_cache/torchinductor"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-4}"
export TRITON_CACHE_DIR="$sm_cache/triton"

sm_driver=$(sed -n 's/.*Kernel Module.*  \([0-9][0-9.]*\) .*/\1/p' /proc/driver/nvidia/version 2>/dev/null || true)
sm_nvml="/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.$sm_driver"
if [[ -n "$sm_driver" && -f "$sm_nvml" ]]; then
  mkdir -p "$sm_cache/nvml"
  ln -sfn "$sm_nvml" "$sm_cache/nvml/libnvidia-ml.so.1"
  ln -sfn "$sm_nvml" "$sm_cache/nvml/libnvidia-ml.so"
fi
export LD_LIBRARY_PATH="$sm_cache/nvml${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

sm_library="${SM_LIBRARY:-data/library/bones_seed_compact.pt}"
sm_run="${SM_RUN_NAME:-bones_seed_48g_$(date +%Y%m%d_%H%M%S)}"
if [[ ! -f "$sm_library" ]]; then
  printf 'Missing packed library: %s\n' "$sm_library" >&2
  exit 1
fi
sm_resume=()
if [[ -n "${SM_RESUME:-}" ]]; then
  sm_resume=(--resume "$SM_RESUME")
fi

exec .venv/bin/python -u -m supermimic.scripts.train \
  --run-name "$sm_run" --data.library "$sm_library" \
  --env.num-envs "${SM_NUM_ENVS:-8192}" \
  --env.replay-capacity "${SM_REPLAY_CAPACITY:-1000}" \
  --wm-trainer.batch-size "${SM_WM_BATCH_SIZE:-2048}" \
  --policy-trainer.batch-size "${SM_POLICY_BATCH_SIZE:-7936}" \
  --schedule.total-iterations "${SM_TOTAL_ITERATIONS:-320000}" \
  --schedule.checkpoint-interval "${SM_CHECKPOINT_INTERVAL:-2000}" \
  --schedule.log-interval "${SM_LOG_INTERVAL:-50}" \
  "${sm_resume[@]}" "$@"
