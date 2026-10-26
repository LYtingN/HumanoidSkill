#!/usr/bin/env bash
# Small, real BONES-SEED run through the paper's collect / WM / BPTT loop.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

sm_python="${SM_PYTHON:-$PWD/.venv/bin/python}"
sm_csv="${SM_CSV_DIR:-/home/eason_er/nyx/g1/csv}"
sm_npz="${SM_NPZ_DIR:-data/npz/bones_seed_smoke}"
sm_run="${SM_RUN_NAME:-bones_seed_smoke_$(date +%Y%m%d_%H%M%S)}"
sm_iterations="${SM_ITERATIONS:-100}"
sm_cache="$PWD/data/runtime"
mkdir -p "$sm_cache" "logs/$sm_run"

unset PYTHONPATH
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export XDG_CACHE_HOME="$sm_cache/cache"
export WARP_CACHE_PATH="$sm_cache/warp"
export TORCHINDUCTOR_CACHE_DIR="$sm_cache/torchinductor"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-4}"
export TRITON_CACHE_DIR="$sm_cache/triton"

# Some hosts retain an older loaded kernel module after a user-library upgrade.
# Select its existing NVML library for monitoring; keep the CUDA library selection
# intact (this host's CUDA 13 library supports forward compatibility).
sm_driver=$(sed -n 's/.*Kernel Module.*  \([0-9][0-9.]*\) .*/\1/p' /proc/driver/nvidia/version 2>/dev/null || true)
sm_nvml="/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.$sm_driver"
if [[ -n "$sm_driver" && -f "$sm_nvml" ]]; then
  mkdir -p "$sm_cache/nvml"
  ln -sfn "$sm_nvml" "$sm_cache/nvml/libnvidia-ml.so.1"
  ln -sfn "$sm_nvml" "$sm_cache/nvml/libnvidia-ml.so"
  export LD_LIBRARY_PATH="$sm_cache/nvml${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

"$sm_python" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.__version__, torch.cuda.get_device_name(0), flush=True)'

# Walking clips provide a bounded first experiment. Source is 120 Hz, cm/deg;
# the existing converter maps to metres/radians and resamples to 50 Hz.
"$sm_python" scripts/data/csv_to_npz_batch.py \
  --input-dir "$sm_csv" --output-dir "$sm_npz" --input-fps 120 \
  --pattern '**/*walk*.csv' --limit 32 --workers 4 \
  2>&1 | tee "logs/$sm_run/convert.log"
"$sm_python" scripts/data/filter_bones_seed.py --npz-dir "$sm_npz" --workers 4 \
  2>&1 | tee "logs/$sm_run/filter.log"

# Preserve the paper's networks, k=8, H=32, UTD=0.4, noise and randomization.
# Reduce environment/replay/batch sizes and run length for this first run.
# At step 40 warmup has just ended: its checkpoint is an untrained baseline.
"$sm_python" -u -m supermimic.scripts.train \
  --run-name "$sm_run" --data.npz-dir "$sm_npz" \
  --data.manifest "$sm_npz/manifest_filtered.csv" \
  --env.num-envs 512 --env.replay-capacity 200 \
  --wm-trainer.batch-size 256 --policy-trainer.batch-size 256 \
  --wm-trainer.compile None --policy-trainer.compile None \
  --schedule.total-iterations "$sm_iterations" --schedule.log-interval 20 \
  --schedule.checkpoint-interval 40 "$@" \
  2>&1 | tee "logs/$sm_run/train.log"

for sm_ckpt in ckpt_0000040 ckpt_latest; do
  "$sm_python" -u scripts/eval_benchmark.py \
    --ckpt "logs/$sm_run/$sm_ckpt.pt" --npz-dir "$sm_npz" \
    --manifest "$sm_npz/manifest_filtered.csv" --num-envs 32 \
    2>&1 | tee "logs/$sm_run/eval_$sm_ckpt.log"
done
printf 'Run artifacts: %s/logs/%s\n' "$PWD" "$sm_run"
