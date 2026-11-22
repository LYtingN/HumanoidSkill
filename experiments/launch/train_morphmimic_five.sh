#!/usr/bin/env bash
# The upstream single-G1 recipe, with the reference mask fixed to five points.
set -euo pipefail
if (( $# < 4 )); then
  echo "Usage: bash $0 CHECKOUT SUBSET_YAML HUMAN_ROOT OUTPUT [TRAIN_OPTIONS...]" >&2
  exit 2
fi
checkout=$(realpath "$1")
subset=$(realpath -e "$2")
human=$(realpath -e "$3")
output=$(realpath -m "$4")
shift 4
cd "$checkout"
test "$(git rev-parse HEAD)" = 41c8382f127a6f5109bfe047445a9180f71f1fc0
# Upstream sets CUDA_VISIBLE_DEVICES from --gpu-ids before spawning local ranks.
count=${MORPHMIMIC_GPU_COUNT:-1}
if [[ ! "$count" =~ ^[1-8]$ ]]; then
  echo "Invalid allocated GPU count: $count" >&2
  exit 1
fi
devices=${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 "$((count - 1))")}
IFS=, read -ra ids <<< "$devices"
if [[ ! "$devices" =~ ^[0-9]+(,[0-9]+)*$ || ${#ids[@]} != "$count" ]]; then
  echo "Expected $count numeric CUDA device IDs, got $devices" >&2
  exit 1
fi
gpu_args=(--gpu "${ids[0]}")
if (( count > 1 )); then
  gpu_args=(--gpu-ids "[$devices]")
fi
export PYTHONUNBUFFERED=1
exec .venv/bin/python -m morphmimic.scripts.train \
  --robot g1 "${gpu_args[@]}" --subset-yaml "$subset" --human-root "$human" \
  --kp-interface --kp-student copy --kp-masking five --no-kp-preview-mix \
  --library-dtype float16 --iterations 60000 --ckpt-every 4000 \
  --log-dir "$output" --run-name g1-five "$@"
