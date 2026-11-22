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
# The upstream CLI sets CUDA_VISIBLE_DEVICES to the numeric id it receives.
gpu=${CUDA_VISIBLE_DEVICES:-0}
if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
  echo "Expected a single numeric CUDA device for the official single-G1 entrypoint; got $gpu" >&2
  exit 1
fi
export PYTHONUNBUFFERED=1
exec .venv/bin/python -m morphmimic.scripts.train \
  --robot g1 --gpu "$gpu" --subset-yaml "$subset" --human-root "$human" \
  --kp-interface --kp-student copy --kp-masking five --no-kp-preview-mix \
  --library-dtype float16 --iterations 60000 --ckpt-every 4000 \
  --log-dir "$output" --run-name g1-five "$@"
