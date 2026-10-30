#!/usr/bin/env bash
# v3 (imagined GRPO + object world model, can_pnp.scripts.train_v3) on 8 local GPUs (torchrun, NCCL).
#
#   TRACKER=<supermimic>.pt DIFFUSION=<Humanoid-Diffusion run>/best.pt \
#     OUTPUT_DIR=logs/v3/run1 bash scripts/train_v3_8gpu.sh [extra train_v3 args]
#
# Extra arguments go to can_pnp.scripts.train_v3 (e.g. --init-noise-checkpoint logs/dsrl/<run>/best.pt).
set -euo pipefail
cd "$(dirname "$0")/.."

: "${TRACKER:?set TRACKER to the SuperMimic tracker checkpoint}"
: "${DIFFUSION:?set DIFFUSION to the Humanoid-Diffusion checkpoint}"
OUTPUT_DIR="${OUTPUT_DIR:-logs/v3/run_$(date +%Y%m%d_%H%M%S)}"
DIFFUSION_REPO="${DIFFUSION_REPO:-../../UMI-Diffusion/Humanoid-Diffusion}"
NPROC="${NPROC:-8}"
# The head camera renders through MuJoCo Warp; no OpenGL context is needed (or present on
# remote-26, which has no libEGL).
export MUJOCO_GL="${MUJOCO_GL:-disable}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

mkdir -p "$OUTPUT_DIR"
# Static rendezvous on 127.0.0.1 and a fixed port rather than --standalone: where localhost
# resolves only to ::1 (as on remote-26) the standalone store's clients time out, and the c10d
# rendezvous listens on an ephemeral port, which fails once those are used up (remote-26: an IDE
# port forward holding ~28k loopback connections).
exec .venv/bin/torchrun --master-addr 127.0.0.1 --master-port "${MASTER_PORT:-29541}" --nnodes 1 \
  --nproc-per-node "$NPROC" -m can_pnp.scripts.train_v3 \
  --tracker-checkpoint "$TRACKER" --diffusion-checkpoint "$DIFFUSION" \
  --diffusion-repo "$DIFFUSION_REPO" --out-dir "$OUTPUT_DIR" "$@"
