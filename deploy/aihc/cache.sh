#!/usr/bin/env bash
# Source from both interactive development and the training entrypoint.
export NYX_CFS_ROOT="${NYX_CFS_ROOT:-/mnt/cfs/usr/nyx}"
export HF_HOME="$NYX_CFS_ROOT/system_config/hf"
export TORCH_HOME="$NYX_CFS_ROOT/system_config/torch"
export XDG_CACHE_HOME="$NYX_CFS_ROOT/system_config/cache"
export UV_CACHE_DIR="$NYX_CFS_ROOT/system_config/uv"
export TRITON_CACHE_DIR="$NYX_CFS_ROOT/system_config/triton"
export TORCHINDUCTOR_CACHE_DIR="$NYX_CFS_ROOT/system_config/torchinductor"
export WARP_CACHE_PATH="$NYX_CFS_ROOT/system_config/warp-cu130"
export TMPDIR="$NYX_CFS_ROOT/system_config/tmp"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$UV_CACHE_DIR" \
  "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$WARP_CACHE_PATH" "$TMPDIR"
