#!/usr/bin/env bash
# Select the NVML library matching the loaded kernel driver for nvtop only.
set -euo pipefail

nvtop_driver=$(sed -n 's/.*Kernel Module.*  \([0-9][0-9.]*\) .*/\1/p' /proc/driver/nvidia/version 2>/dev/null || true)
nvtop_library="/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.$nvtop_driver"
if [[ -n "$nvtop_driver" && -f "$nvtop_library" ]]; then
  nvtop_cache="${XDG_CACHE_HOME:-$HOME/.cache}/nvtop-nvml/$nvtop_driver"
  mkdir -p "$nvtop_cache"
  ln -sfn "$nvtop_library" "$nvtop_cache/libnvidia-ml.so"
  ln -sfn "$nvtop_library" "$nvtop_cache/libnvidia-ml.so.1"
  export LD_LIBRARY_PATH="$nvtop_cache${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

exec /usr/bin/nvtop "$@"
