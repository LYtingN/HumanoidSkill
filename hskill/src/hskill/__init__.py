import os
from pathlib import Path

# The CUDA 12 warp-lang build reports the same version as the CUDA 13 one, so both would share
# ~/.cache/warp/<version>; a CUDA 13 kernel there fails to load on our drivers ("device kernel
# image is invalid"). Keep this package's kernels in their own cache unless one is set.
os.environ.setdefault("WARP_CACHE_PATH", str(Path.home() / ".cache" / "warp-cu12"))
