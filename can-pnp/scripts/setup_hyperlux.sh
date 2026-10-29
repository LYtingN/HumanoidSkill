#!/usr/bin/env bash
# Optional HyperLux (OptiX) RGB backend for the head camera (CanPnpEnvCfgArgs.rgb_backend):
# patches the installed mjlab with the RGB-renderer interface of mjlab_hyperlux_plugin, installs
# hyperlux 0.1.8 + the plugin into .venv and applies HyperLux's own OptiX source patch.
#
#   HYPERLUX_PLUGIN=<clone of github.com/lzyang2000/mjlab_hyperlux_plugin> bash scripts/setup_hyperlux.sh
#
# Needs nvcc (CUDA_HOME), a C++ compiler and CMake; the first render compiles native code into
# HYPERLUX_CACHE_DIR. `uv sync` reinstalls mjlab unpatched: rerun this script afterwards.
# The plugin targets the mjlab commit pinned in pyproject.toml (8bb2bc9).
set -euo pipefail
cd "$(dirname "$0")/.."

PLUGIN="${HYPERLUX_PLUGIN:-../../third_party/mjlab_hyperlux_plugin}"
PY=.venv/bin/python
UV="${UV:-$(command -v uv || echo ~/.local/bin/uv)}"
SITE=$("$PY" -c "import mjlab, os; print(os.path.dirname(os.path.dirname(mjlab.__file__)))")

# Source hunks only: the docs hunks target files of the mjlab repository that a wheel lacks.
PATCH=$(mktemp)
"$PY" - "$PLUGIN/patches/mjlab-rgb-backends.patch" "$PATCH" <<'EOF'
import re, sys
text = open(sys.argv[1]).read()
blocks = re.split(r"(?m)^(?=diff --git )", text)
open(sys.argv[2], "w").write("".join(b for b in blocks if b.startswith("diff --git a/src/")))
EOF
if patch -p2 -R --dry-run -s -f -d "$SITE" < "$PATCH" > /dev/null 2>&1; then
  echo "mjlab already patched ($SITE)"
else
  patch -p2 -N -d "$SITE" < "$PATCH"
fi
rm -f "$PATCH"

"$UV" pip install --python "$PY" "hyperlux==0.1.8" "conta>=0.0.1" "nanobind>=2.0" "cmake>=3.24"
# No dependency resolution: mjlab / warp / torch stay as pinned by this project.
"$UV" pip install --python "$PY" --no-deps "$PLUGIN"

export HYPERLUX_BACKEND=OPTIX
export HYPERLUX_CACHE_DIR="${HYPERLUX_CACHE_DIR:-$HOME/.cache/hyperlux}"
mkdir -p "$HYPERLUX_CACHE_DIR"
"$PY" -m mjlab_hyperlux.patch_hyperlux
echo "HyperLux ready; run with PATH=$PWD/.venv/bin:\$PATH HYPERLUX_BACKEND=OPTIX HYPERLUX_CACHE_DIR=$HYPERLUX_CACHE_DIR"
