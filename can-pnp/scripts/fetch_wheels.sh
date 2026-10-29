#!/usr/bin/env bash
# Fetch the CUDA 12 build of warp-lang into wheels/ (uv find-links, see pyproject.toml).
# PyPI only ships warp-lang built against CUDA 13 (needs driver >= 580); our hosts run
# drivers 550 / 570. On machines with slow GitHub access, copy wheels/ over instead.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p wheels
version=1.18.0
wheel="warp_lang-${version}+cu12-py3-none-manylinux_2_28_x86_64.whl"
if [[ ! -f "wheels/$wheel" ]]; then
    curl -fL --retry 5 -C - -o "wheels/$wheel.part" \
        "https://github.com/NVIDIA/warp/releases/download/v${version}/${wheel}"
    mv "wheels/$wheel.part" "wheels/$wheel"
fi
ls -la "wheels/$wheel"
