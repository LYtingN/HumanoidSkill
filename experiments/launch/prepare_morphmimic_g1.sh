#!/usr/bin/env bash
# Run after fully extracting g1/csv from BONES-SEED's g1.tar.gz.
set -euo pipefail

if [[ $# != 2 ]]; then
  echo "Usage: bash $0 MORPHMIMIC_CHECKOUT DATA_ROOT" >&2
  exit 2
fi
source_dir=$(realpath "$1")
data_dir=$(realpath "$2")
csv_dir="$data_dir/seed/g1/csv"
metadata="$data_dir/seed/metadata/seed_metadata_v003.csv"
output="$data_dir/seed_g1_npz50"
release=41c8382f127a6f5109bfe047445a9180f71f1fc0
if [[ $(git -C "$source_dir" rev-parse HEAD) != "$release" ]]; then
  echo "Expected MorphMimic v0.1 at $release" >&2
  exit 1
fi
if [[ ! -s "$metadata" || ! -d "$csv_dir" ]]; then
  echo "Expected $metadata and $csv_dir" >&2
  exit 1
fi
python="$source_dir/.venv/bin/python"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$python" -m morphmimic.scripts.convert_seed --robot g1 --csv-dir "$csv_dir" \
  --metadata "$metadata" --out-dir "$output" --workers "${WORKERS:-16}"
"$python" -m morphmimic.scripts.build_subsets --manifest "$output/manifest.csv" \
  --out-dir "$output/subsets"
"$python" - "$output" <<'PY'
import sys
from pathlib import Path

from morphmimic.data.manifest import filtered, load_manifest, load_subset

root = Path(sys.argv[1]).resolve()
manifest = load_manifest(str(root / "manifest.csv"))
expected = {root / p for p in filtered(manifest)["path"]}
actual = set(map(Path, load_subset(str(root / "subsets/all.yaml"))))
if not actual or actual != expected:
    raise SystemExit("all.yaml and the officially filtered manifest disagree")
print(f"Verified {len(actual)} clips in {root / 'subsets/all.yaml'}")
PY
