"""Mirror the DSRL / v3 runs' metrics.jsonl into TensorBoard event files, following them as they grow.

    .venv/bin/python tools/metrics_to_tensorboard.py logs/dsrl &
    .venv/bin/tensorboard --logdir logs/dsrl --port 6007 --host 127.0.0.1

Every ``<run>/metrics.jsonl`` under the directory gets a ``<run>/tb`` event directory, rebuilt
from the JSON lines when this script starts (so restarting it does not duplicate points).
Per-iteration keys without a ``prefix/`` are grouped under ``update/``. A train_v3 run's
``dists.jsonl`` (hskill.utils.distributions) goes to the Histograms / Distributions tabs as ``dist/*``
and its outcome shares to ``dist/success_*``.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from torch.utils.tensorboard import SummaryWriter


def _read(path: Path, done: int) -> list[dict]:
    """The complete JSON lines of ``path`` after the first ``done``."""
    rows = []
    for line in path.read_text().splitlines()[done:]:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            break  # line still being written: read it next time
    return rows


def _histograms(writer: SummaryWriter, record: dict) -> None:
    step = record["cycle"]
    counts = np.array(record["success"]["counts"], dtype=float)
    for outcome, share in zip(record["outcomes"], counts / max(counts.sum(), 1)):
        writer.add_scalar(f"dist/success_{outcome}", share, step)
    for name, value in record.items():
        if not isinstance(value, dict) or "edges" not in value:
            continue
        edges, c = np.array(value["edges"]), np.array(value["counts"], dtype=float)
        if c.sum() == 0:
            continue
        centers = 0.5 * (edges[1:] + edges[:-1])
        used = np.flatnonzero(c)
        writer.add_histogram_raw(
            f"dist/{name}", min=float(edges[used[0]]), max=float(edges[used[-1] + 1]), num=int(c.sum()),
            sum=float((c * centers).sum()), sum_squares=float((c * centers**2).sum()),
            bucket_limits=edges[1:].tolist(), bucket_counts=c.tolist(), global_step=step,
        )  # fmt: skip


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("logs/dsrl")
    interval = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
    runs: dict[Path, tuple[SummaryWriter, int]] = {}  # metrics.jsonl -> (writer, lines written)
    dists: dict[Path, int] = {}  # dists.jsonl -> lines written
    while True:
        for path in sorted(root.glob("*/metrics.jsonl")):
            if path not in runs:
                shutil.rmtree(path.parent / "tb", ignore_errors=True)
                runs[path] = (SummaryWriter(path.parent / "tb"), 0)
            writer, done = runs[path]
            lines = path.read_text().splitlines()
            for line in lines[done:]:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    break  # line still being written: read it next time
                step_key = "iteration" if "iteration" in record else "cycle"  # DSRL / train_v3
                step = record[step_key]
                for key, value in record.items():
                    if key != step_key and isinstance(value, (int, float)):
                        writer.add_scalar(key if "/" in key else f"update/{key}", value, step)
                done += 1
            dist_path = path.parent / "dists.jsonl"
            if dist_path.exists():
                new = _read(dist_path, dists.get(dist_path, 0))
                for record in new:
                    _histograms(writer, record)
                dists[dist_path] = dists.get(dist_path, 0) + len(new)
            writer.flush()
            runs[path] = (writer, done)
        time.sleep(interval)


if __name__ == "__main__":
    main()
