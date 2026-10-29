"""Mirror the DSRL runs' metrics.jsonl into TensorBoard event files, following them as they grow.

    .venv/bin/python scripts/metrics_to_tensorboard.py logs/dsrl &
    .venv/bin/tensorboard --logdir logs/dsrl --port 6007 --host 127.0.0.1

Every ``<run>/metrics.jsonl`` under the directory gets a ``<run>/tb`` event directory, rebuilt
from the JSON lines when this script starts (so restarting it does not duplicate points).
Per-iteration keys without a ``prefix/`` are grouped under ``update/``.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("logs/dsrl")
    interval = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
    runs: dict[Path, tuple[SummaryWriter, int]] = {}  # metrics.jsonl -> (writer, lines written)
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
                step = record["iteration"]
                for key, value in record.items():
                    if key != "iteration" and isinstance(value, (int, float)):
                        writer.add_scalar(key if "/" in key else f"update/{key}", value, step)
                done += 1
            writer.flush()
            runs[path] = (writer, done)
        time.sleep(interval)


if __name__ == "__main__":
    main()
