"""Measured gripper openings at every recorded phase start (``assets/grasp_starts_gripper_measured.json``).

``grasp_starts.json`` stores the commanded opening (``action.*_gripper``): 0 while the can is held.
The fingers then rest on the can, so the measured opening (``observation.*_gripper``) is about
50-56 mm. Starting a held phase from the commanded value closes the fingers through the can.

    python extract_measured_gripper.py GRASP_STARTS_JSON LEROBOT_DATA_DIR > grasp_starts_gripper_measured.json

LEROBOT_DATA_DIR is the can_pnp dataset's ``data`` directory (``chunk-XXX/episode_XXXXXX.parquet``).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def main():
    starts = json.loads(Path(sys.argv[1]).read_text())
    root = Path(sys.argv[2])
    out = {}
    for ep, record in sorted(starts.items(), key=lambda kv: int(kv[0])):
        path = root / f"chunk-{int(ep) // 1000:03d}" / f"episode_{int(ep):06d}.parquet"
        table = pq.read_table(path, columns=["observation.left_gripper", "observation.right_gripper"])
        left, right = (np.asarray(table.column(c).to_pylist(), dtype=float).reshape(-1)
                       for c in ("observation.left_gripper", "observation.right_gripper"))
        out[ep] = {phase: [round(left[v["frame"]], 6), round(right[v["frame"]], 6)]
                   for phase, v in record["phases"].items()}
    print(json.dumps(out, separators=(",", ":")))


if __name__ == "__main__":
    main()
