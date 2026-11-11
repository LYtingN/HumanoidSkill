"""The collected demonstrations' can paths, for the green "where the can must go" of tools/plot_layers.py.

    .venv/bin/python tools/extract_demo_can.py <collection>/episodes logs/demo_can_paths.npz

Every QA-passed episode, every ``every``-th step: the time since the episode start, the phase of the
step and the can position relative to the upper box centre (x, y) and its rim (z) of the nominal
layout. ``phase_start`` (episodes, phases) gives the time each phase starts (NaN: never reached),
so a plot can take the demonstrations from any curriculum phase on.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks/can_pnp/scripts"))
sys.path.insert(0, str(ROOT / "tasks/can_pnp/src"))

from can_pnp.scene.config import SceneGeometryCfg  # noqa: E402
from extract_grasp_starts import qpos_layout  # noqa: E402

PHASES = ("REACH", "CLOSE", "LIFT", "TURN_WALK", "PLACE_REACH", "LOWER", "RELEASE", "WITHDRAW")


def main() -> None:
    episodes, out = Path(sys.argv[1]), Path(sys.argv[2])
    every = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    geo = SceneGeometryCfg()
    box = np.array([geo.upper_box_center[0], geo.upper_box_center[1], geo.box_bottom_z + geo.upper_box_size[2]])
    rows, starts = [], []
    for qa in sorted(episodes.glob("slot_*/attempt_*/RTX_VIDEO_QA.json")):
        if not json.loads(qa.read_text()).get("passed"):
            continue
        attempt = qa.parent
        traj = np.load(attempt / "trajectory.npz", allow_pickle=True)
        qpos, phase = traj["qpos"], traj["phase"]
        can = [adr for _, adr, size in qpos_layout(attempt / "scene.xml") if size == 7][1]  # robot, can, ...
        dt = 1.0 / 50
        ep = len(starts)
        starts.append([float(np.argmax(phase == p)) * dt if (phase == p).any() else np.nan for p in PHASES])
        for i in range(0, len(phase), every):
            p = PHASES.index(str(phase[i])) if str(phase[i]) in PHASES else -1
            rows.append((ep, i * dt, p, *(qpos[i, can : can + 3] - box)))
    data = np.array(rows, dtype=np.float32)
    np.savez_compressed(out, rows=data, phase_start=np.array(starts, dtype=np.float32), phases=np.array(PHASES),
                        columns=np.array(("episode", "t", "phase", "x", "y", "z")))  # fmt: skip
    print(f"{out}: {len(starts)} episodes, {len(data)} frames (every {every} steps)")


if __name__ == "__main__":
    main()
