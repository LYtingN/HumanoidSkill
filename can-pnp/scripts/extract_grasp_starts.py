"""Extract the recorded states at the start of the reach / grasp of every collected episode.

    python scripts/extract_grasp_starts.py \\
        /data/nas_ray/home/jessie.lyu/single_can_adapter20_recollect250_20261017/episodes \\
        src/can_pnp/assets/grasp_starts.json

Reads each accepted attempt (``RTX_VIDEO_QA.json`` passed) of the MuJoCo collection: the
physical trajectory (``trajectory.npz``), its layout (``trajectory.json``) and the joint
layout of its model (``scene.xml``). For the first frame of the REACH and CLOSE phases it
stores the robot root pose, the named joint positions, both can poses and the gripper
openings, keyed by the LeRobot ``episode_index``. Needs only numpy.
"""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

PHASES = ("REACH", "CLOSE")
QPOS_SIZE = {"free": 7, "ball": 4, "hinge": 1, "slide": 1}


def qpos_layout(scene_xml: Path) -> list[tuple[str | None, int, int]]:
    """(joint name, qpos address, size) in qpos order: depth-first over the worldbody."""
    worldbody = ET.parse(scene_xml).getroot().find("worldbody")
    layout, adr = [], 0
    for element in worldbody.iter():
        if element.tag == "freejoint":
            kind = "free"
        elif element.tag == "joint":
            kind = element.get("type", "hinge")
        else:
            continue
        layout.append((element.get("name"), adr, QPOS_SIZE[kind]))
        adr += QPOS_SIZE[kind]
    return layout


def extract(attempt: Path) -> dict:
    trajectory = np.load(attempt / "trajectory.npz", allow_pickle=True)
    layout = json.loads((attempt / "trajectory.json").read_text())["layout"]
    joints = qpos_layout(attempt / "scene.xml")
    qpos, phase, gripper = trajectory["qpos"], trajectory["phase"], trajectory["gripper"]
    if sum(size for _, _, size in joints) != qpos.shape[1]:
        raise ValueError(f"{attempt}: scene.xml joints do not add up to qpos {qpos.shape[1]}")
    free = [adr for name, adr, size in joints if size == 7]
    if len(free) != 3:
        raise ValueError(f"{attempt}: expected robot + 2 can free joints, got {len(free)}")
    robot_adr, can_adr, shelf_adr = free
    named = {name: adr for name, adr, size in joints if name and size == 1}

    frames = {}
    for name in PHASES:
        t = int(np.argmax(phase == name))
        if phase[t] != name:
            raise ValueError(f"{attempt}: no {name} phase")
        q = qpos[t]
        frames[name] = {
            "frame": t,
            "root_pos": q[robot_adr : robot_adr + 3].tolist(),
            "root_quat": q[robot_adr + 3 : robot_adr + 7].tolist(),
            "joint_pos": {joint: float(q[adr]) for joint, adr in named.items()},
            "can_pos": q[can_adr : can_adr + 3].tolist(),
            "can_quat": q[can_adr + 3 : can_adr + 7].tolist(),
            "shelf_can_pos": q[shelf_adr : shelf_adr + 3].tolist(),
            "shelf_can_quat": q[shelf_adr + 3 : shelf_adr + 7].tolist(),
            "gripper_opening": gripper[t].tolist(),  # left, right (m)
        }
    return {
        "attempt": str(attempt),
        "table_front_x": layout["table_front_x"],
        "table_center_y": layout["table_center_y"],
        "phases": frames,
    }


def main() -> None:
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    episodes = {}
    for qa in sorted(root.glob("slot_*/attempt_*/RTX_VIDEO_QA.json")):
        info = json.loads(qa.read_text())
        if info.get("passed"):
            episodes[int(info["episode_index"])] = extract(qa.parent)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({str(k): episodes[k] for k in sorted(episodes)}))
    print(f"{len(episodes)} episodes -> {out}")


if __name__ == "__main__":
    main()
