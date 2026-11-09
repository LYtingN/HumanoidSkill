"""Time-to-completion targets of recorded demonstrations, for a progress critic (as RISE).

A demonstration reaches the goal at its last frame, so its frame ``t`` is worth the success
reward discounted by the steps still to go: ``success_reward * gamma ** (T - 1 - t)``. That needs
no object state (demonstrations carry none), only the robot's proprioception, which the
imagined states have too: body joint positions, gravity in the pelvis frame, both wrists' poses
in the pelvis frame and the right gripper opening, plus the share of the episode elapsed since the
start frame.

Reads a LeRobot dataset (``<root>/data/chunk-*/episode_*.parquet``) with the fields
``observation.state`` (joint positions, names in ``meta/info.json``), ``observation.eef_state``
(left / right wrist position and w-first quaternion in the pelvis frame),
``observation.projected_gravity`` and ``observation.right_gripper`` (opening, m).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from mjlab.utils.lab_api.math import matrix_from_quat

PROPRIO_DIM = 29 + 3 + 2 * (3 + 6) + 1


def proprio_features(joint_pos: torch.Tensor, gravity: torch.Tensor, wrist_pos: torch.Tensor,
                     wrist_quat: torch.Tensor, opening: torch.Tensor) -> torch.Tensor:  # fmt: skip
    """(B, 29) joints, (B, 3) gravity in the pelvis frame, (B, 2, 3) / (B, 2, 4) wrists in the
    pelvis frame, (B,) right opening (m) -> (B, PROPRIO_DIM)."""
    rot6d = matrix_from_quat(wrist_quat)[..., :2].flatten(-2)  # (B, 2, 6), sign-free
    return torch.cat([joint_pos, gravity, wrist_pos.flatten(-2), rot6d.flatten(-2), opening[:, None] / 0.065], dim=-1)


class DemoProgress:
    def __init__(self, root: str | Path, start_frames: dict[int, int], joint_names: tuple[str, ...],
                 gamma: float, success_reward: float, episode_steps: int, device: str):  # fmt: skip
        import pyarrow.parquet as pq

        root = Path(root)
        names = json.loads((root / "meta" / "info.json").read_text())["features"]["observation.state"]["names"]
        cols = [names.index(n) for n in joint_names]
        feats, times, targets = [], [], []
        for path in sorted(root.glob("data/chunk-*/episode_*.parquet")):
            table = pq.read_table(path, columns=["episode_index", "observation.state", "observation.eef_state",
                                                 "observation.projected_gravity", "observation.right_gripper"])  # fmt: skip
            episode = int(table["episode_index"][0].as_py())
            if episode not in start_frames:
                continue

            def column(key: str) -> torch.Tensor:
                return torch.as_tensor(np.asarray(table[key].to_pylist(), dtype=np.float32).reshape(len(table), -1))

            start, length = start_frames[episode], len(table)
            sl = slice(start, length)
            eef = column("observation.eef_state")[sl].view(-1, 2, 7)
            feats.append(proprio_features(column("observation.state")[sl][:, cols], column("observation.projected_gravity")[sl],
                                          eef[..., :3], eef[..., 3:], column("observation.right_gripper")[sl, 0]))  # fmt: skip
            t = torch.arange(start, length, dtype=torch.float32)
            times.append((t - start) / episode_steps)
            targets.append(success_reward * gamma ** (length - 1 - t))
        if not feats:
            raise ValueError(f"No demonstration of the start frames under {root}")
        self.proprio = torch.cat(feats).to(device)
        self.time = torch.cat(times).to(device)
        self.target = torch.cat(targets).to(device)
        self.episodes = len(feats)

    def __len__(self) -> int:
        return len(self.target)

    def sample(self, n: int) -> dict[str, torch.Tensor]:
        i = torch.randint(len(self.target), (n,), device=self.target.device)
        return {"proprio": self.proprio[i], "time": self.time[i], "target": self.target[i]}
