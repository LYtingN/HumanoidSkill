"""Diffusion, recorded-dataset and standing reference sources."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from mjlab.utils.lab_api.math import matrix_from_quat

from hskill.embodiment import g1_aloha as C
from hskill.s0.supermimic import matrix_to_rot6d


def load_diffusion_policy(checkpoint: str | Path, repo: str | Path, device: str = "cuda:0"):
    """``humanoid_diffusion.model.load_policy`` from a Humanoid-Diffusion checkout."""
    repo = str(Path(repo).resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from humanoid_diffusion.model import load_policy

    policy, config = load_policy(checkpoint, device=device)
    if not getattr(policy, "rotation_dim", 0):
        raise ValueError(
            "This diffusion checkpoint predicts SONIC targets (action.wbc); train it with "
            "config/can_" "pnp_supermimic.yaml to get SuperMimic reference targets"
        )
    return policy, config

def set_sampling(policy, execute_steps: int | None = None, inference_steps: int | None = None) -> None:
    """Override the checkpoint's control steps per reference (replanning interval, at most the
    reference ``horizon``) and its number of DDIM steps; None keeps the checkpoint's."""
    if execute_steps is not None:
        if not 0 < execute_steps <= policy.horizon:
            raise ValueError(f"execute_steps must be in [1, {policy.horizon}], got {execute_steps}")
        policy.execute_steps = execute_steps
    if inference_steps is not None:
        policy.inference_steps = inference_steps


class StandSource:
    """Hold the default pose with the root orientation of the moment."""

    execute_steps, obs_horizon, horizon = 8, 1, 32

    def __init__(self, default_joint_pos: torch.Tensor):
        self.default_joint_pos = default_joint_pos

    def predict(self, rows: torch.Tensor, images=None, states=None):
        n, device = int(rows.sum()), self.default_joint_pos.device
        joint_pos = self.default_joint_pos[rows][:, None].repeat(1, self.horizon, 1)
        grip = torch.full((n, self.horizon, 2), C.GRIPPER_OPEN_WIDTH, device=device)
        rot6d = torch.tensor((1.0, 0.0, 0.0, 1.0, 0.0, 0.0), device=device).repeat(n, self.horizon, 1)
        return joint_pos, grip, rot6d

class DatasetSource:
    """Targets of one recorded LeRobot episode, exactly as the diffusion policy is trained on.

    Every env replays the episode from frame 0 after its reset (open loop).
    """

    execute_steps, obs_horizon, horizon = 8, 1, 32

    def __init__(self, parquet: str | Path, num_envs: int, device: str):
        import json

        import numpy as np
        import pyarrow.parquet as pq

        # <root>/data/chunk-xxx/episode_xxxxxx.parquet -> <root>/meta/info.json
        info = json.loads((Path(parquet).parents[2] / "meta" / "info.json").read_text())
        names = info["features"]["observation.state"]["names"]
        table = pq.read_table(parquet)

        def column(key: str) -> torch.Tensor:
            value = np.asarray(table[key].to_pylist(), dtype=np.float32).reshape(len(table), -1)
            return torch.as_tensor(value, device=device)

        self.joint_pos = column("observation.state")[:, [names.index(n) for n in C.BODY_JOINT_NAMES]]
        self.grip = torch.cat([column("action.left_gripper"), column("action.right_gripper")], dim=-1)
        self.root = matrix_from_quat(column("observation.root_orientation"))
        self.length = len(table)
        self.time = torch.zeros(num_envs, dtype=torch.long, device=device)

    def reset(self, env_ids: torch.Tensor) -> None:
        self.time[env_ids] = 0

    def predict(self, rows: torch.Tensor, images=None, states=None):
        t0 = self.time[rows].clamp(max=self.length - 1)  # past the end: hold the last frame
        idx = torch.clamp(t0[:, None] + torch.arange(self.horizon, device=t0.device), max=self.length - 1)
        rel = self.root[t0].transpose(-1, -2)[:, None] @ self.root[idx]
        self.time[rows] += self.execute_steps
        return self.joint_pos[idx], self.grip[idx], matrix_to_rot6d(rel)

class DiffusionSource:
    def __init__(self, policy):
        self.policy = policy
        self.execute_steps, self.obs_horizon = policy.execute_steps, policy.obs_horizon
        self.horizon = policy.horizon

    def predict(self, rows: torch.Tensor, images: torch.Tensor, states: torch.Tensor):
        pred = self.policy.predict_action(images[rows], states[rows])
        return pred["body_action_pred"], pred["gripper_width_pred"], pred["root_rotation_pred"]
