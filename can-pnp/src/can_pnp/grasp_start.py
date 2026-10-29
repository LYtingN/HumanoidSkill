"""Start episodes from recorded states at the reach / grasp of the collected episodes.

``assets/grasp_starts.json`` (``scripts/extract_grasp_starts.py``) holds, per LeRobot
``episode_index``, the first frame of the REACH and CLOSE phases of the MuJoCo collection:
robot root pose and joints, target / resident can poses, gripper openings and the table
layout. :func:`apply_grasp_start` writes one of them into each given env (env frame +
env origin), so the policy starts with the arm already at the can.

The collection world shares this scene's frame (table, shelf and box at the same nominal
places), so the recorded poses are used as they are.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from can_pnp import constants as C
from can_pnp.layout import layout_term
from can_pnp.pipeline import gripper_q

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

GRASP_STARTS = Path(__file__).parent / "assets" / "grasp_starts.json"


class GraspStarts:
    def __init__(self, path: str | Path = GRASP_STARTS, phase: str = "CLOSE"):
        episodes = json.loads(Path(path).read_text())
        self.episode_ids = sorted(int(k) for k in episodes)
        self.phase = phase
        rows = [episodes[str(i)] for i in self.episode_ids]
        frames = [row["phases"][phase] for row in rows]

        def stack(key: str) -> torch.Tensor:
            return torch.tensor([f[key] for f in frames], dtype=torch.float32)

        self.table_front_x = torch.tensor([r["table_front_x"] for r in rows], dtype=torch.float32)
        self.table_center_y = torch.tensor([r["table_center_y"] for r in rows], dtype=torch.float32)
        self.root_pos, self.root_quat = stack("root_pos"), stack("root_quat")
        self.can_pos, self.can_quat = stack("can_pos"), stack("can_quat")
        self.shelf_can_pos, self.shelf_can_quat = stack("shelf_can_pos"), stack("shelf_can_quat")
        self.joint_pos = torch.tensor(
            [[f["joint_pos"][name] for name in C.BODY_JOINT_NAMES] for f in frames]
        )
        self.gripper_opening = stack("gripper_opening")  # left, right (m)

    def __len__(self) -> int:
        return len(self.episode_ids)


def apply_grasp_start(
    env: ManagerBasedRlEnv, env_ids: torch.Tensor, starts: GraspStarts, rows: torch.Tensor
) -> None:
    """Write recorded start ``rows`` (indices into ``starts``) into ``env_ids``, then refresh
    the simulation and the observation history of those envs."""
    device = env.device
    rows = rows.cpu()
    n = len(env_ids)
    origin = env.scene.env_origins[env_ids]
    zeros6 = torch.zeros(n, 6, device=device)
    def to(x: torch.Tensor) -> torch.Tensor:
        return x[rows].to(device)

    robot, can, shelf_can, table = (env.scene[k] for k in ("robot", "can", "shelf_can", "table"))
    geo = layout_term(env).scene_cfg.geometry

    robot.write_root_state_to_sim(
        torch.cat((to(starts.root_pos) + origin, to(starts.root_quat), zeros6), -1), env_ids=env_ids
    )
    body_ids = robot.find_joints(C.BODY_JOINT_NAMES, preserve_order=True)[0]
    gripper_ids = robot.find_joints(C.GRIPPER_JOINT_NAMES, preserve_order=True)[0]
    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_pos[:, body_ids] = to(starts.joint_pos)
    joint_pos[:, gripper_ids] = gripper_q(to(starts.gripper_opening))
    robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids)

    identity = torch.tensor((1.0, 0.0, 0.0, 0.0), device=device).repeat(n, 1)
    table_pose = torch.cat((torch.zeros(n, 3, device=device), identity), -1)
    table_pose[:, 0] = to(starts.table_front_x) + geo.table_size[0] / 2
    table_pose[:, 1] = to(starts.table_center_y)
    table_pose[:, :3] += origin
    table.write_mocap_pose_to_sim(table_pose, env_ids=env_ids)
    can.write_root_state_to_sim(
        torch.cat((to(starts.can_pos) + origin, to(starts.can_quat), zeros6), -1), env_ids=env_ids
    )
    # The box jitter of the collection is not recorded: keep the resident can inside this box.
    term = layout_term(env)
    inner = torch.tensor(geo.upper_box_size[:2], device=device) / 2 - geo.box_wall_thickness - geo.can_radius
    box_xy = term.box_center_w[env_ids, :2] - origin[:, :2]
    shelf_pos = to(starts.shelf_can_pos)
    shelf_pos[:, :2] = torch.maximum(torch.minimum(shelf_pos[:, :2], box_xy + inner), box_xy - inner)
    shelf_can.write_root_state_to_sim(
        torch.cat((shelf_pos + origin, to(starts.shelf_can_quat), zeros6), -1), env_ids=env_ids
    )
    # The placement goal is the half of the box the recorded resident can leaves free (as
    # layout.py picks it for a sampled resident), not the one of the reset layout.
    lay = term.scene_cfg.layout
    right = torch.tensor(lay.placement_goal_right, device=device)
    left = torch.tensor(lay.placement_goal_left, device=device)
    goal = torch.where(shelf_pos[:, :1] - term.box_delta[env_ids, :1] < lay.placement_corridor_split_x, right, left)
    term.goal_pos_w[env_ids] = goal + term.box_delta[env_ids] + origin
    term.layout["table_front_x"][env_ids] = to(starts.table_front_x)
    term.layout["table_center_y"][env_ids] = to(starts.table_center_y)
    term.layout["can_xyz"][env_ids] = to(starts.can_pos)

    env.scene.write_data_to_sim()
    env.sim.forward()
    env.sim.sense()
    env.observation_manager.reset(env_ids)
    env.obs_buf = env.observation_manager.compute(update_history=True, env_ids=env_ids)
