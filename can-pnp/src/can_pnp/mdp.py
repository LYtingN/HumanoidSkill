"""Scene-level observation and termination terms.

Only what the scene itself defines lives here: object poses relative to the robot and
the fall / drop terminations of ``envs/isaaclab/config.py`` (``fall_height``,
``can_drop_height``). Stage logic, rewards and the residual-RL observation layout are not
part of the scene port.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse

from can_pnp.layout import layout_term

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def _in_base_frame(env: ManagerBasedRlEnv, pos_w: torch.Tensor) -> torch.Tensor:
    robot = env.scene["robot"]
    return quat_apply_inverse(robot.data.root_link_quat_w, pos_w - robot.data.root_link_pos_w)


def object_pos_b(env: ManagerBasedRlEnv, entity_name: str = "can") -> torch.Tensor:
    """Object root position in the robot pelvis frame."""
    return _in_base_frame(env, env.scene[entity_name].data.root_link_pos_w)


def object_quat_w(env: ManagerBasedRlEnv, entity_name: str = "can") -> torch.Tensor:
    return env.scene[entity_name].data.root_link_quat_w


def goal_pos_b(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Placement goal in the robot pelvis frame."""
    return _in_base_frame(env, layout_term(env).goal_pos_w)


def robot_fallen(env: ManagerBasedRlEnv, fall_height: float = 0.55) -> torch.Tensor:
    origin_z = env.scene.env_origins[:, 2]
    return env.scene["robot"].data.root_link_pos_w[:, 2] - origin_z < fall_height


def can_dropped(
    env: ManagerBasedRlEnv, entity_name: str = "can", drop_height: float = 0.2
) -> torch.Tensor:
    origin_z = env.scene.env_origins[:, 2]
    return env.scene[entity_name].data.root_link_pos_w[:, 2] - origin_z < drop_height
