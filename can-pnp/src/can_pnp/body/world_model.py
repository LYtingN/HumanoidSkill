"""SuperMimic's body world model f_omega on can_pnp robot states.

The tracker checkpoint of supermimic-reproduced also holds the world model it was trained with:
``s' = f(s, a, xi)`` over the 71-d robot state (MuJoCo layout, root position relative to the
env origin)

    [root pos (3), root quat wxyz (4), 29 body joints, root lin vel world (3),
     root ang vel root frame (3), 29 body joint vels]

with ``xi`` = [foot friction (1), encoder bias (29), torso COM offset (3)], and a termination
head on the root height / tilt error against the reference. supermimic is imported from its
checkout (its torch index pins conflict with this project's, so it is no package dependency),
as Humanoid-Diffusion is in ``pipeline.load_diffusion_policy``.

The grippers are not part of the state: the world model was trained on the bare G1 hand, and
the ALOHA gripper's weight is cancelled by gravity compensation (``entities._hand_gravcomp``).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from can_pnp import constants as C

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

STATE_DIM = 71
XI_DIM = 33
DEFAULT_REPO = Path(__file__).resolve().parents[4] / "supermimic-reproduced"


def import_supermimic(repo: str | Path = DEFAULT_REPO) -> None:
    src = str((Path(repo) / "src").resolve())
    if src not in sys.path:
        sys.path.insert(0, src)


class BodyWorldModel:
    """f_omega with its FK, plus the robot state and nominal ``xi`` of ``env``."""

    def __init__(self, env: ManagerBasedRlEnv, checkpoint: str | Path, repo: str | Path = DEFAULT_REPO):
        import_supermimic(repo)
        from supermimic.data.motion_convert import G1_JOINT_NAMES, load_g1_model
        from supermimic.envs.tracking_env_cfg import (
            ANCHOR_BODY,
            FOOT_GEOM_REGEX,
            ROOT_BODY,
            TRACKED_BODIES,
        )
        from supermimic.kinematics.g1_fk import G1Kinematics
        from supermimic.world_model.heads import AnalyticHeads
        from supermimic.world_model.state_spec import StateSpecCfg
        from supermimic.world_model.world_model import WorldModel, WorldModelCfg

        if tuple(G1_JOINT_NAMES) != C.BODY_JOINT_NAMES:
            raise ValueError("SuperMimic's joint order differs from BODY_JOINT_NAMES")
        self.env, self.device = env, env.device
        robot = self.robot = env.scene["robot"]
        body_ids = robot.find_joints(C.BODY_JOINT_NAMES, preserve_order=True)[0]
        idx = robot.indexing
        self.q_adr = torch.cat([idx.free_joint_q_adr, idx.joint_q_adr[body_ids]])
        self.v_adr = torch.cat([idx.free_joint_v_adr, idx.joint_v_adr[body_ids]])
        self.body_ids = body_ids
        self.default_joint_pos = robot.data.default_joint_pos[0, body_ids].clone()

        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cfg = dict(state["config"]["world_model"])
        cfg["hidden"], cfg["term_hidden"] = tuple(cfg["hidden"]), tuple(cfg["term_hidden"])
        cfg["state_spec"] = StateSpecCfg(**{k: tuple(v) if isinstance(v, list) else v
                                            for k, v in cfg["state_spec"].items()})  # fmt: skip
        self.fk = G1Kinematics(load_g1_model(), device=self.device)
        heads = AnalyticHeads(
            self.fk, TRACKED_BODIES, ANCHOR_BODY, ROOT_BODY,
            robot.data.soft_joint_pos_limits[0, body_ids].clone(), self.default_joint_pos,
        )  # fmt: skip
        self.wm = WorldModel(WorldModelCfg(**cfg), heads, self.device)
        self.wm.load_state_dict(state["wm"])
        self.wm.eval().requires_grad_(False)
        self.dt = self.wm.spec.dt
        if abs(self.dt - env.step_dt) > 1e-9:
            raise ValueError(f"World model dt {self.dt} != env step {env.step_dt}")

        # Nominal context: this scene randomizes none of it (encoder bias stays at its default).
        foot = robot.find_geoms(FOOT_GEOM_REGEX)[0][0]
        friction = env.sim.mj_model.geom_friction[int(idx.geom_ids[foot]), 0]
        xi = torch.zeros(XI_DIM, device=self.device)
        xi[0] = float(friction)
        self.xi = xi

    def context(self) -> torch.Tensor:
        """(N, 33) xi of the envs: nominal friction / COM, the robot's encoder bias."""
        xi = self.xi.repeat(self.env.num_envs, 1)
        xi[:, 1:30] = self.robot.data.encoder_bias[:, self.body_ids]
        return xi

    def state(self) -> torch.Tensor:
        """(N, 71) robot state of the envs, root position relative to the env origin."""
        data = self.env.sim.data
        s = torch.cat([data.qpos[:, self.q_adr], data.qvel[:, self.v_adr]], dim=-1)
        s[:, :3] -= self.env.scene.env_origins
        return s

    def step(self, s: torch.Tensor, action: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        return self.wm.step(s, action, xi)
