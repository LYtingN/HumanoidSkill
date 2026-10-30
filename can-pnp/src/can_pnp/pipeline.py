"""Reference source -> SuperMimic tracker -> joint PD, in the can pick-and-place scene.

Every ``execute_steps`` control steps a reference source predicts ``horizon`` frames of
``[29 joint positions | 2 gripper openings | 6D root rotation w.r.t. the root now]``:

* :class:`DiffusionSource` — the diffusion policy (Humanoid-Diffusion,
  ``config/can_pnp_supermimic.yaml``), closed loop on the head camera and robot state;
* :class:`DatasetSource` — the same targets read from a recorded episode (open loop), which
  checks the tracker and goal construction before the diffusion policy is retrained;
* :class:`StandSource` — the default pose.

Each control step the tracker gets the goal of the next 10 reference frames and outputs the
29 body actions; the grippers execute the reference openings directly.

Diffusion observation (48), matching the dataset fields it was trained on:
``observation.state`` body joints (29), ``observation.eef_state`` = left/right
``wrist_yaw_link`` position + w,x,y,z quaternion in the pelvis frame (14),
``observation.projected_gravity`` (3), left/right gripper opening in metres (2); plus the
head camera RGB resized to 224 x 224 (area interpolation, as the training image cache).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from mjlab.utils.lab_api.math import matrix_from_quat, quat_apply_inverse, quat_inv, quat_mul

from can_pnp import constants as C
from can_pnp.odometry import LegKinematics, root_height_and_velocity
from can_pnp.supermimic import (
    FULL_GOAL_DIM,
    ReferenceTrajectory,
    TrackingPolicy,
    matrix_to_rot6d,
    rot6d_to_matrix,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


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
            "config/can_pnp_supermimic.yaml to get SuperMimic reference targets"
        )
    return policy, config


def gripper_opening(q: torch.Tensor) -> torch.Tensor:
    return C.GRIPPER_OPEN_WIDTH * (1.0 - q / C.GRIPPER_CLOSED_Q)


def gripper_q(opening: torch.Tensor) -> torch.Tensor:
    return C.GRIPPER_CLOSED_Q * (1.0 - opening.clamp(0.0, C.GRIPPER_OPEN_WIDTH) / C.GRIPPER_OPEN_WIDTH)


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


def reference_plan(
    joint_pos: torch.Tensor,
    rot6d: torch.Tensor,
    root_quat_w: torch.Tensor,
    default_joint_pos: torch.Tensor,
    dt: float,
    legs: LegKinematics | None = None,
    measured_joint_pos: torch.Tensor | None = None,
    measured_height: torch.Tensor | None = None,
) -> ReferenceTrajectory:
    """Tracker goals of a predicted reference from the robot state at the replan: root rotation
    ``root_quat_w``; with ``legs`` (full-observation tracker) also the reference root height and
    velocity by leg odometry, calibrated on the measured joints and root height (env frame)."""
    height = lin_vel = None
    if legs is not None:
        root = matrix_from_quat(root_quat_w)
        height, lin_vel = root_height_and_velocity(
            legs, joint_pos, root[:, None] @ rot6d_to_matrix(rot6d), measured_joint_pos, root, measured_height, dt
        )
    return ReferenceTrajectory(
        joint_pos, rot6d, root_quat_w, default_joint_pos, dt=dt, root_height=height, root_lin_vel_w=lin_vel
    )


class SuperMimicController:
    def __init__(
        self,
        env: ManagerBasedRlEnv,
        source,
        tracker: TrackingPolicy,
        image_size: int = 224,
        camera: str | None = "head_camera",
    ):
        self.env, self.source, self.tracker = env, source, tracker
        self.robot = env.scene["robot"]
        self.camera = env.scene[camera] if camera else None
        self.image_size = image_size
        self.execute_steps = source.execute_steps
        self.obs_horizon = source.obs_horizon
        self.body_ids = self.robot.find_joints(C.BODY_JOINT_NAMES, preserve_order=True)[0]
        self.gripper_ids = self.robot.find_joints(C.GRIPPER_JOINT_NAMES, preserve_order=True)[0]
        self.pelvis = self.robot.body_names.index("pelvis")
        self.torso = self.robot.body_names.index("torso_link")
        self.wrists = [self.robot.body_names.index(f"{s}_wrist_yaw_link") for s in ("left", "right")]
        self.default_joint_pos = self.robot.data.default_joint_pos[:, self.body_ids]
        # The full-observation tracker also needs the reference root height and velocity.
        self.legs = None
        if tracker.goal_dim == FULL_GOAL_DIM:
            self.legs = LegKinematics(env.sim.mj_model, C.BODY_JOINT_NAMES, env.device, prefix="robot/")
        n = env.num_envs
        self.step_in_chunk = torch.zeros(n, dtype=torch.long, device=env.device)
        self.history: tuple[torch.Tensor, torch.Tensor] | None = None
        self.plan: ReferenceTrajectory | None = None
        self.plan_gripper: torch.Tensor | None = None
    # ------------------------------------------------------------------ observations

    def diffusion_state(self) -> torch.Tensor:
        data = self.robot.data
        pelvis_pos = data.body_link_pos_w[:, self.pelvis]
        pelvis_quat = data.body_link_quat_w[:, self.pelvis]
        eef = []
        for body in self.wrists:
            pos = quat_apply_inverse(pelvis_quat, data.body_link_pos_w[:, body] - pelvis_pos)
            quat = quat_mul(quat_inv(pelvis_quat), data.body_link_quat_w[:, body])
            quat = quat * torch.where(quat[:, :1] < 0, -1.0, 1.0)  # w >= 0, as recorded
            eef += [pos, quat]
        return torch.cat(
            [data.joint_pos[:, self.body_ids], *eef, data.projected_gravity_b,
             gripper_opening(data.joint_pos[:, self.gripper_ids])], dim=-1,
        )  # fmt: skip

    def diffusion_image(self) -> torch.Tensor:
        rgb = self.camera.data.rgb.permute(0, 3, 1, 2).float()
        image = F.interpolate(rgb, size=(self.image_size,) * 2, mode="area")
        return image.round().clamp(0, 255).to(torch.uint8)

    def _push_history(self, reset: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        image, state = self.diffusion_image(), self.diffusion_state()
        if self.history is None:
            self.history = (image[:, None].repeat(1, self.obs_horizon, 1, 1, 1),
                            state[:, None].repeat(1, self.obs_horizon, 1))  # fmt: skip
        images, states = self.history
        images = torch.cat([images[:, 1:], image[:, None]], dim=1)
        states = torch.cat([states[:, 1:], state[:, None]], dim=1)
        # Fresh episodes repeat their first frame, as the dataset clamps index -1 to 0.
        images[reset] = image[reset, None]
        states[reset] = state[reset, None]
        self.history = (images, states)
        return images, states

    # ------------------------------------------------------------------ control

    @torch.no_grad()
    def act(self, obs: dict[str, torch.Tensor], reset: torch.Tensor) -> torch.Tensor:
        """Env action ``[29 tracker actions | 2 gripper joint targets]`` for this step.

        ``reset`` marks envs whose episode starts at this step (all True on the first call).
        """
        n, device = self.env.num_envs, self.env.device
        self.step_in_chunk[reset] = 0
        if hasattr(self.source, "reset"):
            self.source.reset(reset.nonzero().flatten())
        images = states = None
        if self.camera is not None:
            images, states = self._push_history(reset)
        replan = (self.step_in_chunk % self.execute_steps == 0) | reset
        if self.plan is None:
            replan[:] = True
        if bool(replan.any()):
            self._replan(replan, images, states)
            self.step_in_chunk[replan] = 0
        offset = self.step_in_chunk.clone()
        goal = self.plan.goal(offset, self.robot.data.body_link_quat_w[:, self.torso])
        body = self.tracker(obs["policy"], goal)
        gripper = gripper_q(self.plan_gripper[torch.arange(n, device=device), offset])
        self.step_in_chunk += 1
        return torch.cat([body, gripper], dim=-1)

    def _replan(self, rows: torch.Tensor, images, states) -> None:
        joint_pos, grip, rot6d = self.source.predict(rows, images, states)
        data = self.robot.data
        plan = reference_plan(
            joint_pos, rot6d, data.root_link_quat_w[rows], self.default_joint_pos[rows], self.env.step_dt,
            self.legs, data.joint_pos[rows][:, self.body_ids],
            (data.root_link_pos_w[:, 2] - self.env.scene.env_origins[:, 2])[rows],
        )  # fmt: skip
        if self.plan is None:
            self.plan, self.plan_gripper = plan, grip
            return
        for name in ("joint_pos", "joint_vel", "root_w", "gravity_b", "ang_vel_b", "frames"):
            getattr(self.plan, name)[rows] = getattr(plan, name)
        if plan.full:
            self.plan.root_height[rows] = plan.root_height
            self.plan.root_lin_vel_w[rows] = plan.root_lin_vel_w
        self.plan_gripper[rows] = grip
