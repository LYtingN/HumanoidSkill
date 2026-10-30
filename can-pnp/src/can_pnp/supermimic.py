"""SuperMimic tracking policy and its goal features for a predicted reference trajectory.

The tracker is a checkpoint of humanoid_skill/supermimic-reproduced (``policy.nets.Policy``),
either ``deploy_obs`` + ``anchor_ori_obs`` (goal 646, below) or the full-observation default
(goal 680: 10 frames of the 64 below + ref root linear velocity in its yaw frame (3) and
ref root height (1), no anchor term). The diffusion reference has no root translation; for the
full variant both come from leg kinematics (``can_pnp.odometry``). Inputs, read off the
checkpoint's fixed scale vectors and config:

* observation history (930): joint_pos - q0, joint_vel, base_ang_vel, projected_gravity and
  the last action, 10 frames each, term-major (the ``policy`` group of ``env_cfg``);
* goal (646): 10 future reference frames of 64
  ``[q - q0 (29), q_dot (29), projected gravity of the ref root (3), ref root ang vel in the
  ref root frame (3)]`` (``deploy_obs`` drops the 68-dim frame's horizontal velocity and
  height), then the 6D orientation of the reference torso anchor in the robot torso frame
  (first two columns of ``R_robot^T R_ref``, row-major, as mjlab's ``motion_anchor_ori_b``).

Output: 29 joint actions, executed as ``q0 + G1_ACTION_SCALE * a``.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from mjlab.utils.lab_api.math import matrix_from_quat

FUTURE_FRAMES = 10
GOAL_FRAME_DIM = 64
ANCHOR_DIM = 6
GOAL_DIM = FUTURE_FRAMES * GOAL_FRAME_DIM + ANCHOR_DIM
FULL_GOAL_FRAME_DIM = 68
FULL_GOAL_DIM = FUTURE_FRAMES * FULL_GOAL_FRAME_DIM
OBS_DIM = 930
NUM_JOINTS = 29
WAIST = slice(12, 15)  # waist yaw, roll, pitch in the body joint order


class _MLP(nn.Module):
    """Same layout as ``supermimic.world_model.dynamics.MLP`` (Linear / SiLU, ``net.*`` keys)."""

    def __init__(self, dims: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(a, b))
            if i < len(dims) - 2:
                layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _mlp_dims(state: dict[str, torch.Tensor], prefix: str) -> list[int]:
    weights = sorted(
        (int(k[len(prefix) :].split(".")[0]), v)
        for k, v in state.items()
        if k.startswith(prefix) and k.endswith(".weight")
    )
    return [weights[0][1].shape[1], *(w.shape[0] for _, w in weights)]


class _Scaled(nn.Module):
    """An MLP with the fixed per-feature input scale buffer it was trained with."""

    def __init__(self, dims: list[int], scale_dim: int):
        super().__init__()
        self.mlp = _MLP(dims)
        self.register_buffer("scale", torch.ones(scale_dim))


class TrackingPolicy(nn.Module):
    """``a = dec(tanh(enc(goal * s_g)), obs_hist * s_o)`` (``supermimic.policy.nets.Policy``)."""

    def __init__(self, state: dict[str, torch.Tensor]):
        super().__init__()
        enc = _mlp_dims(state, "encoders.motion.mlp.net.")
        dec = _mlp_dims(state, "decoder.mlp.net.")
        self.encoders = nn.ModuleDict({"motion": _Scaled(enc, enc[0])})
        # The decoder input is [latent, scaled observation history].
        self.decoder = _Scaled(dec, dec[0] - enc[-1])
        self.load_state_dict(state)

    @property
    def goal_dim(self) -> int:
        return self.encoders["motion"].scale.numel()

    @property
    def obs_dim(self) -> int:
        return self.decoder.scale.numel()

    def forward(self, obs_hist: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        enc = self.encoders["motion"]
        z = torch.tanh(enc.mlp(goal * enc.scale))
        return self.decoder.mlp(torch.cat([z, obs_hist * self.decoder.scale], dim=-1))


def load_tracking_policy(path: str | Path, device: str = "cuda:0") -> TrackingPolicy:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    env_cfg = checkpoint["config"]["env"]
    trainer_cfg = checkpoint["config"]["policy_trainer"]
    deploy = str(env_cfg.get("deploy_obs")) == "True" and str(trainer_cfg.get("anchor_ori_obs")) == "True"
    full = not env_cfg.get("deploy_obs") and not env_cfg.get("anchor_obs")
    if not (deploy or full):
        raise ValueError("Expected a deploy_obs + anchor_ori_obs or a full-observation SuperMimic checkpoint")
    policy = TrackingPolicy(checkpoint["policy"])
    if policy.goal_dim != (GOAL_DIM if deploy else FULL_GOAL_DIM) or policy.obs_dim != OBS_DIM:
        raise ValueError(f"Unexpected tracker dims goal={policy.goal_dim} obs={policy.obs_dim}")
    return policy.to(device).eval()


# ----------------------------------------------------------------------------- goal features


def rot6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    """Inverse of the row-major first-two-columns 6D layout, Gram-Schmidt re-orthonormalized."""
    m = rot6d.reshape(*rot6d.shape[:-1], 3, 2)
    a = torch.nn.functional.normalize(m[..., 0], dim=-1)
    b = m[..., 1] - (a * m[..., 1]).sum(-1, keepdim=True) * a
    b = torch.nn.functional.normalize(b, dim=-1)
    return torch.stack([a, b, torch.cross(a, b, dim=-1)], dim=-1)


def matrix_to_rot6d(matrix: torch.Tensor) -> torch.Tensor:
    return matrix[..., :2].reshape(*matrix.shape[:-2], 6)


def _axis_rotation(angle: torch.Tensor, axis: int) -> torch.Tensor:
    c, s = angle.cos(), angle.sin()
    one, zero = torch.ones_like(angle), torch.zeros_like(angle)
    if axis == 0:
        rows = ((one, zero, zero), (zero, c, -s), (zero, s, c))
    elif axis == 1:
        rows = ((c, zero, s), (zero, one, zero), (-s, zero, c))
    else:
        rows = ((c, -s, zero), (s, c, zero), (zero, zero, one))
    return torch.stack([torch.stack(r, -1) for r in rows], -2)


def torso_in_pelvis(waist: torch.Tensor) -> torch.Tensor:
    """Rotation of ``torso_link`` in the pelvis frame: yaw (z), roll (x), pitch (y) joints."""
    yaw, roll, pitch = waist.unbind(-1)
    return _axis_rotation(yaw, 2) @ _axis_rotation(roll, 0) @ _axis_rotation(pitch, 1)


class ReferenceTrajectory:
    """A predicted reference ``[t0, t0 + H)`` and its per-frame SuperMimic goal features.

    ``joint_pos`` (N, H, 29) absolute joint positions, ``root_rot6d`` (N, H, 6) the reference
    root rotation relative to the robot root at ``t0`` (whose world quaternion is
    ``root_quat_w``), sampled every ``dt`` seconds. ``root_height`` (N, H) and
    ``root_lin_vel_w`` (N, H, 3) select the full-observation goal (``FULL_GOAL_DIM``), which
    has no anchor term.
    """

    def __init__(
        self,
        joint_pos: torch.Tensor,
        root_rot6d: torch.Tensor,
        root_quat_w: torch.Tensor,
        default_joint_pos: torch.Tensor,
        dt: float,
        root_height: torch.Tensor | None = None,
        root_lin_vel_w: torch.Tensor | None = None,
    ):
        self.horizon, self.dt = joint_pos.shape[1], dt
        self.full = root_height is not None
        self.root_height, self.root_lin_vel_w = root_height, root_lin_vel_w
        self.joint_pos = joint_pos
        self.joint_vel = torch.gradient(joint_pos, spacing=dt, dim=1)[0]
        root_w = matrix_from_quat(root_quat_w)[:, None] @ rot6d_to_matrix(root_rot6d)
        self.root_w = root_w  # (N, H, 3, 3)
        gravity = torch.tensor((0.0, 0.0, -1.0), device=joint_pos.device)
        self.gravity_b = (root_w.transpose(-1, -2) @ gravity)  # (N, H, 3)
        # Body-frame angular velocity from the skew part of R_k^T R_{k+1} per step, then averaged
        # over the two steps around each frame (one-sided at the ends).
        step = root_w[:, :-1].transpose(-1, -2) @ root_w[:, 1:]
        omega = torch.stack(
            (step[..., 2, 1] - step[..., 1, 2], step[..., 0, 2] - step[..., 2, 0],
             step[..., 1, 0] - step[..., 0, 1]), -1,
        ) / (2 * dt)  # fmt: skip
        padded = torch.cat([omega[:, :1], omega, omega[:, -1:]], dim=1)
        self.ang_vel_b = 0.5 * (padded[:, :-1] + padded[:, 1:])
        self.frames = torch.cat(
            [joint_pos - default_joint_pos[:, None], self.joint_vel, self.gravity_b, self.ang_vel_b],
            dim=-1,
        )  # (N, H, 64)
        if self.full:
            # Linear velocity in the reference root's yaw frame (supermimic goal_features).
            yaw = torch.atan2(root_w[..., 1, 0], root_w[..., 0, 0])
            lin_vel_h = _axis_rotation(yaw, 2).transpose(-1, -2) @ root_lin_vel_w[..., None]
            self.frames = torch.cat([self.frames, lin_vel_h[..., 0], root_height[..., None]], dim=-1)  # (N, H, 68)

    def goal(self, offset: torch.Tensor | int, robot_anchor_quat_w: torch.Tensor) -> torch.Tensor:
        """(N, 646) or (N, 680) goal for the control step ``t0 + offset`` (frames clamped at
        the horizon)."""
        n, device = self.joint_pos.shape[0], self.joint_pos.device
        offset = torch.as_tensor(offset, device=device).expand(n)
        idx = torch.clamp(offset[:, None] + torch.arange(FUTURE_FRAMES, device=device), max=self.horizon - 1)
        rows = torch.arange(n, device=device)[:, None]
        frames = self.frames[rows, idx].reshape(n, -1)
        if self.full:
            return frames
        ref_anchor = self.root_w[rows[:, 0], offset] @ torso_in_pelvis(self.joint_pos[rows[:, 0], offset, WAIST])
        anchor = matrix_from_quat(robot_anchor_quat_w).transpose(-1, -2) @ ref_anchor
        return torch.cat([frames, matrix_to_rot6d(anchor)], dim=-1)
