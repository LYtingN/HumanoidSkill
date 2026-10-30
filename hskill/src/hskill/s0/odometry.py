"""Root height and velocity of a joint-space reference by leg kinematics.

The diffusion reference has joint positions and a root rotation but no root translation; the
full-observation SuperMimic goal also wants the reference root's height and horizontal
velocity. Both follow from the legs if the lower foot does not slip: the root moves opposite
to that foot (in the world-aligned pelvis frame), and the root height is the lower sole's depth
below the pelvis plus a ground offset calibrated on the measured robot at planning time.
"""

from __future__ import annotations

import mujoco
import torch
from mjlab.utils.lab_api.math import matrix_from_quat

FEET = ("left_ankle_roll_link", "right_ankle_roll_link")
# Sole centre in the ankle-roll frame: under the foot collision capsules (z -0.025, r 0.01).
SOLE = (0.04, 0.0, -0.035)


def _axis_angle(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rotation matrices (..., 3, 3) about a fixed unit ``axis`` (3,) by ``angle`` (...)."""
    x, y, z = axis
    c, s = angle.cos(), angle.sin()
    t = 1 - c
    rows = (
        (t * x * x + c, t * x * y - s * z, t * x * z + s * y),
        (t * x * y + s * z, t * y * y + c, t * y * z - s * x),
        (t * x * z - s * y, t * y * z + s * x, t * z * z + c),
    )
    return torch.stack([torch.stack(r, -1) for r in rows], -2)


class LegKinematics:
    """Sole positions in the pelvis frame from body joint positions (``BODY_JOINT_NAMES`` order).

    ``prefix`` is the robot's name prefix in a composed scene model (``"robot/"``).
    """

    def __init__(
        self, model: mujoco.MjModel, joint_names: tuple[str, ...], device: str | torch.device, prefix: str = ""
    ):
        self.chains = []
        for foot in FEET:
            body, chain = model.body(prefix + foot).id, []
            while model.body(int(model.body_parentid[body])).name != prefix + "pelvis":
                chain.append(body)
                body = int(model.body_parentid[body])
            chain.append(body)
            links = []
            for b in reversed(chain):
                joint = int(model.body_jntadr[b])
                if model.body_jntnum[b] != 1 or model.jnt_pos[joint].any():
                    raise ValueError(f"{model.body(b).name}: expected one hinge at the body origin")
                links.append((
                    torch.tensor(model.body_pos[b], dtype=torch.float32, device=device),
                    matrix_from_quat(torch.tensor(model.body_quat[b], dtype=torch.float32, device=device)),
                    torch.tensor(model.jnt_axis[joint], dtype=torch.float32, device=device),
                    joint_names.index(model.joint(joint).name.removeprefix(prefix)),
                ))  # fmt: skip
            self.chains.append(links)
        self.sole = torch.tensor(SOLE, device=device)

    def soles(self, joint_pos: torch.Tensor) -> torch.Tensor:
        """(..., 29) -> (..., 2, 3) left / right sole centres in the pelvis frame."""
        out = []
        for links in self.chains:
            rot = torch.eye(3, device=joint_pos.device).expand(*joint_pos.shape[:-1], 3, 3)
            pos = torch.zeros(*joint_pos.shape[:-1], 3, device=joint_pos.device)
            for offset, fixed, axis, index in links:
                pos = pos + (rot @ offset[:, None])[..., 0]
                rot = rot @ fixed @ _axis_angle(axis, joint_pos[..., index])
            out.append(pos + (rot @ self.sole[:, None])[..., 0])
        return torch.stack(out, -2)


def root_height_and_velocity(
    kinematics: LegKinematics,
    joint_pos: torch.Tensor,
    root_w: torch.Tensor,
    measured_joint_pos: torch.Tensor,
    measured_root_w: torch.Tensor,
    measured_height: torch.Tensor,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference root height (N, H) and world linear velocity (N, H, 3).

    ``joint_pos`` (N, H, 29) and world root rotations ``root_w`` (N, H, 3, 3) of the reference;
    the measured joints (N, 29), root rotation (N, 3, 3) and root height (N,) calibrate the
    ground: the measured lower sole is taken to be on the floor.
    """
    soles = (root_w[:, :, None] @ kinematics.soles(joint_pos)[..., None])[..., 0]  # (N, H, 2, 3)
    measured = (measured_root_w[:, None] @ kinematics.soles(measured_joint_pos)[..., None])[..., 0]
    ground = measured_height + measured[..., 2].min(-1).values  # floor z relative to the sole drop
    height = ground[:, None] - soles[..., 2].min(-1).values
    # Stance foot of each step: the lower one at its start; the root moves opposite to it.
    stance = soles[:, :-1, :, 2].argmin(-1)  # (N, H-1)
    pick = stance[..., None, None].expand(-1, -1, 1, 3)
    step = soles[:, 1:].gather(2, pick)[:, :, 0] - soles[:, :-1].gather(2, pick)[:, :, 0]
    vel = -step / dt  # (N, H-1, 3), velocity over [k, k+1]
    # Averaged over the two steps around each frame, one-sided at the ends (as ang_vel_b).
    padded = torch.cat([vel[:, :1], vel, vel[:, -1:]], dim=1)
    vel = 0.5 * (padded[:, :-1] + padded[:, 1:])
    vel[..., 2] = torch.gradient(height, spacing=dt, dim=1)[0]
    return height, vel
