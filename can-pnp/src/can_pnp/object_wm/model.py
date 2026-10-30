"""Object world model: the target can over a chunk, given the right gripper's motion and command.

Object state (env frame, 14): ``[pos (3), quat wxyz (4), lin vel (3), ang vel (3), right finger
opening (1)]``; the opening is the measured one (the command's effect, blocked by the can once
grasped). Each 50 Hz step ``k -> k + 1`` is predicted from

* the can relative to the gripper base at ``k`` (position, 6D orientation, velocities in the
  gripper frame), the opening, the can axis in the world and gravity in the gripper frame;
* the gripper's motion over the step (pose ``k + 1`` in the frame of ``k``) and the commanded
  opening of the step (straight from the reference: the grippers bypass the tracker);
* the box and table-top centres relative to the can (world axes, the scene is axis-aligned).

The network outputs the change of the can's pose relative to the gripper, so its zero output
(the initialization) is a can rigidly held; the velocities are the finite differences of the
new pose plus a learned correction, and a logit for "held by both fingers". Steps are chained
over the chunk (autoregressive), as at use.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.utils.lab_api.math import (
    matrix_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_inv,
    quat_mul,
)
from torch import nn

OBJ_DIM = 14
POSE_DIM = 7  # gripper base pos (3) + quat wxyz (4), env frame
FEAT_DIM = 3 + 6 + 3 + 3 + 1 + 3 + 3 + 3 + 6 + 1 + 3 + 3
OUT_DIM = 3 + 3 + 3 + 3 + 1 + 1
IDENTITY_6D = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def quat_exp(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Rotation vector -> unit quaternion (wxyz), smooth at zero."""
    angle = (v.square().sum(-1, keepdim=True) + eps).sqrt()
    return torch.cat([torch.cos(angle / 2), v * (torch.sin(angle / 2) / angle)], dim=-1)


def quat_log(q: torch.Tensor) -> torch.Tensor:
    """Unit quaternion -> rotation vector (shortest)."""
    q = q * torch.where(q[..., :1] < 0, -1.0, 1.0)
    xyz = q[..., 1:]
    s = (xyz.square().sum(-1, keepdim=True) + 1e-12).sqrt()
    return xyz * (2 * torch.atan2(s, q[..., :1]) / s)


def rot6d(q: torch.Tensor) -> torch.Tensor:
    return matrix_from_quat(q)[..., :2].flatten(-2)


@dataclass
class ObjectWMCfg:
    hidden: tuple[int, ...] = (512, 512, 512)
    dt: float = 0.02
    # Feature / output scales: positions over 0.1 m, motion per step over 1 cm, etc.
    pos_scale: float = 0.1
    vel_scale: float = 0.5
    ang_vel_scale: float = 5.0
    opening_scale: float = 0.065
    context_scale: float = 0.3
    step_pos_scale: float = 0.01
    step_rot_scale: float = 0.05
    out_pos: float = 0.005
    out_rot: float = 0.02
    out_vel: float = 0.1
    out_ang_vel: float = 1.0
    out_opening: float = 0.005


class ObjectWorldModel(nn.Module):
    def __init__(self, cfg: ObjectWMCfg | None = None):
        super().__init__()
        cfg = self.cfg = cfg or ObjectWMCfg()
        layers, d = [], FEAT_DIM
        for h in cfg.hidden:
            layers += [nn.Linear(d, h), nn.SiLU()]
            d = h
        last = nn.Linear(d, OUT_DIM)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        self.net = nn.Sequential(nn.LayerNorm(FEAT_DIM), *layers, last)
        self.register_buffer("identity_6d", torch.tensor(IDENTITY_6D), persistent=False)
        self._step = self.step

    def compile_step(self) -> None:
        """Run :meth:`step` compiled (fused kernels; forward and backward), as in rollouts."""
        self._step = torch.compile(self.step, dynamic=None)

    def step(self, obj: torch.Tensor, grip: torch.Tensor, grip_next: torch.Tensor, command: torch.Tensor,
             box: torch.Tensor, table: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # fmt: skip
        """(B, 14) object, (B, 7) gripper poses at k and k + 1, (B,) command, (B, 3) box / table
        centres -> (B, 14) object at k + 1, (B,) held logit."""
        c = self.cfg
        pos, quat, vel, ang_vel, opening = obj[:, :3], obj[:, 3:7], obj[:, 7:10], obj[:, 10:13], obj[:, 13:]
        g_pos, g_quat = grip[:, :3], grip[:, 3:]
        n_pos, n_quat = grip_next[:, :3], grip_next[:, 3:]
        rel_pos = quat_apply_inverse(g_quat, pos - g_pos)
        rel_quat = quat_mul(quat_inv(g_quat), quat)
        gravity = torch.tensor((0.0, 0.0, -1.0), device=obj.device).expand_as(pos)
        feat = torch.cat([
            rel_pos / c.pos_scale, rot6d(rel_quat),
            quat_apply_inverse(g_quat, vel) / c.vel_scale, quat_apply_inverse(g_quat, ang_vel) / c.ang_vel_scale,
            opening / c.opening_scale, matrix_from_quat(quat)[..., 2], quat_apply_inverse(g_quat, gravity),
            quat_apply_inverse(g_quat, n_pos - g_pos) / c.step_pos_scale,
            (rot6d(quat_mul(quat_inv(g_quat), n_quat)) - self.identity_6d) / c.step_rot_scale,
            command[:, None] / c.opening_scale, (box - pos) / c.context_scale, (table - pos) / c.context_scale,
        ], dim=-1)  # fmt: skip
        out = self.net(feat)
        # New pose relative to the gripper at k + 1 (zero output: held rigidly).
        rel_pos = rel_pos + out[:, 0:3] * c.out_pos
        rel_quat = quat_mul(rel_quat, quat_exp(out[:, 3:6] * c.out_rot))
        new_pos = n_pos + quat_apply(n_quat, rel_pos)
        new_quat = quat_mul(n_quat, rel_quat)
        new_quat = new_quat / new_quat.norm(dim=-1, keepdim=True)
        new_vel = (new_pos - pos) / c.dt + out[:, 6:9] * c.out_vel
        new_ang_vel = quat_log(quat_mul(new_quat, quat_inv(quat))) / c.dt + out[:, 9:12] * c.out_ang_vel
        new_opening = opening + out[:, 12:13] * c.out_opening
        return torch.cat([new_pos, new_quat, new_vel, new_ang_vel, new_opening], dim=-1), out[:, 13]

    def rollout(self, obj0: torch.Tensor, grip: torch.Tensor, command: torch.Tensor, box: torch.Tensor,
                table: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # fmt: skip
        """Chain :meth:`step`: ``obj0`` (B, 14), ``grip`` (H + 1, B, 7), ``command`` (H, B) ->
        objects (H + 1, B, 14) (objects[0] = obj0) and held logits (H, B) of steps 1..H."""
        objs, logits = [obj0], []
        for k in range(command.shape[0]):
            obj, logit = self._step(objs[-1], grip[k], grip[k + 1], command[k], box, table)
            objs.append(obj)
            logits.append(logit)
        return torch.stack(objs), torch.stack(logits)


@dataclass
class ObjectLossCfg:
    # Nominal error scales: 1 cm, ~0.1 rad, 0.1 m/s, 1 rad/s, 5 mm opening.
    pos: float = 0.01
    rot: float = 0.1
    vel: float = 0.1
    ang_vel: float = 1.0
    opening: float = 0.005
    held_weight: float = 0.1


def object_loss(pred: torch.Tensor, logits: torch.Tensor, true: torch.Tensor, held: torch.Tensor, mask: torch.Tensor,
                cfg: ObjectLossCfg | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:  # fmt: skip
    """Normalized squared errors of steps 1..H against the simulated object (world truth in the
    env frame), ``held`` (H, B) labels, ``mask`` (H, B) the steps still inside the episode."""
    cfg = cfg or ObjectLossCfg()
    p, t = pred[1:], true[1:]
    dot = (p[..., 3:7] * t[..., 3:7]).sum(-1)
    terms = {
        "pos": (p[..., :3] - t[..., :3]).square().sum(-1) / cfg.pos**2,
        # 4 (1 - <q1, q2>^2) ~ (rotation angle)^2, smooth at zero error.
        "rot": 4 * (1 - dot.square()).clamp(min=0) / cfg.rot**2,
        "vel": (p[..., 7:10] - t[..., 7:10]).square().sum(-1) / cfg.vel**2,
        "ang_vel": (p[..., 10:13] - t[..., 10:13]).square().sum(-1) / cfg.ang_vel**2,
        "opening": (p[..., 13] - t[..., 13]).square() / cfg.opening**2,
        "held": nn.functional.binary_cross_entropy_with_logits(logits, held.float(), reduction="none"),
    }
    m = mask.float()
    count = m.sum().clamp(min=1)
    means = {k: (v * m).sum() / count for k, v in terms.items()}
    loss = sum(v for k, v in means.items() if k != "held") + cfg.held_weight * means["held"]
    with torch.no_grad():
        err = (p[..., :3] - t[..., :3]).norm(dim=-1)
        metrics = {f"loss_{k}": v.detach() for k, v in means.items()}
        metrics["pos_err_cm_final"] = 100 * (err[-1] * m[-1]).sum() / m[-1].sum().clamp(min=1)
        metrics["pos_err_cm_mean"] = 100 * (err * m).sum() / count
        metrics["held_acc"] = (((logits > 0) == held) * m).sum() / count
    return loss, metrics
