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


def _step_features(c: ObjectWMCfg, identity_6d: torch.Tensor, obj: torch.Tensor, grip: torch.Tensor,
                   grip_next: torch.Tensor, command: torch.Tensor, box: torch.Tensor, table: torch.Tensor):  # fmt: skip
    """Network input of a step and the object's pose relative to the gripper at k."""
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
        (rot6d(quat_mul(quat_inv(g_quat), n_quat)) - identity_6d) / c.step_rot_scale,
        command[:, None] / c.opening_scale, (box - pos) / c.context_scale, (table - pos) / c.context_scale,
    ], dim=-1)  # fmt: skip
    return feat, rel_pos, rel_quat


def _step_next(c: ObjectWMCfg, obj: torch.Tensor, rel_pos: torch.Tensor, rel_quat: torch.Tensor,
               grip_next: torch.Tensor, out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # fmt: skip
    """Object at k + 1 from the network output: the new pose relative to the gripper at k + 1
    (zero output: held rigidly), velocities by finite differences plus corrections."""
    pos, quat, opening = obj[:, :3], obj[:, 3:7], obj[:, 13:]
    n_pos, n_quat = grip_next[:, :3], grip_next[:, 3:]
    rel_pos = rel_pos + out[:, 0:3] * c.out_pos
    rel_quat = quat_mul(rel_quat, quat_exp(out[:, 3:6] * c.out_rot))
    new_pos = n_pos + quat_apply(n_quat, rel_pos)
    new_quat = quat_mul(n_quat, rel_quat)
    new_quat = new_quat / new_quat.norm(dim=-1, keepdim=True)
    new_vel = (new_pos - pos) / c.dt + out[:, 6:9] * c.out_vel
    new_ang_vel = quat_log(quat_mul(new_quat, quat_inv(quat))) / c.dt + out[:, 9:12] * c.out_ang_vel
    new_opening = opening + out[:, 12:13] * c.out_opening
    return torch.cat([new_pos, new_quat, new_vel, new_ang_vel, new_opening], dim=-1), out[:, 13]


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
        feat, rel_pos, rel_quat = _step_features(self.cfg, self.identity_6d, obj, grip, grip_next, command, box, table)
        return _step_next(self.cfg, obj, rel_pos, rel_quat, grip_next, self.net(feat))

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


class EnsembleMLP(nn.Module):
    """``members`` independent MLPs (LayerNorm, SiLU hidden layers, zero-initialized last layer,
    as :class:`ObjectWorldModel`'s) evaluated together with batched matrix products:
    (E, B, d_in) -> (E, B, d_out)."""

    def __init__(self, members: int, d_in: int, hidden: tuple[int, ...], d_out: int):
        super().__init__()
        self.norm_weight = nn.Parameter(torch.ones(members, 1, d_in))
        self.norm_bias = nn.Parameter(torch.zeros(members, 1, d_in))
        self.weights, self.biases = nn.ParameterList(), nn.ParameterList()
        dims = (d_in, *hidden, d_out)
        for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
            bound = 0.0 if i == len(dims) - 2 else a**-0.5  # nn.Linear's default; zero last layer
            self.weights.append(nn.Parameter(torch.empty(members, a, b).uniform_(-bound, bound)))
            self.biases.append(nn.Parameter(torch.empty(members, 1, b).uniform_(-bound, bound)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.layer_norm(x, x.shape[-1:]) * self.norm_weight + self.norm_bias
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = torch.baddbmm(b, x, w)
            if i < len(self.weights) - 1:
                x = nn.functional.silu(x)
        return x


class ObjectWMEnsemble(nn.Module):
    """``members`` object world models trained on their own batches; their spread over a rollout is
    the model's uncertainty (large where its data never went, e.g. hand paths no execution took).

    The members are evaluated together (stacked weights, batched matrix products), so a rollout of
    the ensemble launches as many kernels as one of a single model.
    :meth:`rollout` returns the members' mean object (positions and velocities averaged, the mean
    quaternion re-normalized) and mean held logit, as a single model does.
    """

    def __init__(self, cfg: ObjectWMCfg | None = None, members: int = 3):
        super().__init__()
        cfg = self.cfg = cfg or ObjectWMCfg()
        self.num_members = members
        self.net = EnsembleMLP(members, FEAT_DIM, cfg.hidden, OUT_DIM)
        self.register_buffer("identity_6d", torch.tensor(IDENTITY_6D), persistent=False)
        self._step = self.step

    def compile_step(self) -> None:
        self._step = torch.compile(self.step, dynamic=None)

    def step(self, obj, grip, grip_next, command, box, table) -> tuple[torch.Tensor, torch.Tensor]:
        """As :meth:`ObjectWorldModel.step`, on (E * B, ...) inputs, member-major."""
        feat, rel_pos, rel_quat = _step_features(self.cfg, self.identity_6d, obj, grip, grip_next, command, box, table)
        out = self.net(feat.view(self.num_members, -1, feat.shape[-1])).flatten(0, 1)
        return _step_next(self.cfg, obj, rel_pos, rel_quat, grip_next, out)

    def rollout_members(self, obj0, grip, command, box, table) -> tuple[torch.Tensor, torch.Tensor]:
        """Every member on its own inputs: ``obj0`` (E, B, 14), ``grip`` (H + 1, E, B, 7),
        ``command`` (H, E, B), ``box`` / ``table`` (E, B, 3) -> objects (E, H + 1, B, 14), held
        logits (E, H, B)."""
        E, B = obj0.shape[:2]
        H = command.shape[0]
        obj, box, table = obj0.reshape(E * B, -1), box.reshape(E * B, 3), table.reshape(E * B, 3)
        grip, command = grip.reshape(H + 1, E * B, -1), command.reshape(H, E * B)
        objs, logits = [obj], []
        for k in range(H):
            obj, logit = self._step(objs[-1], grip[k], grip[k + 1], command[k], box, table)
            objs.append(obj)
            logits.append(logit)
        return torch.stack(objs).view(H + 1, E, B, -1).transpose(0, 1), torch.stack(logits).view(H, E, B).transpose(0, 1)

    def rollout_all(self, obj0, grip, command, box, table) -> tuple[torch.Tensor, torch.Tensor]:
        """Every member on the same inputs (as :meth:`ObjectWorldModel.rollout`): objects
        (E, H + 1, B, 14) and held logits (E, H, B)."""
        E = self.num_members

        def rep(x: torch.Tensor, dim: int) -> torch.Tensor:
            return x.unsqueeze(dim).expand(*x.shape[:dim], E, *x.shape[dim:])

        return self.rollout_members(rep(obj0, 0), rep(grip, 1), rep(command, 1), rep(box, 0), rep(table, 0))

    @staticmethod
    def mean(objs: torch.Tensor) -> torch.Tensor:
        quat = objs[..., 3:7] * torch.where((objs[..., 3:7] * objs[:1, ..., 3:7]).sum(-1, keepdim=True) < 0, -1.0, 1.0)
        quat = quat.mean(0)
        return torch.cat([objs[..., :3].mean(0), quat / quat.norm(dim=-1, keepdim=True), objs[..., 7:].mean(0)], dim=-1)

    @staticmethod
    def spread(objs: torch.Tensor) -> torch.Tensor:
        """(B,) disagreement of the members on the final object position (m)."""
        return objs[:, -1, :, :3].std(0).norm(dim=-1) if len(objs) > 1 else torch.zeros(objs.shape[2], device=objs.device)

    def rollout(self, obj0, grip, command, box, table) -> tuple[torch.Tensor, torch.Tensor]:
        objs, logits = self.rollout_all(obj0, grip, command, box, table)
        return self.mean(objs), logits.mean(0)


def load_object_wm(state: dict, dt: float) -> ObjectWorldModel | ObjectWMEnsemble:
    """The object world model of a train_v3 checkpoint, single or ensemble."""
    cfg = ObjectWMCfg(hidden=tuple(state["args"]["wm_hidden"]), dt=dt)
    weights = state["object_wm"]
    members = weights["net.norm_weight"].shape[0] if "net.norm_weight" in weights else 0
    model = ObjectWMEnsemble(cfg, members) if members else ObjectWorldModel(cfg)
    model.load_state_dict(weights)
    return model


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
