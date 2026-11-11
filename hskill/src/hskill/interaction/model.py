"""Object world model: the target can's future, given the robot's keypoints and gripper commands.

One prediction covers ``horizon`` 50 Hz steps after the current step t:

    o_{t+1 .. t+H} = M( K_{t-L+1 .. t}, w_{t-L+1 .. t}, o_{t-L+1 .. t},   history
                        K_{t+1 .. t+H}, u_{t .. t+H-1} )                    condition

* ``K``: the keypoints of ``KEYPOINT_BODIES`` (pelvis, left / right ankle, left / right gripper
  base), each ``[pos (3) | quat wxyz (4)]`` in the env frame;
* ``w``: both measured gripper openings; ``u``: both commanded openings (straight from the
  reference, the grippers bypass the tracker);
* ``o``: the object, history ``[pos (3) | quat (4) | lin vel (3)]`` (the acceleration is their
  finite difference), prediction ``[pos | quat | lin vel | lin acc (3)]``.

The future keypoints are an input, not a prediction: the same history with different future
hands (Phi's for the actual branch J, the reference's for the ideal branch J*, any candidate
intent) gives different futures of the object. No scene object (box, table) is an input.

All inputs are expressed in the heading frame of the pelvis at t (yaw only, origin under the
pelvis, z kept), a frame fixed for the whole window. The output (``ObjectWMCfg.output``):

* ``"delta"`` (default): per-step increments of the object's position and rotation in that
  frame, summed from its pose at t. The zero output (the initialization) is an object at rest; no
  grasp or contact is assumed, so the same model serves held, untouched, released, dropped or
  pushed objects.
* ``"rigid"``: a residual over a can held rigidly by the right gripper base from t on (earlier
  checkpoints).

Velocities and accelerations are the finite differences of the predicted path plus learned
corrections, and a logit per step tells whether both fingers hold the can. Longer windows chain
predictions, the predicted object becoming history.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from mjlab.utils.lab_api.math import (
    matrix_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_inv,
    quat_mul,
    yaw_quat,
)
from torch import nn

from hskill.embodiment.g1_aloha import KEYPOINT_BODIES, RIGHT_EEF

OBJ_DIM = 14  # simulated object record: pos, quat, lin vel, ang vel, right finger opening
HIST_OBJ_DIM = 10  # object history input: pos, quat, lin vel
PRED_DIM = 13  # predicted object: pos, quat, lin vel, lin acc
NUM_KP = len(KEYPOINT_BODIES)
KP_DIM = 7
HIST_FRAME_DIM = NUM_KP * 9 + 2 + 3 + 6 + 3 + 3
FUT_FRAME_DIM = NUM_KP * 9 + 2
OUT_FRAME_DIM = 3 + 3 + 3 + 3 + 1


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


def _apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    batch = torch.broadcast_shapes(q.shape[:-1], v.shape[:-1])
    return quat_apply(q.expand(*batch, 4), v.expand(*batch, 3))


def _apply_inv(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    batch = torch.broadcast_shapes(q.shape[:-1], v.shape[:-1])
    return quat_apply_inverse(q.expand(*batch, 4), v.expand(*batch, 3))


def _mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    shape = torch.broadcast_shapes(a.shape, b.shape)
    return quat_mul(a.expand(shape), b.expand(shape))


def object_targets(obj: torch.Tensor, dt: float) -> torch.Tensor:
    """Simulated object records (T, ..., >= 10) -> prediction targets (T, ..., 13): pos, quat, lin
    vel and the finite-difference acceleration (0 at the first frame)."""
    vel = obj[..., 7:10]
    acc = torch.cat([torch.zeros_like(vel[:1]), (vel[1:] - vel[:-1]) / dt])
    return torch.cat([obj[..., :10], acc], dim=-1)


@dataclass
class ObjectWMCfg:
    history: int = 10  # L frames of keypoints, openings and object, the current one last
    horizon: int = 32  # H predicted frames per call
    hidden: tuple[int, ...] = (512, 512, 512)
    dt: float = 0.02
    # Feature scales (heading frame): positions over 0.5 m, velocities 0.5 m/s, accelerations 10 m/s^2.
    pos_scale: float = 0.5
    vel_scale: float = 0.5
    acc_scale: float = 10.0
    opening_scale: float = 0.065
    # "delta": per-step increments from the current pose (no grasp assumed); "rigid": a residual over
    # the can held rigidly by the right gripper base.
    output: Literal["delta", "rigid"] = "delta"
    # Output scales: per-step increments ("delta") or the residual over the rigid grasp ("rigid").
    out_pos: float = 0.05
    out_rot: float = 0.1
    out_vel: float = 0.1
    out_acc: float = 1.0

    @property
    def in_dim(self) -> int:
        return self.history * HIST_FRAME_DIM + self.horizon * FUT_FRAME_DIM

    @property
    def out_dim(self) -> int:
        return self.horizon * OUT_FRAME_DIM


def _features(c: ObjectWMCfg, hist_kp, hist_width, hist_obj, fut_kp, fut_cmd) -> tuple[torch.Tensor, torch.Tensor]:
    """Time-major inputs (L | H, B, ...) -> network input (B, in_dim) and the heading (B, 4)."""
    pelvis = hist_kp[-1, :, 0]
    head = yaw_quat(pelvis[:, 3:7])
    origin = nn.functional.pad(pelvis[:, :2], (0, 1))
    head_inv = quat_inv(head)

    def kp_feat(kp):  # (T, B, 5, 7) -> (T, B, 45)
        pos = _apply_inv(head[:, None], kp[..., :3] - origin[:, None]) / c.pos_scale
        return torch.cat([pos, rot6d(_mul(head_inv[:, None], kp[..., 3:]))], dim=-1).flatten(-2)

    pos, quat, vel = hist_obj[..., :3], hist_obj[..., 3:7], hist_obj[..., 7:10]
    acc = (vel[1:] - vel[:-1]) / c.dt
    acc = torch.cat([acc[:1], acc]) if len(acc) else torch.zeros_like(vel)
    hist = torch.cat([
        kp_feat(hist_kp), hist_width / c.opening_scale,
        _apply_inv(head, pos - origin) / c.pos_scale, rot6d(_mul(head_inv, quat)),
        _apply_inv(head, vel) / c.vel_scale, _apply_inv(head, acc) / c.acc_scale,
    ], dim=-1)  # fmt: skip
    fut = torch.cat([kp_feat(fut_kp), fut_cmd / c.opening_scale], dim=-1)
    return torch.cat([hist.transpose(0, 1).flatten(1), fut.transpose(0, 1).flatten(1)], dim=-1), head


def _decode(c: ObjectWMCfg, out: torch.Tensor, head: torch.Tensor, hist_kp, hist_obj, fut_kp) -> tuple[torch.Tensor, torch.Tensor]:
    """Network output (B, out_dim) -> objects (H + 1, B, 13) (frame 0: the current one) and held
    logits (H, B)."""
    H = c.horizon
    y = out.view(len(out), H, OUT_FRAME_DIM).transpose(0, 1)  # (H, B, 13)
    p0, q0, v0 = hist_obj[-1, :, :3], hist_obj[-1, :, 3:7], hist_obj[-1, :, 7:10]
    if c.output == "delta":
        # Summed increments from the current pose: displacement and rotation vector in the t frame.
        pos = p0[None] + _apply(head[None], torch.cumsum(y[..., 0:3] * c.out_pos, dim=0))
        quat = _mul(quat_exp(_apply(head[None], torch.cumsum(y[..., 3:6] * c.out_rot, dim=0))), q0[None])
    else:
        g0 = hist_kp[-1, :, RIGHT_EEF]
        rel_pos = _apply_inv(g0[:, 3:], p0 - g0[:, :3])
        rel_quat = _mul(quat_inv(g0[:, 3:]), q0)
        g = fut_kp[:, :, RIGHT_EEF]  # (H, B, 7)
        pos = g[..., :3] + _apply(g[..., 3:], rel_pos[None]) + _apply(head[None], y[..., 0:3] * c.out_pos)
        quat = _mul(quat_exp(_apply(head[None], y[..., 3:6] * c.out_rot)), _mul(g[..., 3:], rel_quat[None]))
    quat = quat / quat.norm(dim=-1, keepdim=True)
    vel = (pos - torch.cat([p0[None], pos[:-1]])) / c.dt + _apply(head[None], y[..., 6:9] * c.out_vel)
    acc = (vel - torch.cat([v0[None], vel[:-1]])) / c.dt + _apply(head[None], y[..., 9:12] * c.out_acc)
    a0 = (v0 - hist_obj[-2, :, 7:10]) / c.dt if len(hist_obj) > 1 else torch.zeros_like(v0)
    first = torch.cat([p0, q0, v0, a0], dim=-1)
    return torch.cat([first[None], torch.cat([pos, quat, vel, acc], dim=-1)]), y[..., 12]


class _ObjectWMBase(nn.Module):
    """Shared prediction, chaining and ensemble statistics; subclasses supply :meth:`_net` over
    (E, B, in_dim) -> (E, B, out_dim)."""

    cfg: ObjectWMCfg
    num_members: int

    def __init__(self, cfg: ObjectWMCfg | None):
        super().__init__()
        self.cfg = cfg or ObjectWMCfg()
        self._predict = self.predict_members

    def compile_step(self) -> None:
        """Run :meth:`predict_members` compiled (forward and backward), as in rollouts."""
        self._predict = torch.compile(self.predict_members, dynamic=None)

    def _net(self, feat: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def predict_members(self, hist_kp, hist_width, hist_obj, fut_kp, fut_cmd) -> tuple[torch.Tensor, torch.Tensor]:
        """One horizon, every member on its own inputs (time-major, member second): ``hist_kp``
        (L, E, B, 5, 7), ``hist_width`` (L, E, B, 2), ``hist_obj`` (L, E, B, 10), ``fut_kp``
        (H, E, B, 5, 7), ``fut_cmd`` (H, E, B, 2) -> objects (E, H + 1, B, 13), logits (E, H, B)."""
        E, B = hist_kp.shape[1:3]

        def flat(x):
            return x.reshape(x.shape[0], E * B, *x.shape[3:])

        hk, hw, ho, fk, fc = map(flat, (hist_kp, hist_width, hist_obj, fut_kp, fut_cmd))
        feat, head = _features(self.cfg, hk, hw, ho, fk, fc)
        out = self._net(feat.view(E, B, -1)).flatten(0, 1)
        objs, logits = _decode(self.cfg, out, head, hk, ho, fk)
        H = self.cfg.horizon
        return objs.view(H + 1, E, B, -1).transpose(0, 1), logits.view(H, E, B).transpose(0, 1)

    def rollout_members(self, hist_kp, hist_width, hist_obj, fut_kp, fut_cmd, fut_width=None) -> tuple[torch.Tensor, torch.Tensor]:
        """Any number W of future frames: whole horizons chained, the predicted object (and the
        future keypoints and openings: ``fut_width`` (W, E, B, 2), else the commands) becoming the
        history of the next one. -> objects (E, W + 1, B, 13), logits (E, W, B)."""
        H, L, W = self.cfg.horizon, self.cfg.history, fut_kp.shape[0]
        if fut_width is None:
            fut_width = fut_cmd
        objs, logits = [], []
        for s in range(0, W, H):
            idx = torch.arange(s, s + H, device=fut_kp.device).clamp(max=W - 1)  # pad the last horizon
            o, lg = self._predict(hist_kp, hist_width, hist_obj, fut_kp[idx], fut_cmd[idx])
            n = min(H, W - s)
            objs.append(o[:, : n + 1] if s == 0 else o[:, 1 : n + 1])
            logits.append(lg[:, :n])
            if s + H < W:
                hist_kp = torch.cat([hist_kp, fut_kp[s : s + H]])[-L:]
                hist_width = torch.cat([hist_width, fut_width[s : s + H]])[-L:]
                hist_obj = torch.cat([hist_obj, o[:, 1:, :, :HIST_OBJ_DIM].transpose(0, 1)])[-L:]
        return torch.cat(objs, dim=1), torch.cat(logits, dim=1)

    def rollout_all(self, hist_kp, hist_width, hist_obj, fut_kp, fut_cmd, fut_width=None) -> tuple[torch.Tensor, torch.Tensor]:
        """Every member on the same inputs (time-major, no member dim): objects (E, W + 1, B, 13),
        logits (E, W, B)."""
        E = self.num_members

        def rep(x):
            return None if x is None else x.unsqueeze(1).expand(x.shape[0], E, *x.shape[1:])

        return self.rollout_members(*map(rep, (hist_kp, hist_width, hist_obj, fut_kp, fut_cmd, fut_width)))

    def rollout(self, hist_kp, hist_width, hist_obj, fut_kp, fut_cmd, fut_width=None) -> tuple[torch.Tensor, torch.Tensor]:
        """The members' mean object (W + 1, B, 13) and mean held logit (W, B)."""
        objs, logits = self.rollout_all(hist_kp, hist_width, hist_obj, fut_kp, fut_cmd, fut_width)
        return self.mean(objs), logits.mean(0)

    @staticmethod
    def mean(objs: torch.Tensor) -> torch.Tensor:
        """Members' mean (E, ..., 13) -> (..., 13): positions and derivatives averaged, the
        quaternions sign-aligned to member 0, averaged and re-normalized."""
        quat = objs[..., 3:7] * torch.where((objs[..., 3:7] * objs[:1, ..., 3:7]).sum(-1, keepdim=True) < 0, -1.0, 1.0)
        quat = quat.mean(0)
        return torch.cat([objs[..., :3].mean(0), quat / quat.norm(dim=-1, keepdim=True), objs[..., 7:].mean(0)], dim=-1)

    @staticmethod
    def spread(objs: torch.Tensor) -> torch.Tensor:
        """(B,) disagreement of the members on the final object position (m)."""
        return objs[:, -1, :, :3].std(0).norm(dim=-1) if len(objs) > 1 else torch.zeros(objs.shape[2], device=objs.device)


class ObjectWorldModel(_ObjectWMBase):
    """A single model (``num_members`` = 1)."""

    num_members = 1

    def __init__(self, cfg: ObjectWMCfg | None = None):
        super().__init__(cfg)
        cfg = self.cfg
        layers, d = [], cfg.in_dim
        for h in cfg.hidden:
            layers += [nn.Linear(d, h), nn.SiLU()]
            d = h
        last = nn.Linear(d, cfg.out_dim)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        self.net = nn.Sequential(nn.LayerNorm(cfg.in_dim), *layers, last)

    def _net(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat[0])[None]


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


class ObjectWMEnsemble(_ObjectWMBase):
    """``members`` object world models trained on their own batches; their spread over a rollout is
    the model's uncertainty (large where its data never went, e.g. hand paths no execution took).
    The members are evaluated together (stacked weights, batched matrix products)."""

    def __init__(self, cfg: ObjectWMCfg | None = None, members: int = 3):
        super().__init__(cfg)
        self.num_members = members
        self.net = EnsembleMLP(members, self.cfg.in_dim, self.cfg.hidden, self.cfg.out_dim)

    def _net(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)


def load_object_wm(state: dict, dt: float) -> ObjectWorldModel | ObjectWMEnsemble:
    """The object world model of a train_v3 checkpoint, single or ensemble."""
    args = state["args"]
    # Checkpoints from before ``wm_output`` predicted the residual over the rigid grasp.
    cfg = ObjectWMCfg(history=args["wm_history"], horizon=args["horizon"], hidden=tuple(args["wm_hidden"]), dt=dt,
                      output=args.get("wm_output", "rigid"))  # fmt: skip
    weights = state["object_wm"]
    members = weights["net.norm_weight"].shape[0] if "net.norm_weight" in weights else 0
    model = ObjectWMEnsemble(cfg, members) if members else ObjectWorldModel(cfg)
    model.load_state_dict(weights)
    return model


@dataclass
class ObjectLossCfg:
    # Nominal error scales: 1 cm, ~0.1 rad, 0.1 m/s, 2 m/s^2.
    pos: float = 0.01
    rot: float = 0.1
    vel: float = 0.1
    acc: float = 2.0
    held_weight: float = 0.1


def object_loss(pred: torch.Tensor, logits: torch.Tensor, true: torch.Tensor, held: torch.Tensor, mask: torch.Tensor,
                cfg: ObjectLossCfg | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:  # fmt: skip
    """Normalized squared errors of frames 1..H of ``pred`` against ``true`` (H + 1, B, 13: see
    :func:`object_targets`), ``held`` (H, B) labels, ``mask`` (H, B) the steps still inside the
    episode."""
    cfg = cfg or ObjectLossCfg()
    p, t = pred[1:], true[1:]
    dot = (p[..., 3:7] * t[..., 3:7]).sum(-1)
    terms = {
        "pos": (p[..., :3] - t[..., :3]).square().sum(-1) / cfg.pos**2,
        # 4 (1 - <q1, q2>^2) ~ (rotation angle)^2, smooth at zero error.
        "rot": 4 * (1 - dot.square()).clamp(min=0) / cfg.rot**2,
        "vel": (p[..., 7:10] - t[..., 7:10]).square().sum(-1) / cfg.vel**2,
        "acc": (p[..., 10:13] - t[..., 10:13]).square().sum(-1) / cfg.acc**2,
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
