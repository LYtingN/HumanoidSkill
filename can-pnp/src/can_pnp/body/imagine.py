"""Phi: the frozen SuperMimic tracker driving the body world model through a reference chunk.

Mirrors ``SuperMimicController.act`` without the simulator: at control step ``k`` of the chunk
the tracker sees the goal of reference frames ``k .. k + 9`` and its 10-frame observation
history, and f_omega integrates the state one 50 Hz step under the action. The history after
the start is rebuilt analytically from the imagined states (the scene's ``policy`` group is
noise-free, so it equals ``supermimic.world_model.heads.analytic_observation``).

:func:`reference_states` is the no-model alternative: the reference itself, as if tracked
perfectly, with the root moved by the leg-odometry velocity of the plan.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.utils.lab_api.math import quat_from_matrix

from can_pnp.body.world_model import BodyWorldModel
from can_pnp.supermimic import FUTURE_FRAMES, ReferenceTrajectory, TrackingPolicy

OBS_FRAME_DIM = 93
OBS_TERMS = (29, 29, 3, 3, 29)  # joint_pos, joint_vel, base_ang_vel, projected_gravity, actions


@dataclass
class BodyRollout:
    states: torch.Tensor  # (H + 1, N, 71), states[0] the start
    actions: torch.Tensor  # (H, N, 29) tracker actions
    fall_prob: torch.Tensor  # (H + 1, N) predicted P(terminated by step k), 0 at the start


def unflatten_history(obs: torch.Tensor, frames: int = FUTURE_FRAMES) -> torch.Tensor:
    """Term-major (N, 930) policy observation -> (frames, N, 93), oldest first."""
    terms, start = [], 0
    for dim in OBS_TERMS:
        terms.append(obs[:, start : start + frames * dim].reshape(-1, frames, dim))
        start += frames * dim
    return torch.cat(terms, dim=-1).transpose(0, 1)


def flatten_history(hist: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`unflatten_history`."""
    out, start = [], 0
    for dim in OBS_TERMS:
        out.append(hist[..., start : start + dim].transpose(0, 1).flatten(1))
        start += dim
    return torch.cat(out, dim=-1)


def _reference_root(plan: ReferenceTrajectory, s0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference root position (N, H, 3) and quaternion (N, H, 4): odometry from the start root
    (or the start position held, for a plan without root height / velocity)."""
    quat = quat_from_matrix(plan.root_w)
    if not plan.full:
        return s0[:, None, :3].expand(-1, plan.horizon, -1), quat
    dt = plan.dt
    xy = s0[:, None, :2] + torch.cumsum(plan.root_lin_vel_w[:, :-1, :2] * dt, dim=1)
    xy = torch.cat([s0[:, None, :2], xy], dim=1)
    return torch.cat([xy, plan.root_height[..., None]], dim=-1), quat


class PhiStep:
    """One Phi step as one compiled function (as SuperMimic's ``imagination.compiled.ImagStep``):
    tracker action, f_omega, termination logit and the next observation history. Shared by the
    imagined chunks and the Phi windows of the collection."""

    def __init__(self, body: BodyWorldModel, tracker: TrackingPolicy, compile: bool = True):
        from supermimic.world_model.heads import ReferenceFrame, analytic_observation

        self.body, self.tracker = body, tracker

        def step(hist, goal, s, xi, ref_pos, ref_quat):
            a = tracker(flatten_history(hist), goal)
            s_next = body.step(s, a, xi)
            logit = body.wm.termination_logits(s_next, ReferenceFrame(None, None, None, None, ref_pos, ref_quat))
            hist = torch.cat([hist[1:], analytic_observation(s_next, a, xi, body.default_joint_pos)[None]])
            return a, s_next, logit, hist

        self.fn = torch.compile(step, dynamic=None) if compile else step

    def __call__(self, hist, goal, s, xi, ref_pos=None, ref_quat=None):
        """(10, B, 93) history, (B, goal) goal, (B, 71) state, (B, 33) xi, reference root
        (default: the state's own, when the termination logit is not needed) -> action,
        next state, termination logit of the next state, next history."""
        if ref_pos is None:
            ref_pos, ref_quat = s[:, :3], s[:, 3:7]
        return self.fn(hist, goal, s, xi, ref_pos, ref_quat)


@torch.no_grad()
def imagine_body(
    phi: PhiStep,
    plan: ReferenceTrajectory,
    s0: torch.Tensor,
    obs_policy: torch.Tensor,
    xi: torch.Tensor,
    horizon: int | None = None,
) -> BodyRollout:
    """Roll the tracker + f_omega ``horizon`` steps (default: the whole plan) from state ``s0``
    (N, 71) with the env's ``policy`` observation ``obs_policy`` (N, 930) at that state."""
    horizon = horizon or plan.horizon
    n = s0.shape[0]
    body = phi.body
    hist = unflatten_history(obs_policy)
    ref_pos, ref_quat = _reference_root(plan, s0)
    torso = body.fk.body_index("torso_link")
    s, survive = s0, torch.ones(n, device=s0.device)
    states, actions, fall = [s0], [], [torch.zeros(n, device=s0.device)]
    for k in range(horizon):
        anchor = s[:, 3:7] if plan.full else body.fk(s[:, :36])["quat"][:, torso]
        # SuperMimic labels state k + 1 against the reference frame of transition k.
        j = min(k, plan.horizon - 1)
        a, s, logit, hist = phi(hist, plan.goal(k, anchor), s, xi, ref_pos[:, j], ref_quat[:, j])
        survive = survive * (1 - torch.sigmoid(logit))
        states.append(s)
        actions.append(a)
        fall.append(1 - survive)
    return BodyRollout(torch.stack(states), torch.stack(actions), torch.stack(fall))


@torch.no_grad()
def reference_states(plan: ReferenceTrajectory, s0: torch.Tensor, horizon: int | None = None) -> torch.Tensor:
    """(H + 1, N, 71) states of the reference tracked perfectly (velocities left at zero),
    states[0] = s0."""
    horizon = horizon or plan.horizon
    pos, quat = _reference_root(plan, s0)
    k = torch.clamp(torch.arange(1, horizon + 1, device=s0.device), max=plan.horizon - 1)
    s = torch.zeros(horizon, *s0.shape, device=s0.device)
    s[..., 0:3] = pos[:, k].transpose(0, 1)
    s[..., 3:7] = quat[:, k].transpose(0, 1)
    s[..., 7:36] = plan.joint_pos[:, k].transpose(0, 1)
    return torch.cat([s0[None], s])
