"""Per-cycle histograms of what training moves: the episodes' outcomes, S1's output and S0's execution.

Over a collection's policy envs (not the N(0, I) ones) at the curriculum frontier:

- ``success``: how each episode ended, placed / failed (dropped or fallen) / timed out, and
  ``success_potential`` the task potential the timed-out ones ended at (0 .. 1, 1 = the can at
  the goal);
- ``s1_noise_norm``: per decision, ``|w| / sqrt(dim)`` of the noise S1 chose (about 1 under N(0, I),
  the unsteered policy);
- ``s1_intent_hand_can_cm``: per step, how far the reference's right hand (``ref_kp``, the hand the
  intent asks for) is from the can;
- ``s0_hand_err_cm``: per step, the right hand's distance to the reference's right hand (S0's
  tracking error, as ``track/err_cm``).

Fixed bins, values past the last edge counted in the last bin, so cycles and ranks add up.
``train_v3`` appends one record per collection to ``<run>/dists.jsonl``; ``tools/plot_distributions.py``
draws them over the cycles and ``tools/metrics_to_tensorboard.py`` mirrors them as histograms.
"""

from __future__ import annotations

import torch

from hskill.data.episodes import Episodes
from hskill.embodiment import g1_aloha as C

EDGES = {
    "success_potential": (0.0, 1.0, 20),
    "s1_noise_norm": (0.0, 4.0, 40),
    "s1_intent_hand_can_cm": (0.0, 100.0, 50),
    "s0_hand_err_cm": (0.0, 20.0, 40),
}
OUTCOMES = ("placed", "failed", "timeout")


def edges(name: str) -> torch.Tensor:
    lo, hi, bins = EDGES[name]
    return torch.linspace(lo, hi, bins + 1)


def _histogram(name: str, values: torch.Tensor) -> torch.Tensor:
    lo, hi, bins = EDGES[name]
    index = ((values.float() - lo) / (hi - lo) * bins).floor().clamp(0, bins - 1).long()
    return torch.bincount(index, minlength=bins).float()


def episode_histograms(ep: Episodes, envs: torch.Tensor) -> dict[str, torch.Tensor]:
    """Counts of the distributions above over the envs of ``envs`` (N,) bool and the policy's envs."""
    envs = envs & ~ep.base
    failed = (ep.dropped | ep.fallen) & ~ep.placed
    timeout = ~ep.placed & ~failed
    out = {"success": torch.stack([(envs & ep.placed).sum(), (envs & failed).sum(), (envs & timeout).sum()]).float()}
    out["success_potential"] = _histogram("success_potential", ep.final_potential[envs & timeout])
    live = ep.decision_live & envs[None]
    norm = ep.noise.norm(dim=-1) / ep.noise.shape[-1] ** 0.5
    out["s1_noise_norm"] = _histogram("s1_noise_norm", norm[live])
    alive = ep.alive & envs[None]
    hand, ref_hand, can = ep.kp[..., C.RIGHT_EEF, :3], ep.ref_kp[..., C.RIGHT_EEF, :3], ep.obj[..., :3]
    out["s1_intent_hand_can_cm"] = _histogram("s1_intent_hand_can_cm", 100 * (ref_hand - can).norm(dim=-1)[alive])
    out["s0_hand_err_cm"] = _histogram("s0_hand_err_cm", 100 * (hand - ref_hand).norm(dim=-1)[alive])
    return out


def record(cycle: int, level: int, counts: dict[str, torch.Tensor]) -> dict:
    """One ``dists.jsonl`` line."""
    out = {"cycle": cycle, "level": level, "outcomes": list(OUTCOMES)}
    for name, c in counts.items():
        out[name] = {"counts": c.tolist()} if name == "success" else {"edges": edges(name).tolist(), "counts": c.tolist()}
    return out


def layer_points(ep: Episodes, envs: torch.Tensor, rim: float, dt: float, every: int = 5) -> dict[str, torch.Tensor]:
    """Where each layer puts the can, every ``every`` steps of the envs of ``envs`` (policy envs only),
    relative to the box centre (x, y) and its rim (z), for tools/plot_layers.py:

    - ``executed``: the can as simulated (S1's intent as S0 executed it);
    - ``intent``: the can carried by the reference's right hand (``ref_kp``, the reference executed
      exactly) with the grip the hand has on it at that step, ``ref_hand + R_ref R_hand^T (can - hand)``;
      only while the hand holds the can (before, the intent does not say where the can goes).

    Each comes with ``*_t``, the time since the episode start (its curriculum phase's start)."""
    from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

    envs = envs & ~ep.base
    steps = torch.arange(0, ep.alive.shape[0], every, device=ep.alive.device)
    alive = ep.alive[steps] & envs[None]
    t = (steps.float() * dt)[:, None].expand_as(alive)
    origin = torch.stack([ep.box[:, 0], ep.box[:, 1], torch.full_like(ep.box[:, 0], rim)], -1)
    can = ep.obj[steps, :, :3] - origin
    hand, ref = ep.kp[steps][..., C.RIGHT_EEF, :], ep.ref_kp[steps][..., C.RIGHT_EEF, :]
    flat = lambda x: x.reshape(-1, x.shape[-1])  # noqa: E731
    grip = quat_apply_inverse(flat(hand[..., 3:7]), flat(ep.obj[steps, :, :3] - hand[..., :3]))
    intent = ref[..., :3] + quat_apply(flat(ref[..., 3:7]), grip).view_as(can) - origin
    held = alive & ep.held[steps]
    return {"executed": can[alive], "executed_t": t[alive], "intent": intent[held], "intent_t": t[held]}
