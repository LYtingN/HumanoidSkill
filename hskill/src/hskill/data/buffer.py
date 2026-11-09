"""Recent collection rollouts, sampled with more weight on newer ones.

Each sample first picks a rollout with probability ``decay ** age`` (age 0 = newest), then a
uniform (decision, env) pair among those alive at the decision with a complete window.
:meth:`windows` gives object-WM training windows, conditioned on the simulated gripper poses or,
for a ``phi_fraction`` of them, on Phi's (the input the model sees at imagination time); the
labels are always the simulated can. :meth:`starts` gives imagination starts and, on the noise
policy's envs only, the critic's samples.
"""

from __future__ import annotations

import torch

from hskill.data.episodes import Episodes

TIME_MAJOR = ("obj", "grip", "command", "held", "mask")  # (time, sample, ...) window fields


class EpisodeBuffer:
    def __init__(self, capacity: int, decay: float, horizon: int):
        self.capacity, self.decay, self.horizon = capacity, decay, horizon
        self.episodes: list[Episodes] = []
        self.valid: list[torch.Tensor] = []  # (M, 2) alive (decision, env) pairs per rollout
        self.valid_policy: list[torch.Tensor] = []  # the same on the noise policy's envs

    def __len__(self) -> int:
        return len(self.episodes)

    def add(self, episodes: Episodes) -> None:
        self.episodes.append(episodes)
        self.valid.append(episodes.decision_alive.nonzero())
        self.valid_policy.append((episodes.decision_alive & ~episodes.base[None]).nonzero())
        if len(self.episodes) > self.capacity:
            self.episodes.pop(0)
            self.valid.pop(0)
            self.valid_policy.pop(0)

    def _pick(self, n: int, policy_only: bool = False) -> list[tuple[Episodes, torch.Tensor, torch.Tensor]]:
        """(rollout, decisions, envs) groups of ``n`` recency-weighted samples."""
        valids = self.valid_policy if policy_only else self.valid
        ages = torch.arange(len(self.episodes) - 1, -1, -1, dtype=torch.float32)
        weight = self.decay**ages * torch.tensor([float(len(v) > 0) for v in valids])
        which = torch.multinomial(weight / weight.sum(), n, replacement=True)
        out = []
        for i in which.unique().tolist():
            m = int((which == i).sum())
            valid = valids[i]
            pick = valid[torch.randint(len(valid), (m,), device=valid.device)]
            out.append((self.episodes[i], pick[:, 0], pick[:, 1]))
        return out

    def windows(self, n: int, phi_fraction: float) -> dict[str, torch.Tensor]:
        H = self.horizon
        parts = []
        for ep, d, e in self._pick(n):
            T = ep.command.shape[0]
            t = d[None] * ep.execute_steps + torch.arange(H + 1, device=d.device)[:, None]  # (H + 1, m)
            inside = t <= T  # windows longer than Phi's may run past the rollout's end: masked
            t = t.clamp(max=T)
            grip = ep.grip[t, e]
            if phi_fraction > 0:  # Phi's windows have Phi's horizon, which is then H
                phi = torch.rand(len(d), device=d.device) < phi_fraction
                grip = torch.where(phi[None, :, None], ep.phi_grip[d, :, e].transpose(0, 1), grip)
            parts.append({
                "obj": ep.obj[t, e], "grip": grip, "command": ep.command[t[:-1].clamp(max=T - 1), e],
                "held": ep.held[t[1:], e], "mask": ep.alive[t[1:], e] & inside[1:],
                "box": ep.box[e], "table": ep.table[e],
            })  # fmt: skip
        return {k: torch.cat([p[k] for p in parts], dim=1 if k in TIME_MAJOR else 0) for k in parts[0]}

    def starts(self, n: int, policy_only: bool = False) -> dict[str, torch.Tensor]:
        """Recorded decisions: imagination starts, and with ``final`` (the episode's return) and
        ``time`` (share of the episode elapsed; ``step_frac`` per step) the critic's samples."""
        parts = []
        for ep, d, e in self._pick(n, policy_only):
            steps = ep.command.shape[0]
            parts.append({
                "cond": ep.cond[d, e], "s0": ep.s0[d, e], "obs0": ep.obs0[d, e], "xi": ep.xi[e],
                "obj0": ep.obj[d * ep.execute_steps, e], "box": ep.box[e], "table": ep.table[e], "goal": ep.goal[e],
                "final": ep.final_potential[e], "time": (d * ep.execute_steps).float() / steps,
                "step_frac": torch.full_like(d, 1, dtype=torch.float32) / steps,
                # Steps from the decision to the episode's end (for discounted critic targets).
                "to_end": ((ep.end_step[e] if ep.end_step is not None else torch.full_like(e, steps)) - d * ep.execute_steps).float(),
            })  # fmt: skip
        return {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
