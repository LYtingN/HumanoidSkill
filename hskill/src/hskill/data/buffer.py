"""Recent collection rollouts, sampled with more weight on newer ones.

Each sample first picks a rollout with probability ``decay ** age`` (age 0 = newest), then a
uniform (decision, env) pair among those alive at the decision with a complete window.
:meth:`windows` gives object-WM training windows: the ``history`` frames up to a decision
(keypoints, openings, can) and the ``horizon`` frames after it, the future keypoints simulated or,
for a ``phi_fraction`` of them, Phi's (the input the model sees at imagination time), or for an
``ideal_fraction`` of the exactly executed windows (right-hand tracking error below ``well_tracked``
throughout) the ideal branch's (the reference executed exactly: there the simulated can is what
that ideal input brings about, so the model also learns the ideal branch's kind of input); the
labels are always the simulated can, and ``fail`` / ``done`` mark the step each episode ended
dropping or toppling the can / placing it (the event head's labels). Frames before an episode's start repeat its first frame. :meth:`starts` gives imagination starts and, on the noise
policy's envs only, the critic's samples.
"""

from __future__ import annotations

import torch

from hskill.data.episodes import Episodes
from hskill.embodiment.g1_aloha import RIGHT_EEF

# (time, sample, ...) window fields
TIME_MAJOR = ("hist_kp", "hist_width", "hist_obj", "fut_kp", "fut_cmd", "obj", "held", "mask", "fail", "done")


class EpisodeBuffer:
    def __init__(self, capacity: int, decay: float, horizon: int, history: int = 10):
        self.capacity, self.decay, self.horizon, self.history = capacity, decay, horizon, history
        self.episodes: list[Episodes] = []
        self.valid: list[torch.Tensor] = []  # (M, 2) alive (decision, env) pairs per rollout
        self.valid_policy: list[torch.Tensor] = []  # the same on the noise policy's envs

    def __len__(self) -> int:
        return len(self.episodes)

    def add(self, episodes: Episodes) -> None:
        self.episodes.append(episodes)
        self.valid.append(episodes.decision_alive.nonzero())
        self.valid_policy.append((episodes.decision_alive & episodes.policy_envs()[None]).nonzero())
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

    def _history(self, ep: Episodes, d: torch.Tensor, e: torch.Tensor) -> dict[str, torch.Tensor]:
        """The ``history`` frames up to decisions ``d`` of envs ``e``, time-major (L, m, ...)."""
        L = self.history
        h = (d[None] * ep.execute_steps + torch.arange(1 - L, 1, device=d.device)[:, None]).clamp(min=0)
        return {"hist_kp": ep.kp[h, e], "hist_width": ep.width[h, e], "hist_obj": ep.obj[h, e, :10]}

    def windows(self, n: int, phi_fraction: float, ideal_fraction: float = 0.0, well_tracked: float = 0.02,
                exact_only: bool = False, phi_only: bool = False) -> dict[str, torch.Tensor]:  # fmt: skip
        """Training windows; ``exact_only`` masks out every window that was not executed exactly,
        ``phi_only`` every window Phi did not run for (``ideal`` (m,) marks the windows given the ideal
        branch's keypoints). Phi's keypoints only go to the envs Phi ran windows for."""
        H = self.horizon
        parts = []
        for ep, d, e in self._pick(n):
            T = ep.command.shape[0]
            t = d[None] * ep.execute_steps + torch.arange(H + 1, device=d.device)[:, None]  # (H + 1, m)
            inside = t <= T  # windows longer than Phi's may run past the rollout's end: masked
            t = t.clamp(max=T)
            fut_kp = ep.kp[t[1:], e]
            mask = ep.alive[t[1:], e] & inside[1:]
            phi = torch.zeros(len(d), dtype=torch.bool, device=d.device)
            has_phi = ep.phi_env[e] if ep.phi_env is not None else torch.ones_like(phi)
            if phi_only:
                mask = mask & has_phi[None]
            if phi_fraction > 0:  # Phi's windows have Phi's horizon, which is then H
                phi = (torch.rand(len(d), device=d.device) < phi_fraction) & has_phi
                fut_kp = torch.where(phi[None, :, None, None], ep.phi_kp[d, 1:, e].transpose(0, 1), fut_kp)
            ideal = torch.zeros_like(phi)
            if (ideal_fraction > 0 or exact_only) and ep.ref_kp is not None:
                err = (ep.ref_kp[t[1:], e, RIGHT_EEF, :3] - ep.kp[t[1:], e, RIGHT_EEF, :3]).norm(dim=-1)
                exact = torch.where(mask, err, 0).amax(0) < well_tracked
                ideal = exact & ~phi & (torch.rand(len(d), device=d.device) < ideal_fraction)
                fut_kp = torch.where(ideal[None, :, None, None], ep.ref_kp[t[1:], e], fut_kp)
                if exact_only:
                    mask = mask & exact[None]
            # The step each episode ended, and how (none of them: a time-out or a fall).
            last = t[1:] == (ep.end_step[e] if ep.end_step is not None else torch.full_like(e, -1))[None]
            fail = last & (ep.dropped[e] if ep.dropped is not None else torch.zeros_like(phi))[None]
            done = last & (ep.placed[e] if ep.placed is not None else torch.zeros_like(phi))[None]
            parts.append({
                **self._history(ep, d, e), "fut_kp": fut_kp, "fut_cmd": ep.command[t[:-1].clamp(max=T - 1), e],
                "obj": ep.obj[t, e], "held": ep.held[t[1:], e], "mask": mask, "ideal": ideal,
                "fail": fail & mask, "done": done & mask, "box": ep.box[e],
            })  # fmt: skip
        return {k: torch.cat([p[k] for p in parts], dim=1 if k in TIME_MAJOR else 0) for k in parts[0]}

    def starts(self, n: int, policy_only: bool = False) -> dict[str, torch.Tensor]:
        """Recorded decisions: imagination starts (with the object WM's history, batch-first
        (m, L, ...)), and with ``final`` (the episode's return) and ``time`` (share of the episode
        elapsed; ``step_frac`` per step) the critic's samples, with the decision's ``noise`` while
        the rollout keeps it."""
        parts = []
        for ep, d, e in self._pick(n, policy_only):
            steps = ep.command.shape[0]
            parts.append({
                **{k: v.transpose(0, 1) for k, v in self._history(ep, d, e).items()},
                "cond": ep.cond[d, e], "s0": ep.s0[d, e], "obs0": ep.obs0[d, e], "xi": ep.xi[e],
                "obj0": ep.obj[d * ep.execute_steps, e], "box": ep.box[e], "table": ep.table[e], "goal": ep.goal[e],
                "final": ep.final_potential[e], "time": (d * ep.execute_steps).float() / steps,
                "step_frac": torch.full_like(d, 1, dtype=torch.float32) / steps,
                # Steps from the decision to the episode's end (for discounted critic targets).
                "to_end": ((ep.end_step[e] if ep.end_step is not None else torch.full_like(e, steps)) - d * ep.execute_steps).float(),
                **({"noise": ep.noise[d, e]} if ep.noise is not None else {}),
            })  # fmt: skip
        # Fields every picked rollout has (not all of them keep the noise).
        return {k: torch.cat([p[k] for p in parts]) for k in parts[0] if all(k in p for p in parts)}
