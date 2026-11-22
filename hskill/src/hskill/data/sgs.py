"""Success-guided sampling over recorded reset configurations (SGS, arXiv:2610.12465)."""

from collections import deque

import torch
import torch.distributed as dist


class SuccessGuidedSampler:
    def __init__(self, starts, train_rows, seed):
        self.rows = torch.as_tensor(train_rows, dtype=torch.long)
        self.phase_ids = starts.phase_ids[self.rows]
        self.phase_count = len(starts.phases)
        self.configs = [(starts.phases[phase], starts.episode_ids[row])
                        for row, phase in zip(self.rows.tolist(), self.phase_ids.tolist(), strict=True)]
        if len(set(self.configs)) != len(self.configs) or not self.configs:
            raise ValueError("SGS needs distinct, nonempty (phase, episode) training starts")
        self.history = {row: deque(maxlen=100) for row in self.rows.tolist()}
        # All ranks select the same phase so a synchronous rollout has one phase-specific time limit.
        self.phase_rng = torch.Generator().manual_seed(seed)

    def weights(self):
        # Manipulation settings: target 0.5, concentration 1, temperature 2, epsilon 1e-4.
        # Unvisited starts get the peak weight until they have a measured outcome.
        p = torch.tensor([sum(h) / len(h) if h else 0.5 for h in self.history.values()])
        eps = 1e-4
        kernel = ((p + eps) * (1 - p + eps)).sqrt()
        return (kernel.clamp_min(eps) + eps).sqrt()

    def sample_phase(self):
        mass = torch.zeros(self.phase_count).scatter_add_(0, self.phase_ids, self.weights())
        return int(torch.multinomial(mass, 1, generator=self.phase_rng))

    def sample(self, count, generator):
        """``count`` configurations drawn i.i.d. from the whole bank by SGS weight (the paper's
        per-environment reset)."""
        indices = torch.multinomial(self.weights(), count, replacement=True, generator=generator)
        return self.rows[indices]

    def phase_metrics(self):
        """Per phase: mean measured success over its visited starts and its share of the sampling mass."""
        p = torch.tensor([sum(h) / len(h) if h else float("nan") for h in self.history.values()])
        mass = self.weights()
        mass = mass / mass.sum()
        out = {}
        for i in range(self.phase_count):
            sel = self.phase_ids == i
            visited = p[sel][~p[sel].isnan()]
            out[i] = (visited.mean().item() if len(visited) else float("nan"), mass[sel].sum().item())
        return out

    def sample_rows(self, phase, count, generator):
        mask = self.phase_ids == phase
        indices = torch.multinomial(self.weights()[mask], count, replacement=True, generator=generator)
        return self.rows[mask][indices]

    def observe(self, rows, placed, policy_mask):
        # Fixed-length packets let all ranks gather outcomes even when policy counts differ.
        outcomes = torch.where(policy_mask, 2 * rows.to(placed.device) + placed.long(), -1)
        if dist.is_initialized():
            merged = outcomes.new_empty(dist.get_world_size() * outcomes.numel())
            dist.all_gather_into_tensor(merged, outcomes.contiguous())
            outcomes = merged
        for code in outcomes.cpu().tolist():
            if code >= 0:
                row, success = divmod(code, 2)
                self.history[row].append(success)

    def metrics(self):
        # Equal weight across the fixed pool for comparing collections from different phases.
        # Unobserved starts have contributed no measured successes.
        n = len(self.history)
        return {"curriculum/coverage": sum(bool(h) for h in self.history.values()) / n,
                "curriculum/pool_success": sum(sum(h) / len(h) for h in self.history.values() if h) / n}

    def state_dict(self):
        return {"configs": self.configs, "history": [list(h) for h in self.history.values()],
                "phase_rng": self.phase_rng.get_state()}

    def load_state_dict(self, state):
        if state["configs"] != self.configs:
            raise ValueError("SGS checkpoint and training reset configurations differ")
        for history, values in zip(self.history.values(), state["history"], strict=True):
            history.clear()
            history.extend(values)
        self.phase_rng.set_state(state["phase_rng"].cpu())
