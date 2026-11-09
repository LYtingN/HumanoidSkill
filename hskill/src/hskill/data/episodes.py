"""Recorded simulated episodes and imagination starts."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Episodes:
    """One collection rollout: T steps x N envs, D = T / 8 decisions, H-step Phi windows."""

    grip: torch.Tensor  # (T + 1, N, 7) right gripper base pose
    command: torch.Tensor  # (T, N) commanded right opening of each step
    obj: torch.Tensor  # (T + 1, N, 14) can pose / velocities, right finger opening
    held: torch.Tensor  # (T + 1, N) bool, both right fingers touch the can
    alive: torch.Tensor  # (T + 1, N) bool, the state belongs to the episode
    phi_grip: torch.Tensor  # (D, H + 1, N, 7) Phi's gripper poses of the window from decision d
    cond: torch.Tensor  # (D, N, tokens, emb) bf16 diffusion observation tokens
    s0: torch.Tensor  # (D, N, 71) robot state at the decision
    obs0: torch.Tensor  # (D, N, 930) tracker observation at the decision
    decision_alive: torch.Tensor  # (D, N) bool
    xi: torch.Tensor  # (N, 33)
    box: torch.Tensor  # (N, 3) box centre
    table: torch.Tensor  # (N, 3) table-top centre
    goal: torch.Tensor  # (N, 3) placement goal
    final_potential: torch.Tensor  # (N,) the episode's return (task potential at its end)
    base: torch.Tensor  # (N,) bool, the env ran on N(0, I) noise
    execute_steps: int = 8
    # Right gripper pose of the reference's arm joints on the actual body after each step (the
    # hand an exact arm tracker would have): the measured counterpart of the ideal branch.
    ref_grip: torch.Tensor | None = None  # (T + 1, N, 7)
    # Per decision, for the simulator's own GRPO update: the noise taken, its log-likelihood and
    # whether the env was still in its episode.
    noise: torch.Tensor | None = None  # (D, N, dim)
    log_prob: torch.Tensor | None = None  # (D, N)
    decision_live: torch.Tensor | None = None  # (D, N) bool
    end_step: torch.Tensor | None = None  # (N,) last step of each episode (done, failed or time-out)
    # How each episode ended (none of them: time-out).
    placed: torch.Tensor | None = None  # (N,) bool
    dropped: torch.Tensor | None = None  # (N,) bool
    fallen: torch.Tensor | None = None  # (N,) bool (or diverged)
