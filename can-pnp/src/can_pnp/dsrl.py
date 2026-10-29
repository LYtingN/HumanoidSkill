"""DSRL: steer the frozen diffusion policy through its initial DDIM noise.

Humanoid-Diffusion samples with DDIM eta=0, so a reference chunk is a deterministic function of
the observation and the initial noise ``w`` (horizon x action_dim). :class:`NoisePolicy` is a
Gaussian policy over ``w`` given the diffusion's own observation tokens; it starts at the
standard normal (mean 0, std 1), i.e. exactly the base policy, and RL moves it from there.
Diffusion, DINOv3 and the SuperMimic tracker stay frozen.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class NoisePolicy(nn.Module):
    """``w ~ N(mu(cond), diag(exp(log_std))^2)`` over the DDIM initial noise."""

    def __init__(self, n_tokens: int, n_emb: int, horizon: int, action_dim: int, hidden: int = 512,
                 init_log_std: float = 0.0, min_log_std: float = -2.5):  # fmt: skip
        super().__init__()
        self.noise_shape = (horizon, action_dim)
        dim = horizon * action_dim
        self.net = nn.Sequential(
            nn.LayerNorm(n_tokens * n_emb),
            nn.Linear(n_tokens * n_emb, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )
        # Zero mean head: the initial policy is the base policy's N(0, I).
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.log_std = nn.Parameter(torch.full((dim,), float(init_log_std)))
        self.min_log_std = min_log_std

    def forward(self, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """cond (B, n_tokens, n_emb) -> mean (B, dim), log_std (dim,)."""
        return self.net(cond.flatten(1).float()), self.log_std.clamp_min(self.min_log_std)

    @staticmethod
    def log_prob(w: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
        z = (w - mean) * torch.exp(-log_std)
        return -(0.5 * z.square() + log_std + 0.5 * math.log(2 * math.pi)).sum(-1)

    @staticmethod
    def kl_to_standard_normal(mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
        """KL(pi(.|cond) || N(0, I)) per sample: distance from the base policy's noise."""
        return 0.5 * (torch.exp(2 * log_std) + mean.square() - 1 - 2 * log_std).sum(-1)


class SteeredDiffusionSource:
    """Reference source of ``SuperMimicController``: diffusion sampled from policy-chosen noise.

    ``mode`` is ``"sample"`` (training rollouts), ``"mean"`` (deterministic evaluation) or
    ``"base"`` (``w ~ N(0, I)``, the unsteered diffusion policy). ``bf16`` runs the diffusion
    (DINOv3 + DiT, trained in bf16) under autocast, ~3x faster than fp32; the noise policy stays fp32. Every call records the
    observation tokens, the noise and its log-probability of the replanned rows (unless
    ``record`` is off, as when only running the policy).
    """

    def __init__(self, diffusion, noise_policy: NoisePolicy, mode: str = "sample", record: bool = True,
                 bf16: bool = False):  # fmt: skip
        self.policy, self.noise_policy, self.mode, self.record = diffusion, noise_policy, mode, record
        self.bf16 = bf16
        self.execute_steps, self.obs_horizon = diffusion.execute_steps, diffusion.obs_horizon
        self.horizon = diffusion.horizon
        self.records: list[dict[str, torch.Tensor]] = []

    @torch.no_grad()
    def predict(self, rows: torch.Tensor, images: torch.Tensor, states: torch.Tensor):
        diffusion = self.policy
        with torch.autocast("cuda", torch.bfloat16, enabled=self.bf16):
            cond = diffusion.encode(diffusion._prepare_image(images[rows]), states[rows])
        mean, log_std = self.noise_policy(cond)
        if self.mode == "sample":
            w = mean + torch.exp(log_std) * torch.randn_like(mean)
        elif self.mode == "mean":
            w = mean
        else:
            w = torch.randn_like(mean)
        if self.record:
            self.records.append({
                "rows": rows.clone(), "cond": cond.to(torch.bfloat16), "noise": w,
                "log_prob": NoisePolicy.log_prob(w, mean, log_std),
            })  # fmt: skip
        with torch.autocast("cuda", torch.bfloat16, enabled=self.bf16):
            pred = diffusion.sample(cond, w.view(-1, *self.noise_policy.noise_shape))
        keys = ("body_action_pred", "gripper_width_pred", "root_rotation_pred")
        return tuple(pred[k].float() for k in keys)
