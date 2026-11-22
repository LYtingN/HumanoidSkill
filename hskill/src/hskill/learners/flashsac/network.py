import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layer import (
    EnsembleCategoricalValue,
    EnsembleFlashSACBlock,
    EnsembleFlashSACEmbedder,
    EnsembleUnitRMSNorm,
    FlashSACBlock,
    FlashSACEmbedder,
    NormalTanhPolicy,
    UnitRMSNorm,
)


def prior_pre_tanh_std(scale: float) -> float:
    """Std of a zero-mean Gaussian u for which ``scale * tanh(u)`` has unit second moment."""
    if scale <= 1:
        raise ValueError("The action scale must exceed 1 to reach a unit-variance latent")
    x, w = np.polynomial.hermite_e.hermegauss(80)
    w = w / w.sum()
    lo, hi = 1e-4, 20.0
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if (w * (scale * np.tanh(mid * x)) ** 2).sum() < 1 else (lo, mid)
    return (lo + hi) / 2


class PriorNormalTanhPolicy(NormalTanhPolicy):
    """Tanh-Gaussian policy that starts near a fixed, observation-independent prior.

    The unit-norm projections are kept (weight normalization), but their outputs pass through
    small learnable gains, so the initial mean is close to zero and the initial std is close to
    ``prior_std`` for every observation. The gains are not weight-normalized and can grow.
    """

    def __init__(self, hidden_dim: int, action_dim: int, prior_std: float, initial_gain: float = 0.01):
        super().__init__(hidden_dim=hidden_dim, action_dim=action_dim)
        self.mean_gain = nn.Parameter(torch.full((action_dim,), initial_gain))
        self.std_gain = nn.Parameter(torch.full((action_dim,), initial_gain))
        unit = (math.log(prior_std) - self.log_std_min) / (self.log_std_max - self.log_std_min)
        if not 0 < unit < 1:
            raise ValueError("prior_std lies outside the policy's std range")
        with torch.no_grad():
            self.std_bias.fill_(math.atanh(2 * unit - 1))

    def get_mean_and_std(self, x: torch.Tensor, training: bool) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean_gain * F.linear(x, self.mean_w.w.weight) + self.mean_bias
        raw_log_std = self.std_gain * F.linear(x, self.std_w.w.weight) + self.std_bias
        log_std = self.log_std_min + (self.log_std_max - self.log_std_min) * 0.5 * (1 + torch.tanh(raw_log_std))
        return mean, torch.exp(log_std)


class FlashSACActor(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        input_dim: int,
        hidden_dim: int,
        action_dim: int,
        prior_std: float | None = None,
    ):
        super().__init__()
        self.embedder = FlashSACEmbedder(input_dim=input_dim, hidden_dim=hidden_dim)
        self.encoder = nn.ModuleList([FlashSACBlock(hidden_dim) for _ in range(num_blocks)])
        self.post_norm = UnitRMSNorm(hidden_dim)
        self.predictor = (NormalTanhPolicy(hidden_dim=hidden_dim, action_dim=action_dim) if prior_std is None
                          else PriorNormalTanhPolicy(hidden_dim, action_dim, prior_std))

    def get_mean_and_std(
        self,
        observations: torch.Tensor,
        training: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = observations
        x = self.embedder(x, training)
        for block in self.encoder:
            x = block(x, training)
        x = self.post_norm(x)
        mean, std = self.predictor.get_mean_and_std(x, training)
        return mean, std

    def forward(
        self,
        observations: torch.Tensor,
        training: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = observations
        x = self.embedder(x, training)
        for block in self.encoder:
            x = block(x, training)
        x = self.post_norm(x)
        actions, info = self.predictor(x, training)
        return actions, info


class FlashSACDoubleCritic(nn.Module):
    """
    Double-Q for Clipped Double Q-learning.
    https://arxiv.org/pdf/1802.09477v3

    Fuses N parallel critic networks into single batched operations.
    All internal computation uses (N, batch, dim) tensor layout.
    """

    def __init__(
        self,
        num_blocks: int,
        input_dim: int,
        hidden_dim: int,
        num_bins: int,
        min_v: float,
        max_v: float,
        num_qs: int = 2,
    ):
        super().__init__()
        self.num_qs = num_qs

        self.embedder = EnsembleFlashSACEmbedder(num_qs, input_dim, hidden_dim)
        self.encoder = nn.ModuleList([EnsembleFlashSACBlock(num_qs, hidden_dim) for _ in range(num_blocks)])
        self.post_norm = EnsembleUnitRMSNorm(num_qs, hidden_dim)
        self.predictor = EnsembleCategoricalValue(
            num_ensemble=num_qs,
            hidden_dim=hidden_dim,
            num_bins=num_bins,
            min_v=min_v,
            max_v=max_v,
        )

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        training: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = torch.cat((observations, actions), dim=-1)  # [B, in_dim]
        x = x.unsqueeze(0).expand(self.num_qs, -1, -1)  # [num_qs, B, in_dim]
        x = self.embedder(x, training)
        for block in self.encoder:
            x = block(x, training)
        x = self.post_norm(x)
        qs, infos = self.predictor(x, training)
        return qs, infos


class FlashSACTemperature(nn.Module):
    def __init__(self, initial_value: float = 0.01):
        super().__init__()
        self.log_temp = nn.Parameter(torch.tensor([math.log(initial_value)], dtype=torch.float32))

    def forward(self) -> torch.Tensor:
        return torch.exp(self.log_temp)
