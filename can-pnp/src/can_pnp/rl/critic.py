"""Critic of the imagined chunk returns: the rest of the episode from a body + can state, no image.

The imagined objective of one 32-frame chunk alone is greedy (holding the can still always pays
in the short run, placing it pays beyond the chunk). The critic predicts how much the task
potential (``can_pnp.task``: done 10 > failed 0 > exp(-d / 0.1)) still changes until the episode
ends, ``final potential - potential now``: the sum of the episode's remaining chunk returns, so
the reward stays that potential. A candidate is then worth ``success_reward`` if it completes the
task inside the chunk, 0 if it drops the can, else the chunk-end potential plus the critic's
remainder; minus the start's potential, as before. The zero-initialized head starts as the greedy
chunk return (ranking by the can's distance to the goal, which the potential still resolves far
from the goal where the episode returns are all ~0) and learns the remainder from the
Monte-Carlo returns of the simulated episodes of the noise policy (not the N(0, I) envs).

Features: the body world model's heading-invariant body state; the can, its velocity, the goal,
the box and the table-top in the robot's heading frame; the can axis, the finger opening and the
share of the episode elapsed (its return depends on the time left).
"""

from __future__ import annotations

import torch
from mjlab.utils.lab_api.math import matrix_from_quat, quat_apply_inverse, yaw_quat
from torch import nn

from can_pnp.body.world_model import BodyWorldModel

FEAT_DIM = 71 + 3 + 3 + 3 + 3 + 3 + 3 + 1 + 1


def critic_features(body: BodyWorldModel, s: torch.Tensor, obj: torch.Tensor, box: torch.Tensor, table: torch.Tensor,
                    goal: torch.Tensor, time: torch.Tensor) -> torch.Tensor:  # fmt: skip
    """(B, 71) body, (B, 14) can, (B, 3) box / table / goal (env frame), (B,) elapsed share."""
    heading = yaw_quat(s[:, 3:7])
    root = s[:, :3]

    def local(x):
        return quat_apply_inverse(heading, x - root)

    return torch.cat([
        body.wm.spec.canonical(s), local(obj[:, :3]), quat_apply_inverse(heading, obj[:, 7:10]),
        local(goal), local(box), local(table), matrix_from_quat(obj[:, 3:7])[..., 2], obj[:, 13:14] / 0.065,
        time[:, None],
    ], dim=-1)  # fmt: skip


class ValueCritic(nn.Module):
    def __init__(self, hidden: tuple[int, ...] = (512, 512, 512)):
        super().__init__()
        layers, d = [nn.LayerNorm(FEAT_DIM)], FEAT_DIM
        for h in hidden:
            layers += [nn.Linear(d, h), nn.SiLU()]
            d = h
        last = nn.Linear(d, 1)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        layers.append(last)
        self.net = nn.Sequential(*layers)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """Predicted remaining change of the task potential (B,)."""
        return self.net(feat).squeeze(-1)
