"""Critic of the imagined chunk returns: the rest of the episode from a body + can state, no image.

The imagined objective of one 32-frame chunk alone is greedy (holding the can still always pays
in the short run, placing it pays beyond the chunk). The critic predicts how much the task
potential (task tracker: done 10 > failed 0 > exp(-d / 0.1)) still changes until the episode
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
from mjlab.utils.lab_api.math import (
    matrix_from_quat,
    quat_apply_inverse,
    quat_inv,
    quat_mul,
    yaw_quat,
)
from torch import nn

from hskill.data.demo_progress import PROPRIO_DIM, proprio_features
from hskill.s0.body_world_model import BodyWorldModel

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


# ----------------------------------------------------------------------------- progress critic

OBJECT_DIM = 3 + 3 + 3 + 3 + 3 + 3
PROGRESS_FEAT_DIM = PROPRIO_DIM + OBJECT_DIM + 1


def object_features(s: torch.Tensor, obj: torch.Tensor, box: torch.Tensor, table: torch.Tensor,
                    goal: torch.Tensor) -> torch.Tensor:  # fmt: skip
    """The object, its velocity, the goal, box and table-top in the robot's heading frame, and the
    object's axis (B, OBJECT_DIM)."""
    heading, root = yaw_quat(s[:, 3:7]), s[:, :3]

    def local(x):
        return quat_apply_inverse(heading, x - root)

    return torch.cat([local(obj[:, :3]), quat_apply_inverse(heading, obj[:, 7:10]), local(goal), local(box),
                      local(table), matrix_from_quat(obj[:, 3:7])[..., 2]], dim=-1)  # fmt: skip


def state_proprio(kin, s: torch.Tensor, opening: torch.Tensor) -> torch.Tensor:
    """Proprioceptive features of 71-d body states (as the demonstrations record them)."""
    poses = kin.poses(s)
    root_pos, root_quat = s[:, None, :3], s[:, None, 3:7].expand(-1, 2, -1)
    wrist_pos = quat_apply_inverse(root_quat, poses["wrist_pos"] - root_pos)
    wrist_quat = quat_mul(quat_inv(root_quat), poses["wrist_quat"])
    gravity = quat_apply_inverse(s[:, 3:7], torch.tensor((0.0, 0.0, -1.0), device=s.device).expand(len(s), 3))
    return proprio_features(s[:, 7:36], gravity, wrist_pos, wrist_quat, opening)


def progress_features(proprio: torch.Tensor, objects: torch.Tensor | None) -> torch.Tensor:
    """Critic input; ``objects`` None (demonstrations: no object state) zeroes and flags that part.

    No elapsed time: every demonstration succeeds, so time alone would predict their targets, and
    it is the same for all candidates of an imagined group."""
    n = len(proprio)
    if objects is None:
        objects, flag = torch.zeros(n, OBJECT_DIM, device=proprio.device), torch.zeros(n, 1, device=proprio.device)
    else:
        flag = torch.ones(n, 1, device=proprio.device)
    return torch.cat([proprio, objects, flag], dim=-1)


class ProgressCritic(nn.Module):
    """``W(state)``: the episode's final task potential, discounted by the steps to the episode's
    end (``success_reward * gamma ** steps`` once done, 0 after a failure). Pre-trained on the
    demonstrations' time to completion (proprioception only, no object state), then on the
    simulated episodes of the noise policy (proprioception and object). In imagination a candidate
    is worth the terminal potential if the task ends inside the chunk, else ``W`` of the chunk end's
    proprioception alone, as the demonstrations': how far the body is along the task by the
    demonstrations' measure. (With the object part, ``W`` would answer for the current policy, ~0
    while it never succeeds, for every candidate alike.) The object's fate inside the chunk (held,
    dropped, closer) comes from the object WM."""

    def __init__(self, hidden: tuple[int, ...] = (512, 512, 512)):
        super().__init__()
        layers, d = [nn.LayerNorm(PROGRESS_FEAT_DIM)], PROGRESS_FEAT_DIM
        for h in hidden:
            layers += [nn.Linear(d, h), nn.SiLU()]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat).squeeze(-1)
