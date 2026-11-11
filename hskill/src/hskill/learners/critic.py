"""Model-based critic: the episode's return from the world models' prediction of a decision.

For a decision and its noise ``w``, the frozen DiT decodes the reference, Phi (tracker + body
world model) executes it for ``horizon`` frames and the object world model predicts the can
(``hskill.executors.imaginer.Imaginer``). The critic sees only that prediction, no simulator
state of the decision:

    Q(s_hat, o_hat, d, t_left) ~ Psi_end

* ``s_hat``: Phi's body state at the window's end (heading-invariant, ``spec.canonical``);
* ``o_hat``: the object WM's can at the window's end in the predicted pelvis heading frame
  (position, axis, velocity), relative to the predicted right gripper base, and its held
  probability;
* ``d``: the predicted can-to-goal vector (heading frame) and distance;
* ``t_left``: the seconds of the episode left at the decision (the return of a time-limited
  episode depends on it; the same for all candidates of a decision; episode lengths differ per
  curriculum phase).

The target is the Monte-Carlo return of the simulated episode, the task potential at its end
(``success_reward`` placed, 0 failed, the distance shaping at the time-out), on the noise
policy's envs. Members of the ensemble differ by initialization; their spread is reported.
"""

from __future__ import annotations

import torch
from mjlab.utils.lab_api.math import matrix_from_quat, quat_apply_inverse, yaw_quat
from torch import nn

from hskill.embodiment.g1_aloha import RIGHT_EEF
from hskill.s0.body_world_model import BodyWorldModel

FEAT_DIM = 71 + 3 + 3 + 3 + 3 + 1 + 3 + 1 + 1


def critic_features(body: BodyWorldModel, s_end: torch.Tensor, kp_end: torch.Tensor, obj_end: torch.Tensor,
                    held_logit: torch.Tensor, goal: torch.Tensor, t_left: torch.Tensor) -> torch.Tensor:  # fmt: skip
    """(B, 71) predicted body, (B, 5, 7) its keypoints, (B, 13) predicted can, (B,) held logit,
    (B, 3) goal (env frame), (B,) seconds of the episode left -> (B, FEAT_DIM)."""
    heading, root = yaw_quat(s_end[:, 3:7]), s_end[:, :3]
    can = obj_end[:, :3]
    to_goal = goal - can
    return torch.cat([
        body.wm.spec.canonical(s_end),
        quat_apply_inverse(heading, can - root),
        quat_apply_inverse(heading, matrix_from_quat(obj_end[:, 3:7])[..., 2]),  # can axis
        quat_apply_inverse(heading, obj_end[:, 7:10]),
        quat_apply_inverse(kp_end[:, RIGHT_EEF, 3:7], can - kp_end[:, RIGHT_EEF, :3]),  # can in the gripper frame
        torch.sigmoid(held_logit)[:, None],
        quat_apply_inverse(heading, to_goal), to_goal.norm(dim=-1, keepdim=True),
        t_left[:, None] / 10.0,
    ], dim=-1)  # fmt: skip


class ModelCritic(nn.Module):
    def __init__(self, hidden: tuple[int, ...] = (512, 512, 512), members: int = 3):
        super().__init__()
        self.members = nn.ModuleList()
        for _ in range(members):
            layers, d = [nn.LayerNorm(FEAT_DIM)], FEAT_DIM
            for h in hidden:
                layers += [nn.Linear(d, h), nn.SiLU()]
                d = h
            layers.append(nn.Linear(d, 1))
            self.members.append(nn.Sequential(*layers))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """Every member's predicted return (E, B)."""
        return torch.stack([m(feat).squeeze(-1) for m in self.members])
