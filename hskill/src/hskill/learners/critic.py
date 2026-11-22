"""Model-based critic: the episode's return from the world models' prediction of a decision.

For a decision and its noise ``w``, the frozen DiT decodes the reference, Phi (tracker + body
world model) executes it for ``horizon`` frames and the object world model predicts the can
(``hskill.executors.imaginer.Imaginer``). The critic sees only that prediction, no simulator
state of the decision:

    Q(s_hat, o_hat, d, t_left) ~ Psi_end

* ``s_hat``: Phi's body state at the window's end (heading-invariant, ``spec.canonical``);
* ``o_hat``: the object WM's can at the window's end in the predicted pelvis heading frame
  (position, axis, velocity), its position and distance from the right gripper base,
  and its held probability;
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

FEAT_DIM = 71 + 3 + 3 + 3 + 3 + 1 + 3 + 1 + 1 + 1


def critic_features(body: BodyWorldModel, s_end: torch.Tensor, kp_end: torch.Tensor, obj_end: torch.Tensor,
                    held_logit: torch.Tensor, goal: torch.Tensor, t_left: torch.Tensor) -> torch.Tensor:  # fmt: skip
    """(B, 71) predicted body, (B, 5, 7) its keypoints, (B, 13) predicted can, (B,) held logit,
    (B, 3) goal (env frame), (B,) seconds of the episode left -> (B, FEAT_DIM)."""
    return task_state_features(body.wm.spec.canonical(s_end), s_end, kp_end, obj_end,
                               torch.sigmoid(held_logit), goal, t_left)


def task_state_features(canonical, s_end, kp_end, obj_end, held_probability, goal, t_left):
    """Shared privileged features for a predicted or measured state, in the environment frame."""
    heading, root = yaw_quat(s_end[:, 3:7]), s_end[:, :3]
    can = obj_end[:, :3]
    gripper_to_can = can - kp_end[:, RIGHT_EEF, :3]
    to_goal = goal - can
    return torch.cat([
        canonical,
        quat_apply_inverse(heading, can - root),
        quat_apply_inverse(heading, matrix_from_quat(obj_end[:, 3:7])[..., 2]),  # can axis
        quat_apply_inverse(heading, obj_end[:, 7:10]),
        quat_apply_inverse(kp_end[:, RIGHT_EEF, 3:7], gripper_to_can),
        held_probability[:, None],
        quat_apply_inverse(heading, to_goal), to_goal.norm(dim=-1, keepdim=True),
        t_left[:, None] / 10.0,
        gripper_to_can.norm(dim=-1, keepdim=True),  # metres from right gripper base to can centre
    ], dim=-1)  # fmt: skip


class CriticInputNorm(nn.LayerNorm):
    """Keep the existing state normalization and pass the additional distance in metres."""

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return torch.cat((super().forward(feat[..., :-1]), feat[..., -1:]), dim=-1)


class ModelCritic(nn.Module):
    def __init__(self, hidden: tuple[int, ...] = (512, 512, 512), members: int = 3):
        super().__init__()
        self.members = nn.ModuleList()
        for _ in range(members):
            layers, d = [CriticInputNorm(FEAT_DIM - 1)], FEAT_DIM
            for h in hidden:
                layers += [nn.Linear(d, h), nn.SiLU()]
                d = h
            layers.append(nn.Linear(d, 1))
            self.members.append(nn.Sequential(*layers))

    def restore_training_state(self, weights, optimizer, optimizer_state):
        """Resume an 89-input critic with zero initial weight and Adam moments for distance."""
        legacy = weights["members.0.1.weight"].shape[1] == FEAT_DIM - 1
        if legacy:
            weights = weights.copy()
            for i in range(len(self.members)):
                key = f"members.{i}.1.weight"
                weights[key] = nn.functional.pad(weights[key], (0, 1))
        self.load_state_dict(weights)
        optimizer.load_state_dict(optimizer_state)
        if legacy:
            for member in self.members:
                moments = optimizer.state.get(member[1].weight, {})
                for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                    if key in moments:
                        moments[key] = nn.functional.pad(moments[key], (0, 1))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """Every member's predicted return (E, B)."""
        return torch.stack([m(feat).squeeze(-1) for m in self.members])
