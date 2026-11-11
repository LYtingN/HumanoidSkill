"""Event head: per step, whether the episode goes on, fails (the can dropped or toppled) or is done
(the can resting in the box), read off the object world model's predicted can.

As SuperMimic reads its termination off the body world model's predicted state, the head sees the
object WM's prediction, not the simulator's state, and is trained on the simulator's outcomes: a
cross-entropy over (continue, fail, done) at every step of the WM's training windows, the step each
episode ended labelled with how it ended. It so learns what a predicted can implies, the WM's error
included, and the imagination scores with probabilities instead of thresholds on predicted states
(where a centimetre of WM error flips "in the box" or "dropped"):

    S_k    = prod_{j <= k} p_continue_j                (the episode still running after step k)
    P_done = sum_k S_{k-1} p_done_k
    J      = P_done * success_reward + S_H * value_H   (value_H: phi of the predicted can, or the critic)

A failed episode is worth 0, so it adds nothing to J. The robot's falls are not events here (Phi
predicts those). Each prediction sees ``context`` steps of features (the done test holds for 0.2 s,
10 steps), frames before the window repeating its first one.
"""

from __future__ import annotations

import torch
from torch import nn

from hskill.embodiment.g1_aloha import RIGHT_EEF
from hskill.interaction.model import rot6d

FRAME_DIM = 16  # can position (3) / rotation (6) / velocity (3) w.r.t. the box, right hand - can (3), opening (1)
CONTINUE, FAIL, DONE = 0, 1, 2


def frame_features(obj: torch.Tensor, kp: torch.Tensor, cmd: torch.Tensor, box: torch.Tensor, rim: float) -> torch.Tensor:
    """(T, B, >= 10) can [pos | quat | lin vel], (T, B, 5, 7) keypoints, (T, B, 2) commanded openings,
    (B, 3) box centre -> (T, B, FRAME_DIM). Positions relative to the box centre (x, y) and its rim (z)."""
    origin = torch.stack([box[:, 0], box[:, 1], torch.full_like(box[:, 0], rim)], -1)
    pos = obj[..., :3]
    return torch.cat([
        (pos - origin) / 0.5, rot6d(obj[..., 3:7]), obj[..., 7:10] / 0.5,
        (kp[..., RIGHT_EEF, :3] - pos) / 0.2, cmd[..., 1:2] / 0.065,
    ], dim=-1)  # fmt: skip


class EventHead(nn.Module):
    def __init__(self, hidden: tuple[int, ...] = (256, 256), context: int = 10):
        super().__init__()
        self.context = context
        dims = [context * FRAME_DIM, *hidden]
        layers: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.SiLU()]
        self.net = nn.Sequential(*layers, nn.Linear(dims[-1], 3))

    def forward(self, obj: torch.Tensor, kp: torch.Tensor, cmd: torch.Tensor, box: torch.Tensor, rim: float) -> torch.Tensor:
        """A window: (H + 1, B, >= 10) can (frame 0: the current one), (H + 1, B, 5, 7) keypoints,
        (H + 1, B, 2) openings, (B, 3) box -> (H, B, 3) logits of (continue, fail, done) at steps 1..H."""
        f = frame_features(obj, kp, cmd, box, rim)
        f = torch.cat([f[:1].expand(self.context - 1, -1, -1), f])  # (H + context, B, F)
        windows = f.unfold(0, self.context, 1)[1:]  # (H, B, F, context): frames k - context + 1 .. k
        return self.net(windows.transpose(-1, -2).flatten(-2))

    @staticmethod
    def window_inputs(pred: torch.Tensor, hist_kp: torch.Tensor, fut_kp: torch.Tensor,
                      fut_cmd: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # fmt: skip
        """The object WM's window (prediction (H + 1, B, 13), history / future keypoints, future
        commands (H, B, 2)) -> the head's (obj, kp, cmd) over frames 0..H."""
        return pred, torch.cat([hist_kp[-1:], fut_kp]), torch.cat([fut_cmd[:1], fut_cmd])


def event_labels(fail: torch.Tensor, done: torch.Tensor) -> torch.Tensor:
    """(H, B) bool event steps -> (H, B) class indices."""
    return torch.where(done, DONE, torch.where(fail, FAIL, CONTINUE))


def event_loss(logits: torch.Tensor, fail: torch.Tensor, done: torch.Tensor,
               mask: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:  # fmt: skip
    """Cross-entropy over the masked steps, and how the predicted event mass compares with the
    labelled events (``*_ratio``: predicted / actual count, 1 when calibrated)."""
    m = mask.float()
    count = m.sum().clamp(min=1)
    ce = nn.functional.cross_entropy(logits.flatten(0, 1), event_labels(fail, done).flatten(), reduction="none")
    loss = (ce * m.flatten()).sum() / count
    with torch.no_grad():
        p = logits.softmax(-1)
        metrics = {"loss": loss.detach()}
        for name, k, label in (("fail", FAIL, fail), ("done", DONE, done)):
            actual = (label.float() * m).sum()
            metrics[f"{name}_rate"] = actual / count
            metrics[f"{name}_ratio"] = (p[..., k] * m).sum() / actual.clamp(min=1)
            metrics[f"{name}_p_at_event"] = (p[..., k] * label * m).sum() / actual.clamp(min=1)
    return loss, metrics


def soft_outcome(logits: torch.Tensor) -> dict[str, torch.Tensor]:
    """(H, B, 3) logits -> per window ``survive`` (S_H), ``p_done`` and ``p_fail`` (B,)."""
    p = logits.softmax(-1)
    survive = torch.cumprod(p[..., CONTINUE], 0)
    before = torch.cat([torch.ones_like(survive[:1]), survive[:-1]])  # S_{k-1}
    return {"survive": survive[-1], "p_done": (before * p[..., DONE]).sum(0), "p_fail": (before * p[..., FAIL]).sum(0)}
