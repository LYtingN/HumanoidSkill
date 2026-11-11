"""Checkpoint-independent coordinate/timing tests against the original tracker math."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
for source in ("downloads/5points_FM/inference_source", "downloads/morphmimic/inference_source"):
    if not (ROOT / source).exists():
        pytest.skip("Downloaded policy source snapshots required", allow_module_level=True)
    sys.path.insert(0, str(ROOT / source))

from hskill.s0.morphmimic import KeypointPlan
from humanoid_diffusion.keypoints import world_to_keypoints
from morphmimic.model.keypoints import targets_features
from morphmimic.train.obs_history import ObsHistory
from morphmimic.train.normalizers import RunningNormalizer, normalize_history


def test_anchor_inverse_and_sliding_preview_match_original_targets_features():
    rng = np.random.default_rng(42)
    pos = rng.normal(size=(2, 32, 5, 3)).astype(np.float32)
    quat = rng.normal(size=(2, 32, 5, 4)).astype(np.float32)
    quat /= np.linalg.norm(quat, axis=-1, keepdims=True)
    anchor_pos = np.array([[3, -2, .8], [-4, 8, .75]], dtype=np.float32)
    anchor_quat = quat[:, 0, 0].copy()
    relative = np.stack([world_to_keypoints(pos[i], quat[i], anchor_pos[i], anchor_quat[i])
                         for i in range(2)])
    pred = torch.cat((torch.from_numpy(relative), torch.zeros(2, 32, 2)), -1).float()
    plan = KeypointPlan(pred, torch.from_numpy(anchor_pos), torch.from_numpy(anchor_quat))
    torch.testing.assert_close(plan.pos_w, torch.from_numpy(pos), atol=2e-6, rtol=1e-5)
    # Nonzero robot roll/pitch is intentional: tracker orientation uses FULL IMU,
    # while FM's anchor transform uses yaw only.
    state = torch.zeros(2, 240)
    state[:, 100:104] = torch.from_numpy(quat[:, 4, 0])
    for offset in (0, 7):
        p, q = plan.window(torch.tensor([offset, offset]))
        expected = targets_features(state, torch.from_numpy(pos[:, offset:offset+10]),
                                    torch.from_numpy(quat[:, offset:offset+10]))
        torch.testing.assert_close(targets_features(state, p, q), expected, atol=1e-5, rtol=1e-5)
    with pytest.raises(ValueError, match="preview"):
        plan.window(24)


def test_time_major_history_reset_and_framewise_normalization():
    h = ObsHistory(2, 10, 93, "cpu")
    first = torch.arange(186).reshape(2, 93).float()
    h.reset(None, first)
    h.push(first + 100)
    h.reset(torch.tensor([1]), first[1:] - 10)
    assert torch.equal(h.flat()[0, -93:], first[0] + 100)
    assert torch.equal(h.flat()[0, :93], first[0])
    assert torch.equal(h.buf[1], (first[1] - 10).expand(10, 93))
    norm = RunningNormalizer(93).eval()
    norm.mean.copy_(torch.arange(93))
    norm.var.fill_(4)
    expected = ((h.buf - norm.mean) / 2.01).flatten(1)
    torch.testing.assert_close(normalize_history(norm, h.flat()), expected)


def test_plan_keeps_original_anchor_when_robot_moves():
    pred = torch.zeros(1, 32, 47)
    identity6 = torch.tensor([1., 0., 0., 1., 0., 0.])
    pred[..., :45].view(1, 32, 5, 9)[..., 3:] = identity6
    pred[..., :45].view(1, 32, 5, 9)[..., 2] = .8
    pelvis = torch.tensor([[1., 2., .7]])
    plan = KeypointPlan(pred, pelvis, torch.tensor([[1., 0., 0., 0.]]))
    pelvis += 100  # Measured robot state changes after the replan.
    p, _ = plan.window(7)
    torch.testing.assert_close(p[..., 0], torch.ones_like(p[..., 0]))
    torch.testing.assert_close(p[..., 1], torch.full_like(p[..., 1], 2))
    torch.testing.assert_close(p[..., 2], torch.full_like(p[..., 2], .8))
