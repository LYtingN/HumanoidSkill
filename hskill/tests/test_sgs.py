from types import SimpleNamespace

import torch

from hskill.data.sgs import SuccessGuidedSampler


def test_sgs_history_sampling_and_resume():
    starts = SimpleNamespace(phases=("RELEASE", "CLOSE"), phase_ids=torch.tensor([0, 0, 0, 1, 1, 1]),
                             episode_ids=[0, 1, 2, 0, 1, 2])
    rows = [0, 2, 3, 5]  # Episode 1 is held out.
    sampler = SuccessGuidedSampler(starts, rows, seed=7)
    uniform = sampler.weights()
    assert uniform.unique().numel() == 1
    sampled = torch.tensor([0] * 120 + [2, 2, 3, 5, 5])
    placed = torch.tensor([False] * 20 + [True] * 100 + [False, True, False, False, True])
    policy = torch.ones_like(placed)
    policy[-2:] = False
    sampler.observe(sampled, placed, policy)
    assert list(sampler.history[0]) == [1] * 100
    assert not sampler.history[5]
    w = sampler.weights()
    assert 0 < w[0] == w[2] < w[1] == w[3]
    assert sampler.metrics() == {"curriculum/coverage": 0.75, "curriculum/pool_success": 0.375}
    sampler.sample_phase()
    restored = SuccessGuidedSampler(starts, rows, seed=22)
    restored.load_state_dict(sampler.state_dict())
    assert restored.metrics() == sampler.metrics()
    assert restored.sample_phase() == sampler.sample_phase()
    selected = restored.sample_rows(1, 200, torch.Generator().manual_seed(0))
    assert set(selected.tolist()) == {3, 5}


def test_sgs_iid_sampling_follows_weights():
    starts = SimpleNamespace(phases=("A", "B"), phase_ids=torch.tensor([0, 0, 1, 1]), episode_ids=[0, 1, 0, 1])
    sampler = SuccessGuidedSampler(starts, [0, 1, 2, 3], seed=0)
    for _ in range(100):  # row 0 always fails, row 1 always succeeds, rows 2/3 half
        sampler.observe(torch.tensor([0, 1, 2, 3]), torch.tensor([False, True, True, False]),
                        torch.ones(4, dtype=torch.bool))
    for _ in range(50):
        sampler.observe(torch.tensor([2, 3]), torch.tensor([False, True]), torch.ones(2, dtype=torch.bool))
    rows = sampler.sample(20000, torch.Generator().manual_seed(0))
    share = torch.bincount(rows, minlength=4).float() / len(rows)
    assert share[2] > 4 * share[0] and share[3] > 4 * share[1]  # the learning zone dominates
    metrics = sampler.phase_metrics()
    assert abs(metrics[1][0] - 0.5) < 1e-6 and metrics[1][1] > 0.8
