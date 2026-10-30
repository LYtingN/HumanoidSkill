"""Tracker observation history packing (CPU)."""

import torch
from hskill.executors.imagine import flatten_history, unflatten_history


def test_history_roundtrip():
    obs = torch.randn(5, 930)
    torch.testing.assert_close(flatten_history(unflatten_history(obs)), obs, atol=0, rtol=0)
