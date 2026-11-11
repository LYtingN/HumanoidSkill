"""Task bookkeeping shared by simulated and imagined episodes."""

import torch
from can_pnp.scene.config import SceneGeometryCfg
from can_pnp.task_spec import TaskCfg, TaskTracker, shaping_potential


def test_task_tracker_done_after_hold_and_potential():
    geo, cfg = SceneGeometryCfg(), TaskCfg()
    box = torch.tensor([[0.5, 0.0, 0.85], [0.5, 0.0, 0.85]])
    goal = box.clone()
    in_box = torch.tensor([[0.5, 0.0, geo.box_bottom_z + 0.05], [0.9, 0.0, 0.9]])  # second: outside
    upright = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1)
    task = TaskTracker(cfg, geo, box, goal, torch.zeros(2), in_box, step_dt=0.02)
    live = torch.ones(2, dtype=torch.bool)
    for _ in range(task.hold_steps):
        task.step(in_box, upright, torch.tensor([True, True]), live)
    assert task.placed.tolist() == [True, False]
    p = task.potential(torch.tensor([False, True]))
    assert p[0] == cfg.success_reward and p[1] == 0.0
    # Neither done nor failed: the shaping of the can's distance to the goal.
    live = task.potential(torch.tensor([False, False]))
    torch.testing.assert_close(live[1], shaping_potential((in_box[1] - goal[1]).norm(), cfg))
    assert task.terminal_reward().tolist() == [cfg.success_reward, 0.0]


def test_two_scale_shaping():
    cfg = TaskCfg()
    d = torch.tensor([0.0, 0.1, 0.5, 0.86])
    phi = shaping_potential(d, cfg)
    expected = 0.5 * torch.exp(-d / 0.1) + 0.5 * torch.exp(-d / 0.5)
    torch.testing.assert_close(phi, expected)
    assert phi[0] == 1.0 and (phi[1:] < phi[:-1]).all()
    assert phi[-1] > 0.08  # still informative far from the goal (exp(-d / 0.1) alone: 2e-4)
