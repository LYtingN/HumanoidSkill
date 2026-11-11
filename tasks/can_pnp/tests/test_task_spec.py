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


def test_toppled_can_fails_unless_held():
    geo, cfg = SceneGeometryCfg(), TaskCfg()
    goal = torch.tensor([[0.5, 0.0, 0.85]]).repeat(3, 1)
    on_table = torch.tensor([[0.2, 0.0, 0.8]]).repeat(3, 1)
    lying = torch.tensor([[0.7071, 0.7071, 0.0, 0.0]]).repeat(3, 1)  # axis horizontal
    task = TaskTracker(cfg, geo, goal, goal, torch.zeros(3), on_table, step_dt=0.02)
    live = torch.ones(3, dtype=torch.bool)
    held = torch.tensor([False, True, False])
    for i in range(task.hold_steps):
        # Env 2 lies only for a moment before it is upright again.
        quat = lying.clone()
        if i == task.hold_steps - 1:
            quat[2] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        task.step(on_table, quat, torch.tensor([True, True, True]), live, held)
    assert task.toppled.tolist() == [True, False, False]
    p = task.potential(torch.zeros(3, dtype=torch.bool))
    assert p[0] == 0.0 and p[1] > 0 and p[2] > 0


def test_can_lying_in_the_box_is_done_unless_upright_release():
    geo = SceneGeometryCfg()
    box = torch.tensor([[0.5, 0.0, 0.85]])
    lying_in_box = torch.tensor([[0.5, 0.0, geo.box_bottom_z + geo.can_radius]])
    lying = torch.tensor([[0.7071, 0.7071, 0.0, 0.0]])
    closed, live = torch.tensor([False]), torch.tensor([True])
    for cfg, done in ((TaskCfg(), True), (TaskCfg(upright_release=True), False)):
        task = TaskTracker(cfg, geo, box, box, torch.zeros(1), lying_in_box, step_dt=0.02)
        for _ in range(task.hold_steps):
            task.step(lying_in_box, lying, closed, live, torch.tensor([False]))
        assert task.placed.item() is done and not task.toppled.item()


def test_moving_can_in_the_box_is_not_done():
    geo, cfg = SceneGeometryCfg(), TaskCfg()
    box = torch.tensor([[0.5, 0.0, 0.85]])
    pos = torch.tensor([[0.45, 0.0, geo.box_bottom_z + 0.05]])
    upright, live = torch.tensor([[1.0, 0.0, 0.0, 0.0]]), torch.tensor([True])
    task = TaskTracker(cfg, geo, box, box, torch.zeros(1), pos, step_dt=0.02)
    for i in range(3 * task.hold_steps):
        moving = pos + torch.tensor([[0.002 * (i % 2), 0.0, 0.0]])  # 0.1 m/s back and forth
        task.step(moving, upright, torch.tensor([True]), live)
    assert not task.placed.item()


def test_can_resting_in_the_box_in_the_hand_is_not_done():
    geo, cfg = SceneGeometryCfg(), TaskCfg()
    box = torch.tensor([[0.5, 0.0, 0.85]]).repeat(4, 1)
    in_box = torch.tensor([[0.5, 0.0, geo.box_bottom_z + geo.can_height / 2]]).repeat(4, 1)
    upright, live = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(4, 1), torch.ones(4, dtype=torch.bool)
    # Held / let go (contact sensing), and without it: gripper commanded closed / open.
    cases = ((torch.tensor([True, False]), torch.tensor([False, False])), (None, torch.tensor([False, True])))
    done = []
    for held, released in cases:
        task = TaskTracker(cfg, geo, box[:2], box[:2], torch.zeros(2), in_box[:2], step_dt=0.02)
        for _ in range(task.hold_steps):
            task.step(in_box[:2], upright[:2], released, live[:2], held)
        done += task.placed.tolist()
    assert done == [False, True, False, True]
