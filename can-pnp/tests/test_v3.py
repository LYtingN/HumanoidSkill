"""v3 pieces that need no simulator: task bookkeeping, object WM prior, buffer sampling, ranking."""

import torch
from mjlab.utils.lab_api.math import quat_apply, quat_mul

from can_pnp.config import SceneGeometryCfg
from can_pnp.object_wm.buffer import EpisodeBuffer
from can_pnp.object_wm.model import ObjectWorldModel, object_loss, quat_exp, quat_log
from can_pnp.rl.grpo import group_advantages
from can_pnp.sim.collect import Episodes, kendall_tau
from can_pnp.task import TaskCfg, TaskTracker


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


def test_object_wm_starts_as_a_rigid_grasp():
    torch.manual_seed(0)
    model = ObjectWorldModel()
    H, B = 8, 3
    gq = torch.nn.functional.normalize(torch.randn(H + 1, B, 4), dim=-1)
    grip = torch.cat([0.1 * torch.randn(H + 1, B, 3), gq], -1)
    offset = torch.tensor([0.05, 0.0, -0.02]).expand(B, 3)
    rel = torch.nn.functional.normalize(torch.randn(B, 4), dim=-1)
    pos = grip[..., :3] + quat_apply(gq.reshape(-1, 4), offset.repeat(H + 1, 1)).view(H + 1, B, 3)
    quat = quat_mul(gq.reshape(-1, 4), rel.repeat(H + 1, 1)).view(H + 1, B, 4)
    obj0 = torch.cat([pos[0], quat[0], torch.zeros(B, 6), torch.full((B, 1), 0.02)], -1)
    objs, logits = model.rollout(obj0, grip, torch.zeros(H, B), torch.zeros(B, 3), torch.zeros(B, 3))
    torch.testing.assert_close(objs[..., :3], pos, atol=1e-5, rtol=0)
    assert ((objs[..., 3:7] * quat).sum(-1).abs() > 1 - 1e-5).all()
    true = torch.cat([pos, quat, objs[..., 7:13].detach(), objs[..., 13:].detach()], -1)
    loss, _ = object_loss(objs, logits, true, torch.zeros(H, B, dtype=torch.bool), torch.ones(H, B))
    loss.backward()
    assert all(p.grad is None or p.grad.isfinite().all() for p in model.parameters())
    v = torch.tensor([[0.3, -0.2, 0.1], [0.0, 0.0, 0.0]])
    torch.testing.assert_close(quat_log(quat_exp(v)), v, atol=1e-5, rtol=0)


def test_kendall_tau_and_relative_ties():
    a = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    b = torch.tensor([[10.0, 20.0, 30.0], [30.0, 20.0, 10.0]])
    torch.testing.assert_close(kendall_tau(a, b), torch.tensor([1.0, -1.0]))
    assert kendall_tau(a, torch.ones(2, 3)).isnan().all()
    returns = torch.tensor([1e-5, 2e-5, 1.0, 2.0])
    _, informative = group_advantages(returns, 2, min_std=torch.tensor([[1e-7], [10.0]]))
    assert informative.tolist() == [1.0, 0.0]


def _episodes(T=32, N=3, D=4, H=8) -> Episodes:
    t = torch.arange(T + 1, dtype=torch.float32)[:, None, None].expand(T + 1, N, 1)
    alive = torch.ones(T + 1, N, dtype=torch.bool)
    alive[20:, 2] = False
    return Episodes(
        grip=t.expand(-1, -1, 7).clone(), command=torch.zeros(T, N), obj=t.expand(-1, -1, 14).clone(),
        held=torch.zeros(T + 1, N, dtype=torch.bool), alive=alive,
        phi_grip=-torch.ones(D, H + 1, N, 7), cond=torch.zeros(D, N, 2, 4), s0=torch.zeros(D, N, 71),
        obs0=torch.zeros(D, N, 930), decision_alive=torch.ones(D, N, dtype=torch.bool), xi=torch.zeros(N, 33),
        box=torch.zeros(N, 3), table=torch.zeros(N, 3), goal=torch.zeros(N, 3),
        final_potential=torch.tensor([10.0, 0.5, 0.0]), base=torch.tensor([False, True, False]),
    )  # fmt: skip


def test_buffer_windows_follow_the_episode():
    buf = EpisodeBuffer(capacity=2, decay=0.5, horizon=8)
    for _ in range(3):
        buf.add(_episodes())
    assert len(buf) == 2
    w = buf.windows(64, phi_fraction=0.0)
    assert w["obj"].shape == (9, 64, 14) and w["command"].shape == (8, 64) and w["box"].shape == (64, 3)
    # Windows are consecutive steps from a decision (multiple of 8), simulated grip = time index.
    steps = w["obj"][:, :, 0]
    assert ((steps[1:] - steps[:-1]) == 1).all() and (steps[0] % 8 == 0).all()
    torch.testing.assert_close(w["grip"][..., 0], steps)
    assert (buf.windows(16, phi_fraction=1.0)["grip"] == -1).all()
    s = buf.starts(5)
    assert s["cond"].shape == (5, 2, 4) and s["obj0"].shape == (5, 14)
    # Critic samples: the policy envs only (env 1 ran on N(0, I)), time = decision step / episode steps.
    c = buf.starts(64, policy_only=True)
    assert set(c["final"].tolist()) <= {10.0, 0.0}
    torch.testing.assert_close(c["time"], torch.round(c["time"] * 32 / 8) * 8 / 32)
    torch.testing.assert_close(c["step_frac"], torch.full((64,), 1 / 32))


def test_value_critic_starts_greedy():
    from can_pnp.rl.critic import FEAT_DIM, ValueCritic

    v = ValueCritic(hidden=(16,))(100 * torch.randn(50, FEAT_DIM))
    assert v.shape == (50,) and (v == 0).all()  # no remainder yet: the greedy chunk return
