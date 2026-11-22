import math

import torch

from hskill.learners.flashsac.learner import FlashSAC, Replay
from hskill.learners.flashsac.update import _compute_categorical_td_target


def test_terminal_projection_does_not_bootstrap():
    log_probs = torch.tensor([[.2, .3, .5], [.2, .3, .5]]).log()
    target = _compute_categorical_td_target(log_probs, torch.zeros(2), torch.tensor([1., 0.]),
                                             torch.zeros(2), 1., 3, -1., 1.)
    torch.testing.assert_close(target, torch.tensor([[0., 1., 0.], [.2, .3, .5]]))


def test_replay_and_delayed_actor_update():
    torch.manual_seed(1)
    replay = Replay(5)
    learner = FlashSAC(12, 9, 4, "cpu", actor_width=8, critic_width=8, blocks=1)
    for t in range(3):
        obs = torch.randn(2, 12)
        action = learner.act(obs, torch.tensor([True, False]))
        batch = dict(actor_observation=obs, actor_next_observation=torch.randn_like(obs),
                     observation=torch.randn(2, 9), next_observation=torch.randn(2, 9),
                     action=action, reward=torch.tensor([float(t), 1.]), terminated=torch.tensor([1., 0.]))
        replay.add(batch)
        learner.normalizer.update_reward_stats(batch["reward"], batch["terminated"], torch.zeros(2))
    assert replay.size == 5
    torch.testing.assert_close(replay.fields["action"][[4, 0]], action)
    before = [p.clone() for p in learner.actor.network.parameters()]
    info = learner.update(replay.sample(4, "cpu"))
    assert all(map(math.isfinite, info.values()))
    assert any(not torch.equal(a, b) for a, b in zip(before, learner.actor.network.parameters()))
    before = [p.clone() for p in learner.actor.network.parameters()]
    info = learner.update(replay.sample(4, "cpu"))
    assert "actor/loss" not in info
    assert all(torch.equal(a, b) for a, b in zip(before, learner.actor.network.parameters()))


def _batch(n, obs=12, state=9, action=4):
    return dict(actor_observation=3 + 2 * torch.randn(n, obs), actor_next_observation=3 + 2 * torch.randn(n, obs),
                observation=torch.randn(n, state), next_observation=torch.randn(n, state),
                action=torch.rand(n, action) * 2 - 1, reward=torch.randn(n), terminated=torch.zeros(n))


def test_prior_initialized_actor_matches_fm_latent_scale():
    torch.manual_seed(0)
    learner = FlashSAC(64, 9, 256, "cpu", actor_width=32, critic_width=8, blocks=1, noise_scale=3.)
    assert abs(learner.config["prior_std"] - 0.36) < 0.02
    obs = 5 + 3 * torch.randn(512, 64)
    mean, std = learner.actor.network.get_mean_and_std(obs, training=False)
    ratio = std / learner.config["prior_std"]
    assert mean.abs().max() < 0.1 and abs(ratio.mean() - 1) < 0.02 and (ratio - 1).abs().max() < 0.3
    latent = 3 * learner.act(obs, torch.ones(512, dtype=torch.bool))
    assert abs(latent.square().mean().sqrt().item() - 1) < 0.05


def test_calibration_sets_running_statistics():
    torch.manual_seed(0)
    learner = FlashSAC(12, 9, 4, "cpu", actor_width=8, critic_width=8, blocks=1, noise_scale=3.)
    batch = _batch(256)
    learner.calibrate(batch)
    norm = learner.actor.network.embedder.norm
    torch.testing.assert_close(norm.running_mean, torch.full((12,), 3.), atol=.4, rtol=0)
    torch.testing.assert_close(norm.running_var, torch.full((12,), 4.), atol=1.2, rtol=0)
    assert learner.calibrated


def test_checkpoint_round_trip_continues_identically(tmp_path):
    torch.manual_seed(0)
    learner = FlashSAC(12, 9, 4, "cpu", actor_width=8, critic_width=8, blocks=1, noise_scale=3.)
    replay = Replay(64)
    replay.add(_batch(40))
    learner.normalizer.update_reward_stats(torch.randn(40), torch.zeros(40), torch.zeros(40))
    learner.calibrate(replay.sample(32, "cpu"))
    for _ in range(3):
        learner.update(replay.sample(16, "cpu"))
    torch.save(dict(learner=learner.checkpoint(), replay=replay.state_dict()), tmp_path / "c.pt")
    state = torch.load(tmp_path / "c.pt", weights_only=True)
    torch.manual_seed(5)
    clone = FlashSAC(12, 9, 4, "cpu", actor_width=8, critic_width=8, blocks=1, noise_scale=3.)
    clone.load_checkpoint(state["learner"])
    restored = Replay(64)
    restored.load_state_dict(state["replay"])
    assert (restored.size, restored.cursor) == (40, 40)
    torch.testing.assert_close(restored.fields["reward"][:40], replay.fields["reward"][:40])
    batch = replay.sample(16, "cpu")
    torch.manual_seed(9)  # the update samples actor actions
    a = learner.update({k: v.clone() for k, v in batch.items()})
    torch.manual_seed(9)
    b = clone.update({k: v.clone() for k, v in batch.items()})
    assert a == b and clone.updates == learner.updates == 4


def _distributed_worker(rank, world, directory):
    import torch.distributed as dist
    dist.init_process_group("gloo", init_method=f"file://{directory}/rendezvous", rank=rank, world_size=world)
    torch.manual_seed(100 + rank)  # different seeds: the constructor must still produce one replica
    learner = FlashSAC(12, 9, 4, "cpu", actor_width=8, critic_width=8, blocks=1, noise_scale=3.)
    replay = Replay(64)
    replay.add(_batch(32))
    learner.normalizer.update_reward_stats(torch.ones(2), torch.zeros(2), torch.zeros(2))
    learner.calibrate(replay.sample(32, "cpu"))
    for _ in range(4):
        learner.update(replay.sample(16, "cpu"))
    flat = torch.cat([t.flatten() for n in (learner.actor, learner.critic, learner.target, learner.temperature)
                      for t in [*n.network.parameters(), *n.network.buffers()]])
    torch.save(flat, f"{directory}/{rank}.pt")
    dist.destroy_process_group()


def test_distributed_replicas_stay_identical(tmp_path):
    import torch.multiprocessing as mp
    mp.spawn(_distributed_worker, args=(2, str(tmp_path)), nprocs=2)
    torch.testing.assert_close(torch.load(tmp_path / "0.pt"), torch.load(tmp_path / "1.pt"), rtol=0, atol=1e-6)


def test_replay_bf16_storage_round_trip():
    replay = Replay(8, {"actor_observation": torch.bfloat16})
    obs = torch.randn(4, 6).bfloat16().float()  # BF16-representable, as the FM tokens are
    replay.add(dict(actor_observation=obs, reward=torch.arange(4.)))
    assert replay.fields["actor_observation"].dtype == torch.bfloat16
    batch = replay.sample(16, "cpu")
    assert batch["actor_observation"].dtype == torch.float32
    restored = Replay(8)
    restored.load_state_dict(replay.state_dict())
    assert restored.fields["actor_observation"].dtype == torch.bfloat16
    assert restored.sample(4, "cpu")["actor_observation"].dtype == torch.float32
    torch.testing.assert_close(replay.fields["actor_observation"][:4].float(), obs)


def test_discounted_return_statistics_reset_after_terminal_reward():
    from hskill.learners.flashsac.reward_normalization import RewardNormalizer

    norm = RewardNormalizer(.99, 5., False, torch.device("cpu"))
    for reward, done in [(1., False), (10., True), (0., False)]:
        norm.update_reward_stats(torch.tensor([reward]), torch.tensor([done]), torch.tensor([False]))
    torch.testing.assert_close(norm.G_r, torch.zeros(1))
    torch.testing.assert_close(norm.G_r_max, torch.tensor([10.99]))
    torch.testing.assert_close(norm.G_rms.mean, torch.tensor([(1 + 10.99) / 3]))


def test_absorbing_observations_never_reach_batch_norm_or_targets():
    for n, done in [(1, [1.]), (3, [0., 1., 0.]), (2, [1., 1.])]:
        batch = _batch(n)
        batch["terminated"] = torch.tensor(done)
        for key in ("actor_next_observation", "next_observation"):
            batch[key][batch["terminated"].bool()] = math.nan
        learner = FlashSAC(12, 9, 4, "cpu", actor_width=8, critic_width=8, blocks=1)
        learner.calibrate(batch)
        assert all(math.isfinite(x) for x in learner.update(batch).values())
        for net in (learner.actor, learner.critic, learner.target):
            assert all(t.isfinite().all() for t in net.network.state_dict().values())
