"""v3 pieces that need no simulator: task bookkeeping, object WM prior, buffer sampling, ranking."""

import torch
from hskill.data.buffer import EpisodeBuffer
from hskill.embodiment.g1_aloha import RIGHT_EEF
from hskill.interaction.model import (
    ObjectWMCfg,
    ObjectWMEnsemble,
    ObjectWorldModel,
    object_loss,
    object_targets,
    quat_exp,
    quat_log,
)
from hskill.learners.grpo import group_advantages
from hskill.sim.collect import Episodes, kendall_tau
from mjlab.utils.lab_api.math import quat_apply, quat_mul


def _held_can(L: int, W: int, B: int, seed: int = 0):
    """Random keypoint paths and a can rigidly held by the right gripper base along them."""
    g = torch.Generator().manual_seed(seed)
    kp = torch.cat([0.3 * torch.randn(L + W, B, 5, 3, generator=g),
                    torch.nn.functional.normalize(torch.randn(L + W, B, 5, 4, generator=g), dim=-1)], -1)  # fmt: skip
    hand = kp[:, :, RIGHT_EEF]
    offset = torch.tensor([0.05, 0.0, -0.02]).repeat(L + W, B, 1)
    rel = torch.nn.functional.normalize(torch.randn(B, 4, generator=g), dim=-1).repeat(L + W, 1, 1)
    pos = hand[..., :3] + quat_apply(hand[..., 3:].reshape(-1, 4), offset.reshape(-1, 3)).view(L + W, B, 3)
    quat = quat_mul(hand[..., 3:].reshape(-1, 4), rel.reshape(-1, 4)).view(L + W, B, 4)
    vel = torch.cat([torch.zeros(1, B, 3), (pos[1:] - pos[:-1]) / 0.02])
    return kp, torch.cat([pos, quat, vel], -1)


def test_object_wm_starts_as_a_rigid_grasp():
    L, H, B = 10, 32, 3
    rigid = ObjectWMCfg(output="rigid")
    for model in (ObjectWorldModel(rigid), ObjectWMEnsemble(rigid, members=3)):
        for W in (H, 2 * H, H + 8):  # one horizon, chained horizons, a partial last horizon
            kp, obj = _held_can(L, W, B)
            objs, logits = model.rollout(kp[:L], torch.zeros(L, B, 2), obj[:L], kp[L:], torch.zeros(W, B, 2))
            assert objs.shape == (W + 1, B, 13) and logits.shape == (W, B)
            torch.testing.assert_close(objs[..., :3], obj[L - 1 :, :, :3], atol=1e-5, rtol=0)
            assert ((objs[..., 3:7] * obj[L - 1 :, :, 3:7]).sum(-1).abs() > 1 - 1e-5).all()
            torch.testing.assert_close(objs[1:, :, 7:10], obj[L:, :, 7:10], atol=1e-4, rtol=0)
    true = object_targets(obj[L - 1 : L + H], 0.02)
    objs, logits = model.rollout(kp[:L], torch.zeros(L, B, 2), obj[:L], kp[L : L + H], torch.zeros(H, B, 2))
    loss, _ = object_loss(objs, logits, true, torch.zeros(H, B, dtype=torch.bool), torch.ones(H, B))
    loss.backward()
    assert all(p.grad is None or p.grad.isfinite().all() for p in model.parameters())
    v = torch.tensor([[0.3, -0.2, 0.1], [0.0, 0.0, 0.0]])
    torch.testing.assert_close(quat_log(quat_exp(v)), v, atol=1e-5, rtol=0)


def test_object_wm_delta_starts_at_rest():
    """The delta output assumes no grasp: untrained, the object stays where it is, whatever the hands do."""
    L, H, B = 10, 32, 3
    for model in (ObjectWorldModel(), ObjectWMEnsemble(members=3)):
        for W in (H, 2 * H, H + 8):
            kp, obj = _held_can(L, W, B)
            objs, logits = model.rollout(kp[:L], torch.zeros(L, B, 2), obj[:L], kp[L:], torch.zeros(W, B, 2))
            assert objs.shape == (W + 1, B, 13) and logits.shape == (W, B)
            torch.testing.assert_close(objs[..., :3], obj[L - 1 : L, :, :3].expand(W + 1, B, 3), atol=1e-6, rtol=0)
            assert ((objs[..., 3:7] * obj[L - 1 : L, :, 3:7]).sum(-1).abs() > 1 - 1e-6).all()
            torch.testing.assert_close(objs[2:, :, 7:10], torch.zeros(W - 1, B, 3), atol=1e-6, rtol=0)


def test_actual_and_ideal_branches_differ_only_by_future_keypoints():
    """J = Psi(M(history, actual future)), J* = Psi(M(history, ideal future)): the same history with
    other future hands gives another future of the can; the same future gives the same one."""
    L, H, B = 10, 32, 4
    for output in ("delta", "rigid"):
        torch.manual_seed(0)
        model = ObjectWMEnsemble(ObjectWMCfg(output=output), members=3)
        torch.nn.init.normal_(model.net.weights[-1], std=0.05)  # a trained-looking model, not just the prior
        _branches(model, L, H, B)


def _branches(model, L: int, H: int, B: int) -> None:
    kp, obj = _held_can(L, H, B)
    hist = (kp[:L], torch.zeros(L, B, 2), obj[:L])
    actual = kp[L:]
    ideal = actual.clone()
    ideal[:, :, RIGHT_EEF, :3] += 0.03  # the reference's hand, 3 cm off what S0 achieved
    cmd = torch.zeros(H, B, 2)
    with torch.no_grad():
        a1, _ = model.rollout(*hist, actual, cmd)
        a2, _ = model.rollout(*hist, actual, cmd)
        i1, _ = model.rollout(*hist, ideal, cmd)
    torch.testing.assert_close(a1, a2)
    torch.testing.assert_close(a1[0], i1[0])  # the present is shared
    # Every future frame differs (by little through an untrained delta head, by the 3 cm through the rigid prior).
    assert (a1[1:, :, :3] - i1[1:, :, :3]).norm(dim=-1).amin() > 1e-9


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
        kp=t[..., None].expand(-1, -1, 5, 7).clone(), width=t.expand(-1, -1, 2).clone(),
        command=t[:-1].expand(-1, -1, 2).clone(), obj=t.expand(-1, -1, 14).clone(),
        held=torch.zeros(T + 1, N, dtype=torch.bool), alive=alive,
        phi_kp=-torch.ones(D, H + 1, N, 5, 7), cond=torch.zeros(D, N, 2, 4), s0=torch.zeros(D, N, 71),
        obs0=torch.zeros(D, N, 930), decision_alive=torch.ones(D, N, dtype=torch.bool), xi=torch.zeros(N, 33),
        box=torch.zeros(N, 3), table=torch.zeros(N, 3), goal=torch.zeros(N, 3),
        final_potential=torch.tensor([10.0, 0.5, 0.0]), base=torch.tensor([False, True, False]),
    )  # fmt: skip


def test_buffer_windows_follow_the_episode():
    buf = EpisodeBuffer(capacity=2, decay=0.5, horizon=8, history=4)
    for _ in range(3):
        buf.add(_episodes())
    assert len(buf) == 2
    w = buf.windows(64, phi_fraction=0.0)
    assert w["obj"].shape == (9, 64, 14) and w["fut_kp"].shape == (8, 64, 5, 7) and w["fut_cmd"].shape == (8, 64, 2)
    assert w["hist_kp"].shape == (4, 64, 5, 7) and w["hist_width"].shape == (4, 64, 2) and w["hist_obj"].shape == (4, 64, 10)
    # Windows are consecutive steps from a decision (multiple of 8), every recorded value = time index.
    steps = w["obj"][:, :, 0]
    assert ((steps[1:] - steps[:-1]) == 1).all() and (steps[0] % 8 == 0).all()
    torch.testing.assert_close(w["fut_kp"][..., 0, 0], steps[1:])  # the keypoints after the decision
    torch.testing.assert_close(w["fut_cmd"][..., 0], steps[:-1])  # the commands from the decision on
    # The history ends at the decision; frames before the episode repeat its first one.
    torch.testing.assert_close(w["hist_obj"][-1, :, 0], steps[0])
    torch.testing.assert_close(w["hist_kp"][..., 0, 0], (steps[0][None] + torch.arange(-3.0, 1.0)[:, None]).clamp(min=0))
    assert (buf.windows(16, phi_fraction=1.0)["fut_kp"] == -1).all()
    s = buf.starts(5)
    assert s["cond"].shape == (5, 2, 4) and s["obj0"].shape == (5, 14) and s["hist_kp"].shape == (5, 4, 5, 7)
    # Critic samples: the policy envs only (env 1 ran on N(0, I)), time = decision step / episode steps.
    c = buf.starts(64, policy_only=True)
    assert set(c["final"].tolist()) <= {10.0, 0.0}
    torch.testing.assert_close(c["time"], torch.round(c["time"] * 32 / 8) * 8 / 32)
    torch.testing.assert_close(c["step_frac"], torch.full((64,), 1 / 32))


def test_buffer_gives_ideal_keypoints_only_to_exactly_executed_windows():
    buf = EpisodeBuffer(capacity=1, decay=1.0, horizon=8, history=4)
    ep = _episodes()
    ep.ref_kp = ep.kp.clone()
    ep.ref_kp[:, [0, 2]] += 0.005  # right hand 0.9 cm off: exactly executed
    ep.ref_kp[:, 1] += 0.1  # 17 cm off: not
    buf.add(ep)
    w = buf.windows(256, phi_fraction=0.0, ideal_fraction=1.0, well_tracked=0.02)
    offset = (w["fut_kp"][..., 0, 0] - w["obj"][1:, :, 0])[0]  # per window: the ideal offset or 0
    assert w["ideal"].any() and (~w["ideal"]).any()
    torch.testing.assert_close(offset, torch.where(w["ideal"], 0.005, 0.0))
    exact = buf.windows(256, phi_fraction=0.0, exact_only=True)
    sim_kp = (exact["fut_kp"][..., 0, 0] == exact["obj"][1:, :, 0]).all(0)
    assert sim_kp.all() and not exact["ideal"].any()  # no ideal input without ideal_fraction
    assert exact["mask"].any() and not exact["mask"].all()  # env 1's windows masked out, the others kept


def test_buffer_gives_phi_keypoints_only_to_phi_envs():
    buf = EpisodeBuffer(capacity=1, decay=1.0, horizon=8, history=4)
    ep = _episodes()
    ep.phi_env = torch.tensor([True, False, True])  # Phi ran windows for envs 0 and 2 only
    buf.add(ep)
    w = buf.windows(256, phi_fraction=1.0)
    from_phi = (w["fut_kp"] == -1).flatten(2).all(-1).all(0)
    assert from_phi.any() and (~from_phi).any()  # env 1's windows keep the simulated keypoints
    torch.testing.assert_close(w["fut_kp"][..., 0, 0][:, ~from_phi], w["obj"][1:, :, 0][:, ~from_phi])
    only = buf.windows(256, phi_fraction=1.0, phi_only=True)
    assert (only["fut_kp"] == -1).flatten(2).all(-1).all(0)[only["mask"].any(0)].all()


def test_masked_group_advantages_ignore_dropped_members():
    from hskill.learners.grpo import masked_group_advantages

    scores = torch.tensor([1.0, 2.0, 100.0, 0.0, 5.0, 5.0, 7.0, 0.0])
    keep = torch.tensor([True, True, False, False, True, False, False, False])
    adv, informative = masked_group_advantages(scores, 4, keep)
    assert adv[2] == 0 and adv[3] == 0 and adv[0] < 0 < adv[1]  # the outlier did not enter the stats
    assert informative.tolist() == [1.0, 0.0]  # one kept member: no signal
    assert (adv[4:] == 0).all()
