"""The batched v0.1 human/five tracker inputs agree with the release's per-robot PolicyRunner."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / "downloads/morphmimic-v0.1"
FM_SOURCE = ROOT / "downloads/5points_FM/inference_source"
pytestmark = pytest.mark.skipif(not (RELEASE / "morphmimic_policy/policy.onnx").exists() or not FM_SOURCE.is_dir(),
                                reason="needs the MorphMimic v0.1 release and the five-point FM source")


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


@pytest.mark.parametrize("window", ["official", "spread"])
def test_batched_tracker_matches_policy_runner(window):
    sys.path[:0] = [str(FM_SOURCE), str(RELEASE / "source")]
    pytest.importorskip("onnxruntime")
    from deploy.sim2sim.policy_node import PolicyRunner
    from hskill.s0.official_five import HumanFiveReference
    from hskill.s0.official_five_batch import BatchedHumanFive

    n, execute, horizon = 4, 8, 32
    rng = np.random.default_rng(0)
    export = RELEASE / "morphmimic_policy"
    tracked = json.loads((export / "g1_amber.json").read_text())["robot"]["tracked_bodies"]
    stand = RELEASE / "source/deploy/vr_teleop/gmr_configs/stand.json"
    batched = BatchedHumanFive(RELEASE, n, execute, horizon, window=window)
    runners = [PolicyRunner(export, "g1_amber",
                            HumanFiveReference(stand, tracked, execute, horizon=horizon, window=window),
                            interface="human", human_points="five") for _ in range(n)]
    robot = batched.robot
    steps = [0] * n
    worst = 0.0
    for chunk in range(3):
        if chunk == 2:  # world 2 starts a new episode, as OfficialFiveController.body_action does
            batched.reset([2])
            runners[2].reset()
            runners[2].prev_step = -1
            steps[2] = 0
        pos = rng.normal(size=(n, horizon, 5, 3)) * 0.2 + np.array([0.0, 0.0, 0.8])
        quat = unit(rng.normal(size=(n, horizon, 5, 4)))
        batched.set_plan(pos, quat)
        for i, runner in enumerate(runners):
            runner.reference.update(pos[i], quat[i])
        for _ in range(execute):
            imu = unit(rng.normal(size=(n, 4)) * 0.1 + np.array([1.0, 0.0, 0.0, 0.0]))
            gyro = rng.normal(size=(n, 3)) * 0.5
            q = robot.default_joint_pos + rng.normal(size=(n, robot.num_native)) * 0.2
            dq = rng.normal(size=(n, robot.num_native))
            expected = np.stack([r.step(steps[i], imu[i], gyro[i], q[i], dq[i]) for i, r in enumerate(runners)])
            steps = [s + 1 for s in steps]
            got = batched.step(imu, gyro, q, dq)
            worst = max(worst, np.abs(got - expected).max())
    print(f"max |batched - PolicyRunner| target difference {worst:.2e} rad")
    assert worst < 1e-4, worst


@pytest.mark.parametrize("window", ["official", "spread"])
def test_torch_graph_matches_onnxruntime(window):
    sys.path[:0] = [str(FM_SOURCE), str(RELEASE / "source")]
    pytest.importorskip("onnxruntime")
    import torch
    from hskill.s0.official_five_batch import BatchedHumanFive

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n, rng = 16, np.random.default_rng(1)
    reference = BatchedHumanFive(RELEASE, n, 8, 32, window=window)
    graph = BatchedHumanFive(RELEASE, n, 8, 32, device=device, window=window)
    if graph.graph is None:  # a CPU device keeps onnxruntime; force the PyTorch graph on the CPU
        from deploy.sim2sim.policy_node import pruned
        import onnx
        from hskill.s0.onnx_torch import OnnxTorch
        model = pruned(onnx.load(RELEASE / "morphmimic_policy/policy.onnx"), "action_human")
        model = onnx.load_from_string(model) if isinstance(model, (bytes, bytearray)) else model
        graph.graph = OnnxTorch(model, "cpu")
        graph.constants = {k: torch.as_tensor(v) for k, v in graph.runner.inputs.items()}
    robot = reference.robot
    pos = rng.normal(size=(n, 32, 5, 3)) * 0.2 + np.array([0.0, 0.0, 0.8])
    quat = unit(rng.normal(size=(n, 32, 5, 4)))
    reference.set_plan(pos, quat)
    graph.set_plan(pos, quat)
    if graph.device_features is None:
        graph.device_features = torch.as_tensor(graph.features)
    worst = 0.0
    for _ in range(8):
        imu = unit(rng.normal(size=(n, 4)) * 0.1 + np.array([1.0, 0.0, 0.0, 0.0]))
        gyro = rng.normal(size=(n, 3)) * 0.5
        q = robot.default_joint_pos + rng.normal(size=(n, robot.num_native)) * 0.2
        dq = rng.normal(size=(n, robot.num_native))
        worst = max(worst, np.abs(graph.step(imu, gyro, q, dq) - reference.step(imu, gyro, q, dq)).max())
    print(f"max |PyTorch graph ({device}) - onnxruntime| target difference {worst:.2e} rad")
    assert worst < 1e-4, worst


def test_official_window_reads_the_next_ten_plan_frames():
    """At execution step k the official window is plan frames k+1 .. k+10, as the release's ClipReference."""
    sys.path[:0] = [str(FM_SOURCE), str(RELEASE / "source")]
    pytest.importorskip("onnxruntime")
    from hskill.s0.official_five_batch import BatchedHumanFive

    n, execute, horizon = 2, 8, 32
    rng = np.random.default_rng(2)
    batched = BatchedHumanFive(RELEASE, n, execute, horizon)
    robot = batched.robot
    pos = rng.normal(size=(n, horizon, 5, 3)) * 0.2 + np.array([0.0, 0.0, 0.8])
    quat = unit(rng.normal(size=(n, horizon, 5, 4)))
    batched.set_plan(pos, quat)
    for k in range(execute):
        imu = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
        q = np.tile(robot.default_joint_pos, (n, 1))
        batched.step(imu, np.zeros((n, 3)), q, np.zeros_like(q))
        np.testing.assert_array_equal(batched.windows, batched.plan_frames[:, k + 1:k + 11])


def test_device_goal_features_match_numpy():
    """The on-device goal and heading features equal the release's NumPy ones at every execution step."""
    sys.path[:0] = [str(FM_SOURCE), str(RELEASE / "source")]
    import torch
    from hskill.s0.official_five import HumanFiveReference
    from hskill.s0.official_five_batch import (heading_features_batch, heading_features_torch, human_features_batch,
                                               human_features_torch)
    from deploy.sim2sim.human import HUMAN_SPINE3, _anchor, point_mask

    rng = np.random.default_rng(3)
    n, h = 5, 32
    tracked = json.loads((RELEASE / "morphmimic_policy/g1_amber.json").read_text())["robot"]["tracked_bodies"]
    ref = HumanFiveReference(RELEASE / "source/deploy/vr_teleop/gmr_configs/stand.json", tracked, 8, horizon=h)
    ref.update((rng.normal(size=(n * h, 5, 3)) * 0.2 + np.array([0.0, 0.0, 0.8])), unit(rng.normal(size=(n * h, 5, 4))))
    frames = ref.frames.reshape(n, h, -1)
    mask = point_mask("five")
    torso = unit(rng.normal(size=(n, 4)))
    slot = HUMAN_SPINE3 if mask[HUMAN_SPINE3] else _anchor(mask)
    for k in range(8):
        win = frames[:, ref.window_indices(k)]
        t = torch.as_tensor(win, dtype=torch.float64)
        np.testing.assert_allclose(human_features_torch(t, mask).numpy(), human_features_batch(win, mask), atol=1e-9)
        got = heading_features_torch(torch.as_tensor(torso), t.unflatten(-1, (24, 7))[..., slot, 3:]).numpy()
        np.testing.assert_allclose(got, heading_features_batch(torso, win, mask), atol=1e-9)
