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


def test_batched_tracker_matches_policy_runner():
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
    batched = BatchedHumanFive(RELEASE, n, execute, horizon)
    runners = [PolicyRunner(export, "g1_amber", HumanFiveReference(stand, tracked, execute, future_horizon=horizon),
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


def test_torch_graph_matches_onnxruntime():
    sys.path[:0] = [str(FM_SOURCE), str(RELEASE / "source")]
    pytest.importorskip("onnxruntime")
    import torch
    from hskill.s0.official_five_batch import BatchedHumanFive

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n, rng = 16, np.random.default_rng(1)
    reference = BatchedHumanFive(RELEASE, n, 8, 32)
    graph = BatchedHumanFive(RELEASE, n, 8, 32, device=device)
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
