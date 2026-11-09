"""Demonstration features of the progress critic match the simulator's at a recorded start."""

from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
DEMOS = ROOT / "data" / "residual_rl" / "can_pnp"
TRACKER = ROOT.parent / "supermimic_0100000.pt"


@pytest.mark.skipif(not torch.cuda.is_available() or not TRACKER.exists() or not DEMOS.exists(),
                    reason="needs CUDA, the tracker and the demonstrations")  # fmt: skip
def test_demo_and_sim_proprio_agree_at_the_grasp_start():
    from can_pnp.scene.env_cfg import CanPnpEnvCfgArgs, make_env
    from can_pnp.starts import GraspStarts, apply_grasp_start
    from hskill.data.demo_progress import DemoProgress
    from hskill.embodiment.g1_aloha import BODY_JOINT_NAMES
    from hskill.learners.critic import state_proprio
    from hskill.s0.body_world_model import BodyWorldModel
    from hskill.s0.controller import gripper_opening
    from hskill.s0.kinematics import BodyKinematics

    starts = GraspStarts(phase="CLOSE")
    available = {int(p.stem.split("_")[-1]) for p in DEMOS.glob("data/chunk-*/episode_*.parquet")}
    rows = [r for r, e in enumerate(starts.episode_ids) if e in available][:2]
    env = make_env(CanPnpEnvCfgArgs(num_envs=len(rows), camera=False, randomize=False))
    env.reset()
    apply_grasp_start(env, torch.arange(len(rows), device=env.device), starts, torch.tensor(rows))
    body = BodyWorldModel(env, TRACKER)
    kin = BodyKinematics(env, body, compile=False)
    robot = env.scene["robot"]
    right = robot.find_joints(("right_gripper",), preserve_order=True)[0][0]
    sim = state_proprio(kin, body.state(), gripper_opening(robot.data.joint_pos[:, right]))
    for i, r in enumerate(rows):
        episode = starts.episode_ids[r]
        demo = DemoProgress(DEMOS, {episode: int(starts.frame[r])}, BODY_JOINT_NAMES, 0.997, 10.0, 700, env.device)
        first = demo.proprio[0]
        torch.testing.assert_close(sim[i, :29], first[:29], atol=1e-3, rtol=0)  # joints as recorded
        torch.testing.assert_close(sim[i, 29:32], first[29:32], atol=0.02, rtol=0)  # gravity in the pelvis frame
        torch.testing.assert_close(sim[i, 32:38], first[32:38], atol=0.01, rtol=0)  # wrist positions (m)
        torch.testing.assert_close(sim[i, 38:50], first[38:50], atol=0.02, rtol=0)  # wrist orientations
        assert float(demo.target[-1]) == pytest.approx(10.0) and bool((demo.target[:-1] < demo.target[1:]).all())
