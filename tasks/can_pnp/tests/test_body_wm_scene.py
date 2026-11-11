"""Body world model glue (GPU): robot state, FK poses and the tracker observation it rebuilds."""

from pathlib import Path

import pytest
import torch
from hskill.executors.imagine import unflatten_history

FULL_TRACKER = Path(__file__).resolve().parents[4] / "supermimic_0100000.pt"


@pytest.mark.skipif(not torch.cuda.is_available() or not FULL_TRACKER.exists(), reason="needs CUDA and the tracker")
def test_state_poses_and_observation_match_the_env():
    from can_pnp.scene.env_cfg import CanPnpEnvCfgArgs, make_env
    from hskill.s0.body_world_model import BodyWorldModel
    from hskill.s0.kinematics import BodyKinematics

    env = make_env(CanPnpEnvCfgArgs(num_envs=4, camera=False))
    env.reset()
    torch.manual_seed(0)
    for _ in range(10):  # move off the default pose
        obs, *_ = env.step(0.3 * torch.randn(4, env.action_manager.total_action_dim, device=env.device))
    body = BodyWorldModel(env, FULL_TRACKER)
    from supermimic.world_model.heads import analytic_observation  # on the path once loaded
    kin = BodyKinematics(env, body)
    s = body.state()
    poses = kin.poses(s)
    robot = env.scene["robot"]
    origin = env.scene.env_origins
    for i, side in enumerate(("left", "right")):
        b = robot.body_names.index(f"{side}_gripper_base")
        torch.testing.assert_close(poses["gripper_pos"][:, i], robot.data.body_link_pos_w[:, b] - origin, atol=1e-4, rtol=0)
        q, q_env = poses["gripper_quat"][:, i], robot.data.body_link_quat_w[:, b]
        assert ((q * q_env).sum(-1).abs() > 1 - 1e-5).all()
    t = robot.body_names.index("torso_link")
    torch.testing.assert_close(poses["torso_pos"], robot.data.body_link_pos_w[:, t] - origin, atol=1e-4, rtol=0)
    # The object world model's keypoints (pelvis, ankles, gripper bases) as the simulator has them.
    from hskill.embodiment.g1_aloha import KEYPOINT_BODIES

    kp = kin.keypoints(s)
    ids = [robot.body_names.index(b) for b in KEYPOINT_BODIES]
    torch.testing.assert_close(kp[..., :3], robot.data.body_link_pos_w[:, ids] - origin[:, None], atol=1e-4, rtol=0)
    assert ((kp[..., 3:] * robot.data.body_link_quat_w[:, ids]).sum(-1).abs() > 1 - 1e-5).all()
    # The newest frame of the policy history is the analytic observation of the state.
    last = unflatten_history(obs["policy"])[-1]
    rebuilt = analytic_observation(s, last[:, -29:], body.context(), body.default_joint_pos)
    torch.testing.assert_close(rebuilt, last, atol=1e-4, rtol=0)
