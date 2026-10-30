"""Body world model glue (GPU): robot state, FK poses and the tracker observation it rebuilds."""

from pathlib import Path

import pytest
import torch

from can_pnp.body.imagine import flatten_history, unflatten_history

FULL_TRACKER = Path(__file__).resolve().parents[3] / "supermimic_0100000.pt"


def test_history_roundtrip():
    obs = torch.randn(5, 930)
    torch.testing.assert_close(flatten_history(unflatten_history(obs)), obs, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available() or not FULL_TRACKER.exists(), reason="needs CUDA and the tracker")
def test_state_poses_and_observation_match_the_env():
    from can_pnp.body.kinematics import BodyKinematics
    from can_pnp.body.world_model import BodyWorldModel
    from can_pnp.env_cfg import CanPnpEnvCfgArgs, make_env

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
    # The newest frame of the policy history is the analytic observation of the state.
    last = unflatten_history(obs["policy"])[-1]
    rebuilt = analytic_observation(s, last[:, -29:], body.context(), body.default_joint_pos)
    torch.testing.assert_close(rebuilt, last, atol=1e-4, rtol=0)
