"""SuperMimic goal features (CPU)."""

import math
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch
from mjlab.utils.lab_api.math import matrix_from_quat, quat_from_euler_xyz

from can_pnp import constants as C
from can_pnp.supermimic import (
    FULL_GOAL_DIM,
    GOAL_DIM,
    ReferenceTrajectory,
    matrix_to_rot6d,
    rot6d_to_matrix,
    torso_in_pelvis,
)

TRACKER = Path(__file__).resolve().parents[3] / (
    "logs/supermimic/1023_SM_Reproduced_LafanBones_DeployObs_Shards_2xA800_N4096_4278/ckpt_0268000.pt"
)
FULL_TRACKER = Path(__file__).resolve().parents[3] / "supermimic_0100000.pt"


def test_rot6d_roundtrip():
    torch.manual_seed(0)
    q = torch.nn.functional.normalize(torch.randn(64, 4), dim=-1)
    m = matrix_from_quat(q)
    torch.testing.assert_close(rot6d_to_matrix(matrix_to_rot6d(m)), m, atol=1e-6, rtol=0)


def test_torso_rotation_matches_mujoco_fk():
    model = mujoco.MjModel.from_xml_path(str(C.ROBOT_XML))
    data = mujoco.MjData(model)
    torch.manual_seed(0)
    for _ in range(5):
        waist = (torch.rand(3) - 0.5) * 0.8
        data.qpos[:] = model.qpos0
        for name, value in zip(C.BODY_JOINT_NAMES[12:15], waist):
            data.qpos[model.joint(name).qposadr[0]] = float(value)
        mujoco.mj_forward(model, data)
        pelvis = torch.tensor(data.xmat[model.body("pelvis").id].reshape(3, 3))
        torso = torch.tensor(data.xmat[model.body("torso_link").id].reshape(3, 3))
        torch.testing.assert_close(pelvis.T @ torso, torso_in_pelvis(waist.double()), atol=1e-6, rtol=0)


def _quat(matrix: torch.Tensor) -> torch.Tensor:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, matrix.double().numpy().reshape(-1))
    return torch.tensor(q, dtype=torch.float32)


def test_reference_goal_features():
    n, h, dt = 2, 32, 0.02
    q0 = torch.tensor(C.DEPLOY_DEFAULT_ANGLES).repeat(n, 1)
    t = torch.arange(h) * dt
    # Joint 0 ramps at 1 rad/s; the reference root yaws at 0.5 rad/s relative to the root now,
    # which is rolled by 30 deg in env 0 and upright in env 1.
    joint_pos = q0[:, None].repeat(1, h, 1)
    joint_pos[..., 0] += t
    yaw = 0.5 * t
    rel = matrix_from_quat(quat_from_euler_xyz(0 * yaw, 0 * yaw, yaw))
    roll = math.radians(30)
    root_quat = quat_from_euler_xyz(torch.tensor([roll, 0.0]), torch.zeros(2), torch.zeros(2))
    ref = ReferenceTrajectory(joint_pos, matrix_to_rot6d(rel)[None].repeat(n, 1, 1), root_quat, q0, dt)
    torch.testing.assert_close(ref.frames[..., 0], t.expand(n, h))
    torch.testing.assert_close(ref.frames[..., 29], torch.ones(n, h), atol=1e-4, rtol=0)
    # Gravity in the reference root frame: upright -> (0, 0, -1); rolled -> tilt preserved by yaw.
    torch.testing.assert_close(ref.gravity_b[1], torch.tensor([0.0, 0.0, -1.0]).expand(h, 3), atol=1e-6, rtol=0)
    torch.testing.assert_close(ref.gravity_b[0, :, 2], torch.full((h,), -math.cos(roll)), atol=1e-6, rtol=0)
    torch.testing.assert_close(ref.gravity_b[0, :, :2].norm(dim=-1), torch.full((h,), math.sin(roll)), atol=1e-6, rtol=0)
    # Body angular velocity: 0.5 rad/s about the reference root z axis.
    torch.testing.assert_close(ref.ang_vel_b[..., 2], torch.full((n, h), 0.5), atol=1e-3, rtol=0)
    # Anchor: the robot torso already at the reference -> identity rotation.
    robot_anchor = torch.stack(
        [_quat(ref.root_w[i, 0] @ torso_in_pelvis(joint_pos[i, 0, 12:15])) for i in range(n)]
    )
    goal = ref.goal(0, robot_anchor)
    assert goal.shape == (n, GOAL_DIM)
    torch.testing.assert_close(goal[:, -6:], torch.tensor([1.0, 0, 0, 1, 0, 0]).expand(n, 6), atol=1e-5, rtol=0)
    # Frames past the horizon clamp to the last one.
    late = ref.goal(h - 2, robot_anchor)[:, :640].reshape(n, 10, 64)
    torch.testing.assert_close(late[:, 1:], ref.frames[:, -1:].expand(n, 9, 64))


def test_full_goal_features():
    n, h, dt = 2, 32, 0.02
    q0 = torch.tensor(C.DEPLOY_DEFAULT_ANGLES).repeat(n, 1)
    rot6d = torch.tensor([1.0, 0, 0, 1, 0, 0]).repeat(n, h, 1)
    height = torch.tensor([0.78, 0.70])[:, None].expand(n, h)
    # World velocity +x; env 1's root is yawed by 90 deg, so in its yaw frame that is -y.
    vel = torch.tensor([0.5, 0.0, 0.0]).expand(n, h, 3)
    root_quat = torch.tensor([[1.0, 0, 0, 0], [math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)]])
    ref = ReferenceTrajectory(q0[:, None].repeat(1, h, 1), rot6d, root_quat, q0, dt,
                              root_height=height, root_lin_vel_w=vel)
    goal = ref.goal(0, root_quat).reshape(n, 10, 68)
    assert goal.reshape(n, -1).shape[1] == FULL_GOAL_DIM
    torch.testing.assert_close(goal[0, :, 64:67], torch.tensor([0.5, 0, 0]).expand(10, 3), atol=1e-6, rtol=0)
    torch.testing.assert_close(goal[1, :, 64:67], torch.tensor([0, -0.5, 0]).expand(10, 3), atol=1e-6, rtol=0)
    torch.testing.assert_close(goal[..., 67], height[:, :10])
    # The first 64 features are the deploy_obs frame.
    deploy = ReferenceTrajectory(q0[:, None].repeat(1, h, 1), rot6d, root_quat, q0, dt)
    torch.testing.assert_close(goal[..., :64], deploy.frames[:, :10])


def test_leg_odometry_matches_mujoco():
    from can_pnp.odometry import LegKinematics, root_height_and_velocity

    model = mujoco.MjModel.from_xml_path(str(C.ROBOT_XML))
    data = mujoco.MjData(model)
    legs = LegKinematics(model, C.BODY_JOINT_NAMES, "cpu")
    torch.manual_seed(0)
    q = torch.tensor(C.DEPLOY_DEFAULT_ANGLES) + (torch.rand(29) - 0.5) * 0.4
    data.qpos[:] = model.qpos0
    for name, value in zip(C.BODY_JOINT_NAMES, q):
        data.qpos[model.joint(name).qposadr[0]] = float(value)
    mujoco.mj_forward(model, data)
    pelvis = model.body("pelvis").id
    for i, foot in enumerate(("left_ankle_roll_link", "right_ankle_roll_link")):
        b = model.body(foot).id
        sole = data.xpos[b] + data.xmat[b].reshape(3, 3) @ np.array([0.04, 0.0, -0.035])
        expected = data.xmat[pelvis].reshape(3, 3).T @ (sole - data.xpos[pelvis])
        torch.testing.assert_close(legs.soles(q)[i], torch.tensor(expected, dtype=torch.float32), atol=1e-5, rtol=0)
    # Standing still: measured height everywhere, zero velocity.
    eye = torch.eye(3)[None]
    height, vel = root_height_and_velocity(legs, q.expand(1, 8, 29), eye[:, None].expand(1, 8, 3, 3),
                                           q[None], eye, torch.tensor([0.78]), 0.02)
    torch.testing.assert_close(height, torch.full((1, 8), 0.78), atol=1e-6, rtol=0)
    torch.testing.assert_close(vel, torch.zeros(1, 8, 3), atol=1e-6, rtol=0)


@pytest.mark.skipif(not TRACKER.exists(), reason="SuperMimic checkpoint not present")
def test_tracker_checkpoint_loads():
    from can_pnp.supermimic import OBS_DIM, load_tracking_policy

    policy = load_tracking_policy(TRACKER, device="cpu")
    action = policy(torch.zeros(3, OBS_DIM), torch.zeros(3, GOAL_DIM))
    assert action.shape == (3, 29) and torch.isfinite(action).all()


@pytest.mark.skipif(not FULL_TRACKER.exists(), reason="full-observation SuperMimic checkpoint not present")
def test_full_tracker_checkpoint_loads():
    from can_pnp.supermimic import OBS_DIM, load_tracking_policy

    policy = load_tracking_policy(FULL_TRACKER, device="cpu")
    assert policy.goal_dim == FULL_GOAL_DIM
    action = policy(torch.zeros(3, OBS_DIM), torch.zeros(3, FULL_GOAL_DIM))
    assert action.shape == (3, 29) and torch.isfinite(action).all()
