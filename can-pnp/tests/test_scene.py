"""Scene geometry and layout sampling against the Isaac Lab values (CPU, no simulation)."""

import mujoco
import numpy as np
import pytest
import torch

from can_pnp.config import CanPnpSceneCfg
from can_pnp.layout import _hsv_to_rgb, sample_layout


@pytest.fixture(scope="module")
def scene_cfg():
    return CanPnpSceneCfg()


@pytest.fixture(scope="module")
def model(scene_cfg):
    from mjlab.scene import Scene

    from can_pnp.env_cfg import CanPnpEnvCfgArgs, make_scene_cfg

    return Scene(make_scene_cfg(CanPnpEnvCfgArgs(num_envs=1, scene=scene_cfg)), "cpu").compile()


def _geom_top(model, data, name):
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    return data.geom_xpos[gid][2] + model.geom_size[gid][2]


def test_robot_actuation(model):
    from can_pnp.constants import BODY_JOINT_NAMES, GRIPPER_JOINT_NAMES

    # mjlab G1 actuators on the 29 body joints + 2 gripper actuators; mirrors follow by equality.
    assert model.nu == 31
    assert model.neq == 2
    targets = {model.joint(model.actuator(i).trnid[0]).name for i in range(model.nu)}
    assert targets == {f"robot/{n}" for n in BODY_JOINT_NAMES + GRIPPER_JOINT_NAMES}


def test_robot_is_mjlab_g1_with_aloha_hands(model):
    def body_id(name):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"robot/{name}")

    # SuperMimic's capsule feet and the ALOHA finger contacts (condim 6) both collide.
    for name, condim in (("left_foot1_collision", 3), ("right_gripper_left_collision", 6)):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"robot/{name}")
        assert model.geom_contype[gid] and model.geom_condim[gid] == condim
    assert model.body_parentid[body_id("left_gripper_base")] == body_id("left_wrist_yaw_link")
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "robot/left_hand_collision") == -1


def test_prop_heights(model, scene_cfg):
    g = scene_cfg.geometry
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert _geom_top(model, data, "table/top") == pytest.approx(g.table_height, abs=1e-9)
    assert _geom_top(model, data, "room/shelf_layer2") == pytest.approx(g.box_bottom_z, abs=1e-9)
    # Box floor and rim match the reward config (box_floor_z 0.814, box_rim_z 0.91).
    assert _geom_top(model, data, "box/bottom") == pytest.approx(0.814, abs=1e-9)
    assert _geom_top(model, data, "box/xp") == pytest.approx(0.91, abs=1e-9)


def test_can_mass_and_size(model, scene_cfg):
    g = scene_cfg.geometry
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "can/can")
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "can/collider")
    assert model.body_mass[bid] == pytest.approx(g.can_mass)
    np.testing.assert_allclose(model.geom_size[gid][:2], (g.can_radius, g.can_height / 2))
    # Resting on the tabletop: center height = table + half height.
    assert g.can_initial_pos[2] == pytest.approx(g.table_height + g.can_height / 2)


def test_nominal_layout(scene_cfg):
    g = scene_cfg.geometry
    lay = sample_layout(scene_cfg, 4, randomize=False, device="cpu")
    torch.testing.assert_close(lay["table_front_x"], torch.full((4,), g.table_front_x))
    torch.testing.assert_close(lay["table_center_y"], torch.full((4,), g.table_center_y))
    # Spawn bounds: inset [0.08, 0.26] -> midpoint 0.17; robot 1.0 m in front of the table.
    torch.testing.assert_close(lay["can_xyz"][0], torch.tensor([g.table_front_x + 0.17, g.table_center_y, g.can_initial_pos[2]]))
    torch.testing.assert_close(lay["robot_xy"][0], torch.tensor([g.table_front_x - 1.0, g.table_center_y + 0.14]))
    assert (lay["robot_yaw_deg"] == 0).all()


def test_random_layout_ranges(scene_cfg):
    lay_cfg, g = scene_cfg.layout, scene_cfg.geometry
    torch.manual_seed(0)
    lay = sample_layout(scene_cfg, 4096, randomize=True, device="cpu")
    (inset_lo, lat_lo), (inset_hi, lat_hi) = lay_cfg.can_spawn_bounds()
    # Ranges of the recorded episodes (inset 0.080-0.259 m, lateral -0.200-0.190 m).
    assert (inset_lo, lat_lo, inset_hi, lat_hi) == (0.08, -0.2, 0.26, 0.2)
    inset = lay["can_xyz"][:, 0] - lay["table_front_x"]
    lateral = lay["can_xyz"][:, 1] - lay["table_center_y"]
    assert inset.min() >= inset_lo and inset.max() <= inset_hi
    assert lateral.min() >= lat_lo and lateral.max() <= lat_hi
    # Table back edge stays 0.3-0.6 m in front of the shelf front (y = 0.598).
    gap = lay_cfg.shelf_front_y - (lay["table_center_y"] + 0.3)
    assert gap.min() >= 0.3 - 1e-6 and gap.max() <= 0.6 + 1e-6
    fore = lay["table_front_x"] - g.table_front_x
    assert fore.abs().max() <= 0.25
    assert lay["robot_yaw_deg"].abs().max() <= 45.0
    front = lay["table_front_x"] - lay["robot_xy"][:, 0]
    assert front.min() >= 0.5 and front.max() <= 1.5


def test_hsv_matches_colorsys():
    import colorsys

    torch.manual_seed(0)
    hsv = torch.rand(256, 3)
    expected = torch.tensor([colorsys.hsv_to_rgb(*row.tolist()) for row in hsv])
    torch.testing.assert_close(_hsv_to_rgb(hsv), expected)
