"""G1 + ALOHA construction with SuperMimic training dynamics."""

from __future__ import annotations

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_ARTICULATION, G1_XML
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

from hskill.embodiment import g1_aloha as C
from hskill.utils.colors import srgb

ROBOT_BLACK = (0.05, 0.05, 0.05)
# Link-name fragments whose visuals envs/isaaclab/config.py repaints ROBOT_BLACK.
ROBOT_BLACK_PARTS = ("gripper", "finger", "aloha2", "head", "waist", "hip_pitch", "ankle_roll")


# FULL_COLLISION of mjlab's G1 (SuperMimic's training contacts: self collisions on, condim 1,
# capsule feet condim 3 / friction 0.6), plus the ALOHA finger geoms with their MJCF
# friction (1, 0.005, 0.0005) and condim 6 so grasps hold.
_FOOT = r"^(left|right)_foot[1-7]_collision$"
_GRIPPER = r"^(left|right)_gripper_.*$"
ROBOT_COLLISION = CollisionCfg(
    geom_names_expr=(".*_collision", _GRIPPER),
    contype=1,
    conaffinity=1,
    condim={_FOOT: 3, _GRIPPER: 6, ".*": 1},
    priority={_FOOT: 1, ".*": 0},
    friction={_FOOT: (0.6,)},
)


def _hand_gravcomp(spec: mujoco.MjSpec) -> None:
    """Cancel the weight the ALOHA gripper adds over the hand SuperMimic is trained with.

    The mjlab G1 wrist_yaw_link (rubber hand) weighs 0.255 kg; with the adapter and gripper the
    hand weighs 1.05 kg. Against the arm PD gains SuperMimic was trained with (shoulder / elbow
    kp 14 Nm/rad) that sags the hand ~11 cm, more than the recorded lift. Gravity
    compensation (MuJoCo ``gravcomp``, an upward force equal to the body weight, as a
    feed-forward term on the real arm) leaves the arm the training hand's weight.
    """
    trained = mujoco.MjSpec.from_file(str(G1_XML)).body("left_wrist_yaw_link").mass
    for side in ("left", "right"):
        wrist = spec.body(f"{side}_wrist_yaw_link")
        wrist.gravcomp = max(0.0, 1.0 - trained / wrist.mass)
        for name in (f"{side}_gripper_base", f"{side}_aloha2_finger_left", f"{side}_aloha2_finger_right"):
            spec.body(name).gravcomp = 1.0


def _robot_spec(hand_gravcomp: bool = True) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(C.ROBOT_XML))
    # mjlab's G1 carries a top light tracking the robot; the Isaac scene has no such light and it
    # washes out the tabletop in the head camera.
    for light in list(spec.lights):
        spec.delete(light)
    if hand_gravcomp:
        _hand_gravcomp(spec)
    black = srgb(ROBOT_BLACK)
    for geom in spec.geoms:
        if geom.contype or geom.conaffinity:
            continue
        body = geom.parent.name.lower()
        mesh = (geom.meshname or "").lower()
        if any(p in body for p in ROBOT_BLACK_PARTS) or any(
            p in mesh for p in ("head", "waist_support")
        ):
            # G1 visuals carry a material (which overrides rgba); the ALOHA ones do not.
            if geom.material:
                geom.material = "black"
            else:
                geom.rgba = black
    return spec


def robot_cfg(hand_gravcomp: bool = True) -> EntityCfg:
    grippers = BuiltinPositionActuatorCfg(
        target_names_expr=C.GRIPPER_JOINT_NAMES,
        stiffness=C.GRIPPER_STIFFNESS,
        damping=C.GRIPPER_DAMPING,
        effort_limit=C.GRIPPER_EFFORT_LIMIT,
    )
    joint_pos = dict(zip(C.BODY_JOINT_NAMES, C.DEPLOY_DEFAULT_ANGLES))
    joint_pos.update({name: 0.0 for name in C.GRIPPER_JOINT_NAMES + C.MIRROR_GRIPPER_JOINT_NAMES})
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, C.ROBOT_INIT_HEIGHT), joint_pos=joint_pos, joint_vel={".*": 0.0}
        ),
        spec_fn=lambda: _robot_spec(hand_gravcomp),
        collisions=(ROBOT_COLLISION,),
        articulation=EntityArticulationInfoCfg(
            actuators=(*G1_ARTICULATION.actuators, grippers),
            soft_joint_pos_limit_factor=G1_ARTICULATION.soft_joint_pos_limit_factor,
        ),
    )


