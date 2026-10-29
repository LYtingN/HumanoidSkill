"""Robot constants.

The robot is mjlab's G1 (the model SuperMimic is trained on) with the ALOHA 2 grippers of
motion_token_residual_rl (``assets/robots/g1_aloha``, built by ``scripts/build_assets.py``).
The 29 body joints use mjlab's G1 actuators so the SuperMimic tracker sees its training
dynamics; the grippers keep the can_pnp actuator (``envs/isaaclab/config.py``).
"""

from __future__ import annotations

from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
ROBOT_XML = ASSETS_DIR / "robots" / "g1_aloha" / "g1_aloha.xml"
SCENE_ASSETS_DIR = ASSETS_DIR / "scene"

# IsaacLab / SONIC / mjlab-G1 joint order: the SuperMimic action and the diffusion state.
BODY_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
# Actuated finger slides; the MJCF equality makes each mirror finger follow.
GRIPPER_JOINT_NAMES = ("left_gripper", "right_gripper")
MIRROR_GRIPPER_JOINT_NAMES = ("left_aloha2_finger_mirror", "right_aloha2_finger_mirror")

# SONIC deploy pose == mjlab KNEES_BENT_KEYFRAME == SuperMimic's default joint position q0.
DEPLOY_DEFAULT_ANGLES = (
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    0.0, 0.0, 0.0,
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
)  # fmt: skip
ROBOT_INIT_HEIGHT = 0.76  # KNEES_BENT_KEYFRAME pelvis height

GRIPPER_STIFFNESS = 1800.0
GRIPPER_DAMPING = 20.0
GRIPPER_EFFORT_LIMIT = 35.0
# Gripper opening (diffusion / dataset units, metres; 0.065 = open) <-> slide joint q:
# q = 0.035 (1 - opening / 0.065)  (dataset ``gripper_action_decode``).
GRIPPER_OPEN_WIDTH = 0.065
GRIPPER_CLOSED_Q = 0.035

# Bodies named in the Isaac Lab contact sensors (right-hand fingers against the target can).
RIGHT_FINGER_BODIES = ("right_aloha2_finger_left", "right_aloha2_finger_right")
