"""Can pick-and-place scene on mjlab (port of motion_token_residual_rl ``CanPnpEnvCfg``),
set up for the SuperMimic tracker.

Scene: Isaac Lab can_pnp props and layout, finger contact sensors against the target can
and an RGB head camera on ``torso_link``. Robot and control follow SuperMimic's training
env (``supermimic.envs.tracking_env_cfg``): mjlab G1 actuators, 5 ms steps, decimation 4
(50 Hz), solver iterations 10 / 20, and the same ``policy`` observation group. The action
is ``[29 tracker actions (q0 + G1_ACTION_SCALE * a) | 2 gripper joint targets]``.
Stages and rewards of the residual-RL task are not part of the port.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import mujoco
from mjlab.asset_zoo.robots import G1_ACTION_SCALE
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs import mdp as base_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import CameraSensorCfg, ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from can_pnp import constants as C
from can_pnp import entities, mdp
from can_pnp.config import CanPnpSceneCfg
from can_pnp.layout import reset_can_pnp_layout


@dataclass
class CanPnpEnvCfgArgs:
    num_envs: int = 64
    # Layout / resident / box randomization (Isaac Lab: domain_randomization.train_enabled).
    randomize: bool = True
    camera: bool = True
    # Head-camera RGB renderer: MuJoCo Warp, or HyperLux (OptiX, scripts/setup_hyperlux.sh). HyperLux
    # takes geom colors from the compiled model, so per-env color randomization is not rendered.
    rgb_backend: Literal["mujoco_warp", "hyperlux"] = "mujoco_warp"
    # Dataset episodes run up to ~20 s (navigation + pick + place).
    episode_length_s: float = 30.0
    fall_height: float = 0.55
    can_drop_height: float = 0.2
    seed: int | None = 0
    # Per-world contact / constraint buffers. mjlab's heuristic gives njmax = 64, which overflows
    # here (dropped constraints sink the cans and blow up the robot); a fallen robot with both
    # cans resting peaks at ~40 contacts / ~170 constraint rows.
    nconmax: int | None = 128
    # Cancel the extra weight of the ALOHA gripper over the hand SuperMimic is trained with
    # (entities._hand_gravcomp); without it the hand sags ~11 cm and cannot lift the can.
    hand_gravcomp: bool = True
    njmax: int | None = 600
    scene: CanPnpSceneCfg = field(default_factory=CanPnpSceneCfg)


def _edit_scene_spec(spec: mujoco.MjSpec, scene: CanPnpSceneCfg) -> None:
    """Ground plane color / friction and the global lights."""
    ground = spec.geom("terrain")
    ground.rgba = entities.srgb(entities.FLOOR)
    ground.friction = (scene.geometry.ground_static_friction, 0.005, 0.0001)
    spec.visual.headlight.ambient = entities.HEADLIGHT_AMBIENT
    spec.visual.headlight.diffuse = entities.HEADLIGHT_DIFFUSE
    spec.visual.headlight.specular = (0.0, 0.0, 0.0)


def make_scene_cfg(a: CanPnpEnvCfgArgs) -> SceneCfg:
    g = a.scene.geometry
    finger_sensors = tuple(
        ContactSensorCfg(
            name=f"finger_{side}_can_contacts",
            primary=ContactMatch(mode="body", pattern=body, entity="robot"),
            secondary=ContactMatch(mode="body", pattern="can", entity="can"),
            fields=("found", "force"),
            reduce="netforce",
            history_length=4,
        )
        for side, body in zip(("l", "r"), C.RIGHT_FINGER_BODIES)
    )
    camera_cfg = CameraSensorCfg
    extra = {}
    if a.rgb_backend == "hyperlux":
        from mjlab_hyperlux import HyperLuxCameraSensorCfg

        camera_cfg = HyperLuxCameraSensorCfg
        extra = {"hyperlux_anti_aliasing_grid": 1, "hyperlux_batch_size": "auto"}
    camera = camera_cfg(
        name="head_camera",
        parent_body="robot/torso_link",
        pos=a.scene.camera.pos,
        quat=a.scene.camera.quat,
        fovy=a.scene.camera.fovy,
        width=a.scene.camera.width,
        height=a.scene.camera.height,
        data_types=("rgb",),
        use_shadows=a.scene.camera.use_shadows,
        **extra,
    )
    shelf_xy = a.scene.layout.shelf_center_xy_nominal
    return SceneCfg(
        num_envs=a.num_envs,
        env_spacing=g.env_spacing,
        terrain=TerrainEntityCfg(
            terrain_type="plane", textures=(), materials=(), lights=entities.SCENE_LIGHTS
        ),
        entities={
            "robot": entities.robot_cfg(a.hand_gravcomp),
            "can": entities.can_cfg(g, g.can_initial_pos),
            "shelf_can": entities.can_cfg(g, (*shelf_xy, a.scene.layout.shelf_center_z)),
            "table": entities.table_cfg(g),
            "box": entities.box_cfg(g),
            "room": entities.room_cfg(g),
        },
        sensors=finger_sensors + ((camera,) if a.camera else ()),
        spec_fn=lambda spec: _edit_scene_spec(spec, a.scene),
    )


def make_env_cfg(a: CanPnpEnvCfgArgs) -> ManagerBasedRlEnvCfg:
    body = SceneEntityCfg("robot", joint_names=C.BODY_JOINT_NAMES)
    observations = {
        # SuperMimic's policy group without noise: 5 terms x 10 frames, term-major (930).
        "policy": ObservationGroupCfg(
            terms={
                "joint_pos": ObservationTermCfg(func=base_mdp.joint_pos_rel, params={"asset_cfg": body}),
                "joint_vel": ObservationTermCfg(func=base_mdp.joint_vel_rel, params={"asset_cfg": body}),
                "base_ang_vel": ObservationTermCfg(func=base_mdp.base_ang_vel),
                "projected_gravity": ObservationTermCfg(func=base_mdp.projected_gravity),
                "actions": ObservationTermCfg(func=base_mdp.last_action, params={"action_name": "joint_pos"}),
            },
            concatenate_terms=True,
            enable_corruption=False,
            history_length=10,
            flatten_history_dim=True,
        ),
        "scene": ObservationGroupCfg(
            terms={
                "can_pos_b": ObservationTermCfg(func=mdp.object_pos_b, params={"entity_name": "can"}),
                "can_quat_w": ObservationTermCfg(func=mdp.object_quat_w, params={"entity_name": "can"}),
                "shelf_can_pos_b": ObservationTermCfg(func=mdp.object_pos_b, params={"entity_name": "shelf_can"}),
                "goal_pos_b": ObservationTermCfg(func=mdp.goal_pos_b),
            },
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }  # fmt: skip
    actions = {
        # SuperMimic output; mjlab orders the targets by joint definition = BODY_JOINT_NAMES.
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=C.BODY_JOINT_NAMES,
            scale=G1_ACTION_SCALE,
            use_default_offset=True,
        ),
        # Absolute finger slide targets (0 = open, 0.035 = closed).
        "gripper": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=C.GRIPPER_JOINT_NAMES,
            scale=1.0,
            use_default_offset=False,
        ),
    }
    events = {
        "reset_layout": EventTermCfg(
            func=reset_can_pnp_layout,
            mode="reset",
            params={"scene_cfg": a.scene, "randomize": a.randomize},
        )
    }
    terminations = {
        "time_out": TerminationTermCfg(func=base_mdp.time_out, time_out=True),
        "fallen": TerminationTermCfg(func=mdp.robot_fallen, params={"fall_height": a.fall_height}),
        "can_dropped": TerminationTermCfg(
            func=mdp.can_dropped, params={"entity_name": "can", "drop_height": a.can_drop_height}
        ),
    }
    return ManagerBasedRlEnvCfg(
        scene=make_scene_cfg(a),
        observations=observations,
        actions=actions,
        events=events,
        terminations=terminations,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY, entity_name="robot",
            body_name="torso_link", distance=3.0, fovy=55.0, elevation=-25.0, azimuth=200.0,
        ),  # fmt: skip
        sim=SimulationCfg(
            nconmax=a.nconmax,
            njmax=a.njmax,
            mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
        ),
        decimation=4,
        episode_length_s=a.episode_length_s,
        seed=a.seed,
    )


def make_env(a: CanPnpEnvCfgArgs, device: str = "cuda:0") -> ManagerBasedRlEnv:
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()
    return ManagerBasedRlEnv(make_env_cfg(a), device=device)
