"""mjlab entities of the can pick-and-place scene (port of ``envs/isaaclab/scene.py``).

Isaac Lab spawned USD prims per env; here every prop is an mjlab entity:

* ``robot``     mjlab G1 (SuperMimic's model and actuators) with ALOHA 2 grippers (``g1_aloha``).
* ``can``       target can on the table (free body).
* ``shelf_can`` resident can inside the box (free body).
* ``table``     tabletop + 4 legs (mocap: moved by the layout at every reset).
* ``box``       open cardboard box on the top shelf (mocap: moved by the ``box`` DR jitter).
* ``room``      shelf, carpet and floor cable (mocap: placed at the env origin).

Colors: Isaac's UsdPreviewSurface ``diffuseColor`` is linear; MuJoCo ``rgba`` is used as
display (sRGB) values, so scene colors go through :func:`srgb`.
"""

from __future__ import annotations

import math

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_ARTICULATION, G1_XML
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg, LightCfg, MaterialCfg, TextureCfg

from can_pnp import constants as C
from can_pnp.config import SceneGeometryCfg

_BOX = mujoco.mjtGeom.mjGEOM_BOX
_CYLINDER = mujoco.mjtGeom.mjGEOM_CYLINDER
_MESH = mujoco.mjtGeom.mjGEOM_MESH
_PLANE = mujoco.mjtGeom.mjGEOM_PLANE

# Visual-only geoms: no contacts, rendered by the camera (groups 0-2 are visible).
_VISUAL = dict(contype=0, conaffinity=0, group=2, density=0.0)
# Collision geoms that are also what is rendered (plain colored props).
_PROP = dict(contype=1, conaffinity=1, group=0)

# Linear diffuse colors of envs/isaaclab/scene.py.
FLOOR = (0.035, 0.04, 0.04)
WHITE = (0.72, 0.72, 0.69)
TABLE_LEG = (0.72, 0.72, 0.7)
BLACK = (0.026, 0.031, 0.029)
CARDBOARD = (0.55, 0.36, 0.18)
CABLE = (0.23, 0.48, 0.04)
CAN_BODY = (0.065, 0.025, 0.105)
CAN_LID = (0.105, 0.065, 0.155)
CAN_SEAM = (0.065, 0.012, 0.095)
ROBOT_BLACK = (0.05, 0.05, 0.05)
# Link-name fragments whose visuals envs/isaaclab/config.py repaints ROBOT_BLACK.
ROBOT_BLACK_PARTS = ("gripper", "finger", "aloha2", "head", "waist", "hip_pitch", "ankle_roll")


def srgb(linear: tuple[float, float, float], alpha: float = 1.0) -> tuple[float, ...]:
    def encode(c: float) -> float:
        return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055

    return (*(encode(c) for c in linear), alpha)


def _rotate(quat: tuple[float, float, float, float], v: tuple[float, float, float]):
    w, x, y, z = quat
    vx, vy, vz = v
    # v + 2 w (q x v) + 2 q x (q x v)
    cx, cy, cz = y * vz - z * vy, z * vx - x * vz, x * vy - y * vx
    ccx, ccy, ccz = y * cz - z * cy, z * cx - x * cz, x * cy - y * cx
    return (vx + 2 * (w * cx + ccx), vy + 2 * (w * cy + ccy), vz + 2 * (w * cz + ccz))


# ----------------------------------------------------------------------------- robot

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


# ----------------------------------------------------------------------------- cans

# purple_can.usda is authored at radius 0.029, height 0.115 and scaled to the task size.
_CAN_USD_RADIUS = 0.029
_CAN_USD_HEIGHT = 0.115
CAN_VISUAL_GEOMS = ("body_visual", "seam_visual", "lid_visual")


def _can_spec(g: SceneGeometryCfg) -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    sxy, sz = g.can_radius / _CAN_USD_RADIUS, g.can_height / _CAN_USD_HEIGHT
    body = spec.worldbody.add_body(name="can")
    body.add_freejoint(name="can_root")
    body.add_geom(
        name="collider",
        type=_CYLINDER,
        size=(g.can_radius, g.can_height / 2, 0.0),
        mass=g.can_mass,
        friction=(g.can_static_friction, g.can_torsional_friction, g.can_rolling_friction),
        condim=g.can_condim,
        solref=g.can_solref,
        rgba=(1.0, 1.0, 1.0, 0.0),
        contype=1,
        conaffinity=1,
        group=3,
    )
    for part, color in (("body", CAN_BODY), ("seam", CAN_SEAM), ("lid", CAN_LID)):
        spec.add_mesh(
            name=f"purple_can_{part}",
            file=str(C.SCENE_ASSETS_DIR / f"purple_can_{part}.obj"),
            scale=(sxy, sxy, sz),
            # The seam is a zero-thickness band; shell inertia avoids a volume error.
            inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL,
        )
        body.add_geom(
            name=f"{part}_visual", type=_MESH, meshname=f"purple_can_{part}", rgba=srgb(color),
            **_VISUAL,
        )
    return spec


def can_cfg(g: SceneGeometryCfg, pos: tuple[float, float, float]) -> EntityCfg:
    return EntityCfg(init_state=EntityCfg.InitialStateCfg(pos=pos), spec_fn=lambda: _can_spec(g))


# ----------------------------------------------------------------------------- table


def _table_spec(g: SceneGeometryCfg) -> mujoco.MjSpec:
    """Table centered on its body frame; the layout writes the body at (front_x + 0.6, y, 0)."""
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="table")
    sx, sy, sz = g.table_size
    friction = (g.prop_static_friction, 0.005, 0.0001)
    body.add_geom(
        name="top", type=_BOX, size=(sx / 2, sy / 2, sz / 2), pos=(0, 0, g.table_height - sz / 2),
        rgba=srgb(WHITE), friction=friction, **_PROP,
    )  # fmt: skip
    leg_length = g.table_height - sz
    for k, (dx, dy) in enumerate(g.table_leg_offsets):
        body.add_geom(
            name=f"leg{k}", type=_CYLINDER, size=(g.table_leg_radius, leg_length / 2, 0),
            pos=(dx, dy, leg_length / 2), rgba=srgb(TABLE_LEG), friction=friction, **_PROP,
        )  # fmt: skip
    return spec


def table_cfg(g: SceneGeometryCfg) -> EntityCfg:
    pos = (g.table_front_x + g.table_size[0] / 2, g.table_center_y, 0.0)
    return EntityCfg(init_state=EntityCfg.InitialStateCfg(pos=pos), spec_fn=lambda: _table_spec(g))


# ----------------------------------------------------------------------------- box


def _add_open_box(body, prefix, center, size, bottom_z, thickness, friction) -> None:
    """Bottom plate plus four walls, as the ``box()`` loop of envs/isaaclab/scene.py."""
    cx, cy = center
    dx, dy, h = size
    t = thickness
    kw = dict(type=_BOX, rgba=srgb(CARDBOARD), friction=friction, **_PROP)
    body.add_geom(
        name=f"{prefix}bottom", size=(dx / 2, dy / 2, t / 2), pos=(cx, cy, bottom_z + t / 2), **kw
    )
    for s, tag in ((-1, "m"), (1, "p")):
        body.add_geom(
            name=f"{prefix}x{tag}", size=(t / 2, dy / 2, h / 2),
            pos=(cx + s * (dx / 2 - t / 2), cy, bottom_z + h / 2), **kw,
        )  # fmt: skip
        body.add_geom(
            name=f"{prefix}y{tag}", size=(dx / 2 - t, t / 2, h / 2),
            pos=(cx, cy + s * (dy / 2 - t / 2), bottom_z + h / 2), **kw,
        )  # fmt: skip


def _box_spec(g: SceneGeometryCfg) -> mujoco.MjSpec:
    """Open box centered on its body frame in xy; walls sit on the top shelf layer."""
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="box")
    friction = (g.prop_static_friction, 0.005, 0.0001)
    _add_open_box(
        body, "", (0.0, 0.0), g.upper_box_size, g.box_bottom_z, g.box_wall_thickness, friction
    )
    return spec


def box_cfg(g: SceneGeometryCfg) -> EntityCfg:
    pos = (g.upper_box_center[0], g.upper_box_center[1], 0.0)
    return EntityCfg(init_state=EntityCfg.InitialStateCfg(pos=pos), spec_fn=lambda: _box_spec(g))


# ----------------------------------------------------------------------------- room


def _room_spec(g: SceneGeometryCfg) -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="room")
    friction = (g.prop_static_friction, 0.005, 0.0001)
    shelf = dict(type=_BOX, rgba=srgb(BLACK), friction=friction, **_PROP)
    sx, sy = g.shelf_center
    lx, ly, lz = g.shelf_layer_size
    for k, z in enumerate(g.shelf_layer_z):
        body.add_geom(name=f"shelf_layer{k}", size=(lx / 2, ly / 2, lz / 2), pos=(sx, sy, z), **shelf)
    px, py, pz = g.shelf_post_size
    for k, (dx, dy) in enumerate(g.shelf_post_offsets):
        body.add_geom(
            name=f"shelf_post{k}", size=(px / 2, py / 2, pz / 2), pos=(sx + dx, sy + dy, pz / 2), **shelf
        )
    for name, dy in (("shelf_top_front", -0.19), ("shelf_top_back", 0.19)):
        body.add_geom(name=name, size=(lx / 2, 0.012, 0.012), pos=(sx, sy + dy, 1.51), **shelf)
    if g.show_lower_box:
        _add_open_box(
            body, "lower_box_", (0.25, 0.74), (0.30, 0.24, 0.075), 0.37, g.box_wall_thickness,
            friction,
        )  # fmt: skip
    if g.show_power_box:
        body.add_geom(
            name="power_box", type=_BOX, size=(0.085, 0.055, 0.0375), pos=(1.62, 0.42, 0.039),
            rgba=srgb(WHITE), **_VISUAL,
        )  # fmt: skip

    # Carpet: one plane carrying the baked 10 x 10 patch texture (scripts/build_assets.py).
    half = g.carpet_tiles * g.carpet_tile_size / 2
    cx, cy = g.carpet_origin
    body.add_geom(
        name="carpet", type=_PLANE, size=(half, half, 0.01), pos=(cx + half, cy + half, 0.001),
        rgba=(1.0, 1.0, 1.0, 1.0), **_VISUAL,
    )  # fmt: skip

    # Floor cable: 40 thin segments along y = -0.49 + 0.13 sin(7a).
    for i in range(40):
        a, b = i / 40, (i + 1) / 40
        p = (0.2 + 1.7 * a, -0.49 + 0.13 * math.sin(a * 7), 0.006)
        q = (0.2 + 1.7 * b, -0.49 + 0.13 * math.sin(b * 7), 0.006)
        body.add_geom(
            name=f"cable{i}", type=_CYLINDER, size=(0.0015, 0, 0), fromto=(*p, *q),
            rgba=srgb(CABLE), **_VISUAL,
        )  # fmt: skip
    return spec


def room_cfg(g: SceneGeometryCfg) -> EntityCfg:
    return EntityCfg(
        spec_fn=lambda: _room_spec(g),
        textures=(TextureCfg(name="carpet", type="2d", file=str(C.SCENE_ASSETS_DIR / "carpet.png")),),
        materials=(
            MaterialCfg(
                name="carpet", texture="carpet", rgba=(CARPET_GAIN, CARPET_GAIN, CARPET_GAIN, 1.0),
                geom_names_expr=("carpet",),
            ),
        ),
    )


# ----------------------------------------------------------------------------- lights

# Isaac Lab: dome light (450, bluish) plus a key (1050) and fill (450) distant light.
# Distant lights shine along their prim's -z axis. Intensities, the dome (= ambient) and the
# carpet gain are fitted to the recorded RTX head-camera frames of 8 recorded states (episodes
# 0-3 at reach / grasp; mean |diff| 33 -> 14 / 255, no blown-out tabletop, DINOv3 cosine of the
# diffusion policy's encoder 0.944 -> 0.968). The key light casts the ray-traced shadows.
_KEY_QUAT = (0.9238795, 0.2209424, -0.2209424, 0.2209424)
_FILL_QUAT = (0.7071068, -0.4082483, 0.4082483, -0.4082483)
SCENE_LIGHTS = (
    LightCfg(
        name="key", type="directional", dir=_rotate(_KEY_QUAT, (0, 0, -1)), pos=(0, 0, 4),
        diffuse=(0.84, 0.826, 0.798), specular=(0.1, 0.1, 0.1), castshadow=True,
    ),
    LightCfg(
        name="fill", type="directional", dir=_rotate(_FILL_QUAT, (0, 0, -1)), pos=(0, 0, 4),
        diffuse=(0.2, 0.216, 0.239), specular=(0.0, 0.0, 0.0), castshadow=False,
    ),
)  # fmt: skip
# The dome light as ambient; no camera-attached light (Isaac has none).
HEADLIGHT_AMBIENT = (0.3, 0.31, 0.32)
HEADLIGHT_DIFFUSE = (0.0, 0.0, 0.0)
# Baked carpet texture is brighter than the RTX floor.
CARPET_GAIN = 0.30
