"""Scene and layout parameters of the can pick-and-place task.

Values are the resolved ``configs/experiments/can_pnp_full_task.yaml`` of
motion_token_residual_rl (``scene`` and ``task.collection`` sections, plus the
``box`` / ``resident`` / ``layout`` domain-randomization features). Positions are in
the environment frame; mjlab adds the env origin at reset.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class SceneGeometryCfg:
    # Table: 1.2 x 0.6 top, front edge at table_front_x, four legs.
    table_front_x: float = 0.56
    table_center_y: float = -0.04356482716407909
    table_height: float = 0.74
    table_size: tuple[float, float, float] = (1.2, 0.6, 0.025)
    table_leg_radius: float = 0.024
    table_leg_offsets: tuple[tuple[float, float], ...] = (
        (-0.53, -0.24), (-0.53, 0.24), (0.53, -0.24), (0.53, 0.24),
    )  # fmt: skip

    # Shelf: three black layers, four posts, two top rails.
    shelf_center: tuple[float, float] = (0.55, 0.8)
    shelf_layer_size: tuple[float, float, float] = (1.6, 0.4, 0.02)
    shelf_layer_z: tuple[float, ...] = (0.025, 0.36, 0.8)
    shelf_post_offsets: tuple[tuple[float, float], ...] = (
        (-0.79, -0.19), (-0.79, 0.19), (0.79, -0.19), (0.79, 0.19),
    )  # fmt: skip
    shelf_post_size: tuple[float, float, float] = (0.024, 0.024, 1.52)

    # Open cardboard box on the top shelf layer (the placement target).
    upper_box_center: tuple[float, float, float] = (0.25, 0.74, 0.8665)
    upper_box_size: tuple[float, float, float] = (0.255, 0.225, 0.1)
    box_wall_thickness: float = 0.004
    box_bottom_z: float = 0.81
    show_lower_box: bool = False
    show_power_box: bool = False

    # Target can on the table and the resident can already in the box.
    can_initial_pos: tuple[float, float, float] = (0.7, 0.05, 0.7925)
    can_radius: float = 0.0275
    can_height: float = 0.105
    can_mass: float = 0.62
    can_static_friction: float = 1.0
    can_dynamic_friction: float = 0.8
    # Can contacts as in the MuJoCo scene the diffusion dataset was collected in
    # (single_can_adapter20_recollect250 ``scene.xml``): torsional friction, condim 6, solref.
    can_torsional_friction: float = 0.005
    can_rolling_friction: float = 0.0005
    can_condim: int = 6
    can_solref: tuple[float, float] = (0.01, 1.0)

    # Static props: table/shelf/box friction; the floor uses the collection scene's 1.0.
    prop_static_friction: float = 0.8
    ground_static_friction: float = 1.0

    # Carpet: 10 x 10 patches of 0.5 m starting at (-1.75, -2.25).
    carpet_origin: tuple[float, float] = (-1.75, -2.25)
    carpet_tiles: int = 10
    carpet_tile_size: float = 0.5

    env_spacing: float = 3.5


@dataclass
class HeadCameraCfg:
    # Offset on torso_link in OpenGL convention (-z forward, +y up), same as MuJoCo cameras.
    pos: tuple[float, float, float] = (0.0576235, 0.01753, 0.41987)
    quat: tuple[float, float, float, float] = (
        0.6592524821011074, 0.2557071857487173, -0.2557071857487173, -0.6592524821011074,
    )  # fmt: skip
    width: int = 640
    height: int = 480
    focal_length: float = 24.0
    horizontal_aperture: float = 26.509667991878086
    # Ray-traced shadows of the shadow-casting lights (the Isaac RTX images have them).
    use_shadows: bool = True

    @property
    def fovy(self) -> float:
        """Vertical FOV of the Isaac Lab pinhole (vertical aperture = horizontal * h / w)."""
        vertical_aperture = self.horizontal_aperture * self.height / self.width
        return math.degrees(2.0 * math.atan(vertical_aperture / (2.0 * self.focal_length)))


@dataclass
class LayoutCfg:
    """Per-episode layout sampling (``tasks/can_pnp/collection_sampling.py``)."""

    table_fore_aft_offset_m: tuple[float, float] = (-0.25, 0.25)
    table_shelf_edge_gap_m: tuple[float, float] = (0.3, 0.6)
    robot_yaw_deg: tuple[float, float] = (-45.0, 45.0)
    robot_table_front_distance_m: tuple[float, float] = (0.5, 1.5)
    # As recorded in the 250 collected episodes (trajectory.json ``layout``): inset
    # 0.080-0.259 m, lateral -0.200-0.190 m. The Isaac Lab sampler's 0.15 m edge margin
    # would push every can 0.15-0.30 m deep and within +-0.15 m.
    can_lateral_m: tuple[float, float] = (-0.2, 0.2)
    can_front_inset_m: tuple[float, float] = (0.08, 0.26)
    can_table_edge_margin_m: float = 0.08
    # Measured shelf front; tabletop half-depth is 0.3.
    shelf_front_y: float = 0.598
    # Robot spawns this far to +y of the can, facing +x.
    robot_can_lateral_offset: float = 0.14
    # "collection": robot 0.5-1.5 m in front of the table and walks to the can, as collected.
    # "pregrasp": robot already standing at the table, facing +x, with the can where it is at
    # the start of the grasp in the recorded episodes (CLOSE phase, median of 250 episodes:
    # 0.40 m ahead, 0.14 m to the right of the pelvis), so the policy starts with the reach.
    robot_start: str = "collection"
    pregrasp_can_in_pelvis: tuple[float, float] = (0.40, -0.14)

    # Resident can inside the box (``resident`` DR feature).
    shelf_center_xy_low: tuple[float, float] = (0.158, 0.663)
    shelf_center_xy_high: tuple[float, float] = (0.342, 0.817)
    shelf_center_xy_nominal: tuple[float, float] = (0.33, 0.78)
    shelf_center_z: float = 0.8665
    shelf_body_hsv: tuple[tuple[float, float], ...] = ((0.0, 1.0), (0.45, 0.85), (0.3, 0.8))
    shelf_lid_hsv: tuple[tuple[float, float], ...] = ((0.0, 1.0), (0.3, 0.85), (0.35, 0.9))

    # Placement goal: the half of the box not occupied by the resident can.
    placement_corridor_split_x: float = 0.25
    placement_goal_right: tuple[float, float, float] = (0.33, 0.695, 0.8665)
    placement_goal_left: tuple[float, float, float] = (0.17, 0.695, 0.8665)

    # ``box`` DR feature: box (with resident can and goal) jitters in xy.
    box_xy_m: tuple[float, float] = (0.03, 0.02)
    strength: float = 1.0

    def can_spawn_bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Can-center support relative to the tabletop front edge (``tasks/can_pnp/interior.py``)."""
        margin = self.can_table_edge_margin_m
        if not 0 < margin < 0.3:
            raise ValueError("Can center edge margin must be in (0, 0.3) metres")
        front, lateral = self.can_front_inset_m, self.can_lateral_m
        low = (max(front[0], margin), max(lateral[0], -0.3 + margin))
        high = (min(front[1], 1.2 - margin), min(lateral[1], 0.3 - margin))
        if any(a >= b for a, b in zip(low, high)):
            raise ValueError("No nonzero-area can spawn region remains inside table")
        return low, high


@dataclass
class CanPnpSceneCfg:
    geometry: SceneGeometryCfg = field(default_factory=SceneGeometryCfg)
    camera: HeadCameraCfg = field(default_factory=HeadCameraCfg)
    layout: LayoutCfg = field(default_factory=LayoutCfg)
