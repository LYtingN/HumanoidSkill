"""Per-episode scene layout (port of ``tasks/can_pnp/{collection_sampling,interior,reset}.py``
and the scene part of ``tasks/can_pnp/randomization.py``).

On every reset, for each env:

1. sample table offset / table-shelf gap / robot distance + yaw / can inset + lateral;
2. place the table, the target can on it, and the robot facing +x in front of the table;
3. place the resident can in the box (uniform in ``shelf_center_xy_*`` when randomized),
   pick the placement goal in the other half of the box, and recolor the resident can;
4. jitter the box together with the resident can and the goal (``box`` DR feature).

With ``randomize=False`` every quantity takes its nominal value, as the evaluation path
(``domain_randomization.eval_enabled: false``) does.

Isaac Lab's ``reset.py`` also draws the resident xy and colors from a per-layout numpy
seed, but ``TaskRandomization.reset_geometry`` / ``reset_parameters`` then overwrite both
with fresh uniform draws; only that effective behavior is reproduced here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.manager_base import ManagerTermBase

from can_pnp.config import CanPnpSceneCfg
from can_pnp.entities import CAN_BODY, CAN_LID

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def _uniform(n: int, low, high, device) -> torch.Tensor:
    low = torch.as_tensor(low, dtype=torch.float32, device=device)
    high = torch.as_tensor(high, dtype=torch.float32, device=device)
    return low + torch.rand((n, *low.shape), device=device) * (high - low)


def _hsv_to_rgb(hsv: torch.Tensor) -> torch.Tensor:
    """Vectorized ``colorsys.hsv_to_rgb`` for (..., 3) tensors in [0, 1]."""
    h, s, v = hsv.unbind(-1)
    i = torch.floor(h * 6.0)
    f = h * 6.0 - i
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    i = i.long() % 6
    table = torch.stack(
        (
            torch.stack((v, t, p), -1), torch.stack((q, v, p), -1), torch.stack((p, v, t), -1),
            torch.stack((p, q, v), -1), torch.stack((t, p, v), -1), torch.stack((v, p, q), -1),
        ),
        dim=-2,
    )  # fmt: skip
    return table.gather(-2, i[..., None, None].expand(*i.shape, 1, 3)).squeeze(-2)


def _srgb(linear: torch.Tensor) -> torch.Tensor:
    return torch.where(
        linear <= 0.0031308, 12.92 * linear, 1.055 * linear.clamp_min(0) ** (1 / 2.4) - 0.055
    )


def sample_layout(cfg: CanPnpSceneCfg, n: int, randomize: bool, device) -> dict[str, torch.Tensor]:
    """Batched ``collection_sampling.sample_layout`` (env frame, metres / degrees)."""
    lay, geo = cfg.layout, cfg.geometry

    def draw(key: str) -> torch.Tensor:
        low, high = getattr(lay, key)
        if randomize:
            return _uniform(n, low, high, device)
        return torch.full((n,), (low + high) / 2, device=device)

    zeros = torch.zeros(n, device=device)
    fore = draw("table_fore_aft_offset_m") if randomize else zeros
    if randomize:
        gap = draw("table_shelf_edge_gap_m")
    else:
        gap = torch.full((n,), lay.shelf_front_y - 0.3 - geo.table_center_y, device=device)
    front = draw("robot_table_front_distance_m")
    yaw = draw("robot_yaw_deg") if randomize else zeros
    (inset_lo, lateral_lo), (inset_hi, lateral_hi) = lay.can_spawn_bounds()
    if randomize:
        inset = _uniform(n, inset_lo, inset_hi, device)
        lateral = _uniform(n, lateral_lo, lateral_hi, device)
    else:
        inset = torch.full((n,), (inset_lo + inset_hi) / 2, device=device)
        lateral = zeros
    table_front_x = geo.table_front_x + fore
    table_center_y = lay.shelf_front_y - 0.3 - gap if randomize else zeros + geo.table_center_y
    can_xyz = torch.stack(
        (table_front_x + inset, table_center_y + lateral, zeros + geo.can_initial_pos[2]), -1
    )
    robot_xy = torch.stack(
        (table_front_x - front, can_xyz[:, 1] + lay.robot_can_lateral_offset), -1
    )
    if lay.robot_start == "pregrasp":
        robot_xy = can_xyz[:, :2] - torch.tensor(lay.pregrasp_can_in_pelvis, device=device)
        front = table_front_x - robot_xy[:, 0]
        yaw = zeros
    elif lay.robot_start != "collection":
        raise ValueError(f"robot_start must be 'collection' or 'pregrasp', got {lay.robot_start!r}")
    return dict(
        table_front_x=table_front_x,
        table_center_y=table_center_y,
        table_fore_aft_offset_m=fore,
        table_shelf_edge_gap_m=gap,
        robot_table_front_distance_m=front,
        robot_yaw_deg=yaw,
        robot_xy=robot_xy,
        can_xyz=can_xyz,
        can_front_inset_m=inset,
        can_lateral_m=lateral,
    )


class reset_can_pnp_layout(ManagerTermBase):
    """Reset event that lays out every entity of the scene.

    Replaces mjlab's ``reset_scene_to_default``. The sampled layout stays readable on the
    term (``layout``, ``goal_pos_w``, ``box_delta``) for observations and rewards.
    """

    model_fields = ("geom_rgba",)

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        self.scene_cfg: CanPnpSceneCfg = cfg.params["scene_cfg"]
        n, device = env.num_envs, env.device
        self.robot = env.scene["robot"]
        self.can = env.scene["can"]
        self.shelf_can = env.scene["shelf_can"]
        self.table = env.scene["table"]
        self.box = env.scene["box"]
        self.room = env.scene["room"]

        self.layout: dict[str, torch.Tensor] = {
            k: v.repeat(n, *([1] * (v.ndim - 1)))
            for k, v in sample_layout(self.scene_cfg, 1, False, device).items()
        }
        self.goal_pos_w = torch.zeros(n, 3, device=device)
        self.box_center_w = torch.zeros(n, 3, device=device)
        self.box_delta = torch.zeros(n, 3, device=device)
        self.shelf_colors = torch.zeros(n, 2, 3, device=device)  # linear (body, lid)

        geom_ids = self.shelf_can.indexing.geom_ids
        names = self.shelf_can.geom_names
        self._body_geom = geom_ids[names.index("body_visual")]
        self._lid_geom = geom_ids[names.index("lid_visual")]
        self._nominal_colors = torch.tensor((CAN_BODY, CAN_LID), device=device)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | None,
        scene_cfg: CanPnpSceneCfg,
        randomize: bool = True,
    ) -> None:
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device)
        n, device = len(env_ids), env.device
        lay, geo = scene_cfg.layout, scene_cfg.geometry
        origin = env.scene.env_origins[env_ids]
        identity = torch.tensor((1.0, 0.0, 0.0, 0.0), device=device).repeat(n, 1)

        layout = sample_layout(scene_cfg, n, randomize, device)
        for key, value in layout.items():
            self.layout[key][env_ids] = value

        # Robot: default joints, facing +x rotated by the sampled yaw.
        half_yaw = torch.deg2rad(layout["robot_yaw_deg"]) / 2
        root = torch.zeros(n, 13, device=device)
        root[:, :2] = layout["robot_xy"]
        root[:, 2] = self.robot.data.default_root_state[env_ids, 2]
        root[:, :3] += origin
        root[:, 3] = torch.cos(half_yaw)
        root[:, 6] = torch.sin(half_yaw)
        self.robot.write_root_state_to_sim(root, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(
            self.robot.data.default_joint_pos[env_ids],
            self.robot.data.default_joint_vel[env_ids],
            env_ids=env_ids,
        )

        # Table and the target can on it.
        table_pose = torch.cat((torch.zeros(n, 3, device=device), identity), -1)
        table_pose[:, 0] = layout["table_front_x"] + geo.table_size[0] / 2
        table_pose[:, 1] = layout["table_center_y"]
        table_pose[:, :3] += origin
        self.table.write_mocap_pose_to_sim(table_pose, env_ids=env_ids)
        can = torch.cat((layout["can_xyz"] + origin, identity, torch.zeros(n, 6, device=device)), -1)
        self.can.write_root_state_to_sim(can, env_ids=env_ids)

        # Resident can and placement goal (env frame), then the box jitter moves all three.
        if randomize:
            shelf_xy = _uniform(n, lay.shelf_center_xy_low, lay.shelf_center_xy_high, device)
        else:
            shelf_xy = torch.tensor(lay.shelf_center_xy_nominal, device=device).repeat(n, 1)
        shelf = torch.cat((shelf_xy, torch.full((n, 1), lay.shelf_center_z, device=device)), -1)
        right = torch.tensor(lay.placement_goal_right, device=device)
        left = torch.tensor(lay.placement_goal_left, device=device)
        goal = torch.where((shelf_xy[:, 0] < lay.placement_corridor_split_x)[:, None], right, left)
        delta = torch.zeros(n, 3, device=device)
        if randomize:
            box_xy = torch.tensor(lay.box_xy_m, device=device) * lay.strength
            delta[:, :2] = _uniform(n, -box_xy, box_xy, device)
        self.box_delta[env_ids] = delta
        shelf += delta
        goal += delta
        self.goal_pos_w[env_ids] = goal + origin

        shelf_state = torch.cat((shelf + origin, identity, torch.zeros(n, 6, device=device)), -1)
        self.shelf_can.write_root_state_to_sim(shelf_state, env_ids=env_ids)
        box_pose = torch.cat((torch.zeros(n, 3, device=device), identity), -1)
        box_pose[:, :2] = torch.tensor(geo.upper_box_center[:2], device=device) + delta[:, :2]
        box_pose[:, :3] += origin
        self.box.write_mocap_pose_to_sim(box_pose, env_ids=env_ids)
        self.box_center_w[env_ids] = box_pose[:, :3]
        self.box_center_w[env_ids, 2] = geo.upper_box_center[2]
        self.room.write_mocap_pose_to_sim(torch.cat((origin, identity), -1), env_ids=env_ids)

        # Resident can colors: HSV ranges are applied to Isaac's linear diffuseColor.
        if randomize:
            body = _hsv_to_rgb(_uniform(n, *zip(*lay.shelf_body_hsv), device))
            lid = _hsv_to_rgb(_uniform(n, *zip(*lay.shelf_lid_hsv), device))
            colors = torch.stack((body, lid), 1)
        else:
            colors = self._nominal_colors.repeat(n, 1, 1)
        self.shelf_colors[env_ids] = colors
        rgba = env.sim.model.geom_rgba
        rgba[env_ids, self._body_geom, :3] = _srgb(colors[:, 0])
        rgba[env_ids, self._lid_geom, :3] = _srgb(colors[:, 1])


def layout_term(env: ManagerBasedRlEnv, name: str = "reset_layout") -> reset_can_pnp_layout:
    """The live :class:`reset_can_pnp_layout` instance of an environment."""
    return env.event_manager.get_term_cfg(name).func  # type: ignore[return-value]

