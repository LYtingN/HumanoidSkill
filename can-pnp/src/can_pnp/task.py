"""Task bookkeeping of the can pick-and-place episode, shared by simulated and imagined rollouts.

Done: the can rests in the box, upright (axis within ``acos(upright_cos)`` of vertical) and the
right gripper commanded open (> ``release_opening``), all for ``success_hold_s``. The return is
a function of the final state: ``success_reward`` once done, else ``exp(-d / dist_scale)`` of the
can's distance ``d`` to the placement goal, 0 after a fall or a dropped can. A chunk of an
episode is worth the change of that potential over it (potential-based shaping), so the chunks
of an episode sum to its return minus the start's potential.

Positions only need a common frame (world, or env frame for the imagined rollouts).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from can_pnp.config import SceneGeometryCfg


@dataclass
class TaskCfg:
    success_reward: float = 10.0
    dist_scale: float = 0.1  # m
    success_hold_s: float = 0.2
    upright_cos: float = 0.866
    release_opening: float = 0.03  # m; the recorded release opens to 0.065, the grasp closes to 0
    drop_height: float = 0.2  # env ``can_dropped`` termination (can z above the env origin)


class TaskTracker:
    """Per-step task flags of ``n`` episodes; ``box_center`` / ``goal`` (n, 3) in the frame of
    the can positions, ``origin_z`` (n,) the env origin height in that frame."""

    def __init__(self, cfg: TaskCfg, geo: SceneGeometryCfg, box_center: torch.Tensor, goal: torch.Tensor,
                 origin_z: torch.Tensor, can_pos: torch.Tensor, step_dt: float):  # fmt: skip
        self.cfg, self.geo = cfg, geo
        self.box, self.goal, self.origin_z = box_center, goal, origin_z
        self.inner = torch.tensor(geo.upper_box_size[:2], device=goal.device) / 2 - geo.box_wall_thickness
        self.hold_steps = max(1, round(cfg.success_hold_s / step_dt))
        n, device = goal.shape[0], goal.device
        self.lifted, self.entered, self.placed = (torch.zeros(n, dtype=torch.bool, device=device) for _ in range(3))
        self.hold = torch.zeros(n, dtype=torch.long, device=device)
        self.final_dist = (can_pos - goal).norm(dim=-1)
        self.min_dist = self.final_dist.clone()

    def step(self, can_pos: torch.Tensor, can_quat: torch.Tensor, released: torch.Tensor, live: torch.Tensor) -> None:
        """Account the state of ``live`` episodes (can pose after the step, gripper command of it)."""
        geo = self.geo
        can_z = can_pos[:, 2] - self.origin_z
        in_box = ((can_pos[:, :2] - self.box[:, :2]).abs() < self.inner).all(-1) & (can_z < geo.box_bottom_z + geo.can_height)
        # Can axis z component of (w, x, y, z): 1 - 2(x^2 + y^2).
        upright = 1 - 2 * (can_quat[:, 1].square() + can_quat[:, 2].square()) > self.cfg.upright_cos
        self.lifted |= live & (can_z > geo.table_height + geo.can_height / 2 + 0.05)
        self.entered |= live & in_box
        self.hold = torch.where(live & in_box & upright & released, self.hold + 1, 0)
        self.placed |= self.hold >= self.hold_steps
        dist = (can_pos - self.goal).norm(dim=-1)
        self.final_dist = torch.where(live, dist, self.final_dist)
        self.min_dist = torch.where(live, torch.minimum(self.min_dist, dist), self.min_dist)

    def potential(self, failed: torch.Tensor) -> torch.Tensor:
        """Return of the episodes so far: done > failed (fall / drop / divergence) > distance."""
        shaped = torch.where(failed, 0.0, torch.exp(-self.final_dist / self.cfg.dist_scale))
        return torch.where(self.placed, self.cfg.success_reward, shaped).float()
