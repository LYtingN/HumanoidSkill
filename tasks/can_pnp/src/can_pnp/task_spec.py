"""Task bookkeeping of the can pick-and-place episode, shared by simulated and imagined rollouts.

Done: the can rests in the box (slower than ``rest_speed``) for ``success_hold_s``, whatever its
pose and the gripper; with ``upright_release`` (the earlier, stricter definition) it must also be
upright (axis within ``acos(upright_cos)`` of vertical) and the right gripper commanded open
(> ``release_opening``). About half of the cans lowered into the box end up lying on their side in it
with the gripper still closed, which the stricter definition counted as neither done nor failed.

Toppled: outside the box, the can not held and its axis tilted past ``acos(topple_cos)`` for
``success_hold_s``. The shaping alone would still reward a can knocked over towards the box (its
centre moves by up to half its height), so a toppled can fails the episode like a dropped one.

Reward: a terminal reward and a potential-based shaping, kept apart,

    r_k    = R_term(s_{k+1}) + gamma * phi(s_{k+1}) - phi(s_k)
    R_term = success_reward once done, 0 after a fall, a dropped or a toppled can; all end the episode,
             and phi = 0 at an episode's end
    phi(s) = sum_i w_i * exp(-d / scale_i)   (``dist_weights``, ``dist_scales``; d the can's
             distance to the placement goal)

A failure costs no explicit penalty: the episode ends, and with it every reward still to come. A
time-out does not end the episode (a critic bootstraps it). The shaping of an episode that ends
telescopes to -phi(s_0), whatever the policy did, so only the terminal reward and, at a time-out,
phi of the last state tell episodes from the same start apart. Undiscounted (gamma = 1), the
rewards of an episode or of a chunk sum to :meth:`TaskTracker.potential` at its end minus phi at
its start: success_reward once done, 0 after a failure, else phi. Two scales keep the shaping
informative far from the goal (the 0.5 m term) and precise near it (the 0.1 m term).

Positions only need a common frame (world, or env frame for the imagined rollouts).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from can_pnp.scene.config import SceneGeometryCfg


@dataclass
class TaskCfg:
    success_reward: float = 10.0
    dist_scales: tuple[float, ...] = (0.1, 0.5)  # m, the shaping's exponential length scales
    dist_weights: tuple[float, ...] = (0.5, 0.5)  # their weights (phi = 1 at the goal)
    success_hold_s: float = 0.2
    upright_cos: float = 0.866
    release_opening: float = 0.03  # m; the recorded release opens to 0.065, the grasp closes to 0
    drop_height: float = 0.2  # env ``can_dropped`` termination (can z above the env origin)
    topple_cos: float = 0.5  # a can not held with its axis tilted past 60 degrees has toppled
    rest_speed: float = 0.05  # m/s: a can slower than this rests
    upright_release: bool = False  # done also needs the can upright and the gripper open (stricter)


def shaping_potential(dist: torch.Tensor, cfg: TaskCfg) -> torch.Tensor:
    """phi of the can's distance to the placement goal (m): the weighted exponentials, 1 at the goal."""
    return sum(w * torch.exp(-dist / s) for s, w in zip(cfg.dist_scales, cfg.dist_weights, strict=True))


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
        self.lifted, self.entered, self.placed, self.toppled = (torch.zeros(n, dtype=torch.bool, device=device)
                                                                for _ in range(4))  # fmt: skip
        self.hold = torch.zeros(n, dtype=torch.long, device=device)
        self.tilted = torch.zeros(n, dtype=torch.long, device=device)
        self.final_dist = (can_pos - goal).norm(dim=-1)
        self.min_dist = self.final_dist.clone()
        self.last_pos, self.dt = can_pos.clone(), step_dt

    def step(self, can_pos: torch.Tensor, can_quat: torch.Tensor, released: torch.Tensor, live: torch.Tensor,
             held: torch.Tensor | None = None) -> None:  # fmt: skip
        """Account the state of ``live`` episodes (can pose after the step, gripper command of it,
        whether the hand holds the can: None counts it as not held)."""
        geo = self.geo
        can_z = can_pos[:, 2] - self.origin_z
        in_box = ((can_pos[:, :2] - self.box[:, :2]).abs() < self.inner).all(-1) & (can_z < geo.box_bottom_z + geo.can_height)
        # Can axis z component of (w, x, y, z): 1 - 2(x^2 + y^2).
        axis_z = 1 - 2 * (can_quat[:, 1].square() + can_quat[:, 2].square())
        upright = axis_z > self.cfg.upright_cos
        rest = (can_pos - self.last_pos).norm(dim=-1) < self.cfg.rest_speed * self.dt
        self.last_pos = torch.where(live[:, None], can_pos, self.last_pos)
        loose = live & ~in_box & (axis_z < self.cfg.topple_cos) & (True if held is None else ~held)
        self.tilted = torch.where(loose, self.tilted + 1, 0)
        self.toppled |= self.tilted >= self.hold_steps
        self.lifted |= live & (can_z > geo.table_height + geo.can_height / 2 + 0.05)
        self.entered |= live & in_box
        settled = live & in_box & rest & ((upright & released) if self.cfg.upright_release else True)
        self.hold = torch.where(settled, self.hold + 1, 0)
        self.placed |= self.hold >= self.hold_steps
        dist = (can_pos - self.goal).norm(dim=-1)
        self.final_dist = torch.where(live, dist, self.final_dist)
        self.min_dist = torch.where(live, torch.minimum(self.min_dist, dist), self.min_dist)

    def shaping(self) -> torch.Tensor:
        """phi of the can's latest live position (at construction: of the start)."""
        return shaping_potential(self.final_dist, self.cfg)

    def terminal_reward(self) -> torch.Tensor:
        """R_term of the episodes so far: success_reward once done, else 0 (a failure as well)."""
        return torch.where(self.placed, self.cfg.success_reward, 0.0).float()

    def potential(self, failed: torch.Tensor) -> torch.Tensor:
        """The undiscounted rewards so far plus phi of the start: R_term of an ended episode (done or
        failed: fall / drop / divergence, or the can toppled; phi = 0 there), else phi of the can's
        latest position."""
        live = torch.where(failed | self.toppled, 0.0, self.shaping())
        return torch.where(self.placed, self.terminal_reward(), live).float()
