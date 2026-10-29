"""Future trajectories of five keypoints in the current reference plan, for the live viewer.

The reference (diffusion policy / dataset) gives joint positions and the root orientation for
the next ``horizon`` frames but no root translation. Forward kinematics of each frame on a
robot-only model gives the keypoints relative to the root; the root translation is then
integrated by keeping the lower (stance) foot fixed in the world between consecutive frames,
starting at the measured pelvis position when the plan was made.

For display the paths are smoothed along the horizon (moving average, start pinned), eased
towards each new plan instead of jumping at replans, and drawn as Catmull-Rom tubes that fade
towards the end of the horizon.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import numpy as np

from can_pnp import constants as C
from can_pnp.entities import _robot_spec

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

    from can_pnp.pipeline import SuperMimicController

# name: (body, rgb)
KEYPOINTS = {
    "pelvis": ("pelvis", (1.0, 0.85, 0.1)),
    "left hand": ("left_wrist_yaw_link", (0.2, 0.6, 1.0)),
    "right hand": ("right_wrist_yaw_link", (1.0, 0.25, 0.25)),
    "left foot": ("left_ankle_roll_link", (0.3, 0.9, 0.9)),
    "right foot": ("right_ankle_roll_link", (1.0, 0.5, 0.9)),
}
FEET = (3, 4)  # indices into KEYPOINTS


def _smooth(path: np.ndarray, window: int) -> np.ndarray:
    """Moving average along axis 0 (edge padded), keeping the first point."""
    if window <= 1 or len(path) < 3:
        return path
    pad = window // 2
    padded = np.concatenate([path[:1].repeat(pad, 0), path, path[-1:].repeat(pad, 0)])
    kernel = np.ones(window) / window
    out = np.stack([np.convolve(padded[:, j], kernel, mode="valid") for j in range(path.shape[1])], -1)
    out += path[0] - out[0]  # pinned start, shifted smoothly
    return out


def _catmull_rom(points: np.ndarray, samples: int) -> np.ndarray:
    """Uniform Catmull-Rom spline through ``points``, ``samples`` points per segment."""
    if len(points) < 2:
        return points
    p = np.concatenate([points[:1], points, points[-1:]])
    t = np.linspace(0.0, 1.0, samples, endpoint=False)[:, None]
    t2, t3 = t * t, t * t * t
    out = []
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
        out.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))  # fmt: skip
    out.append(points[-1:])
    return np.concatenate(out)


class PlanKeypoints:
    def __init__(
        self,
        env: ManagerBasedRlEnv,
        controller: SuperMimicController,
        smooth_window: int = 5,
        ease: float = 0.3,
        samples: int = 3,
        radius: float = 0.004,
    ):
        self.env, self.controller = env, controller
        self.smooth_window, self.ease, self.samples, self.radius = smooth_window, ease, samples, radius
        self.model = _robot_spec().compile()
        self.data = mujoco.MjData(self.model)
        self.qadr = np.array([self.model.joint(n).qposadr[0] for n in C.BODY_JOINT_NAMES])
        self.bodies = [self.model.body(body).id for body, _ in KEYPOINTS.values()]
        self.rgb = [np.array(rgb, dtype=np.float32) for _, rgb in KEYPOINTS.values()]
        self.points: np.ndarray | None = None  # (num_envs, horizon, 5, 3) plan, world
        self.display: np.ndarray | None = None  # (num_envs, horizon, 5, 3) eased, from now on
        self.remaining: np.ndarray | None = None  # plan frames left per env

    def _relative(self, joint_pos: np.ndarray, root_w: np.ndarray) -> np.ndarray:
        """(H, 5, 3) keypoints relative to the root origin, world-aligned axes."""
        out = np.zeros((len(joint_pos), len(self.bodies), 3))
        quat = np.zeros(4)
        for k in range(len(joint_pos)):
            self.data.qpos[:] = 0.0
            mujoco.mju_mat2Quat(quat, root_w[k].reshape(-1))
            self.data.qpos[3:7] = quat
            self.data.qpos[self.qadr] = joint_pos[k]
            mujoco.mj_kinematics(self.model, self.data)
            out[k] = self.data.xpos[self.bodies]
        return out

    def update(self) -> None:
        """Recompute envs that replanned in the last ``controller.act`` call, then ease the
        displayed paths one control step towards the current plans."""
        ctrl = self.controller
        if ctrl.plan is None:
            return
        n, horizon = ctrl.plan.joint_pos.shape[:2]
        if self.points is None:
            self.points = np.zeros((n, horizon, len(self.bodies), 3))
            fresh = np.ones(n, dtype=bool)
        else:
            fresh = (ctrl.step_in_chunk == 1).cpu().numpy()
        if fresh.any():
            joint_pos = ctrl.plan.joint_pos.cpu().numpy()
            root_w = ctrl.plan.root_w.cpu().numpy()
            pelvis = ctrl.robot.data.root_link_pos_w.cpu().numpy()
            for i in np.flatnonzero(fresh):
                rel = self._relative(joint_pos[i], root_w[i])
                root = np.zeros((horizon, 3))
                root[0] = pelvis[i]
                for k in range(1, horizon):
                    stance = min(FEET, key=lambda f: rel[k - 1, f, 2])
                    root[k] = root[k - 1] + rel[k - 1, stance] - rel[k, stance]
                path = root[:, None] + rel
                self.points[i] = _smooth(path.reshape(horizon, -1), self.smooth_window).reshape(path.shape)

        # Target from now on: the plan from the frame being tracked next, held at its end.
        start = ctrl.step_in_chunk.cpu().numpy().clip(0, horizon - 1)
        target = np.stack([np.concatenate([p[s:], p[-1:].repeat(s, 0)]) for p, s in zip(self.points, start)])
        self.remaining = horizon - start
        if self.display is None:
            self.display = target.copy()
            return
        shifted = np.concatenate([self.display[:, 1:], self.display[:, -1:]], 1)
        self.display = shifted + self.ease * (target - shifted)
        # The current position is not a prediction: follow it exactly.
        self.display[:, 0] = target[:, 0]

    def draw(self, scn: mujoco.MjvScene) -> None:
        """Add each env's eased plan to ``scn`` as tubes fading towards the horizon end."""
        if self.display is None:
            return
        eye = np.eye(3).reshape(-1)
        for i, path in enumerate(self.display):
            for b, rgb in enumerate(self.rgb):
                curve = _catmull_rom(path[: max(int(self.remaining[i]), 2), b], self.samples)
                if scn.ngeom + len(curve) + 1 > scn.maxgeom:
                    return
                for k in range(1, len(curve)):
                    alpha = 0.9 - 0.65 * k / len(curve)
                    geom = scn.geoms[scn.ngeom]
                    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), eye,
                                        np.append(rgb, alpha).astype(np.float32))  # fmt: skip
                    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, self.radius, curve[k - 1], curve[k])
                    scn.ngeom += 1
                mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.full(3, 2.5 * self.radius),
                                    curve[-1], eye, np.append(rgb, 0.6).astype(np.float32))  # fmt: skip
                scn.ngeom += 1

    @staticmethod
    def legend() -> str:
        names = {"pelvis": "yellow", "left hand": "blue", "right hand": "red",
                 "left foot": "cyan", "right foot": "pink"}  # fmt: skip
        return "plan: " + "  ".join(f"{k} {v}" for k, v in names.items())
