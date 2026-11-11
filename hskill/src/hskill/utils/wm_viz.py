"""World-model preview for the live viewer: what the models expect the current plan to do.

At every replan, from the state of that moment, Phi (tracker + SuperMimic body world model)
executes the new reference for its whole horizon and the object world model predicts the target
object from the last frames (keypoints, gripper openings, object) and Phi's future keypoints with
the reference's commanded openings. The viewer then
shows, until the next replan, the predicted gripper path, the predicted object path and the
object's actual path since the replan; the prediction error at the last executed step of each
plan is averaged for the overlay.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch

from hskill.embodiment import g1_aloha as C
from hskill.executors.imagine import PhiStep, imagine_body
from hskill.interaction.model import ObjectWMEnsemble, ObjectWorldModel
from hskill.s0.controller import gripper_opening

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

    from hskill.s0.body_world_model import BodyWorldModel
    from hskill.s0.controller import SuperMimicController
    from hskill.s0.kinematics import BodyKinematics

GRIPPER_RGB = (0.1, 0.9, 0.9)  # Phi's right gripper
PREDICTED_RGB = (1.0, 0.55, 0.0)  # object world model
ACTUAL_RGB = (1.0, 1.0, 1.0)  # simulated object


class WorldModelPreview:
    def __init__(self, env: ManagerBasedRlEnv, controller: SuperMimicController, body: BodyWorldModel,
                 kin: BodyKinematics, object_wm: ObjectWorldModel | ObjectWMEnsemble,
                 object_name: str = "can", radius: float = 0.012):  # fmt: skip
        self.env, self.controller, self.body, self.kin = env, controller, body, kin
        self.object_wm, self.radius = object_wm.eval(), radius
        self.phi = PhiStep(body, controller.tracker, compile=False)
        robot = env.scene["robot"]
        self.robot, self.object = robot, env.scene[object_name]
        self.grippers = robot.find_joints(C.GRIPPER_JOINT_NAMES, preserve_order=True)[0]
        self.kp_bodies = [robot.body_names.index(b) for b in C.KEYPOINT_BODIES]
        self.origins = env.scene.env_origins
        self.pending: tuple[torch.Tensor, ...] | None = None
        self.reset()

    def reset(self) -> None:
        n = self.env.num_envs
        self.pred_object: np.ndarray | None = None  # (N, H + 1, 3) world
        self.pred_gripper: np.ndarray | None = None
        self.actual = [[] for _ in range(n)]  # object positions since each env's last replan
        self.errors: list[float] = []  # cm, at the last executed step of each plan
        self.frames: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []  # the object WM's history

    def frame(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Keypoints (N, 5, 7), gripper openings (N, 2) and object (N, 10), env frame, as
        ``hskill.sim.collect``."""
        robot, data = self.robot.data, self.object.data
        kp = torch.cat([robot.body_link_pos_w[:, self.kp_bodies] - self.origins[:, None],
                        robot.body_link_quat_w[:, self.kp_bodies]], dim=-1)  # fmt: skip
        obj = torch.cat([data.root_link_pos_w - self.origins, data.root_link_quat_w, data.root_link_lin_vel_w], dim=-1)
        return kp, gripper_opening(robot.joint_pos[:, self.grippers]), obj

    def before_act(self, obs: dict[str, torch.Tensor]) -> None:
        """The replan state: robot state, tracker observation and history, before ``act``."""
        if not self.frames:
            self.frames = [self.frame()]
        L = self.object_wm.cfg.history
        frames = [self.frames[0]] * (L - len(self.frames)) + self.frames[-L:]  # before the start: its first frame
        hist = tuple(torch.stack([f[i] for f in frames]) for i in range(3))
        self.pending = (self.body.state(), obs["policy"].clone(), *hist)

    @torch.no_grad()
    def after_act(self) -> None:
        """Predict for the envs that replanned in the ``act`` call just made."""
        fresh = (self.controller.step_in_chunk == 1).cpu().numpy()
        if self.pred_object is not None and not fresh.any():
            return
        s0, obs0, hist_kp, hist_width, hist_obj = self.pending
        plan = self.controller.plan
        rollout = imagine_body(self.phi, plan, s0, obs0, self.body.context())
        fut_kp = self.kin.keypoints(rollout.states[1:])  # (H, N, 5, 7)
        command = self.controller.plan_gripper.transpose(0, 1)  # (H, N, 2) commanded openings
        objs, _ = self.object_wm.rollout(hist_kp, hist_width, hist_obj, fut_kp, command)
        gripper = torch.cat([hist_kp[-1:, :, C.RIGHT_EEF], fut_kp[:, :, C.RIGHT_EEF]])
        origins = self.origins.cpu().numpy()[:, None]
        pred_object = objs[..., :3].transpose(0, 1).cpu().numpy() + origins
        pred_gripper = gripper[..., :3].transpose(0, 1).cpu().numpy() + origins
        if self.pred_object is None:
            self.pred_object, self.pred_gripper = pred_object, pred_gripper
            fresh[:] = True
        for i in np.flatnonzero(fresh):
            done = self.actual[i]
            if len(done) > 1:  # error of the plan just ended, at its last executed step
                self.errors.append(100 * float(np.linalg.norm(done[-1] - self.pred_object[i, len(done) - 1])))
            self.pred_object[i], self.pred_gripper[i] = pred_object[i], pred_gripper[i]
            self.actual[i] = [pred_object[i, 0]]

    def after_step(self) -> None:
        self.frames = (self.frames + [self.frame()])[-self.object_wm.cfg.history :]
        positions = self.object.data.root_link_pos_w.cpu().numpy()
        for i, path in enumerate(self.actual):
            if len(path) < self.pred_object.shape[1]:
                path.append(positions[i])

    def _add(self, scn: mujoco.MjvScene, pos, size: float, rgba) -> bool:
        if scn.ngeom >= scn.maxgeom:
            return False
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.full(3, size), pos,
                            np.eye(3).reshape(-1), np.asarray(rgba, dtype=np.float32))  # fmt: skip
        scn.ngeom += 1
        return True

    def draw(self, scn: mujoco.MjvScene) -> None:
        if self.pred_object is None:
            return
        horizon = self.pred_object.shape[1]
        for i in range(self.env.num_envs):
            for k in range(0, horizon, 2):
                fade = 0.9 - 0.6 * k / horizon
                if not (self._add(scn, self.pred_gripper[i, k], 0.4 * self.radius, (*GRIPPER_RGB, fade))
                        and self._add(scn, self.pred_object[i, k], self.radius, (*PREDICTED_RGB, fade))):  # fmt: skip
                    return
            for pos in self.actual[i][::2]:
                if not self._add(scn, pos, 0.6 * self.radius, (*ACTUAL_RGB, 0.9)):
                    return

    def legend(self) -> str:
        err = f"{np.mean(self.errors[-50:]):.1f} cm (last {min(len(self.errors), 50)} plans)" if self.errors else "-"
        return ("world model: gripper (Phi) cyan, object predicted orange, object actual white\n"
                f"object WM error at the last executed step: {err}")
