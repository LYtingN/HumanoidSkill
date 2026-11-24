"""Pico Student 2.5 low-level policy for one mjlab can_pnp world."""
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import onnxruntime as ort
import torch
from scipy.spatial.transform import Rotation

from can_pnp.scene.env_cfg import make_env_cfg
from hskill.embodiment import g1_aloha as C
from hskill.s0.fivepoint import FivePointPlanner
from humanoid_diffusion.keypoints import KP_BODY_NAMES
from mjlab.actuator import DcMotorActuatorCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor.builtin_sensor import ObjRef
from mjlab.sensor.raycast_sensor import RayCastSensorCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

PACKAGE = Path(__file__).resolve().parents[2] / "pico5_student25_sim2sim_20261123"
sys.path.insert(0, str(PACKAGE))
from experiments.pico_markers import PicoMarkers
from pico5.observation import body_state
from pico5.prepare import digest
from pico5.sim2sim import DEPTH_AGES, INPUTS, ONNX
from sim2sim.camera import MOUNT, OPTICAL_XYZW, official_filter


class DepthPattern:
    """The released Pico camera's 18 x 32 crop in torso coordinates."""
    def __init__(self):
        y, x = np.mgrid[:18, :32]
        u = (x + .5 - 16) * np.tan(np.deg2rad(89.51 / 2)) / 32
        v = (y + .5 - 9) * np.tan(np.deg2rad(58.29 / 2)) / 18
        rays = np.stack((u, v, np.ones_like(u)), -1).reshape(-1, 3)
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        self.optical_z = rays[:, 2].reshape(18, 32).copy()
        self.directions = Rotation.from_quat(OPTICAL_XYZW).apply(rays)

    def generate_rays(self, mj_model, device):
        directions = torch.tensor(self.directions, dtype=torch.float32, device=device)
        return directions.new_tensor(MOUNT).expand_as(directions).contiguous(), directions


def make_pico_env(args, device):
    cfg = make_env_cfg(args)
    cfg.viewer.azimuth, cfg.viewer.elevation, cfg.viewer.distance = 140., -17., 2.7
    metadata = json.loads(ONNX.with_name("deployment.json").read_text())
    control = metadata["controller"]
    if tuple(control["jointNames"]) != C.BODY_JOINT_NAMES or metadata["inputs"] != INPUTS or digest(ONNX) != metadata["onnx_sha256"]:
        raise ValueError("Unexpected Pico joint order, input layout or actor checksum")
    model = mujoco.MjModel.from_xml_path(str(PACKAGE / "assets/robot/g1_29dof.xml"))
    robot = cfg.scene.entities["robot"]
    # Use the packaged DC torque-speed curve at each 5 ms physics substep.
    motors = []
    for i, name in enumerate(control["jointNames"]):
        dof = model.joint(name).dofadr[0]
        motors.append(DcMotorActuatorCfg(
            target_names_expr=(name,), stiffness=control["kp"][i], damping=control["kd"][i],
            effort_limit=control["effortLimit"][i], saturation_effort=control["effortLimit"][i],
            velocity_limit=control["velocityLimit"][i], armature=float(model.dof_armature[dof]),
            frictionloss=float(model.dof_frictionloss[dof]), viscous_damping=float(model.dof_damping[dof]),
        ))
    gripper, = [a for a in robot.articulation.actuators if tuple(a.target_names_expr) == C.GRIPPER_JOINT_NAMES]
    robot.articulation.actuators = (*motors, gripper)
    robot.init_state.joint_pos.update(zip(control["jointNames"], control["defaultJointPosition"], strict=True))
    cfg.actions["joint_pos"].scale = dict(zip(control["jointNames"], control["actionScale"], strict=True))
    cfg.scene.sensors += (RayCastSensorCfg(name="pico_depth", frame=ObjRef("body", "torso_link", "robot"),
                                           pattern=DepthPattern(), max_distance=3., include_geom_groups=(0, 1, 2)),)
    return ManagerBasedRlEnv(cfg, device=device)


class PicoController(FivePointPlanner):
    def __init__(self, env, fm_path):
        super().__init__(env, fm_path)
        self.control = json.loads(ONNX.with_name("deployment.json").read_text())["controller"]
        if tuple(env.action_manager.get_term("joint_pos").target_names) != tuple(self.control["jointNames"]):
            raise ValueError("Unexpected can_pnp body action order")
        if not np.isclose(env.step_dt, self.control["controlDt"]):
            raise ValueError("Pico needs a 50 Hz control loop")
        self.default = np.array(self.control["defaultJointPosition"])
        model = mujoco.MjModel.from_xml_path(str(PACKAGE / "assets/robot/g1_29dof.xml"))
        data = mujoco.MjData(model)
        data.qpos[2] = .79
        data.qpos[[model.joint(n).qposadr[0] for n in C.BODY_JOINT_NAMES]] = self.default
        mujoco.mj_forward(model, data)
        self.markers = PicoMarkers(model, data, np.array([model.body(n).id for n in KP_BODY_NAMES]))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.actor = ort.InferenceSession(str(ONNX), options, providers=["CPUExecutionProvider"])
        if {v.name: list(v.shape[1:]) for v in self.actor.get_inputs()} != INPUTS:
            raise ValueError("Unexpected Pico ONNX inputs")
        self.depth = env.scene["pico_depth"]
        self.optical_z = torch.tensor(self.depth.cfg.pattern.optical_z, device=env.device, dtype=torch.float32)
        self.state_hist = np.zeros((1, 10, 64), np.float32)
        self.action_hist = np.zeros((1, 10, 29), np.float32)
        self.point_hist = np.zeros((1, 11, 5, 8), np.float32)
        self.depth_hist = np.empty((max(DEPTH_AGES) + 1, 18, 32), np.float32)
        self.ages = np.array(DEPTH_AGES)
        self.tick = 0
        self.floor = float(env.scene.env_origins[0, 2])

    def body_action(self, reset, window):
        d = self.robot.data
        quat = d.body_link_quat_w[:, self.pelvis]
        gyro = quat_apply_inverse(quat, d.body_link_ang_vel_w[:, self.pelvis])
        q, qd = d.joint_pos[0, self.body_ids], d.joint_vel[0, self.body_ids]
        state, yaw = body_state(q.cpu().numpy(), qd.cpu().numpy(), quat[0].cpu().numpy(),
                                gyro[0].cpu().numpy(), self.default)
        if reset:
            self.tick = 0
            self.state_hist.fill(0)
            self.action_hist.fill(0)
            self.start_yaw = yaw
            self.to_start = Rotation.from_euler("z", -yaw)
            self.origin = d.body_link_pos_w[0, self.pelvis].cpu().numpy().copy()
            self.origin[2] = self.floor
        pos = self.to_start.apply(self.plan_pos[0, self.offset].cpu().numpy() - self.origin)
        rot = self.to_start * Rotation.from_quat(self.plan_quat[0, self.offset].cpu().numpy(), scalar_first=True)
        point = self.markers.encode(pos, rot.as_quat(scalar_first=True))
        self.point_hist[:, :-1] = self.point_hist[:, 1:].copy()
        self.point_hist[:, -1] = point
        distance = self.depth.data.distances.reshape(18, 32)
        z = torch.where(distance >= 0, (distance * self.optical_z).clamp(max=2.5), 2.5)
        image = (official_filter(z[None, None])[0, 0] * (255 / 2.5)).round().cpu().numpy() / 255
        if reset:
            self.point_hist[:] = point
            self.depth_hist[:] = image
        self.depth_hist[self.tick % len(self.depth_hist)] = image
        feeds = dict(state=state[None], state_history=self.state_hist, action_history=self.action_hist,
                     depth_image=self.depth_hist[(self.tick - self.ages) % len(self.depth_hist)][None],
                     pico_window=self.point_hist, robot_yaw=np.array([yaw - self.start_yaw], np.float32))
        raw, = self.actor.run(["action"], feeds)
        limit = self.control["actionClip"]
        command = (raw * self.control["actionMultiplier"]).clip(-limit, limit)
        self.state_hist[:, 1:] = self.state_hist[:, :-1].copy()
        self.state_hist[:, 0] = state
        self.action_hist[:, 1:] = self.action_hist[:, :-1].copy()
        self.action_hist[:, 0] = command
        self.tick += 1
        return torch.from_numpy(command).to(q.device)
