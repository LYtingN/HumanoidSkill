"""Run fivepoints_fm.pt -> MorphMimic -> G1 in the can pick-and-place scene."""
import argparse
import json
import math
from contextlib import ExitStack
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import torch
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
from mjlab.viewer import NativeMujocoViewer

from can_pnp.scene.config import CanPnpSceneCfg, HeadCameraCfg, SceneGeometryCfg
from can_pnp.scene.env_cfg import CanPnpEnvCfgArgs, make_env
from hskill.s0.fivepoint import FivePointController
from can_pnp.starts import GraspStarts, apply_grasp_start
from can_pnp.scene.layout import layout_term
from experiments.view_policy import _Overview


class FivePointTrajectories:
    # Same order as the FM output: pelvis, left/right ankle, left/right wrist.
    colors = ((1., .82, .05, 1.), (.05, .95, .9, 1.), (1., .28, .82, 1.),
              (.16, .53, 1., 1.), (1., .22, .12, 1.))

    def __init__(self, controller):
        self.controller = controller
        self.replan = -1
        self.points = None

    def debug_vis(self, visualizer):
        if self.controller.plan_pos is None:
            return
        first_draw = self.points is None
        if self.replan != self.controller.replans:
            self.replan = self.controller.replans
            self.points = self.controller.plan_pos[0].detach().cpu().numpy()
        start = self.controller.offset - 1
        remaining = self.points[start:]
        for path, color in zip(remaining.transpose(1, 0, 2), self.colors, strict=True):
            visualizer.add_sphere(path[0], .032, color)
            for a, b in zip(path[:-1], path[1:]):
                if np.linalg.norm(b - a) > 1e-5:
                    visualizer.add_cylinder(a, b, .009, color)
            visualizer.add_sphere(path[-1], .022, color)
        future_indices = getattr(getattr(self.controller, "reference", None), "future_indices", None)
        if future_indices is not None:
            for frame in self.points[future_indices]:
                for point in frame:
                    visualizer.add_sphere(point, .014, (1., 1., 1., 1.))
        if first_draw:
            print(f"Native five-point overlay drawn ({len(remaining)} predicted frames)", flush=True)


def run_native(env, controller, start_episode, limit):
    wrapped = RslRlVecEnvWrapper(env)
    env.manager_visualizers["five_point_trajectories"] = FivePointTrajectories(controller)
    step = 0

    def policy(_obs):
        nonlocal step
        new_episode = env.episode_length_buf[0].item() == 0
        if new_episode:
            start_episode()
            step = 0
        if step in (0, 50, 150, 750):
            pelvis = env.scene["robot"].data.root_link_pos_w[0]
            front = layout_term(env).layout["table_front_x"][0] + env.scene.env_origins[0, 0]
            print(f"t={step * env.step_dt:.2f}s  pelvis_to_table_front={float(front - pelvis[0]):.3f}m"
                  f"  pelvis_height={float(pelvis[2]):.3f}m", flush=True)
        action = controller.act(reset=new_episode)
        if step in (0, 48, 152, 752):
            plan = controller.plan_pos[0]
            dx = (plan[-1, :, 0] - plan[0, :, 0]).cpu().tolist()
            print(f"FM plan t={step * env.step_dt:.2f}s, "
                  f"predicted forward displacement over {len(plan)} frames (metres): "
                  + " ".join(f"{value:+.3f}" for value in dx), flush=True)
        step += 1
        return action

    print(f"NativeMujocoViewer  device={env.device}  terminations={env.termination_manager.active_terms}",
          flush=True)
    NativeMujocoViewer(wrapped, policy).run(num_steps=limit)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fm", type=Path, default=Path("downloads/5points_FM/fivepoints_fm.pt"))
    parser.add_argument("--flashsac", type=Path, help="Use the trained deterministic FM noise actor")
    backend = parser.add_mutually_exclusive_group()
    backend.add_argument("--official-human-five", dest="tracker", action="store_const", const=None,
                         help="Use the official v0.1 human/five ONNX tracker (default)")
    backend.add_argument("--tracker", type=Path, help="Use a G1 keypoint tracker checkpoint")
    parser.add_argument("--sample-future", action=argparse.BooleanOptionalAction, default=None,
                        help="Hold ten evenly spaced future frames until the next FM replan "
                             "(human/five default); --no-sample-future uses a sliding window")
    parser.add_argument("--episode", type=int, default=3)
    parser.add_argument("--phase", choices=["WALK", "APPROACH", "REACH", "CLOSE", "LIFT", "TURN_WALK", "PLACE_REACH", "LOWER", "RELEASE"],
                        help="WALK uses the scene layout; APPROACH uses the recorded episode start")
    parser.add_argument("--seconds", type=float, help="Optional run duration in simulation seconds (recording default: training cap for FlashSAC, otherwise 14 s)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--record", type=Path, help="Save one rollout to this MP4 and exit")
    parser.add_argument("--measured-gripper", action=argparse.BooleanOptionalAction, default=True,
                        help="Start held phases from the measured gripper opening, as FlashSAC training does")
    parser.add_argument("--gripper-hysteresis", type=float, nargs="*", default=[0.02, 0.05], metavar="M",
                        help="Right gripper closes below the first FM width, reopens above the second (m), as in "
                             "training; no values: raw FM widths. A --flashsac checkpoint's own setting wins")
    args = parser.parse_args()
    if args.flashsac and (args.tracker or args.sample_future is False):
        parser.error("--flashsac requires the official v0.1 tracker with sampled future frames")
    if args.sample_future is not None and args.tracker is not None:
        parser.error("Future-frame sampling options apply only to the official human/five tracker")
    if args.gripper_hysteresis and not (len(args.gripper_hysteresis) == 2
                                        and 0 <= args.gripper_hysteresis[0] <= args.gripper_hysteresis[1] <= 0.065):
        parser.error("--gripper-hysteresis takes CLOSE REOPEN widths in [0, 0.065] m")
    if args.seconds is not None and (not math.isfinite(args.seconds) or args.seconds <= 0):
        parser.error("--seconds must be positive")
    phase = args.phase or ("APPROACH" if args.flashsac else "CLOSE" if args.record else "WALK")
    seconds = args.seconds
    torch.manual_seed(args.seed)
    geometry = SceneGeometryCfg(can_mass=.1)
    scene = CanPnpSceneCfg(geometry=geometry, camera=HeadCameraCfg(width=320, height=240))
    env = make_env(CanPnpEnvCfgArgs(num_envs=1, randomize=False, seed=args.seed,
                                    enable_time_out=False, scene=scene), device="cuda:0")
    with ExitStack() as stack:
        stack.callback(env.close)
        can_bodies = env.scene["can"].indexing.body_ids.cpu().numpy()
        can_mass = env.sim.mj_model.body_mass[can_bodies].sum()
        print(f"Target can mass (compiled model): {can_mass:.6f} kg", flush=True)
        if args.tracker is None:
            from hskill.s0.official_five import OfficialFiveController
            controller = OfficialFiveController(env, args.fm, sample_future=args.sample_future is not False)
            tracker_label = "v0.1 human/five"
        else:
            controller = FivePointController(env, args.fm, args.tracker)
            tracker_label = args.tracker.stem
        controller.gripper_hysteresis = tuple(args.gripper_hysteresis) if args.gripper_hysteresis else None
        if args.flashsac:
            training_seconds = controller.load_flashsac_actor(args.flashsac)
        if args.record and seconds is None:
            seconds = training_seconds if args.flashsac else 14.
        starts = GraspStarts(phase=phase, measured_gripper=args.measured_gripper) if phase != "WALK" else None
        ids = torch.tensor([0], device=env.device)
        if starts is not None:
            episode_row = torch.tensor([starts.episode_ids.index(args.episode)])
            print(f"Recorded start: episode={args.episode} phase={phase} frame={starts.frame[episode_row].item()}", flush=True)

        def start_episode():
            if starts is not None:
                apply_grasp_start(env, ids, starts, episode_row)

        limit = max(1, round(seconds / env.step_dt)) if seconds is not None else None
        if args.record is None:
            run_native(env, controller, start_episode, limit)
            return

        args.record.parent.mkdir(parents=True, exist_ok=True)
        overview = _Overview(env, 0)
        stack.callback(overview.renderer.close)
        writer = imageio_ffmpeg.write_frames(str(args.record), (960, 480), fps=round(1 / env.step_dt),
                                             codec="libx264", output_params=["-movflags", "+faststart"])
        writer.send(None)
        stack.callback(writer.close)
        env.reset()
        start_episode()
        start_height = env.scene["can"].data.root_link_pos_w[0, 2].item()
        max_height = start_height
        replans = controller.replans
        inside = False
        for step in range(limit):
            action = controller.act(reset=step == 0)
            _, _, terminated, truncated, _ = env.step(action)
            done = bool((terminated | truncated).item())
            if done:  # mjlab has already reset the world.
                inside = False
                break
            xyz = env.scene["can"].data.root_link_pos_w[0]
            max_height = max(max_height, xyz[2].item())
            half_box = xyz.new_tensor(geometry.upper_box_size[:2]) / 2 - geometry.box_wall_thickness
            inside = bool(((xyz[:2] - layout_term(env).box_center_w[0, :2]).abs() < half_box).all()
                          and geometry.box_bottom_z <= xyz[2] <= geometry.box_bottom_z + geometry.can_height)
            head = env.scene["head_camera"].data.rgb.cpu().numpy()
            panel = np.zeros((480, 960, 3), dtype=np.uint8)
            panel[:, :640] = overview.frame()
            panel[120:360, 640:] = head[0]
            writer.send(panel)
        report = dict(tracker=tracker_label, episode=args.episode if starts is not None else None, phase=phase, seed=args.seed, steps=step + 1,
                      replans=controller.replans - replans, max_can_rise_m=round(max_height-start_height, 4),
                      can_center_in_box_at_end=inside, terminated=done)
        if done:
            report["termination_terms"] = [name for name in env.termination_manager.active_terms
                                           if env.termination_manager.get_term(name).item()]
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
