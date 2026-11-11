"""The same flow-matching policy executed by SuperMimic and by the whole-body ideal executor E*.

Envs come in pairs (2p, 2p + 1) from the same start (``collection``: the nominal layout, the robot
walks to the table; ``grasp``: a recorded grasp start): env 2p runs the frozen SuperMimic tracker,
env 2p + 1 the ideal executor (``hskill.executors.ideal_body``: all body joints on stiff servos, the
root held on the reference root). Both replan with the same policy (the diffusion policy, steered
by a DSRL noise policy or not); a pair that replans together shares the noise, so the first plan
is identical and the plans differ only through what each executor made of the earlier ones.

``switch_at`` makes the groups four envs: SuperMimic throughout, E* throughout, E* until
``switch_at`` then SuperMimic, and SuperMimic until ``switch_at`` then E*. Whether a policy stuck
under SuperMimic after the approach is stuck because of how it got there (the walk's tracking
errors) or because of what SuperMimic does at the table shows in the two switched envs.

Per env and step: the right hand's and the root's distance to the reference (the reference frame
the step servos to, the root by the plan's leg odometry from the replan root, as the ideal branch
of the imagination), the right hand's and the reference's right hand's distance to the can (did the
policy ask to reach it, did the body get there), the can's height, and the episode's outcome
(fallen / dropped / touched: the can moved > 1 cm / lifted / placed). Writes side-by-side overview videos of the first ``video_pairs`` pairs, ``compare.png``
and ``summary.json``.

    uv run python -m experiments.compare_executors --tracker-checkpoint ../supermimic_0100000.pt \\
      --diffusion-checkpoint ../downloads/job85845_20261107/can_pnp_supermimic_flow_matching_best_ema.pt \\
      --start collection --noise-mode base   # the policy without fine-tuning, from the beginning
"""

from __future__ import annotations

import dataclasses
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from experiments.view_policy import _Overview  # first: selects EGL for the offscreen renders

import numpy as np
import torch
import tyro
from can_pnp.scene.config import CanPnpSceneCfg, SceneGeometryCfg
from can_pnp.scene.env_cfg import CanPnpEnvCfgArgs, make_env
from can_pnp.scene.layout import layout_term
from can_pnp.starts import GraspStarts, apply_grasp_start
from hskill.embodiment.g1_aloha import RIGHT_EEF
from hskill.executors.arm_servo import ArmServo
from hskill.executors.ideal_body import RootPin, ideal_executor
from hskill.s0.body_world_model import BodyWorldModel
from hskill.s0.controller import SuperMimicController
from hskill.s0.kinematics import BodyKinematics
from hskill.s0.supermimic import load_tracking_policy
from hskill.s1.noise_policy import NoisePolicy, SteeredDiffusionSource
from hskill.s1.sources import load_diffusion_policy, set_sampling
from hskill.utils.plan_viz import PlanKeypoints

MODES = ("SuperMimic", "ideal E*", "E* -> SuperMimic", "SuperMimic -> E*")
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")  # categorical slots 1-4


@dataclass
class Args:
    tracker_checkpoint: Path
    diffusion_checkpoint: Path
    # A train_v3 / train_dsrl checkpoint; None: the unsteered policy (noise N(0, I)).
    noise_checkpoint: Path | None = None
    noise_mode: Literal["mean", "sample", "base"] = "base"
    diffusion_repo: Path = Path(__file__).resolve().parents[2] / "Diffusion" / "Humanoid-Diffusion"
    execute_steps: int = 16  # replan interval, as train_v3
    bf16: bool = True
    pairs: int = 4  # groups
    start: Literal["collection", "grasp"] = "collection"
    grasp_phase: Literal["REACH", "CLOSE", "LIFT", "TURN_WALK", "PLACE_REACH", "LOWER", "RELEASE"] = "CLOSE"
    grasp_episodes: tuple[int, ...] = ()  # LeRobot episode per group (default: the first ``pairs``)
    seconds: float = 24.0
    switch_at: float | None = None  # s: groups of four with the two switched executors (see above)
    can_mass: float = 0.1
    render_size: tuple[int, int] = (320, 240)  # head camera, as train_v3
    # E*: full = all body joints servoed + root held; arms = only the arm joints servoed (legs and
    # balance stay SuperMimic's); root = only the root held (joints stay SuperMimic's).
    ideal: Literal["full", "arms", "root"] = "full"
    root: Literal["wrench", "kinematic"] = "wrench"  # E*: how the root is held (see RootPin)
    kp_scale: float = 10.0  # E*: joint servo stiffness x the actuators' own
    effort_scale: float = 4.0  # E*: joint force limits x the actuators' own
    video_pairs: int = 2
    overview_size: tuple[int, int] = (480, 360)
    overview_camera: tuple[float, float, float] = (125.0, -25.0, 2.6)  # azimuth 215 looks through the shelf
    device: str = "cuda:0"
    seed: int = 0
    out_dir: Path = Path(f"logs/executors/{time.strftime('%Y%m%d_%H%M%S')}")


class GroupedSource(SteeredDiffusionSource):
    """Every env of a group of ``group`` consecutive envs takes the noise of the group's first env
    whenever both replan in the same call."""

    group = 2

    @torch.no_grad()
    def predict(self, rows, images, states):
        diffusion = self.policy
        with torch.autocast("cuda", torch.bfloat16, enabled=self.bf16):
            cond = diffusion.encode(diffusion._prepare_image(images[rows]), states[rows])
        mean, log_std = self.noise_policy(cond)
        if self.mode == "sample":
            w = mean + torch.exp(log_std) * torch.randn_like(mean)
        else:
            w = mean if self.mode == "mean" else torch.randn_like(mean)
        envs = rows.nonzero().flatten() if rows.dtype == torch.bool else rows  # the controller passes a mask
        index = torch.full((images.shape[0],), -1, dtype=torch.long, device=envs.device)
        index[envs] = torch.arange(len(envs), device=envs.device)
        leader = index[envs - envs % self.group]
        take = (envs % self.group != 0) & (leader >= 0)
        w[take] = w[leader[take]]
        return self.decode(cond, w)


def _label(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=14)
    for i, line in enumerate(lines):
        draw.text((8, 6 + 18 * i), line, font=font, fill=(240, 239, 236), stroke_width=1, stroke_fill=(11, 11, 11))
    return np.asarray(image)


def _plot(path: Path, t: np.ndarray, series: dict[str, np.ndarray], end: np.ndarray, groups: int, group: int,
          names, switch_at: float | None) -> None:  # fmt: skip
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = (("hand_err_cm", "right hand to reference (cm)"), ("hand_can_cm", "right hand to can (cm)"),
              ("ref_hand_can_cm", "reference right hand to can (cm)"), ("can_z", "can height (m)"))  # fmt: skip
    fig, axes = plt.subplots(groups, len(panels), figsize=(4.0 * len(panels), 2.4 * groups), squeeze=False,
                             sharex=True)  # fmt: skip
    for g in range(groups):
        for c, (key, title) in enumerate(panels):
            ax = axes[g, c]
            for e in range(group):
                env = group * g + e
                stop = int(end[env])
                ax.plot(t[:stop], series[key][:stop, env], color=COLORS[e], lw=1.3, label=MODES[e])
            if switch_at is not None:
                ax.axvline(switch_at, color="#52514e", lw=0.8, ls="--")
            ax.grid(color="#e8e8e6", lw=0.6)
            ax.spines[["top", "right"]].set_visible(False)
            if g == 0:
                ax.set_title(title, fontsize=10, color="#0b0b0b")
            if c == 0:
                ax.set_ylabel(str(names[g]), fontsize=9, color="#52514e")
            if g == groups - 1:
                ax.set_xlabel("time (s)", fontsize=9, color="#52514e")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


@torch.no_grad()
def main(args: Args | None = None) -> None:
    args = args or tyro.cli(Args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    group = 2 if args.switch_at is None else 4
    n = group * args.pairs
    env = make_env(
        CanPnpEnvCfgArgs(
            num_envs=n, randomize=False, camera=True, episode_length_s=args.seconds + 1.0, seed=args.seed,
            scene=CanPnpSceneCfg(
                geometry=SceneGeometryCfg(can_mass=args.can_mass),
                camera=dataclasses.replace(CanPnpSceneCfg().camera, width=args.render_size[0], height=args.render_size[1]),
            ),
        ),
        device=args.device,
    )  # fmt: skip
    device = env.device
    tracker = load_tracking_policy(args.tracker_checkpoint, device=args.device)
    policy, _ = load_diffusion_policy(args.diffusion_checkpoint, args.diffusion_repo, args.device)
    set_sampling(policy, args.execute_steps, None)
    hidden = 512
    if args.noise_checkpoint is not None:
        state = torch.load(args.noise_checkpoint, map_location=args.device, weights_only=False)
        hidden = state["args"]["hidden"]
    noise_policy = NoisePolicy(policy.n_tokens, policy.n_emb, policy.horizon, policy.action_dim, hidden=hidden)
    if args.noise_checkpoint is not None:
        noise_policy.load_state_dict(state["noise_policy"])
    elif args.noise_mode != "base":
        raise SystemExit("--noise-mode mean / sample need --noise-checkpoint")
    source = GroupedSource(policy, noise_policy.to(args.device).eval(), args.noise_mode, record=False, bf16=args.bf16)
    source.group = group
    controller = SuperMimicController(env, source, tracker)
    mode = torch.arange(n, device=device) % group
    ideal = (mode == 1) | (mode == 2)  # E* from the start
    none = torch.zeros_like(ideal)
    if args.ideal == "full":
        servo, pin = ideal_executor(env, controller, ideal.clone(), args.kp_scale, args.effort_scale, args.root)
    else:
        servo = ArmServo(env, controller, ideal.clone() if args.ideal == "arms" else none.clone(), args.kp_scale,
                         args.effort_scale)  # fmt: skip
        pin = RootPin(env, controller, ideal.clone() if args.ideal == "root" else none.clone(), args.root)
    body = BodyWorldModel(env, args.tracker_checkpoint)
    kin = BodyKinematics(env, body, compile=False)

    obs, _ = env.reset()
    if args.start == "grasp":
        grasp = GraspStarts(phase=args.grasp_phase)
        episodes = args.grasp_episodes or tuple(grasp.episode_ids[: args.pairs])
        if len(episodes) != args.pairs:
            raise SystemExit(f"--grasp-episodes needs {args.pairs} episodes, got {len(episodes)}")
        rows = torch.tensor([grasp.episode_ids.index(e) for e in episodes]).repeat_interleave(group)
        apply_grasp_start(env, torch.arange(n, device=device), grasp, rows)
        obs = env.obs_buf
        names = tuple(f"episode {e}" for e in episodes)
        where = f"{args.grasp_phase} of episodes {list(episodes)}"
    else:
        names = tuple(f"noise {g}" for g in range(args.pairs))  # one layout, a noise draw per group
        where = "the beginning (nominal layout)"
    print(f"{args.pairs} groups of {MODES[:group]} from {where}; noise {args.noise_mode} "
          f"({args.noise_checkpoint})", flush=True)  # fmt: skip

    geo = env.cfg.events["reset_layout"].params["scene_cfg"].geometry
    can = env.scene["can"]
    can0 = can.data.root_link_pos_w.clone()
    steps = int(round(args.seconds / env.step_dt))
    switch = None if args.switch_at is None else int(round(args.switch_at / env.step_dt))
    alive = torch.ones(n, dtype=torch.bool, device=device)
    end = torch.full((n,), steps, dtype=torch.long, device=device)
    outcome = {k: torch.zeros(n, dtype=torch.bool, device=device)
               for k in ("fallen", "dropped", "touched", "lifted", "placed")}  # fmt: skip
    series = {k: torch.zeros(steps, n, device=device)
              for k in ("hand_err_cm", "root_err_cm", "joint_err_rad", "hand_can_cm", "ref_hand_can_cm", "can_z")}  # fmt: skip
    plan_points = PlanKeypoints(env, controller)
    overviews = []
    for g in range(min(args.video_pairs, args.pairs)):
        overviews.append([_Overview(env, group * g + e, height=args.overview_size[1], width=args.overview_size[0],
                                    camera=args.overview_camera) for e in range(group)])  # fmt: skip
    frames = [[] for _ in overviews]
    reset = torch.ones(n, dtype=torch.bool, device=device)
    env_ids = torch.arange(n, device=device)
    tic = time.perf_counter()
    for step in range(steps):
        if step == switch:
            to_tracker, to_ideal = (mode == 2).nonzero().flatten(), (mode == 3).nonzero().flatten()
            for executor in ((servo, pin) if args.ideal == "full" else (servo,) if args.ideal == "arms" else (pin,)):
                executor.set_active(to_tracker, False)
                executor.set_active(to_ideal, True)
            print(f"step {step}: switched executors", flush=True)
        action = controller.act(obs, reset)
        plan_points.update()
        obs, _, terminated, truncated, _ = env.step(action)
        reset = terminated | truncated

        # Distances to the reference frame this step servoed to (pin.start was set in the step).
        k = torch.clamp(controller.step_in_chunk, max=controller.plan.horizon - 1)
        pos, quat, _, _ = pin.reference(1.0)
        ref = torch.zeros(n, 71, device=device)
        ref[:, 0:3], ref[:, 3:7] = pos, quat
        ref[:, 7:36] = controller.plan.joint_pos[env_ids, k]
        actual = body.state()
        hand, ref_hand = kin.keypoints(actual)[:, RIGHT_EEF, :3], kin.keypoints(ref)[:, RIGHT_EEF, :3]
        can_pos = can.data.root_link_pos_w - env.scene.env_origins
        series["hand_err_cm"][step] = 100 * (hand - ref_hand).norm(dim=-1)
        series["root_err_cm"][step] = 100 * (actual[:, :3] - pos).norm(dim=-1)
        series["joint_err_rad"][step] = (actual[:, 7:36] - ref[:, 7:36]).square().mean(-1).sqrt()
        series["hand_can_cm"][step] = 100 * (hand - can_pos).norm(dim=-1)
        series["ref_hand_can_cm"][step] = 100 * (ref_hand - can_pos).norm(dim=-1)
        can_z = can_pos[:, 2]
        series["can_z"][step] = can_z

        outcome["fallen"] |= env.termination_manager.get_term("fallen") & alive
        outcome["dropped"] |= env.termination_manager.get_term("can_dropped") & alive
        outcome["touched"] |= alive & ((can.data.root_link_pos_w - can0).norm(dim=-1) > 0.01)
        outcome["lifted"] |= alive & (can_z > geo.table_height + geo.can_height / 2 + 0.05)
        box = layout_term(env).box_center_w
        inner = torch.tensor(geo.upper_box_size[:2], device=device) / 2 - geo.box_wall_thickness
        rel = (can.data.root_link_pos_w[:, :2] - box[:, :2]).abs()
        outcome["placed"] |= alive & (rel < inner).all(-1) & (can_z < geo.box_bottom_z + geo.can_height)
        done = alive & reset
        end[done] = step + 1
        alive &= ~reset

        for g, views in enumerate(overviews):
            tiles = []
            for e, view in enumerate(views):
                i = group * g + e
                state = "running" if alive[i] else ("fallen" if outcome["fallen"][i] else "dropped" if outcome["dropped"][i] else "ended")
                tiles.append(_label(view.frame(plan_points.draw), [
                    f"{MODES[e]}   t {(step + 1) * env.step_dt:5.2f} s   {state}",
                    f"hand err {float(series['hand_err_cm'][step, i]):4.1f} cm  to can {float(series['hand_can_cm'][step, i]):5.1f} cm",
                    f"touched {int(outcome['touched'][i])}  lifted {int(outcome['lifted'][i])}  placed {int(outcome['placed'][i])}",
                ]))  # fmt: skip
            frames[g].append(np.concatenate(tiles, axis=1))
        if not bool(alive.any()):
            break
        if step % 100 == 0:
            print(f"step {step}/{steps}  alive {alive.cpu().tolist()}  {time.perf_counter() - tic:.0f} s", flush=True)
    steps_run = step + 1
    end = torch.clamp(end, max=steps_run)

    t = np.arange(1, steps_run + 1) * env.step_dt
    host = {k: v[:steps_run].cpu().numpy() for k, v in series.items()}
    end_np = end.cpu().numpy()
    _plot(args.out_dir / "compare.png", t, host, end_np, args.pairs, group, names, args.switch_at)
    if frames and frames[0]:
        import mediapy

        for g, video in enumerate(frames):
            path = args.out_dir / f"group{g}.mp4"
            mediapy.write_video(path, video, fps=int(round(1.0 / env.step_dt)))
            print(f"wrote {path}")

    def mean_until_end(key):
        return [float(host[key][: max(end_np[i], 1), i].mean()) for i in range(n)]

    def min_until_end(key):
        return [float(host[key][: max(end_np[i], 1), i].min()) for i in range(n)]

    summary = {
        "args": {k: str(v) for k, v in dataclasses.asdict(args).items()}, "groups": list(names),
        "executor": [MODES[i % group] for i in range(n)], "end_s": (end_np * env.step_dt).tolist(),
        **{k: v.cpu().tolist() for k, v in outcome.items()},
        **{f"{k}_mean": mean_until_end(k) for k in ("hand_err_cm", "root_err_cm", "joint_err_rad")},
        **{f"{k}_min": min_until_end(k) for k in ("hand_can_cm", "ref_hand_can_cm")},
    }  # fmt: skip
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for e in range(group):
        envs = slice(e, n, group)
        print(f"{MODES[e]:>18}: hand err {np.mean(summary['hand_err_cm_mean'][envs]):5.2f} cm  "
              f"root err {np.mean(summary['root_err_cm_mean'][envs]):5.2f} cm  "
              f"closest hand-can {np.mean(summary['hand_can_cm_min'][envs]):5.1f} cm  "
              f"(reference {np.mean(summary['ref_hand_can_cm_min'][envs]):5.1f} cm)  "
              + "  ".join(f"{k} {sum(summary[k][envs])}/{args.pairs}" for k in outcome))  # fmt: skip
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
