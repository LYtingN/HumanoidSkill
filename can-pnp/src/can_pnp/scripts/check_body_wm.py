"""How well does Phi (tracker + SuperMimic body world model) predict the robot in this scene?

    uv run --extra diffusion python -m can_pnp.scripts.check_body_wm \\
        --tracker-checkpoint <supermimic>.pt --diffusion-checkpoint <Humanoid-Diffusion>/best.pt \\
        [--noise-checkpoint logs/dsrl/<run>/last.pt --noise-mode sample]

Grasp-start episodes of the (steered) diffusion policy, each reference executed for the whole
``chunk`` (32 frames) before the next replan, so the simulator follows the same reference that
Phi imagines from the replan state. Per step ``k`` of the chunk the errors of the imagined
state against the simulated one are averaged over the envs still alive at ``k``: root position,
joint RMS, right gripper pose (world and torso frame) and head-camera pose. The same errors of
the reference itself, tracked perfectly with leg-odometry root motion
(``body.imagine.reference_states``), show what the world model adds. The predicted fall
probability over the chunk is compared with the falls that happened.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro
from mjlab.utils.lab_api.math import quat_apply_inverse

from can_pnp.body.imagine import PhiStep, imagine_body, reference_states
from can_pnp.body.kinematics import BodyKinematics
from can_pnp.body.world_model import DEFAULT_REPO, BodyWorldModel
from can_pnp.config import CanPnpSceneCfg, SceneGeometryCfg
from can_pnp.dsrl import NoisePolicy, SteeredDiffusionSource
from can_pnp.env_cfg import CanPnpEnvCfgArgs, make_env
from can_pnp.grasp_start import GraspStarts, apply_grasp_start
from can_pnp.pipeline import SuperMimicController, load_diffusion_policy
from can_pnp.supermimic import load_tracking_policy

REPORT_STEPS = (1, 4, 8, 16, 24, 32)


@dataclass
class Args:
    tracker_checkpoint: Path
    diffusion_checkpoint: Path
    # DSRL noise policy (train_dsrl checkpoint); without one the noise is N(0, I).
    noise_checkpoint: Path | None = None
    noise_mode: Literal["sample", "mean", "base"] = "sample"
    diffusion_repo: Path = Path("../../Diffusion/Humanoid-Diffusion")
    supermimic_repo: Path = DEFAULT_REPO
    num_envs: int = 64
    episodes: int = 1  # rollouts of num_envs episodes
    episode_s: float = 14.0
    chunk: int = 32  # control steps per reference = imagined horizon
    grasp_phase: Literal["REACH", "CLOSE"] = "CLOSE"
    can_mass: float = 0.1
    render_size: tuple[int, int] = (320, 240)
    bf16: bool = True
    seed: int = 0
    out: Path = Path("logs/body_wm_check")
    device: str = "cuda:0"


def _angle_deg(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    dot = (q1 * q2).sum(-1).abs().clamp(max=1.0)
    return torch.rad2deg(2 * torch.acos(dot))


def errors(kin: BodyKinematics, pred: torch.Tensor, true: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-step errors (H + 1, N) of predicted against simulated states (H + 1, N, 71)."""
    p, t = kin.poses(pred), kin.poses(true)
    R = 1  # right gripper

    def in_torso(x):
        return quat_apply_inverse(x["torso_quat"], x["gripper_pos"][..., R, :] - x["torso_pos"])

    return {
        "root_pos_cm": 100 * (pred[..., :3] - true[..., :3]).norm(dim=-1),
        "joint_rms_rad": (pred[..., 7:36] - true[..., 7:36]).square().mean(-1).sqrt(),
        "gripper_pos_cm": 100 * (p["gripper_pos"][..., R, :] - t["gripper_pos"][..., R, :]).norm(dim=-1),
        "gripper_pos_torso_cm": 100 * (in_torso(p) - in_torso(t)).norm(dim=-1),
        "gripper_rot_deg": _angle_deg(p["gripper_quat"][..., R, :], t["gripper_quat"][..., R, :]),
        "camera_pos_cm": 100 * (p["camera_pos"] - t["camera_pos"]).norm(dim=-1),
        "camera_rot_deg": _angle_deg(p["camera_quat"], t["camera_quat"]),
    }


def main(args: Args) -> None:
    torch.manual_seed(args.seed)
    device = args.device
    env = make_env(
        CanPnpEnvCfgArgs(
            num_envs=args.num_envs, randomize=False, camera=True, episode_length_s=1e6, seed=args.seed,
            scene=CanPnpSceneCfg(
                geometry=SceneGeometryCfg(can_mass=args.can_mass),
                camera=dataclasses.replace(CanPnpSceneCfg().camera, width=args.render_size[0], height=args.render_size[1]),
            ),
        ),
        device=device,
    )  # fmt: skip
    env.reset()
    diffusion, _ = load_diffusion_policy(args.diffusion_checkpoint, args.diffusion_repo, device)
    diffusion.requires_grad_(False)
    hidden = 512
    state = None
    if args.noise_checkpoint is not None:
        state = torch.load(args.noise_checkpoint, map_location=device, weights_only=False)
        hidden = state["args"]["hidden"]
    noise_policy = NoisePolicy(diffusion.n_tokens, diffusion.n_emb, diffusion.horizon, diffusion.action_dim,
                               hidden=hidden).to(device)  # fmt: skip
    if state is not None:
        noise_policy.load_state_dict(state["noise_policy"])
    mode = args.noise_mode if state is not None else "base"
    tracker = load_tracking_policy(args.tracker_checkpoint, device=device)
    body = BodyWorldModel(env, args.tracker_checkpoint, args.supermimic_repo)
    kin = BodyKinematics(env, body)
    phi = PhiStep(body, tracker)
    starts = GraspStarts(phase=args.grasp_phase)
    sensors = env.sim._sensor_context
    K = args.chunk
    if K > diffusion.horizon:
        raise SystemExit(f"--chunk {K} exceeds the reference horizon {diffusion.horizon}")

    sums: dict[str, dict[str, torch.Tensor]] = {"phi": {}, "reference": {}}
    count = torch.zeros(K + 1, device=device)
    fall_pred, fall_true = [], []
    n = env.num_envs
    for episode in range(args.episodes):
        source = SteeredDiffusionSource(diffusion, noise_policy.eval(), mode, record=False, bf16=args.bf16)
        source.execute_steps = K  # execute each whole reference: the simulator follows what Phi imagines
        controller = SuperMimicController(env, source, tracker)
        env.episode_length_buf.zero_()
        env.action_manager.reset()
        rows = torch.randint(len(starts), (n,))
        apply_grasp_start(env, torch.arange(n, device=device), starts, rows)
        obs = env.obs_buf
        reset = torch.ones(n, dtype=torch.bool, device=device)
        alive = torch.ones(n, dtype=torch.bool, device=device)
        steps = round(args.episode_s / env.step_dt)
        for step in range(steps):
            k = step % K
            if k == 0:
                s0, obs0, xi = body.state(), obs["policy"].clone(), body.context()
            action = controller.act(obs, reset)
            reset = torch.zeros_like(reset)
            if k == 0:
                plan = controller.plan
                rollout = imagine_body(phi, plan, s0, obs0, xi, K)
                ref = reference_states(plan, s0, K)
                true, valid = [s0], [alive.clone()]
                fell = torch.zeros(n, dtype=torch.bool, device=device)
                chunk_alive = alive.clone()
            env.sim._sensor_context = sensors if k >= K - 2 else None  # images of the next replan
            obs, _, terminated, _, _ = env.step(action)
            fell |= alive & env.termination_manager.get_term("fallen")
            alive &= ~terminated
            true.append(body.state())
            valid.append(alive.clone())
            if k == K - 1 or step == steps - 1:
                h = len(true) - 1
                t, v = torch.stack(true), torch.stack(valid).float()
                for name, pred in (("phi", rollout.states[: h + 1]), ("reference", ref[: h + 1])):
                    for key, e in errors(kin, pred, t).items():
                        e = torch.nan_to_num(e) * v
                        sums[name].setdefault(key, torch.zeros(K + 1, device=device))[: h + 1] += e.sum(-1)
                count[: h + 1] += v.sum(-1)
                if h == K:
                    fall_pred.append(rollout.fall_prob[-1][chunk_alive])
                    fall_true.append(fell[chunk_alive])
        print(f"episode {episode + 1}/{args.episodes} done, {int(alive.sum())}/{n} alive at the end", flush=True)
    env.sim._sensor_context = sensors

    mean = {name: {key: (v / count.clamp(min=1)).tolist() for key, v in d.items()} for name, d in sums.items()}
    fp, ft = torch.cat(fall_pred), torch.cat(fall_true).float()
    falls = {
        "chunks": len(fp), "actual_fall_rate": float(ft.mean()), "mean_pred_fall_prob": float(fp.mean()),
        "mean_pred_when_fell": float(fp[ft > 0].mean()) if bool((ft > 0).any()) else None,
        "mean_pred_when_not": float(fp[ft == 0].mean()),
    }  # fmt: skip
    print(f"\nmode={mode}  samples per step: {count[1]:.0f} (step 1) .. {count[K]:.0f} (step {K})")
    keys = list(mean["phi"])
    print(f"{'step':>5} " + " ".join(f"{k:>22}" for k in keys))
    for k in (k for k in REPORT_STEPS if k <= K):
        row = " ".join(f"{mean['phi'][key][k]:>10.2f} / {mean['reference'][key][k]:<9.2f}" for key in keys)
        print(f"{k:>5} {row}")
    print("(phi / perfectly tracked reference)")
    print("falls:", falls)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{mode}_{args.noise_checkpoint.parent.name if args.noise_checkpoint else 'base'}.json"
    path.write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()}, "count": count.tolist(),
                                "errors": mean, "falls": falls}, indent=1))  # fmt: skip
    print("wrote", path)


if __name__ == "__main__":
    main(tyro.cli(Args))
