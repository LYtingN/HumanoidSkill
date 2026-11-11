"""Attribution of a competent policy's failures: was the intent bad, or its execution?

For every failed episode of the noise policy (e.g. DSRL's), the object world model replays the
moments around the failure twice from the simulated can state:

- actual branch: the simulated keypoints (what S0 did);
- ideal branch: the reference executed exactly (``ideal_branch``: the whole body, or only the
  arms on the actual body; recorded during the rollout as ``ref_kp``).

Both replays give the object WM the same kind of history (the ``wm_history`` frames before the
window) and differ in the future keypoints; windows longer than the WM's horizon are chained.

A dropped can whose actual replay loses the can but whose ideal replay keeps it is an execution
loss (S0's); one that loses the can in both is an intent loss (S1's); one the actual replay does
not reproduce is unattributed (model miss), and one where the ensemble disagrees on the ideal
replay is uncertain. Successful episodes give the false-alarm rates of both replays at random
held moments. No Phi: the keypoints are the simulator's, so only the object WM is a model here.

    torchrun ... -m experiments.diagnose_attribution --tracker-checkpoint <supermimic>.pt \\
      --diffusion-checkpoint <dp>/best.pt --init-noise-checkpoint <dsrl>/last.pt --out-dir logs/attr/<run>

The object WM is trained on ``wm_collections`` rollouts of the policy (real hands only, windows of
``pre + post`` steps), or loaded with ``--object-wm-checkpoint``; then ``eval_collections`` fresh
rollouts are attributed. Per-rank results go to ``attribution_rank<r>.pt``; rank 0 writes
``summary.json``.
"""

from __future__ import annotations

import dataclasses
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import tyro
from hskill.data.buffer import TIME_MAJOR, EpisodeBuffer
from hskill.data.episodes import Episodes
from hskill.executors.imagine import PhiStep
from hskill.embodiment.g1_aloha import RIGHT_EEF
from hskill.interaction.model import (
    ObjectWMCfg,
    ObjectWMEnsemble,
    load_object_wm,
    object_loss,
    object_targets,
    quat_exp,
    quat_log,
)
from hskill.learners.grpo import Distributed
from hskill.s0.body_world_model import BodyWorldModel
from hskill.s0.kinematics import BodyKinematics
from hskill.s1.noise_policy import NoisePolicy
from hskill.sim.collect import SimCollector
from hskill.sim.rollouts import Rollouts
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, quat_inv, quat_mul

from experiments._task_setup import rollout_dependencies
from experiments.train_v3 import Args as V3Args


@dataclass
class Args(V3Args):
    out_dir: Path = Path("logs/attr/run")
    object_wm_checkpoint: Path | None = None  # skip the WM training
    wm_collections: int = 6
    wm_steps_per_collection: int = 600
    eval_collections: int = 2
    # Replay window around the failure: ``pre`` steps before the last held step, ``post`` after.
    pre: int = 32
    post: int = 32
    slip: float = 0.03  # m: can displacement in the gripper frame that counts as losing it
    uncertain: float = 0.03  # m: ensemble spread of the final can position above which a replay is unsure
    smooth_seams: bool = True  # ideal branch: the reference without the jumps between plans


def _gather(x: torch.Tensor, t: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    """x (T + 1, N, ...) at times t (W + 1, M) of envs n (M,) -> (W + 1, M, ...)."""
    return x[t, n[None].expand_as(t)]


def smooth_seams(ref: torch.Tensor, k: int) -> torch.Tensor:
    """Reference hand path (T + 1, N, 7) without the jumps between consecutive plans.

    Each replan's first frame lands on average ~1.8 cm off the previous plan's path (p95 ~3.6 cm):
    an exact tracker of that reference would jerk the can loose at every replan, which no
    physical arm does (S0 smooths it out). Here the jump, measured against the previous step's
    motion carried on, is taken out of the new plan's frames and faded back in over its ``k``
    executed steps: the hand of a continuous intent tracked exactly. Frame 8j + 1 is the first of
    plan j (the plan made at step 8j drives the step to 8j + 1).
    """
    out = ref.clone()
    pos, quat = ref[..., :3], ref[..., 3:]
    for s in range(k + 1, ref.shape[0], k):
        jump_pos = pos[s] - (2 * pos[s - 1] - pos[s - 2])
        carried = quat_mul(quat_mul(quat[s - 1], quat_inv(quat[s - 2])), quat[s - 1])
        jump_rot = quat_log(quat_mul(quat[s], quat_inv(carried)))
        for i in range(min(k, ref.shape[0] - s)):
            w = 1 - i / k
            out[s + i, :, :3] = pos[s + i] - w * jump_pos
            out[s + i, :, 3:] = quat_mul(quat_exp(-w * jump_rot), quat[s + i])
    return out


def _move(obj: torch.Tensor, frm: torch.Tensor, to: torch.Tensor) -> torch.Tensor:
    """Object records (..., >= 10) carried rigidly from hand poses ``frm`` to ``to`` (..., 7)."""
    turn = quat_mul(to[..., 3:], quat_inv(frm[..., 3:]))
    pos = to[..., :3] + quat_apply(to[..., 3:], quat_apply_inverse(frm[..., 3:], obj[..., :3] - frm[..., :3]))
    return torch.cat([pos, quat_mul(turn, obj[..., 3:7]), quat_apply(turn, obj[..., 7:10]), obj[..., 10:]], dim=-1)


class Judge:
    """Outcome of a replay window: the task's own placement test (in the box, upright, released,
    for its hold time), entering the box, losing the can (no longer held, slipped in the gripper
    frame or below the drop height) and the distance to the goal, at each env's last step e."""

    def __init__(self, rollouts: Rollouts, args: Args, dt: float):
        self.ro, self.args, self.dt = rollouts, args, dt

    def __call__(self, obj: torch.Tensor, hands: torch.Tensor, held_p: torch.Tensor, released: torch.Tensor,
                 e: torch.Tensor, ep: Episodes, n: torch.Tensor) -> dict[str, torch.Tensor]:  # fmt: skip
        """obj (W + 1, M, >= 7), hands (W + 1, M, 7), held_p (W + 1, M) in [0, 1], released (W, M)."""
        task, W, m = self.ro.task, obj.shape[0] - 1, torch.arange(obj.shape[1], device=obj.device)
        tracker = self.ro.task_tracker(task, self.ro.geo, ep.box[n], ep.goal[n], torch.zeros(len(n), device=obj.device),
                                       obj[0, :, :3], self.dt)  # fmt: skip
        live = torch.ones(len(n), dtype=torch.bool, device=obj.device)
        for k in range(1, W + 1):
            live &= (k <= e) & (obj[k, :, 2] >= task.drop_height)
            tracker.step(obj[k, :, :3], obj[k, :, 3:7], released[k - 1], live)
        rel = quat_apply_inverse(hands[..., 3:], obj[..., :3] - hands[..., :3])
        disp = (rel[e, m] - rel[0]).norm(dim=-1)
        lost = (held_p[e, m] < 0.5) | (disp > self.args.slip) | (obj[e, m, 2] < task.drop_height)
        return {"placed": tracker.placed.clone(), "entered": tracker.entered.clone(), "lost": lost,
                "dist": (obj[e, m, :3] - ep.goal[n]).norm(dim=-1)}  # fmt: skip


@torch.no_grad()
def replay(ep: Episodes, anchor: torch.Tensor, object_wm: ObjectWMEnsemble, judge: Judge, args: Args) -> dict:
    """Actual / ideal replays and the simulated truth of the window around ``anchor`` (N,)."""
    device = ep.kp.device
    T, N = ep.command.shape[:2]
    W, L = args.pre + args.post, object_wm.cfg.history
    n = torch.arange(N, device=device)
    t0 = (anchor - args.pre).clamp(min=0)
    # Judged at the window end, or at the episode's end where the env was reset after it (failures);
    # a placed episode's env runs on unreset, so its window may extend past the placement.
    last = torch.where(ep.placed, torch.full_like(ep.end_step, T), ep.end_step)
    e = (last - t0).clamp(min=1, max=W)
    t = (t0[None] + torch.arange(W + 1, device=device)[:, None]).clamp(max=T)  # (W + 1, N)
    h = (t0[None] + torch.arange(1 - L, 1, device=device)[:, None]).clamp(min=0)  # (L, N) the history
    ref = ep.ref_kp
    if args.smooth_seams:
        ref = ref.clone()
        for i in range(ref.shape[2]):  # every keypoint: the whole reference jumps at a replan
            ref[:, :, i] = smooth_seams(ep.ref_kp[:, :, i], ep.execute_steps)
    kp, ideal_kp = _gather(ep.kp, t, n), _gather(ref, t, n)  # (W + 1, N, 5, 7)
    command = _gather(ep.command, t[:-1].clamp(max=T - 1), n)  # (W, N, 2)
    released = command[..., 1] > judge.ro.task.release_opening
    width = _gather(ep.width, t, n)
    hist_kp, hist_width, hist_obj = _gather(ep.kp, h, n), _gather(ep.width, h, n), _gather(ep.obj, h, n)[..., :10]
    # The ideal branch's history is on the reference's hand: a can held at t0 sits in that hand as it
    # sits in the actual one (had the arm tracked exactly, it would have carried the can there).
    # Starting it on the actual hand instead makes the hand jump by the tracking error in the first
    # step, which the object WM reads as a jerk that shakes the can loose. A free can keeps the
    # measured history.
    held0 = (ep.held & ep.alive)[t0, n]
    ideal_hist_kp = torch.where(held0[None, :, None, None], _gather(ref, h, n), hist_kp)
    moved = _move(hist_obj, hist_kp[:, :, RIGHT_EEF], ideal_hist_kp[:, :, RIGHT_EEF])
    ideal_hist_obj = torch.where(held0[None, :, None], moved, hist_obj)
    branches = {"actual": (hist_kp, hist_obj, kp[1:]), "ideal": (ideal_hist_kp, ideal_hist_obj, ideal_kp[1:])}
    hands = {name: torch.cat([hk[-1:, :, RIGHT_EEF], fk[:, :, RIGHT_EEF]]) for name, (hk, _, fk) in branches.items()}
    out = {}
    for name, (hk, ho, fk) in branches.items():
        objs, logits = object_wm.rollout_all(hk, hist_width, ho, fk, command, width[1:])
        held_p = torch.cat([torch.ones_like(logits[0, :1]), torch.sigmoid(logits).mean(0)])  # (W + 1, N)
        res = judge(ObjectWMEnsemble.mean(objs), hands[name], held_p, released, e, ep, n)
        res["spread"] = objs[:, e, n, :3].std(0).norm(dim=-1)
        out.update({f"{name}_{k}": v for k, v in res.items()})
    held = (ep.held & ep.alive).float()
    truth = judge(_gather(ep.obj, t, n), hands["actual"], _gather(held, t, n), released, e, ep, n)
    out.update({f"true_{k}": v for k, v in truth.items()})
    in_window = torch.arange(W + 1, device=device)[:, None] <= e[None]
    track = torch.where(in_window, (hands["ideal"][..., :3] - hands["actual"][..., :3]).norm(dim=-1), 0)
    out["track_max"] = track.amax(0)
    # Largest per-step hand motion in the window (a jerky ideal path would explain ideal losses).
    for name, path in hands.items():
        step = torch.where(in_window[1:], (path[1:, :, :3] - path[:-1, :, :3]).norm(dim=-1), 0)
        out[f"{name}_step_max"] = step.amax(0)
    return out


@torch.no_grad()
def attribute(ep: Episodes, object_wm: ObjectWMEnsemble, judge: Judge, args: Args, gen: torch.Generator) -> dict:
    """Per-episode replays of one rollout (policy envs only)."""
    device = ep.kp.device
    T, N = ep.command.shape[:2]
    steps = torch.arange(T + 1, device=device)[:, None]
    held = ep.held & ep.alive
    last_held = torch.where(held, steps, -1).amax(0)  # (N,)
    first_held = torch.where(held, steps, T + 1).amin(0)
    end = ep.end_step
    placed, dropped, fallen = ep.placed, ep.dropped, ep.fallen
    timeout = ~(placed | dropped | fallen)
    held_at_end = last_held >= end - 1
    # Anchors: the last held step (drops; releases of time-outs and successes), the grasp start
    # (drops never held), the end (time-outs still holding, falls).
    anchor = torch.where(last_held >= 0, last_held, torch.zeros_like(end))
    anchor = torch.where((timeout & held_at_end) | fallen, (end - args.post).clamp(min=0), anchor)
    main = replay(ep, anchor, object_wm, judge, args)
    # Control: successes at a random held step whose window ends before the release.
    rand = torch.rand(N, generator=gen).to(device)
    span = (last_held - args.post - first_held).clamp(min=0)
    control = replay(ep, first_held.clamp(max=T) + (rand * span).long(), object_wm, judge, args)
    out = {
        "placed": placed, "dropped": dropped, "fallen": fallen, "timeout": timeout, "never_held": last_held < 0,
        "held_at_end": held_at_end, "end": end,
        "ever_commanded_release": ((ep.command[..., 1] > judge.ro.task.release_opening) & ep.alive[1:]).any(0),
        **main, **{f"control_{k}": v for k, v in control.items()},
    }  # fmt: skip
    policy = ~ep.base
    return {k: v[policy].cpu() for k, v in out.items()}


def summarize(r: dict[str, torch.Tensor], args: Args) -> dict:
    def share(mask, of):
        return round(float(mask[of].float().mean()), 4) if bool(of.any()) else None

    def cm(of):
        return round(100 * float(r["track_max"][of].median()), 2) if bool(of.any()) else None

    n = len(r["placed"])
    every = torch.ones(n, dtype=torch.bool)
    sure = r["ideal_spread"] < args.uncertain

    def split(of, reproduced, ideal_ok):
        """Execution / intent / uncertain among the failures the actual replay reproduces."""
        rep = of & reproduced
        execution, intent = rep & ideal_ok & sure, rep & ~ideal_ok
        return {
            "count": int(of.sum()), "reproduced_by_wm": share(reproduced, of),
            "execution_loss": share(execution, rep), "intent_loss": share(intent, rep),
            "uncertain": share(rep & ideal_ok & ~sure, rep),
            "track_max_cm_median": {"all": cm(of), "execution": cm(execution), "intent": cm(intent)},
        }  # fmt: skip

    drop, to = r["dropped"], r["timeout"]
    released = to & ~r["held_at_end"]
    holding = to & r["held_at_end"]
    ok = r["placed"]
    return {
        "episodes": n,
        "outcomes": {k: share(r[k], every) for k in ("placed", "dropped", "fallen", "timeout")},
        "failure_mix": {
            "drop_after_holding": share(drop & ~r["never_held"], ~ok), "drop_never_held": share(drop & r["never_held"], ~ok),
            "timeout_released_not_placed": share(released, ~ok), "timeout_still_holding": share(holding, ~ok),
            "fall": share(r["fallen"], ~ok),
        },  # fmt: skip
        # Lost the can: does the ideal arm keep it?
        "drop_after_holding": {**split(drop & ~r["never_held"], r["actual_lost"], ~r["ideal_lost"]),
                               "sim_lost_in_window": share(r["true_lost"], drop & ~r["never_held"])},  # fmt: skip
        # Released but not placed: does the ideal arm place it?
        "timeout_released_not_placed": {**split(released, ~r["actual_placed"], r["ideal_placed"]),
                                        "entered_box_sim": share(r["true_entered"], released)},  # fmt: skip
        # Never released: does the ideal arm bring the can into the box (else: S1 did not get there
        # or did not open)?
        "timeout_still_holding": {
            **split(holding & ~r["true_entered"], ~r["actual_entered"], r["ideal_entered"]),
            "in_box_but_not_released": share(r["true_entered"], holding),
            "ever_commanded_release": share(r["ever_commanded_release"], holding),
            "dist_true_m_median": round(float(r["true_dist"][holding].median()), 3) if bool(holding.any()) else None,
        },  # fmt: skip
        # Checks of the object WM on successes: placing at the release, keeping the can before it.
        "control_successes": {
            "count": int(ok.sum()),
            "release_placed_actual": share(r["actual_placed"], ok), "release_placed_ideal": share(r["ideal_placed"], ok),
            "hold_false_loss_actual": share(r["control_actual_lost"], ok),
            "hold_false_loss_ideal": share(r["control_ideal_lost"], ok),
            "track_max_cm_median_release": cm(ok),
            "step_max_cm_median": {"actual": round(100 * float(r["control_actual_step_max"][ok].median()), 2),
                                   "ideal": round(100 * float(r["control_ideal_step_max"][ok].median()), 2)}
            if bool(ok.any()) else None,
        },  # fmt: skip
        "ideal_uncertain_share_failures": share(~sure, ~ok),
    }


def main(args: Args | None = None) -> None:
    args = args or tyro.cli(Args)
    if args.init_noise_checkpoint is None:
        raise ValueError("--init-noise-checkpoint: the policy whose failures are attributed")
    ddp = Distributed()
    device = ddp.device
    torch.manual_seed(args.seed + 1000 * ddp.rank)
    gen = torch.Generator().manual_seed(args.seed + 7919 * ddp.rank)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if ddp.rank == 0:
        (args.out_dir / "args.json").write_text(json.dumps(dataclasses.asdict(args), indent=2, default=str))
    rollouts = Rollouts(args, device, ddp.rank, num_envs=args.num_envs, **rollout_dependencies(args))
    env = rollouts.env
    body = BodyWorldModel(env, args.tracker_checkpoint, args.supermimic_repo)
    kin = BodyKinematics(env, body, compile=args.compile)
    phi = PhiStep(body, rollouts.tracker, compile=args.compile)
    collector = SimCollector(rollouts, phi, kin, args.horizon, args.well_tracked, args.wm_history, args.ideal_branch)
    diffusion = rollouts.diffusion
    noise_policy = NoisePolicy(diffusion.n_tokens, diffusion.n_emb, diffusion.horizon, diffusion.action_dim,
                               hidden=args.hidden, init_log_std=args.init_log_std).to(device)  # fmt: skip
    noise_policy.load_state_dict(torch.load(args.init_noise_checkpoint, map_location=device, weights_only=False)["noise_policy"])
    train_rows = torch.tensor(rollouts.train_rows)
    n = env.num_envs
    judge = Judge(rollouts, args, env.step_dt)

    def collect() -> tuple[Episodes, dict]:
        rows = train_rows[torch.randint(len(train_rows), (n,), generator=gen)]
        ep, stats = collector.collect(noise_policy, rows, torch.zeros(n, dtype=torch.bool, device=device))
        ep.cond, ep.noise, ep.log_prob = ep.cond.new_zeros(0), None, None  # not needed: free the tokens
        return ep, stats

    def log(msg: str) -> None:
        if ddp.rank == 0:
            print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    if args.object_wm_checkpoint is not None:
        state = torch.load(args.object_wm_checkpoint, map_location=device, weights_only=False)
        object_wm = load_object_wm(state, env.step_dt).to(device)
    else:
        cfg = ObjectWMCfg(history=args.wm_history, horizon=args.horizon, hidden=args.wm_hidden, dt=env.step_dt)
        object_wm = ObjectWMEnsemble(cfg, args.wm_ensemble).to(device)
        if args.compile:
            object_wm.compile_step()
        optimizer = torch.optim.Adam(object_wm.parameters(), lr=args.wm_lr)
        ddp.broadcast(object_wm)
        H = args.horizon
        buffer = EpisodeBuffer(args.wm_collections, 1.0, H, args.wm_history)
        E, size = object_wm.num_members, args.wm_batch
        matmul = torch.backends.cuda.matmul
        for c in range(args.wm_collections):
            ep, stats = collect()
            buffer.add(ep)
            log(f"WM collection {c}: success {stats['sim/success_rate']:.3f} drop {stats['sim/drop_rate']:.3f}")
            prev, matmul.allow_tf32 = matmul.allow_tf32, True
            object_wm.train()
            metrics = []
            for _ in range(args.wm_steps_per_collection):
                b = buffer.windows(E * size, 0.0)  # real hands only, as in the replays
                perm = torch.randperm(E * size, device=device)
                b = {k: v[:, perm] if k in TIME_MAJOR else v[perm] for k, v in b.items()}
                inputs = [b[k].view(b[k].shape[0], E, size, *b[k].shape[2:])
                          for k in ("hist_kp", "hist_width", "hist_obj", "fut_kp", "fut_cmd")]  # fmt: skip
                objs, logits = object_wm.rollout_members(*inputs)
                loss, m = object_loss(objs.transpose(0, 1).reshape(H + 1, E * size, -1),
                                      logits.transpose(0, 1).reshape(H, E * size), object_targets(b["obj"], env.step_dt),
                                      b["held"], b["mask"])  # fmt: skip
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                ddp.average_grads(object_wm)
                grad = torch.nn.utils.clip_grad_norm_(object_wm.parameters(), 10.0)
                if bool(ddp.sum((~(loss.isfinite() & grad.isfinite())).float().reshape(1)) > 0):
                    continue
                optimizer.step()
                metrics.append({k: float(v) for k, v in m.items() if k in ("pos_err_cm_final", "held_acc")})
            object_wm.eval()
            matmul.allow_tf32 = prev
            last = {k: round(sum(x[k] for x in metrics[-20:]) / len(metrics[-20:]), 3) for k in metrics[-1]} if metrics else {}
            log(f"WM after collection {c}: {last}")
        if ddp.rank == 0:
            torch.save({"object_wm": object_wm.state_dict(), "args": dataclasses.asdict(args)}, args.out_dir / "object_wm.pt")
        del buffer

    results = []
    for c in range(args.eval_collections):
        ep, stats = collect()
        log(f"eval collection {c}: success {stats['sim/success_rate']:.3f} drop {stats['sim/drop_rate']:.3f} "
            f"fall {stats['sim/fall_rate']:.3f}")  # fmt: skip
        results.append(attribute(ep, object_wm, judge, args, gen))
        del ep
    merged = {k: torch.cat([r[k] for r in results]) for k in results[0]}
    torch.save(merged, args.out_dir / f"attribution_rank{ddp.rank}.pt")
    if ddp.enabled:
        dist.barrier()
    if ddp.rank == 0:
        files = sorted(args.out_dir.glob("attribution_rank*.pt"))
        parts = [torch.load(f, weights_only=False) for f in files]
        everything = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
        summary = summarize(everything, args)
        (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), flush=True)
    if ddp.enabled:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
