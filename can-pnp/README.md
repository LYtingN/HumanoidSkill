# can-pnp

Can pick-and-place scene from `motion_token_residual_rl` (`tasks/can_pnp`, Isaac Lab
`envs/isaaclab/scene.py`), ported to [mjlab](https://github.com/mujocolab/mjlab) / MuJoCo Warp
at the same commit as `supermimic-reproduced` (`8bb2bc9`), and driven by

```
diffusion policy (Humanoid-Diffusion, head camera + state)
  -> 32-frame reference [29 joint positions | 2 gripper openings | 6D root rotation]
  -> SuperMimic tracker (deploy_obs checkpoint) -> 29 joint targets -> PD
```

## Scene

| Entity | Content |
|---|---|
| `robot` | mjlab G1 — the model, actuators and collision setup SuperMimic is trained on — with the rubber hands replaced by the ALOHA 2 grippers of the can_pnp robot (`assets/robots/g1_aloha`, built by `scripts/build_assets.py`); grippers kp 1800 / kd 20 / 35 N, mirror fingers coupled by equality |
| `can` | target purple can, r 0.0275 m, h 0.105 m, 0.62 kg, on the table |
| `shelf_can` | resident can in the box, randomized position and colors |
| `table` | 1.2 × 0.6 m top at 0.74 m; moved by the layout |
| `box` | open cardboard box 0.255 × 0.225 × 0.1 m on the top shelf; xy jitter |
| `room` | black shelf, carpet patches, floor cable |
| sensors | `finger_{l,r}_can_contacts`, `head_camera` RGB 640×480 on `torso_link` (fovy 45°) |

Physics: dt 5 ms, decimation 4 (50 Hz), solver 10 / 20 iterations (SuperMimic's training values),
per-world buffers 128 contacts / 600 constraint rows (mjlab's default `njmax = 64` overflows here).
Can and floor contacts follow the MuJoCo scene the diffusion dataset was collected in (can condim 6,
friction (1, 0.005, 0.0005), solref (0.01, 1); floor friction 1).

Layout per reset (`layout.py`, port of `collection_sampling.py` / `reset.py` / `randomization.py`):
table fore-aft ±0.25 m and 0.3–0.6 m gap to the shelf, robot 0.5–1.5 m in front of the table with
yaw ±45°, can inset 0.15–0.3 m and lateral ±0.15 m, resident can uniform in the box with HSV
colors, placement goal in the free half of the box, box jitter ±(0.03, 0.02) m.
`randomize=False` gives the nominal layout. Layout and goal: `can_pnp.layout.layout_term(env)`.

Checked against the collection scene: shelf / box / table geometry and the head camera pose match;
the head-camera image of episode 0's initial state lines up with the dataset frame (can, table edge,
shelf post within ~15 px). Shading differs — the dataset was rendered with Isaac RTX, this scene with
MuJoCo — which is a visual domain gap for the diffusion policy.

## Control pipeline

- `supermimic.py` — loads the tracker checkpoint and builds its 646-dim goal from a reference:
  10 frames × `[q − q0, q̇, gravity in the ref root frame, ref root angular velocity]` + the 6D
  reference torso orientation in the robot torso frame. The layout was read off the checkpoint's
  scale vectors (`deploy_obs` drops horizontal velocity and height from the 68-dim frame). The
  anchor term is assumed to be mjlab's `motion_anchor_ori_b`. Full-observation trackers (goal 680,
  e.g. `supermimic_0100000.pt`) use the whole 68-dim frame — `[…, ref root linear velocity in its
  yaw frame, ref root height]` — and no anchor term.
- `odometry.py` — the diffusion reference has no root translation; for the full tracker the root
  height and velocity come from leg kinematics: the lower sole is the stance foot and does not
  slip, and the ground is calibrated on the measured robot at each replan.
- `pipeline.py` — `SuperMimicController` with three reference sources:
  - `DiffusionSource`: the diffusion policy trained with `Humanoid-Diffusion/config/can_pnp_supermimic.yaml`
    (targets: measured joints, gripper commands, root rotation relative to the prediction time).
    Its observation is built as recorded in the dataset: body joints, `wrist_yaw_link` poses in the
    pelvis frame, projected gravity, gripper openings in metres, head camera resized to 224².
  - `DatasetSource`: the same targets from a recorded episode, open loop.
  - `StandSource`: the default pose.
- The `policy` observation group of `env_cfg.py` is SuperMimic's training group (930 dims); the
  action is `[29 tracker actions (q0 + G1_ACTION_SCALE · a) | 2 gripper slide targets]`.

Checks so far (tracker `ckpt_0268000.pt`): standing 5 s and replaying episode 0 (18 s: walk, turn,
reach) — no falls in 4 envs, joint error ~0.06 / ~0.09 rad, heading error ~1° / <1°.
Tracker `supermimic_0100000.pt` (full observation, odometry): standing 5 s — no falls, joint error
~0.037 rad; episode 0 replay 18 s — no falls, heading error ~0.35°. Diffusion from the CLOSE grasp
starts of episodes 0–7, 14 s, 100 g can: 7/8 lifted, 0/8 in the box, 1 can dropped.

## Setup

```bash
cd humanoid_skill/can-pnp
bash scripts/fetch_wheels.sh            # warp-lang 1.18.0+cu12 into wheels/ (or copy wheels/ over)
UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple uv sync --extra diffusion
```

- Drivers 550 / 570 support CUDA ≤ 12.8: torch comes from the cu126 index and warp from the CUDA 12
  release build (PyPI's warp-lang targets CUDA 13). `pyproject.toml` overrides mjlab's warp source.
- Both warp builds report version 1.18.0 and would share `~/.cache/warp/1.18.0`; the package sets
  `WARP_CACHE_PATH=~/.cache/warp-cu12` so CUDA 13 kernels from other environments are not loaded.
- PyPI is very slow from these hosts; the Aliyun mirror is ~16 MB/s (USTC is faster but returns 429
  on parallel downloads). If GitHub is slow too, redirect the mjlab fetch to a local checkout:
  `GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=url.file:///path/to/mjlab.insteadOf GIT_CONFIG_VALUE_0=https://github.com/mujocolab/mjlab`.
- Local workstation only: its kernel module is 570.172.18 while the user-space libraries were upgraded
  to 580, so CUDA fails to initialize. Run with the 570 libraries (no system change):
  `export LD_LIBRARY_PATH=$HOME/.local/lib/nvidia-570.172.18` (symlinks to the `*.570.172.18` libs).

## Run

```bash
MUJOCO_GL=egl uv run can-pnp-check --num-envs 8 --out-dir logs/scene_check   # scene + renders
uv run pytest                                                                  # CPU tests

TRACKER=<supermimic run>/ckpt_0268000.pt
uv run can-pnp-run --source stand   --tracker-checkpoint $TRACKER --seconds 5
uv run can-pnp-run --source dataset --tracker-checkpoint $TRACKER \
    --episode ../data/residual_rl/can_pnp/data/chunk-000/episode_000000.parquet --seconds 18
uv run --extra diffusion can-pnp-run --source diffusion --tracker-checkpoint $TRACKER \
    --diffusion-checkpoint <Humanoid-Diffusion run>/best.pt --diffusion-repo ../../Diffusion/Humanoid-Diffusion
```

`can-pnp-run` reports falls, joint / heading tracking error and (closed loop) can lifted / placed,
and writes head-camera and overview videos of env 0 to `logs/pipeline/`.

## DSRL + GRPO post-training

`dsrl.py` / `scripts/train_dsrl.py` (`can-pnp-dsrl`) steer the frozen diffusion policy through its
DDIM initial noise (DSRL): a Gaussian `NoisePolicy` on the diffusion's observation tokens picks the
32 × 37 noise of every replan; it starts at N(0, I), i.e. the base policy. Diffusion, DINOv3 and the
tracker stay frozen. GRPO: each group of envs restarts from one recorded grasp start, the episodic
return (`task.py`: 10 once the can rests released and upright in the box for 0.2 s, else
exp(−d / 0.1 m) of the final can-goal distance, 0 after a fall or a dropped can) is normalized
within the group, and PPO-clip updates the noise policy with a KL term to N(0, I) (`rl/grpo.py`).
Grasp starts of the diffusion's 12 validation episodes are held out for the deterministic (mean
noise) evaluation.

```bash
# 8 GPUs; extra arguments go to can_pnp.scripts.train_dsrl (see its Args)
TRACKER=../checkpoints/supermimic_0100000.pt \
DIFFUSION=../../UMI-Diffusion/Humanoid-Diffusion/outputs/<run>/best.pt \
OUTPUT_DIR=logs/dsrl/run1 bash scripts/train_dsrl_8gpu.sh
```

`metrics.jsonl` logs per iteration the train success / in-box / lift / fall / drop / physics-divergence
rates, the final and closest can-goal distances, the share of groups with a learning signal, PPO
statistics and, every `--eval-every` iterations and at a run's first one, `eval_mean/*` (and
`eval_base/*`, the unsteered policy, at the first). `best.pt` holds the noise policy with the best
eval success (then return). The head camera renders at `--render-size` (320 x 240) and only on the
two control steps that feed the next replan.

Live curves: mirror the runs into TensorBoard on the training host and forward the port.

```bash
.venv/bin/python scripts/metrics_to_tensorboard.py logs/dsrl &
.venv/bin/tensorboard --logdir logs/dsrl --port 6007 --host 127.0.0.1 &
ssh -N -L 16007:127.0.0.1:6007 remote-26   # on the workstation, then open http://localhost:16007
```

## v3: imagined post-training with an object world model

`scripts/train_v3.py` trains the same noise policy mostly without rendering or physics. The frozen
S0 side becomes a model (`body/`): SuperMimic's body world model f_ω (the `wm` weights of the tracker
checkpoint, imported from `../supermimic-reproduced`) driven by the tracker through a reference
(Φ, `body/imagine.py`) and the gripper / head-camera FK of its states (φ, `body/kinematics.py`).
An object world model (`object_wm/model.py`) rolls the can over a chunk from the right gripper's
motion and commanded opening; the imagined chunk return is the change of the task potential of
`task.py` over the whole 32-frame reference (`rl/imagine.py`). Each cycle: one simulated rollout
(`sim/collect.py`, 25 % of the envs on N(0, I) noise, failures kept) into a recency-weighted buffer
(`object_wm/buffer.py`); object-WM steps on windows conditioned on simulated or Φ gripper poses
(labels always simulated); a ranking check (identical sim states, candidates executed in sim vs
imagined, Kendall τ) that halves / doubles the imagined updates per sim rollout; GRPO updates on
recorded decisions × imagined candidates.

```bash
TRACKER=../checkpoints/supermimic_0100000.pt \
DIFFUSION=../../UMI-Diffusion/Humanoid-Diffusion/outputs/<run>/best.pt \
OUTPUT_DIR=logs/v3/run1 bash scripts/train_v3_8gpu.sh --init-noise-checkpoint logs/dsrl/<run>/best.pt
# How well Φ predicts the robot here (vs the perfectly tracked reference):
.venv/bin/python -m can_pnp.scripts.check_body_wm --tracker-checkpoint $TRACKER --diffusion-checkpoint $DIFFUSION
```

`metrics.jsonl` per cycle: `sim/*` (the current policy in the simulator) and `sim_base/*`,
`wm_fresh/*` (object-WM error on the newest rollout before training on it, simulated vs Φ body),
`wm/*`, `rank/tau` and the clone spread of the ranking groups, `imag/*`, `grpo/*` and timings.

## Not ported

Stage logic, rewards, the SONIC decoder / residual composer, and the non-scene domain randomization
(mass, friction, PD gains, joint bias, camera, photometric) of the residual-RL task.

## Layout

```
src/can_pnp/
  config.py        scene geometry, head camera and layout parameters
  constants.py     joint order, deploy pose, gripper actuator and opening <-> joint mapping
  entities.py      robot (mjlab G1 + ALOHA), cans, table, box, room; lights
  layout.py        per-episode layout sampling + reset event
  mdp.py           object/goal observations, fall and drop terminations
  env_cfg.py       ManagerBasedRlEnvCfg (CanPnpEnvCfgArgs, make_env)
  supermimic.py    tracker loading and goal features
  pipeline.py      reference sources and SuperMimicController
  odometry.py      reference root height / velocity from leg kinematics (full tracker)
  dsrl.py          DSRL noise policy and the steered diffusion source
  task.py          task flags and episode potential (sim and imagination)
  rl/              grpo.py (advantages, PPO-clip update, DDP), imagine.py (imagined chunk returns)
  sim/             rollouts.py (GRPO episodes), collect.py (v3 data collection, ranking check)
  body/            world_model.py (f_ω), imagine.py (Φ), kinematics.py (φ)
  object_wm/       model.py (object world model, loss), buffer.py (recent rollouts)
  scripts/         check_scene.py (can-pnp-check), run_pipeline.py (can-pnp-run),
                   train_dsrl.py (can-pnp-dsrl), train_v3.py, check_body_wm.py
  assets/          robots/g1_aloha (generated), scene meshes/texture, source/ (Isaac + adapter20 inputs)
scripts/           build_assets.py, fetch_wheels.sh, train_dsrl_8gpu.sh, train_v3_8gpu.sh
tests/             scene geometry, layout sampling, goal features, tracker loading, body WM, v3 pieces
```
