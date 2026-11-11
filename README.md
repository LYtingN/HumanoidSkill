# HumanoidSkill

Humanoid skill learning and world-model experiments using
[SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced).

```text
humanoid_skill/
├── pyproject.toml, uv.lock       # workspace configuration and shared dependency lock
├── hskill/src/hskill/
│   ├── embodiment/              # G1 + ALOHA constants, construction and robot assets
│   ├── s0/                      # SuperMimic tracker, body world model and kinematics
│   ├── s1/                      # reference sources and DSRL noise policy
│   ├── executors/               # body imagination (Φ) and imagined chunk returns
│   ├── interaction/             # object world model
│   ├── learners/                # GRPO
│   ├── data/                    # recorded episodes and replay buffer
│   ├── sim/                     # rollouts and collection with supplied task callbacks
│   └── utils/                   # colors and plan visualization
├── tasks/can_pnp/               # scene, assets, task scoring, grasp starts and task tools
├── experiments/                # train_dsrl, train_v3, check_body_wm, view_policy
│   └── launch/                 # eight-GPU launch scripts
├── tools/                      # wheel download, HyperLux setup and TensorBoard conversion
├── supermimic-reproduced/       # separate package and environment; outside the workspace
└── wheels/, data/, logs/        # local files, excluded from Git
```

Dependencies run from `experiments` through `can_pnp` to `hskill` and SuperMimic.
The experiment entry points supply the scene factory, grasp starts, layout lookup and task
tracker to the core. SuperMimic and Humanoid-Diffusion continue to load from their checkouts.
SuperMimic keeps its own cu130 environment; this workspace uses cu126.

## Setup

Run these commands from `humanoid_skill/`:

```bash
bash tools/fetch_wheels.sh
export UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple
uv sync --all-extras
export TMPDIR="$HOME/.cache/hskill-reorg/tmp"
export TORCHINDUCTOR_CACHE_DIR="$HOME/.cache/hskill-reorg/torchinductor"
mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR"
uv run pytest -q
```

The workspace uses one `.venv` and `uv.lock` at its root. Dependency versions, the mjlab Git
revision and the CUDA 12 wheel indexes match the former `can-pnp` environment. The `diffusion`
extra contains timm, diffusers and torchvision. HyperLux remains optional; install it with
`bash tools/setup_hyperlux.sh` when that renderer is needed.

On the local workstation with the NVIDIA 570 kernel module and newer user-space libraries,
prefix GPU commands with `LD_LIBRARY_PATH=$HOME/.local/lib/nvidia-570.172.18:$LD_LIBRARY_PATH`,
as described in the [task documentation](tasks/can_pnp/README.md).

## Run

All commands below run from the workspace root. The existing `can-pnp-check`, `can-pnp-run`
and `can-pnp-dsrl` commands remain available.

```bash
uv run python -m experiments.view_policy --help
uv run can-pnp-check --help

TRACKER=../supermimic_0100000.pt
DIFFUSION="../Diffusion/Humanoid-Diffusion/outputs/<run>/best.pt"

uv run --extra diffusion python -m experiments.train_dsrl \
  --tracker-checkpoint "$TRACKER" --diffusion-checkpoint "$DIFFUSION"
uv run --extra diffusion python -m experiments.train_v3 \
  --tracker-checkpoint "$TRACKER" --diffusion-checkpoint "$DIFFUSION"
uv run --extra diffusion python -m experiments.check_body_wm \
  --tracker-checkpoint "$TRACKER" --diffusion-checkpoint "$DIFFUSION"
uv run --extra diffusion can-pnp-run --source dsrl --start grasp \
  --tracker-checkpoint "$TRACKER" --diffusion-checkpoint "$DIFFUSION" \
  --noise-checkpoint can-pnp/logs/dsrl/latest_view/last.pt --viewer

TRACKER="$TRACKER" DIFFUSION="$DIFFUSION" \
  bash experiments/launch/train_dsrl_8gpu.sh
TRACKER="$TRACKER" DIFFUSION="$DIFFUSION" \
  bash experiments/launch/train_v3_8gpu.sh
```

The diffusion checkout defaults to `../Diffusion/Humanoid-Diffusion`, resolved from the
experiment file. The body world model defaults to `supermimic-reproduced/` in this workspace.
Launch scripts keep static rendezvous at `127.0.0.1`, ports 29531 and 29541 respectively.

Existing logs and checkpoints stay at their original paths, including the local
`can-pnp/logs/` directory. New runs use the CLI's unchanged `logs/...` defaults relative to
the working directory. Scene assets live under `tasks/can_pnp/src/can_pnp/scene/assets/`;
generated robot assets live under `hskill/src/hskill/embodiment/assets/`. Rebuild them with
`uv run python tasks/can_pnp/scripts/build_assets.py`.

See the [task documentation](tasks/can_pnp/README.md),
[migration verification](docs/REORG_RESULTS.md), and the separate
[SuperMimic README](supermimic-reproduced/README.md). Its MIT notice is retained in
[LICENSE](supermimic-reproduced/LICENSE).
