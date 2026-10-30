# 给 Codex 的任务：humanoid_skill 代码排版重构（只搬代码，不改算法）

## 背景

`humanoid_skill/` 是一个人形机器人分层控制的研究仓库：

- **S0**：SuperMimic 全身跟踪器和它的身体世界模型，代码在 `supermimic-reproduced/`（独立的包 `supermimic`，有自己的环境）。
- **S1**：diffusion policy（DiT），代码在外部仓库 `../Diffusion/Humanoid-Diffusion`，通过 `sys.path` 导入。
- **can-pnp**：一个任务场景（G1 + ALOHA 夹爪抓罐子放进箱子），但目前几乎所有核心算法也都写在 `can-pnp/src/can_pnp/` 里。

本次任务**只做代码排版**：把核心算法从任务场景里分离出来，放到与任务无关的包里。
**不新增任何算法，不修改任何算法逻辑、数值、默认参数。** 新算法之后由另一位开发者写进新结构。

## 目标结构

```text
humanoid_skill/
├── pyproject.toml            # 新：uv workspace 根，成员为 hskill 和 tasks/can_pnp；唯一的 .venv 和 uv.lock
├── README.md                 # 更新：新结构与运行方式
├── docs/                     # 已有：ARCHITECTURE_PLAN.md、本文
├── supermimic-reproduced/    # 不动（不加入 workspace，保持自己的环境和 cu130 设置）
├── hskill/                   # 新：与任务无关的核心包
│   ├── pyproject.toml
│   ├── src/hskill/
│   │   ├── embodiment/       # 机器人本体：G1 + ALOHA
│   │   ├── s0/               # S0 适配
│   │   ├── s1/               # S1 适配
│   │   ├── executors/        # 身体推演（Φ）与参考运动学
│   │   ├── interaction/      # 物体世界模型
│   │   ├── learners/         # GRPO、critic
│   │   ├── data/             # 回合数据与 buffer
│   │   ├── sim/              # 仿真 rollout / 采集
│   │   └── utils/
│   └── tests/
├── tasks/
│   └── can_pnp/              # 由 can-pnp/ 收缩而来：只剩场景与任务定义
│       ├── pyproject.toml
│       ├── src/can_pnp/
│       │   ├── scene/        # config、entities（非机器人部分）、layout、mdp、env_cfg、assets/
│       │   ├── task_spec.py  # 原 task.py
│       │   ├── starts.py     # 原 grasp_start.py
│       │   └── scripts/      # check_scene.py
│       ├── scripts/          # build_assets.py、extract_grasp_starts.py（任务专属工具）
│       └── tests/            # test_scene.py
├── experiments/              # 新：可运行的实验入口
│   ├── train_dsrl.py
│   ├── train_v3.py
│   ├── check_body_wm.py
│   ├── view_policy.py        # 原 run_pipeline.py
│   └── launch/               # train_dsrl_8gpu.sh、train_v3_8gpu.sh
├── tools/                    # metrics_to_tensorboard.py、fetch_wheels.sh、setup_hyperlux.sh
├── wheels/                   # 由 can-pnp/wheels 移来（不进 Git）
└── data/ logs/               # 本地，不进 Git
```

**依赖方向只能向下：** `experiments → can_pnp → hskill → supermimic`。

- `hskill` 内部不允许出现任何 `can_pnp` 的 import（迁移完成后 `grep -rn "can_pnp" hskill/` 必须为空）。
- `tasks/can_pnp` 里不保留任何算法代码。

## 文件迁移对照表

当前路径都相对于 `can-pnp/src/can_pnp/`。

| 当前 | 目标 | 说明 |
|---|---|---|
| `constants.py` 中关节名、默认姿态、夹爪开度与关节映射、`ROBOT_XML`、`RIGHT_FINGER_BODIES` | `hskill/embodiment/g1_aloha.py` | 机器人本体相关 |
| `constants.py` 中其余（场景路径等） | `can_pnp/scene/` | — |
| `entities.py` 中机器人构建（`robot_cfg`、`_robot_spec`、`_hand_gravcomp`、颜色等） | `hskill/embodiment/g1_aloha.py` 或同目录模块 | 机器人本体相关 |
| `entities.py` 中罐子、桌子、箱子、房间、灯光 | `can_pnp/scene/entities.py` | — |
| `scripts/build_assets.py` 中生成机器人资产的部分 | 留在 `tasks/can_pnp/scripts/`，保持可用即可 | 不要求拆分 |
| `assets/robots/` | 跟随 embodiment，或保留在原处并由 embodiment 引用 | 选更简单的，保证路径正确 |
| `assets/scene/` `assets/source/` `assets/grasp_starts.json` | `can_pnp/scene/assets/` 或原位 | — |
| `config.py` `layout.py` `mdp.py` `env_cfg.py` | `can_pnp/scene/` | — |
| `grasp_start.py` | `can_pnp/starts.py` | — |
| `task.py` | `can_pnp/task_spec.py` | 内容不改 |
| `supermimic.py` `odometry.py` | `hskill/s0/` | — |
| `body/world_model.py` | `hskill/s0/body_world_model.py` | 其中导入 supermimic 的默认路径要随新位置更新 |
| `body/kinematics.py` | `hskill/s0/kinematics.py` | — |
| `body/imagine.py` | `hskill/executors/` | — |
| `pipeline.py` 中 `SuperMimicController`、`reference_plan`、`gripper_opening`、`gripper_q` | `hskill/s0/controller.py` | — |
| `pipeline.py` 中 `load_diffusion_policy`、`DiffusionSource`、`DatasetSource`、`StandSource` | `hskill/s1/sources.py` | — |
| `dsrl.py` | `hskill/s1/noise_policy.py` | — |
| `plan_viz.py` | `hskill/utils/` 或 `experiments/` 旁 | 只用于可视化 |
| `rl/grpo.py` | `hskill/learners/grpo.py` | — |
| `rl/critic.py` | `hskill/learners/critic.py` | — |
| `rl/imagine.py` | `hskill/executors/imaginer.py` | — |
| `object_wm/model.py` | `hskill/interaction/model.py` | — |
| `object_wm/buffer.py` | `hskill/data/buffer.py` | — |
| `sim/rollouts.py` `sim/collect.py` | `hskill/sim/` | 它们目前直接引用了 can_pnp 的场景与任务（见下文"难点"） |
| `scripts/train_dsrl.py` `train_v3.py` `check_body_wm.py` `run_pipeline.py` | `experiments/` | `run_pipeline.py` 改名为 `view_policy.py` |
| `scripts/check_scene.py` | `can_pnp/scripts/` | 任务专属 |
| `can-pnp/scripts/*.sh`、`metrics_to_tensorboard.py` 等 | `experiments/launch/`、`tools/` | — |
| `can-pnp/tests/` | 按被测代码的位置分到 `hskill/tests/` 和 `tasks/can_pnp/tests/` | — |

## 难点：`hskill/sim` 和 `hskill/executors` 目前依赖任务代码

`sim/rollouts.py`、`sim/collect.py`、`rl/imagine.py` 里直接 import 了 can_pnp 的 `make_env`、`CanPnpSceneCfg`、`layout_term`、`GraspStarts`、`TaskTracker` 等。

本次**不要**为它们设计新的抽象接口（这属于算法设计，之后由别人做）。请用最小改动保证依赖方向：

- **做法**：把这些依赖改成**由调用方传入**（构造函数参数或回调），默认值由 `experiments/` 的入口脚本从 `can_pnp` 里取出后传进去。
- 例如 `Rollouts` 现在内部调用 `make_env(CanPnpEnvCfgArgs(...))`，改为接收一个已经构造好的 env（或一个 `make_env` 函数），以及 `starts`、`task_cfg`、`layout` 读取函数等。
- 函数体内部的逻辑一行都不改，只改"东西从哪来"。
- 如果某处确实无法在不改逻辑的前提下解耦，在 PR 说明里列出，暂时保留在 `tasks/can_pnp/` 并加 `# TODO(decouple)` 注释，**不要自行发明抽象**。

## 环境

- 根目录建 uv workspace：`[tool.uv.workspace] members = ["hskill", "tasks/can_pnp"]`。
- 把 `can-pnp/pyproject.toml` 里的依赖、`override-dependencies`、`[tool.uv.sources]`（mjlab 的 git 版本、torch / torchvision 的 cu126 源、warp-lang 的本地 flat 索引）和 `[[tool.uv.index]]` 整体搬到根 `pyproject.toml`；**所有版本钉死保持不变**。
- `hskill` 的 `pyproject.toml` 只声明它直接 import 的依赖，不要写任何 `[tool.uv.sources]`，避免和根冲突。
- `supermimic-reproduced` **不加入 workspace**。`hskill/s0/body_world_model.py` 里继续用 `sys.path` 导入 supermimic，只更新默认路径。原因：它的 torch 源是 cu130，和这里的 cu126 冲突。
- `diffusion` 这个 extra（timm、diffusers、torchvision）保留。
- 原来 `can-pnp/.venv` 里手动打过 HyperLux 补丁的 mjlab 不需要迁移；新环境用 `uv sync --all-extras` 生成即可。HyperLux 是可选后端。

## 必须保持不变的东西

1. 所有函数、类的名字、参数、默认值和逻辑（只允许改 import 路径，以及上面"难点"里描述的依赖注入）。
2. 命令行参数名与默认值，除了以下路径默认值：
   - `diffusion_repo` 默认值随新位置调整，仍指向 `/home/eason_er/nyx/Humanoid_Posttraining/Diffusion/Humanoid-Diffusion`；
   - supermimic 的默认路径随新位置调整，仍指向 `humanoid_skill/supermimic-reproduced`。
3. `can_pnp` 资产的加载路径（场景 XML、网格、`grasp_starts.json`）必须仍然有效。
4. 远端多卡启动脚本里的静态 rendezvous 写法：`--master-addr 127.0.0.1 --master-port <固定端口>`。不要改回 `--standalone` 或 c10d。

## 不要做的事

- 不要新增算法、抽象基类或协议（`hskill/interfaces.py` 由别人写）。
- 不要改 `supermimic-reproduced/` 的任何文件。
- 不要改 `Humanoid-Diffusion` 仓库。
- 不要动任何 `logs/`、`data/`、checkpoint，也不要同步到远端服务器。远端 remote-26 上正在用旧结构跑训练。
- 不要 push。

## 工作方式

- 新建分支 `refactor/layout`，每完成下面一步提交一次，提交说明写清楚做了什么。
  1. 建 workspace 与空包骨架；
  2. 迁移 embodiment；
  3. 迁移 s0 / s1；
  4. 迁移 executors / interaction / learners / data / sim（含依赖注入）；
  5. 收缩 `tasks/can_pnp`；
  6. 迁移 experiments / tools，更新 README，删除空的旧 `can-pnp/` 目录。
- 本地临时目录：根分区容易满，跑测试和编译时设置 `TMPDIR` 和 `TORCHINDUCTOR_CACHE_DIR` 到 `/home/eason_er/.cache/` 下。

## 验收

在**迁移前**先跑一遍下面的命令记下输出，迁移后再跑一遍，逐项对比。路径：

- tracker：`/home/eason_er/nyx/Humanoid_Posttraining/supermimic_0100000.pt`
- diffusion：`/home/eason_er/nyx/Humanoid_Posttraining/Diffusion/Humanoid-Diffusion/outputs/can_pnp_supermimic_8xa800_bs128_n2_compile_20261027_074701/best.pt`
- 噪声策略：`can-pnp/logs/dsrl/latest_view/last.pt`（迁移后若移动了 logs，用新路径）

1. **测试**：迁移前 `can-pnp` 下 `pytest -q tests` 共 21 项通过；迁移后在根目录跑全部测试，仍是 21 项通过（GPU 测试需要 CUDA）。
2. **依赖方向**：`grep -rn "can_pnp" hskill/` 为空。
3. **DSRL 冒烟**：用 `train_dsrl` 从上面的噪声策略恢复，`--groups 4 --group-size 8 --episode-s 14 --no-compile --iterations <起始轮+1> --eval-every 0`，跑 1 轮。迁移前后的 `train/*` 指标一致；GPU 物理可能有浮点级的非确定性，差异应在同一量级内。
4. **v3 冒烟**：`train_v3` 用 `--num-envs 32 --episode-s 3 --no-compile --wm-warmup-rollouts 1 --wm-warmup-steps 20 --wm-steps 10 --wm-batch 256 --critic-warmup-steps 20 --critic-steps 10 --critic-batch 256 --imag-starts 16 --imag-group 4 --imag-iters 2 --rank-group 4 --rank-max-delay 1 --cycles 3 --collect-every 2`，能完整跑完且无报错。
5. **Φ 精度**：`check_body_wm --episodes 1 --num-envs 16` 的误差表迁移前后一致（同量级）。
6. **可视化**：`view_policy` 能起来（有显示器时），或至少 `--help` 能运行。
7. **启动脚本**：`bash -n experiments/launch/*.sh` 通过，里面的路径都指向新位置。

## 交付

- 分支上的 6 个提交；
- 一份简短说明：迁移对照的实际结果（与上表不同之处）、所有 `TODO(decouple)` 的位置和原因、验收 1–7 的结果。
