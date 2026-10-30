# HumanoidSkill 代码结构规划与任务拆分指南

本文是重构前的目标结构与任务清单。
原则一句话：**核心算法与任务场景分离；S0、S1、核心算法、具体任务各在一层，依赖只向下。**

分工：代码排版（第 5 节阶段 1 的纯迁移）交给 Codex，指令见 [CODEX_REORG_TASK.md](CODEX_REORG_TASK.md)；
层间协议、通用采集循环和新 pipeline（阶段 2–5）由 Claude 在排版完成后写进新结构。

---

## 1. 现状与问题

```text
humanoid_skill/
├── supermimic-reproduced/   # S0：SuperMimic tracker + 身体 WM 的训练（独立包 supermimic）
├── can-pnp/                 # 一个任务场景，但里面塞了几乎所有核心算法
│   └── src/can_pnp/
│       ├── 场景：config.py constants.py entities.py layout.py mdp.py env_cfg.py grasp_start.py assets/
│       ├── S0 适配：supermimic.py odometry.py body/（f_ω、Φ、φ）pipeline.py 的 SuperMimicController
│       ├── S1 适配：pipeline.py 的 reference source、dsrl.py（噪声策略）
│       ├── 核心算法：rl/（GRPO、critic、想象打分）object_wm/（交互模型、buffer）sim/（采集、排序校验）
│       ├── 任务定义：task.py（势函数、成功判定）
│       └── 入口：scripts/train_dsrl.py train_v3.py check_body_wm.py run_pipeline.py
└── data/                    # 本地数据（不进 Git）
```

问题：

1. 换一个任务（比如开抽屉、搬箱子）就得复制整个 can-pnp，核心算法无法复用。
2. 机器人本体（G1 + ALOHA 夹爪：关节名、夹爪偏移、重力补偿、相机挂载）和任务混在一起。
3. 想落地的新 pipeline（理想/实际双执行者 + 跨层归因 + S0 适应）没有合适的位置放。
4. supermimic（torch cu130）和 can-pnp（torch cu126）是两个环境，can-pnp 只能靠 `sys.path` 导入 supermimic。

---

## 2. 目标结构

```text
humanoid_skill/
├── pyproject.toml              # uv workspace 根：唯一的环境与锁文件，统一 torch 源
├── README.md
├── docs/
│   ├── ARCHITECTURE_PLAN.md    # 本文
│   └── ideas/                  # idea 文档（反事实跨层归因等）
│
├── supermimic/                 # 【S0 层】原 supermimic-reproduced，训练 tracker 与身体 WM
│   └── src/supermimic/         #   基本保持上游结构，只新增"外部参考动作输入"和"任务加权跟踪"的扩展点
│
├── hskill/                     # 【核心算法层】与任务无关、与具体场景无关
│   └── src/hskill/
│       ├── interfaces.py       # 各层之间的协议（见第 3 节），所有模块只依赖这里的抽象
│       ├── embodiment/         # 机器人本体描述：G1 + ALOHA 的关节、默认姿态、夹爪映射、手/相机挂载偏移
│       ├── s0/                 # S0 适配：tracker 加载、目标特征、腿部里程计、身体 WM、Φ（PhiStep）、φ（正运动学）、控制器
│       ├── s1/                 # S1 适配：diffusion 加载、reference source、噪声策略（DSRL）
│       ├── executors/          # 执行者：实际（Φ）、理想（全身/仅手/逐部件替换）、仿真真实执行、仿真"木偶"执行的统一接口
│       ├── interaction/        # 交互模型 M（手 → 物体）、损失
│       ├── attribution/        # 跨层归因：J*、J、Δh、标签（意图错/S0 可修/不可行）、误差预算
│       ├── learners/           # 学习器：噪声空间 GRPO、critic、S0 适应（任务加权跟踪）、两层调度
│       ├── data/               # 任务无关的回合记录格式、buffer、采样
│       ├── sim/                # 通用的仿真采集循环（通过 TaskSpec 与具体任务对接）
│       └── utils/              # 分布式、NaN 保护、日志/TensorBoard
│
├── tasks/                      # 【任务层】每个任务一个包，只描述"场景 + 任务"
│   └── can_pnp/
│       └── src/can_pnp/
│           ├── scene/          # mjlab 场景：config、entities、layout、mdp、env_cfg、assets
│           ├── task_spec.py    # 实现 TaskSpec：物体状态、上下文、势函数、成功/失败判定
│           ├── starts.py       # 起点（grasp starts 等）
│           └── scripts/        # 只属于这个任务的工具：build_assets、extract_grasp_starts、check_scene
│
├── experiments/                # 【实验层】把核心 + 任务组合成可运行的实验
│   ├── train_dsrl.py           # 纯仿真 DSRL
│   ├── train_v3.py             # 想象 + 物体 WM
│   ├── diagnose_attribution.py # 新：双执行者诊断 + 误差预算（只诊断，不更新）
│   ├── coadapt.py              # 新：按归因分派的两层联合训练
│   ├── check_body_wm.py        # Φ 精度
│   ├── view_policy.py          # 原 run_pipeline：可视化
│   ├── configs/                # 每个实验的配置（任务名、检查点路径、超参）
│   └── launch/                 # 8 卡启动脚本（静态 rendezvous，见第 6 节）
│
├── tools/                      # 与算法无关的工具：metrics_to_tensorboard、fetch_wheels、setup_hyperlux
└── data/ logs/ checkpoints/    # 本地，不进 Git
```

**依赖方向（只能向下，禁止反向引用）：**

```text
experiments ──► tasks/* ──► hskill ──► supermimic
                    └────────► hskill.interfaces
```

- `hskill` 不允许 import 任何 `tasks.*`；
- `tasks/can_pnp` 不放任何算法，只实现 `hskill.interfaces` 里的协议；
- `supermimic` 不知道 `hskill` 的存在（保持可以单独训练 S0）；
- Humanoid-Diffusion（S1 的 DiT）仍是外部仓库，只通过 `hskill/s1` 适配。

---

## 3. 层间协议（`hskill/interfaces.py`）

只定义"谁向谁提供什么"，不写实现。每个协议都对应 pipeline 里的一个方框。

| 协议 | 职责 | 由谁实现 |
|---|---|---|
| `Embodiment` | 关节顺序、默认姿态、夹爪开度与关节映射、手/相机在身体上的挂载偏移 | `hskill/embodiment/g1_aloha` |
| `TaskSpec` | 从环境读物体状态和场景上下文；势函数 R；成功、失败（掉落/摔倒）判定；起点采样 | `tasks/can_pnp/task_spec.py` |
| `S1Policy` | 给定观测，提出 1 个或 K 个意图（reference）；可选：返回可学习部分的 log prob | `hskill/s1` |
| `Executor` | 给定 reference 和起点状态，返回身体/手的轨迹 | `hskill/executors`（实际 Φ、理想、仿真真实、仿真木偶） |
| `InteractionModel` | 给定手的轨迹和夹爪指令，预测物体轨迹 | `hskill/interaction` |
| `Attributor` | 对同一个 reference 跑理想和实际两个分支，输出 J*、J、Δh 和归因标签 | `hskill/attribution` |
| `Learner` | 接收带标签的样本，更新某一层（S1 或 S0） | `hskill/learners` |

**关键约束（来自 idea）：** `Executor` 的输出必须显式包含"手的轨迹"，这是两个分支做替换的接口。

---

## 4. 现有文件迁移对照表

| 现在（`can-pnp/src/can_pnp/`） | 迁移到 | 备注 |
|---|---|---|
| `constants.py` 里关节名、默认姿态、夹爪映射 | `hskill/embodiment/g1_aloha.py` | 场景几何常量留在任务里 |
| `entities.py` 里机器人与夹爪重力补偿 | `hskill/embodiment/` | 桌子、罐子、箱子留在任务里 |
| `config.py` `layout.py` `mdp.py` `env_cfg.py` `assets/` | `tasks/can_pnp/scene/` | — |
| `grasp_start.py` | `tasks/can_pnp/starts.py` | — |
| `task.py` | `tasks/can_pnp/task_spec.py` | 实现 `TaskSpec` |
| `supermimic.py` `odometry.py` | `hskill/s0/` | — |
| `body/`（world_model、imagine、kinematics） | `hskill/s0/`（f_ω、Φ）+ `hskill/executors/`（实际、理想） | 夹爪偏移改为从 `Embodiment` 读 |
| `pipeline.py` 的 `SuperMimicController` | `hskill/s0/controller.py` | 观测拼接改为从 `Embodiment` 读 |
| `pipeline.py` 的 reference source | `hskill/s1/sources.py` | — |
| `dsrl.py` | `hskill/s1/noise_policy.py` | — |
| `rl/grpo.py` `rl/critic.py` | `hskill/learners/` | — |
| `rl/imagine.py` | `hskill/executors/` + `hskill/attribution/` | 想象打分即"实际分支" |
| `object_wm/model.py` | `hskill/interaction/` | 输入只依赖手的轨迹 + 上下文 |
| `object_wm/buffer.py` | `hskill/data/` | — |
| `sim/rollouts.py` `sim/collect.py` | `hskill/sim/`（通用循环）+ `tasks/can_pnp/`（读物体、接触的适配） | 排序校验 → `hskill/attribution`（模型误差项） |
| `scripts/train_dsrl.py` `train_v3.py` `check_body_wm.py` `run_pipeline.py` | `experiments/` | `check_scene.py` 留在任务里 |
| `scripts/*.sh` `metrics_to_tensorboard.py` 等 | `experiments/launch/`、`tools/` | — |

---

## 5. 给 Codex 的任务拆分

分阶段，每个任务是一个独立的 PR。**前两个阶段只搬代码、不改行为**，便于验证。

### 阶段 0：统一环境

**T0. 建立 uv workspace**
- 内容：根目录 `pyproject.toml` 声明成员 `supermimic`、`hskill`、`tasks/can_pnp`；在根上统一 torch / warp 的源（remote-26 驱动只支持 CUDA ≤ 12.8，用 cu126）；去掉 can-pnp 里用 `sys.path` 导入 supermimic 的做法。
- 验收：`uv sync` 一次成功；三个包的现有测试全部通过。
- **需要先确认**：supermimic 改用 cu126 后，S0 的训练速度和数值是否不受影响。

### 阶段 1：纯迁移（不改行为）

**T1. 抽出 `hskill/embodiment`**：迁出关节、默认姿态、夹爪映射、挂载偏移、重力补偿。
**T2. 抽出 `hskill/s0` 和 `hskill/s1`**：迁出 tracker、目标特征、里程计、身体 WM、Φ、φ、控制器；diffusion 加载、reference source、噪声策略。
**T3. 把 can-pnp 收缩为纯任务包**：只保留 scene、task_spec、starts、任务工具。
**T4. 迁移入口到 `experiments/`**：train_dsrl、train_v3、check_body_wm、view_policy、启动脚本。

阶段 1 每个任务的验收：
- 现有 21 项测试全部通过；
- 固定随机种子后，`train_dsrl` 的冒烟运行（小规模、1 轮）指标与迁移前一致；
- `check_body_wm` 的输出数字与迁移前一致；
- `hskill` 内 `grep -r "can_pnp"` 无结果。

### 阶段 2：接口化

**T5. 写 `hskill/interfaces.py`**，让 `hskill/sim`、`learners`、`interaction` 只依赖协议。
**T6. 通用采集循环**：`hskill/sim` 不再写死罐子、箱子、桌子，全部通过 `TaskSpec` 读取。
- 验收：写一个只有一个方块的玩具 `TaskSpec`，能跑通采集循环（不要求能学会）。

### 阶段 3：新 pipeline（诊断优先）

**T7. 仿真"木偶"执行者**：让机器人身体按 reference 运动学驱动，物体照常做物理仿真，用来提供理想分支的真值。放在 `hskill/executors`，场景相关的部分通过 `TaskSpec` / `Embodiment` 获取。
**T8. 理想执行者的三种定义**：全身、仅手（身体用实际，手相对躯干用 reference）、逐部件替换。
**T9. 归因模块 + 误差预算**：算 J*、J、Δh，打标签，记录"意图 / 执行损失 / 模型误差"三项。
**T10. `experiments/diagnose_attribution.py`**：拿 DSRL 当前策略只做诊断、不做更新，输出各类失败的占比。
- 验收：对归因为"S0 可修"的失败，换成木偶执行后成功率明显提高；对归因为"意图错"的失败，木偶执行也救不回来。这是**干预检验**。

### 阶段 4：S0 适应

**T11. supermimic 的扩展点**：①训练参考动作可以来自外部（S1 生成的 reference）；②跟踪 reward 支持按部位、按时间加权；③支持在有负载的场景里训练；④适应后同步重拟合 f_ω。
**T12. `hskill/learners/s0_adapt.py`**：从归因模块拿到"S0 可修"的样本，按任务敏感度生成权重，调用 T11。
- 验收：在"S0 可修"的那批 reference 上，适应后执行损失下降；在原始动作数据上的跟踪误差不明显变差。

### 阶段 5：联合训练

**T13. `experiments/coadapt.py`**：两个时间尺度交替，按归因把样本分给 S1 或 S0，每次 S0 更新后刷新 Φ。
- 验收：与"只改 S1（DSRL）""只改 S0""人为交替"三种基线比较成功率和样本效率。

---

## 6. 给 Codex 的通用约束

写进每个任务的说明里：

1. **迁移任务不改行为**：只搬代码、改 import；同时改行为的，拆成两个 PR。
2. **依赖方向**：`hskill` 不能 import `tasks`；任务包里不放算法。
3. **不进 Git**：`data/`、`logs/`、`checkpoints/`、`.venv/`、`wheels/`。
4. **远端（remote-26）多卡启动用静态 rendezvous**：`torchrun --master-addr 127.0.0.1 --master-port <固定端口>`，不要用 `--standalone` 或 c10d（那里 localhost 只解析到 ::1，临时端口还会被占满）。
5. **本地临时文件**：根分区容易满，`TMPDIR`、`TORCHINDUCTOR_CACHE_DIR` 放到 `/home` 下。
6. **每个 PR 附上验收命令和结果**：测试、冒烟运行、关键指标对比。
7. **配置不写死路径**：checkpoint、diffusion 仓库路径都走 `experiments/configs/`。

---

## 7. 需要你先拍板的问题

1. **统一环境用哪个 torch 源**：建议 cu126，因为 remote-26 驱动是 550；需要确认 supermimic 在 cu126 下训练正常。
2. **supermimic 是否保持"可独立复现上游"**：建议保持，所有扩展都做成可选开关，默认行为与上游一致。
3. **包名**：`hskill` 只是占位，可以换成你想要的名字。
4. **Humanoid-Diffusion 是否也纳入 workspace**：建议暂时不纳入，只通过 `hskill/s1` 适配。
