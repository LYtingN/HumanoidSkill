# 跨层归因训练架构（v3：双分支 + 物体世界模型 + progress critic）

本文记录当前实现的算法架构：S1（action head）与 S0（全身控制器）在任务结果空间里对齐，
每次失败归因到"意图"或"执行"，学习信号分派给负责的那一层。
代码入口：`experiments/train_v3.py`；核心模块在 `hskill/`；任务场景在 `tasks/can_pnp/`。

最后更新：2026-10-30。

---

## 1. 问题与思想

- **S1**：冻结的 DiT diffusion policy（DINOv3 视觉 + 本体状态）+ 可训练的噪声策略 μθ（DSRL）。
  每 8 个控制步重新规划一次，输出 reference `r`：32 帧的关节空间轨迹，含两个夹爪的开度。
- **S0**：冻结的 SuperMimic tracker，50 Hz 跟踪 `r`；夹爪开度直接按 `r` 执行，不经过 tracker。
- 两层各自可用，但 S1 的一部分 reference 落在 S0 训练分布的边缘，执行会走样；任务失败时分不清该修哪一层。

**核心思想**：对同一个 `r` 评估两次——"S0 实际会做到什么"与"这个意图本身值多少"，两者之差就是
执行丢掉的价值。对齐的标准是任务结果（手在关键时刻的位置、物体最终状态），不是关节轨迹处处一致。

## 2. 符号

| 符号 | 含义 |
|---|---|
| `Φ` | S0 的模型：tracker + SuperMimic 身体世界模型 f_ω 交替推演。`Φ(r, s)` = S0 执行 `r` 时的身体运动 |
| `φ` | 正运动学（无参数）：71 维身体状态 → 手腕、夹爪、躯干、相机位姿 |
| `M` | 物体世界模型（3 个成员的集成）：手的轨迹 + 夹爪指令 + 场景 → 物体轨迹；成员分歧 = 不确定度 |
| `h = φ(Φ(r))` | 实际分支的右手轨迹 |
| `h* = φ(Φ(r) 的身体 + r 的手臂关节)` | 理想分支的右手轨迹（"只在手上理想"：手臂完美跟踪，身体与平衡仍由 S0 决定） |
| `Ψ` | 任务势函数：完成 10 > 失败（掉落、摔倒）0 > exp(−d/0.1)，d 为物体到目标的距离 |
| `W` | progress critic：从某个状态起，回合最终 Ψ 按到回合结束的步数折扣后的值（完成 10·γ^步数，失败 0） |
| `J` | 实际分支的价值：chunk 内完成取 10、掉落取 0，否则取 `W(32 帧末)`；记录时减去 `Ψ(现在)` |
| `J*` | 理想分支的价值：同上，用 `h*` |

reward 就是每一步 Ψ 的变化；J 是从现在到回合结束的 reward 总和：一个 chunk 用想象，之后用 critic。

## 3. 误差分解

```
J_real  =  J*   −   (J* − J)   −   (J − J_real)
          意图      执行损失        模型误差
          (S1)       (S0)        (Φ、M、critic)
```

## 4. 整体流程

```
┌─ 数据 ──────────────────────────────────────────────────────────────────────┐
│ demo（LeRobot，只有本体，无物体状态）   仿真回合（8 卡 × 512 env × 14 s）    │
│   └→ 时间进度目标                         ├ 每步：实际右手 h、reference 右手 h_ref、│
│                                            │      夹爪指令、物体真值、场景      │
│                                            ├ 每个决策点：S1 观测 token、身体状态、│
│                                            │      tracker 观测、噪声、log prob   │
│                                            └ 每 4 个决策点一个 Φ 窗口（32 帧）   │
└───────────────┬─────────────────────────────┬─────────────────────────────────┘
                ▼                             ▼
┌─ 模型 ──────────────────────────────────────────────────────────────────────┐
│ W：demo 预训练 → demo + 回合混合训练   M：回合训练（真实夹爪 / Φ 夹爪各半）   │
│ Φ：冻结                                                                     │
└───────────────┬─────────────────────────────────────────────────────────────┘
                ▼
┌─ 评估（想象，不渲染、不跑物理）──────────────────────────────────────────────┐
│ 起点（buffer 里的真实决策点）× K 个噪声 → DiT 解码出 K 个 r                   │
│   实际分支：Φ → h → M → chunk 结果 / W → J，不确定度 u                       │
│   理想分支：Φ 的身体 + r 的手臂 → h* → M → J*，不确定度 u*                   │
└───────────────┬─────────────────────────────────────────────────────────────┘
                ▼
┌─ 学习 ──────────────────────────────────────────────────────────────────────┐
│ S1：仿真锚（回合真实回报的 GRPO）+ 想象 GRPO（按 J 打分，丢弃不确定候选）     │
│ S0：任务加权的跟踪适应（阶段 B，未实现）                                     │
└───────────────┬─────────────────────────────────────────────────────────────┘
                ▼
┌─ 校验 ──────────────────────────────────────────────────────────────────────┐
│ 排序校验：同状态的候选在仿真里真实执行一个 chunk                               │
│   τ：想象 chunk 结果 vs 仿真 chunk 结果（门控想象更新）                        │
│   S0 执行得准的候选：想象 J* 的 chunk 结果 vs 仿真结果（检查理想分支）          │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 5. 各组件

### 5.1 Φ 与 φ（S0 侧，冻结）

- 身体世界模型 f_ω 来自 tracker checkpoint 的 `wm`：输入 71 维身体状态（骨盆位姿、29 关节及速度）、
  29 维动作、33 维仿真参数 ξ，输出状态增量；另有终止头（骨盆高度、倾斜）。训练环境中没有物体。
- `PhiStep` 把"tracker 动作 → f_ω 一步 → 终止概率 → 观测历史"编译成一个函数。
- φ 只算骨盆 → 腰 → 躯干 → 双臂 → 手腕这条链（18 个身体），再加夹爪和相机的固定偏移。
- 精度：推 32 帧，右夹爪位置误差约 1.8 cm（比"假设完美跟踪 reference"准 3.4 倍）。

### 5.2 M：物体世界模型（集成）

- 物体状态 14 维：位置、四元数、线速度、角速度、右手手指实际开度。
- 每一步的输入：物体相对右夹爪的位姿和速度、开度、物体轴向、夹爪系下的重力、夹爪这一步的运动、
  这一步的指令开度、箱子和桌面相对物体的位置。不看图像，不看完整身体状态。
- 输出物体相对夹爪的变化量；零初始化时等于"物体被夹爪刚性夹住"。
- 3 个成员各自在自己的批次上训练；最终物体位置的成员间标准差作为不确定度。
  成员的权重堆叠在一起，用批量矩阵乘法一次算完（`EnsembleMLP`），推演一次只发一份 kernel；
  拟合 M 和 critic 时矩阵乘法用 TF32，想象和策略仍用完整 fp32。旧的分成员 checkpoint 由
  `ObjectWMEnsemble.stack_members` 转换。
- 训练窗口一半用仿真真实的夹爪轨迹，一半用 Φ 预测的夹爪轨迹；标签始终是仿真物体真值。
- 精度：推 32 步，末尾物体位置误差约 1.0–1.6 cm。

### 5.3 理想执行的真值从哪里来

**不依赖 demo 的物体状态**（demo 没有、以后也不会有物体状态），**不使用人造的执行者**。

- `M` 学的是"手这样动，物体会怎么动"，与这条手轨迹由谁产生无关，所以所有仿真回合都用来训练 `M`。
- 当 S0 把某个意图执行得足够准（右手相对 reference 的手的误差 < `well_tracked`，默认 2 cm）时，
  实际执行就是理想执行：这次的物体结果就是 `J*` 的真值。
- S1 探索到新的意图，S0 执行它，执行得准的部分就补上了 `M` 在这个意图附近的数据：
  **理想执行的数据随训练自己增长**。
- `M` 在没有数据的地方成员分歧大，这些候选不参与 S1 的更新。

### 5.4 W：progress critic（参考 RISE）

从零开始时策略没有成功回合，只从仿真回合学的 critic 学不到"放进箱子"的价值，想象回报没有区分度。
所以 critic 先在 demo 上预训练：demo 都在最后一帧完成任务，第 t 帧的目标是 `10 · γ^(T−1−t)`。

- **输入（70 维）**：
  - 本体（51 维，demo 和想象里都有）：29 个关节角、骨盆系下的重力、两个手腕在骨盆系下的位置和 6D 朝向、
    右手夹爪开度；
  - 物体（18 维，只有仿真样本有）：物体、目标、箱子、桌面在机器人朝向系下的位置，物体速度，物体轴向；
    demo 样本置零；
  - 标志位（1 维）：是否有物体信息。
  - 不含回合已过去的时间：demo 全部成功，只看时间就能预测 demo 的目标值，而且同一起点的候选时间都一样，
    critic 就不会去区分姿态。
- **demo 数据**：只用训练集 episode 从 CLOSE 起点到结尾的帧（远端 238 条、约 15 万帧）。
- **训练**：新 run 先只用 demo 训练 `critic_demo_steps` 步；之后每个 batch 一半 demo、一半策略回合。
  仿真样本的目标是 `最终 Ψ · γ^(到回合结束的步数)`（完成 10，失败 0，超时 exp(−d/0.1)）。
  γ = `critic_gamma`（默认每控制步 0.997）。
- **想象里**：chunk 内完成取 10、掉落取 0，否则取 `W(32 帧末的身体)`：**只用本体，物体部分置零、
  标志位为 0**，和 demo 样本一样，裁剪到 [0, 10]。
  - 原因：仿真样本的目标几乎全是 0（策略还没成功过），带物体信息时网络学成"标志位 = 1 → 0 分"，
    想象里所有候选都约为 0，critic 区分不了候选。只用本体时，critic 回答的是"这个身体姿态按 demo 的标准
    进行到了哪一步"；物体在 chunk 内有没有被抓住、有没有掉，由 M 判断。
  - 风险：只看本体，罐子没被抓住时手臂摆出后期姿态也会得高分；靠 chunk 内的掉落判断和仿真锚的真实回报约束。
  - 诊断：`critic_fresh/sim_progress_corr`（只用本体的 critic 值与仿真真实结果的相关系数）、
    `imag/value_group_std` vs `imag/chunk_return_group_std`（同一起点的候选在 critic 和 chunk 结果上的差异）。
- **对齐检查**：`tasks/can_pnp/tests/test_demo_progress.py` 在记录的抓取起点比较仿真算出的本体特征
  与 demo 该帧的记录（关节一致，手腕位置 1 cm 内，重力与手腕朝向一致）。
- 旧的 residual critic（只用仿真回合、预测剩余的 Ψ 变化）保留为 `critic_kind=residual`。

### 5.5 S1 的打分与更新

```
score = J + intent_weight × max(J* − J, 0)
```

- `intent_weight = 0`（默认）：**S0 冻结时必须为 0**。S0 不会变，S1 只能按 S0 实际能做到的 J 计分；
  否则会被推向 S0 做不到的意图。J* 与差距只用于诊断。
- `intent_weight > 0`：阶段 B（S0 能适应）之后再打开，对 S0 能补上的差距按意图本身计分。
- 不确定度超过 `uncertainty_max`（默认 3 cm）的候选：优势为 0，且不进入组内的均值与方差。
- 组内差异小于起点势函数 0.1% 的组视为没有信号。
- GRPO：PPO-clip + 对 N(0, I) 的 KL；单次想象更新在 `imag_target_kl` 早停，
  一个 cycle 内所有想象更新的累计漂移不超过 `cycle_kl`（在固定探测样本上精确计算的 KL）。

## 6. 训练循环（`experiments/train_v3.py`）

启动：加载 demo，critic 先在 demo 上预训练；采集 `wm_warmup_rollouts` 轮仿真填 buffer，
M 预训练 `wm_warmup_steps` 步，critic 混合训练 `critic_warmup_steps` 步。

之后每个 cycle：

1. **仿真采集**：每 `collect_every` 个 cycle 一次，或者上一个 cycle 的想象被门控时。
   - 每组 `sim_group` 个环境共用一个抓取起点；`base_fraction` 的组用 N(0, I) 噪声；失败回合保留。
   - **仿真锚**：在策略组上，用回合的真实回报做一次 GRPO 更新（与 DSRL 相同），效果下限即纯仿真 DSRL。
   - 新数据进 buffer（按新旧加权）；M 的每个成员各训练 `wm_steps` 步，critic 训练 `critic_steps` 步。
   - 回滚：成功率比历史最好低 `rollback_margin` 以上时，恢复最好的策略，cycle KL 预算减半。
2. **排序校验**：每组 `rank_group` 个环境先用同一噪声走到同一状态，再各执行一个候选 32 帧。
3. **τ 门控**：`τ < tau_min`（默认 0.2）时本 cycle 不做想象更新，下一个 cycle 直接采集。
   τ 在 `[tau_min, tau_low)` 时每个 cycle 的想象更新次数减半，高于 `tau_high` 时加倍。
4. **想象更新**：每次从 buffer 取 `imag_starts` 个真实决策点，各采 `imag_group` 个噪声，按第 5.5 节打分做 GRPO。

## 7. 日志指标（`metrics.jsonl`）

| 前缀 | 含义 |
|---|---|
| `sim/*`、`sim_base/*` | 策略组 / N(0, I) 组的回合结果：成功率、掉落率、最终距离、被夹住的步数占比 |
| `track/*` | 右手跟踪误差（均值、95 分位）、执行得准的窗口占比、失败回合中执行得准的占比 |
| `anchor/*` | 仿真锚 GRPO 更新的统计 |
| `wm/*`、`wm_fresh/*` | 物体 WM 的训练损失，以及在新回合上（训练前）的误差：仿真夹爪 / Φ 夹爪 |
| `critic/*`、`critic_fresh/*` | critic 在 demo 与仿真样本上的解释方差和均值 |
| `rank/tau` | 想象 chunk 结果与仿真 chunk 结果的组内 Kendall τ |
| `rank/exact_share`、`rank/ideal_mae_exact`、`rank/gap_*` | S0 执行得准的候选占比；理想分支在这些候选上的误差；两类候选的预测差距 |
| `imag/*` | 想象的 J、J*、差距、不确定度、保留比例、chunk 内完成 / 掉落比例 |
| `grpo/*` | 想象 GRPO 更新的统计 |
| `imag_skipped`、`rollback` | τ 门控生效、发生回滚 |

`rank/tau` 两边都只比较 chunk 内能观测到的结果（M 在 Φ 的手上推出的物体 vs 仿真里的物体），
不含 critic：它检查的是 M 与 Φ 的预测是否可靠。

## 8. 代码对应

| 部分 | 模块 |
|---|---|
| Φ、φ | `hskill/s0/body_world_model.py`、`hskill/executors/imagine.py`（`PhiStep`、`imagine_body`）、`hskill/s0/kinematics.py` |
| M（集成） | `hskill/interaction/model.py`（`ObjectWorldModel`、`ObjectWMEnsemble`、`load_object_wm`） |
| critic | `hskill/learners/critic.py`（`ProgressCritic`、`state_proprio`、`object_features`、`progress_features`）、`hskill/data/demo_progress.py`（`DemoProgress`） |
| 双分支评估 | `hskill/executors/imaginer.py`（`Imaginer._chunk`、`_score`、`ARM_JOINTS`） |
| 采集、跟踪诊断、排序校验 | `hskill/sim/collect.py`（`SimCollector`、`tracking_stats`） |
| 回合数据 | `hskill/data/episodes.py`、`hskill/data/buffer.py` |
| GRPO | `hskill/learners/grpo.py`（`group_advantages`、`masked_group_advantages`、`update`） |
| 任务 | `tasks/can_pnp/`（场景、`task_spec.py` 势函数与完成判定、`starts.py` 抓取起点） |
| 训练入口 | `experiments/train_v3.py`；8 卡：`experiments/launch/train_v3_8gpu.sh` |
| 可视化 | `experiments/view_policy.py --object-wm-checkpoint`（`hskill/utils/wm_viz.py`） |

## 9. 运行

本地（workspace 根目录有 `.venv`）：

```bash
.venv/bin/python -m experiments.train_v3 --tracker-checkpoint ../supermimic_0100000.pt \
  --diffusion-checkpoint <Humanoid-Diffusion>/outputs/<run>/best.pt --out-dir logs/v3/<run>
```

remote-26：访问不了 GitHub，无法为新结构 `uv sync`。根目录 `.venv` 是指向 `can-pnp/.venv` 的软链接
（依赖版本相同），新代码通过 `PYTHONPATH` 加载：

```bash
PYTHONPATH=hskill/src:tasks/can_pnp/src:. TRACKER=checkpoints/supermimic_0100000.pt \
DIFFUSION=../UMI-Diffusion/Humanoid-Diffusion/outputs/<run>/best.pt \
DIFFUSION_REPO=../UMI-Diffusion/Humanoid-Diffusion OUTPUT_DIR=logs/v3/<run> MUJOCO_GL=disable \
  bash experiments/launch/train_v3_8gpu.sh [--init-noise-checkpoint <dsrl>/last.pt]
```

8 卡启动脚本用静态 rendezvous（`--master-addr 127.0.0.1 --master-port <固定端口>`）。

## 10. 未实现（阶段 B）

- **S0 适应**：在"S0 能补上差距"的样本上，用按任务敏感度加权的手部跟踪误差微调 tracker
  （SuperMimic 训练流程加外部参考动作与加权跟踪项），适应后重拟合 f_ω。
- **打开意图项**：S0 能适应之后，`intent_weight > 0`。
- **可行性判断**：区分"S0 能补上"与"物理上做不到"的差距（目前只用 M 的不确定度过滤）。
- **两层调度**：两个时间尺度交替更新 S1 与 S0。
- **失败归因只看失败前的一段**：`track/fail_well_tracked_share` 现在要求整个回合都执行得准，过于严格。

## 11. 实验记录与结论

| 实验 | 设置 | 结果 |
|---|---|---|
| Φ 精度 | DSRL 策略，每个 reference 执行满 32 帧 | 右夹爪 1.8 cm，比完美跟踪假设准 3.4 倍 |
| v3 贪心（无 critic） | 从 DSRL 第 610 轮初始化 | τ 0.6–0.77；成功率 61.7% → 57.9%，掉落率下降，策略变保守 |
| v3 residual critic | 从零开始，12 小时 | τ ≈ 0；只出现 1 次成功；最好时掉落 4.7%、距离 0.38 m，之后退化到 0.86 m |
| v3 双分支 + 仿真锚 + τ 门控（residual critic） | 从零开始，约 1 小时 | τ 0.07–0.18，想象全部被门控；掉落 48% → 41%，成功率 0 |
| 纯仿真 DSRL | 第 595 → 935 轮 | 训练成功率约 61% → 80% |

结论：

- 想象只能推一个 chunk（32 帧、0.64 s）；离目标远时候选差异小于模型误差，τ 低。
  物体 WM 的精度在训练中慢慢提高，但不是限制 τ 的主要因素。
- 从零开始时没有成功样本，只从回合学的 critic 学不到放置的价值：这是 progress critic 要解决的问题。
- 仿真锚保证不比 DSRL 差，但每个 cycle 的额外开销（WM 与 critic 训练、Φ 窗口、排序校验）让单位时间的更新
  比纯 DSRL 少。
- 身体 WM 只建模身体，不建模物体对身体的反作用。

| v3 progress critic（排序只比 chunk 结果） | 从零开始，12 个 cycle（约 35 分钟） | τ 0.6–0.72，每个 cycle 都有想象更新；掉落 48% → 29%，距离 0.67 → 0.57 m，成功率 0 |

当前运行：`logs/v3/progress_body_20261030_125250`（物体 WM 加速 + 想象只用本体的 progress critic，从零开始）。
