# 从人视频学全身操作：扩散 encoder 驱动冻结的 SuperMimic decoder

> 状态：idea 记录（2026-11-24），**可行性未验证**。
> 2026-11-25（v6）：stage 3 改成和仓库现有实现一致的 FlashSAC + SGS，DP 按轮次用成功 rollout 自模仿更新（§3.8）；补充 uni model 的 mask 语义（§3.5）。
> 2026-11-25（v7）：**部署时不再输入示范视频**，人视频只当训练数据（与部署任务相同，示范不提供额外信息）；去掉 video query 和进度 φ；stage 2 改为单条视频。场景流目标改回 EgoWAM 的原始定义（全图，**包含执行者**），作为弥合人机差异的主要手段；输入图像不做 mask。
> 架构页：[IDEA_VIDEO_IMITATION_DP.html](IDEA_VIDEO_IMITATION_DP.html)（浏览器直接打开；与 claude.ai 上的 artifact 同步）。
> 与其他文档的关系：独立于 [IDEA_WM_IDM_BRIDGE.md](IDEA_WM_IDM_BRIDGE.md) 的 S1/S0 级联优化。这里复用 SuperMimic 的 tracker 和身体 WM f_ω，但不依赖现有 S1。

---

## 0. 一句话

部署时 G1 只看自己的头部相机画面和本体状态。一个扩散 encoder（下称 DP encoder）先回归未来的 3D 场景流，再以它为条件生成头和双手的未来轨迹 kp̂，最后生成 SuperMimic 的 32 维 latent z，交给冻结的 SuperMimic decoder 执行。训练只用人 ego 视频、mocap，以及 stage 3 在仿真里自己收集的 rollout，不用任何遥操作。

---

## 1. Problem

### 1.1 手上的资源

| 资源 | 内容 |
|---|---|
| SuperMimic tracker | `a = dec(enc_motion(g), o_hist)`。g 是 680 维（10 帧 × 68），z 是 32 维、经 tanh、**没有 KL 正则**（`supermimic/policy/nets.py`） |
| 身体 WM f_ω | 71 维特权状态 + 动作 + ξ → 下一状态，可微，带摔倒 head；不含物体、不含夹爪 |
| BONES-SEED | SuperMimic 的训练集，G1 关节空间，约 13 万条、264 h；过滤时**去掉了含物体接触的 clip** |
| 人的 ego 视频 | 约 100 条，与部署任务相同，只当训练数据 |
| 仿真 | can-pnp 场景 |
| **没有的** | 机器人遥操作轨迹、任务相关的 mocap |

### 1.2 难点

**(a) 三份数据两两不配对。**

| 数据 | 图像 | 头 + 双手 3 点 | z | 本体 |
|---|---|---|---|---|
| BONES-SEED | ✗ | ✓（FK） | ✓（enc_motion） | ✓ |
| 人 ego 视频 | ✓ | ✓（SLAM + HaMeR 伪标签） | ✗ | ✗ |
| 仿真 rollout | ✓（渲染） | ✓ | ✓ | ✓ |

前两份数据唯一的交集是头和双手的 3 个点。仿真 rollout 是唯一四者齐全的来源，但它只在 RL 阶段才产生。

**(b) 稀疏接口是一对多的。** 同样的头和手轨迹，下半身和躯干可以有很多种合理的配合（站、走、蹲、弯腰）。用 MSE 回归 z 会把几种模式平均掉，平均出来的 z 可能落在 decoder 没见过的区域。

**(c) 人和机器人外观不同。** 视觉通路在 stage 2 只见过人手，部署时画面里是夹爪；相机高度也不同。输入端和目标端都受影响：如果预测目标是像素或 VAE latent，这些差异会全部进入监督信号。

**(d) 下层的 OOD。** 前倾够物、下蹲这类任务姿态在 BONES-SEED 里很少；抓起物体后的负载动力学 tracker 也没见过。

### 1.3 核心问题

在没有遥操作、三份数据两两不配对的条件下，如何用人视频和 mocap 训练一个视觉条件的生成式 encoder，让它在冻结 tracker 的 latent 空间里生成可执行、并且符合眼前场景的全身动作。

---

## 2. Novelty

1. **以 3 点为枢纽的组合式训练。** ego → 3 点从人视频学，3 点 → z 从 mocap 学，两段的端到端耦合只靠仿真 RL。三份数据两两不配对，组合本身就是要验证的主张。
2. **扩散 encoder 锚定在冻结 tracker 的 latent 上。** z 的标签全部来自 enc_motion，生成结果天然落在 decoder 认识的流形上。stage 1 用 MorphMimic 式的想象蒸馏，把 latent 回归换成 flow matching。
3. **零遥操作的人 → 机器人迁移，以 3D 场景流为桥。** 视觉通路在人视频上学"先预测相机稳定的 3D 场景流，再生成动作"，场景流与外观无关，DiT 直接读它的隐状态。机器人数据只来自 stage 3 自己在仿真里收集的 rollout，不用遥操作示范（EgoWAM、ω-0 都依赖遥操作）。
4. **噪声空间 RL + 按轮次自模仿的后训练。** FlashSAC 只引导 kp 的初始噪声，成功的 rollout 按轮次蒸馏回 DP 的 kp 通路，z 通路保持 mocap 先验。

**骨干不是贡献。** 结构上它就是一个多模态 masked DiT，加上 ω-0 式的未来预测 query。审稿人会说"ω-0 加 MaskedMimic"，这在架构层面是对的，贡献要放在上面四点。

### 相关工作

| 工作 | 做了什么 | 和本文的区别 |
|---|---|---|
| ω-0（2608.06375） | VLM + V-JEPA 2.1；motion query 和 video query（预测机器人自己的未来视频 latent）；action DiT 输出 SONIC latent | 用 2,220 条遥操作；语言条件；没有 RL |
| EgoWAM（2607.08436） | 人视频 + 遥操作共训，只换世界预测目标：Pixel 最差，DINO 在 OOD 上最好，相机稳定的 3D 场景流在 ID 上最好、对人机不对齐最稳。场景流取全图查询点，包含手 / 机械臂，世界 head 推理时丢掉 | 双臂机械臂；依赖遥操作数据；纯 BC |
| AnyBody（2606.29209） | 5 个关键点随机 mask → 16 维球面 latent → 冻结 decoder，确定性，RL 在 latent 上学残差 | 最接近 stage 1；确定性；没有视频 |
| ViT-H（2610.32966） | 单目视频 → 扩散生成 G1 参考动作 → SONIC 派生的 tracker；用 8k 条网络视频做 DPO | 以视频为输入复现身体动作；我们部署时不需要视频，人视频只当训练数据 |
| DreamVLA（2611.10872）、E2S2H | VLA 输出 SONIC token | 数据来自生成动作或 real-to-sim |
| MorphMimic（github: lzyang2000/morphmimic） | SuperMimic 框架上的 student 蒸馏：冻结 decoder，只训新 encoder，标签在想象里当场计算 | stage 1 直接沿用它的流程 |
| MaskedMimic、PredActor（2609.24840）、OneSteer（2610.41354） | 稀疏约束 → 全身控制；状态-动作空间的扩散控制器 | 不在冻结 tracker 的 latent 里做 |

---

## 3. Method

### 3.1 总体结构

```text
                         ┌── 上下文侧（每次重规划算一次）──────────────────┐
当前帧 ─► V-JEPA 2.1 ───►│ flow query × N_f → h_f（确定性回归 3D 场景流）   │──┐
       （图像，冻结）  │  └──────────────────────────────────────────────────┘  │
                       └──────────────── 当前帧特征 ─────────────────────────────┤ 干净条件
本体 + zₜ₋₁ ─────────────────────────────────────────────────────────────────────┤
                                                                                 ▼
                         ┌── 生成侧：masked DiT（flow matching）──────────┐
                         │ ① 去噪 kp tokens → kp̂                           │
                         │ ② 固定 kp̂，去噪 z tokens → z[t : t+K]           │
                         └──────────────────────────────────────────────────┘
                                                                                 ▼
                                      SuperMimic decoder（冻结）+ obs_hist → a → PD → G1
```

推理路径上只有 V-JEPA 2.1、flow query、masked DiT 和 decoder。f_ω 只在 stage 1 的想象蒸馏里用，stage 3 不用。

### 3.2 视觉组件

| 位置 | 选择 | 依据 | 消融 |
|---|---|---|---|
| 当前帧输入 | V-JEPA 2.1 图像模式，冻结，**不做 mask** | ω-0 用它；把当前帧编码器换成 Wan，成功率 79.1 → 63.6 | DINOv3 |
| 预测目标：执行者 | kp̂（头 + 双手），扩散生成 | 要执行，必须是合法样本；同一场景下的做法不止一种 | — |
| 预测目标：效果 | 相机稳定的 3D 场景流，全图查询点，**包含执行者** | EgoWAM：与外观无关，对人机不对齐最稳 | 去掉执行者点（只留物体）；EgoWAM 式只读不写；DINO 未来特征；Wan latent |

### 3.3 上下文侧：flow query 是人机之间的桥

- **flow query（方案 B）**：可学习 query 读当前帧特征，**确定性回归**场景流，隐状态 h_f 作为 DiT 的条件，即"先预测未来，再决定动作"。
- **为什么它能弥合人机差异**：场景流描述的是物理上发生了什么（手或夹爪怎么动、罐子怎么动），表示在扣除相机运动后的 3D 坐标里。人手和夹爪做出相同的物理动作，得到的监督相近（EgoWAM 的 D2）。DiT 读的是 h_f 而不是只读原始特征，所以生成直接建立在一个与外观无关的未来表示上。
- **为什么包含执行者的点**：静止点被过滤后，运动点大部分落在手 / 机械臂上，这是场景流里信号最密的部分（EgoWAM 的做法）。只留物体点时，接触物体前几乎没有运动点。
- 选回归而不是生成，是因为 h_f 被当作特征读，条件均值就够；z 和 kp̂ 要被执行，所以必须是生成式。
- 依据还有 ω-0 的消融：去掉预测自身未来的 video query，成功率 79.1 → 64.5。
- 没有场景流标签的 stage 不算损失，并对 h_f 做条件 dropout。
- 对照方案 A：EgoWAM 式，场景流在 DiT 里扩散、只读不写、部署时丢掉。

### 3.4 生成侧：masked DiT

在普通 DiT 上改三处：

1. **逐 token 的时间步。** kp 和 z 的噪声水平各自独立（Diffusion Forcing / UniDiffuser 式），所以时间步 embedding 按 token 给，不用全局 adaLN。
2. **交叉注意力读条件。** 目标 token 之间做自注意力，再对条件 token 做交叉注意力。条件天然不读目标，KV 能在去噪步之间缓存（RDT、DiT-Policy 式）。
3. **分模态投影。** kp 和 z 各自归一化，各接一个线性层进出，加模态 embedding。

规模：总共 150–400 个 token，8–12 层、宽 512，约 5000 万到 1 亿参数。图像由外部冻结编码器和 flow query 处理，不进去噪序列。

采样是两步：先只对 kp 去噪，得到 kp̂ 后把它当干净条件固定，再对 z 去噪。一次输出一段 z，长度 K 和每 8 步重规划一次对齐，并以 zₜ₋₁ 为条件，避免模式来回切换。

### 3.5 Mask 与噪声配置

| | Stage 1 · mocap | Stage 2 · 人视频 | Stage 3 · 仿真 RL | 部署 |
|---|---|---|---|---|
| 当前帧 | 空 | 视频的第 t 帧 | G1 相机渲染 | G1 真实相机 |
| h_f | 空 | flow query 输出 | flow query 输出 | flow query 输出 |
| 本体 + zₜ₋₁ | student 访问到的状态 | 空 | 仿真本体 | 真实本体 |
| 来源 token（当前帧） | mocap | 人 | 机器人 | 机器人 |
| kp tokens | FK 未来 3 点（作条件或作目标） | 同一视频的未来 3 点 | RL：π_w 给 ε_kp；蒸馏：成功 rollout 下发的 kp̂ | 先采样 |
| z tokens | z\* = enc_motion(g) | 缺失（τ=1，权重 0），混入 stage 1 batch | RL：DP 自采样；蒸馏：缺失，混入 stage 1 batch | 后采样 |
| flow query | 不训练 | Track4World 标签（全图） | 仿真真值（全图，全部 rollout） | 只算 h_f |

DP 是一个 uni model，每个 token 的角色由噪声水平 τ 和损失权重决定：

| 状态 | 输入 | loss | 含义 |
|---|---|---|---|
| 条件 | 干净值，τ=0 | 无 | 已知，只被读 |
| 目标 | 加噪，τ∈(0,1) | flow matching | 要生成 |
| 缺失 | 纯噪声，τ=1 | 0 | 既不当条件，也不监督 |

只当条件、不生成的输入（当前帧、h_f、本体、zₜ₋₁）只有"给 / null token"两种状态。

规则：

- **生成模态缺数据时，用 τ=1、权重 0，不用 null token。** τ=1 等价于边缘化这个模态，部署第 ① 步去噪 kp 时 z 也正处在 τ=1，训练和推理一致。两步采样对应的两种配置（kp 目标 + z 缺失；kp 条件 + z 目标）要显式、高概率地采到。
- **条件 dropout 在所有 stage 都做。** 有数据也按概率换成空 token。这样模型不能靠"哪些输入是空的"反推所处的 stage，部署时某一路信号缺失也有对应的处理方式。
- **kp̂ 的 exposure bias。** 训练时 z 看到的是加噪的真值 kp，推理时是自己采出来的 kp̂。按比例换成模型自己的预测，或者让真值噪声的幅度匹配 kp̂ 的实测误差。

### 3.6 Stage 1：mocap 蒸馏（MorphMimic 式）

- **teacher**：enc_motion + decoder，冻结。
- **student**：DP encoder，输入本体和 zₜ₋₁；kp tokens 是 FK 的未来 3 点（脚几乎总是 mask，手偶尔 mask）。
- **主要配置是 kp 作条件 → z 作目标**，即 p(z | kp, 本体, zₜ₋₁)，对应 MorphMimic HumanEncoder 的角色。同时以较低概率采 kp 作目标（只给本体），学一个不看图像的 kp 先验。
- **数据流**：
  - 仿真里一半环境由 student 执行，rollout 进 replay buffer，f_ω 交替重拟合。
  - 从 buffer 采起点，在 f_ω 里用 student 推进想象 rollout，在 student 自己访问到的状态上当场计算 teacher 的 z_motion 和 a_teacher。
- **损失**，在 MorphMimic copy 模式的基础上改：
  - latent 项 `‖z_h − z_motion‖²` 换成以 z_motion 为目标的 **flow matching**；
  - 动作项 `‖dec(·) − a_teacher‖²` 用单步 x0 估计 ẑ₀ 计算，**只加在低噪声的 τ 上**。直接对采样结果做 MSE 会把多峰压垮。
- **成本**：copy 模式每一步都对状态做 detach，想象里的 DP 采样**不需要梯度**，成本接近训练普通 DP。return 模式（加 −J）需要可微采样，先不做。
- **对照**：同一 teacher、同一 decoder，DP encoder 对比 MLP HumanEncoder（MorphMimic 原版，属于 AnyBody 一类）。在只给 3 点的设置下比较跟踪误差和摔倒率，作为 stage 1 的主实验。

注意：MorphMimic 是独立复现，goal 是 670 维，和本仓库 supermimic-reproduced 的 680 维不同，两边的 z 空间不能混用。要移植的是 student 的流程，teacher 和 decoder 用自己的 checkpoint。

### 3.7 Stage 2：人视频（必需）

- **单条视频**：第 t 帧 → 同一条视频的未来 3 点（kp 目标）和 3D 场景流（flow query 回归目标）。
- 训练 flow query、当前帧通路和 DiT 的 kp 分支。z tokens 设为缺失、损失为 0，靠混入 stage 1 的 batch 保持"3 点 → z"不被遗忘。
- 这是视觉通路唯一的先验来源：没有这一步，DP 在仿真里的初始成功率接近 0，stage 3 的 RL 就没有起点。

### 3.8 Stage 3：仿真 RL 后训练（FlashSAC + SGS，按轮次自模仿）

参考实现是仓库里现有的 five-point 流程：[train_fivepoint.py](../experiments/train_fivepoint.py)、[learners/flashsac/](../hskill/src/hskill/learners/flashsac/)、[data/sgs.py](../hskill/src/hskill/data/sgs.py)。把其中冻结的 five-point FM 换成 DP。现有的"five-point FM + MorphMimic HumanEncoder"两段，正好对应 DP 的 kp 分支和 kp → z 分支；去掉示范输入后，actor 的输入结构和现有代码一致。

**单轮内的 RL（DP 冻结）**

| 部分 | 做法 |
|---|---|
| 场景 | can-pnp，当前观测始终是 G1 相机渲染 |
| actor π_w（FlashSAC） | 输入 DP 的条件 tokens（h_f、当前帧特征、本体、zₜ₋₁），输出有界噪声 ×3，作为 **ε_kp**。初始化成和观测无关的 tanh-Gaussian，方差与 DP 先验一致 |
| z | DP 读 kp̂ 后去噪得到。ε_z 随机，不受 RL 控制 |
| critic | 两个分类分布 Q，输入 90 维特权任务特征 + action（非对称） |
| 奖励 | 放置的终止奖励 + 势函数 shaping（罐子到目标、手到罐子） |
| 重置 | SGS：重置库由 100 条人示范的 3 点经 stage 1 DP → decoder 在仿真里回放得到，按 phase 取起点；按成功率加权，峰值在 0.5，从终点往回学 |

只引导 ε_kp：RL 只改"执行者接下来怎么动"，z 由 DP 忠实执行。

**按轮次交替**

```text
第 k 轮
 ① RL 阶段：DP_k 冻结
    FlashSAC 训练 π_w 和 critic，SGS 选起点，成功 rollout 存进成功池
    （成功率稳定或到固定步数就停）
 ② 蒸馏阶段：π_w 不再更新
    用成功池监督训练 DP（见下表），混入 stage 1/2 数据，加 KL 拉向 DP_k → 得到 DP_{k+1}
 ③ 重置
    π_w 回到初始化；critic 重新初始化；replay 清空
    SGS 的成功率历史保留（它记录的是起点难度，和 DP 参数关系不大）
 → 第 k+1 轮在 DP_{k+1} 上重新跑 ①
```

**为什么要重置。** π_w 的输出是 DP 的初始噪声，critic 学的是 Q(s, ε)，replay 存的是 (s, ε, r, s′)，三者都依赖 DP 当前的参数。DP 一更新，同一个 ε 生成的 kp̂ 就变了，三者一起失效。重置不会白学：π_w 学到的改进已经在 ② 里蒸馏进 DP_{k+1}，下一轮从更高的先验起步。"慢更新"的意思就是 DP 只在轮与轮之间更新一次。

**蒸馏阶段更新哪一块**

| 模块 | 是否更新 | 监督 |
|---|---|---|
| 当前帧通路 + DiT 的 kp 分支 | ✓ 主体 | kp 做 FM，目标是成功 rollout 中**实际下发的 kp̂**（π_w 引导后的采样） |
| flow query + flow head | ✓ | 回归仿真场景流真值（全图），所有 rollout 都能用。这一步把机器人渲染帧接到同一个场景流空间里 |
| DiT 的 z 分支 | 基本不动 | rollout batch 里 z 设为缺失（τ=1，权重 0），混入 stage 1 batch 守住 kp → z |
| V-JEPA、decoder | 冻结 | — |

- **kp 标签取下发的 kp̂。** 成功就是它作为命令带来的，蒸馏的是完整的策略 π_w∘DP。如果 tracker 跟踪误差大、下发和实际差得远，再改用实际 FK 做 hindsight 重标注。
- **z 分支不更新。** π_w 不控制 ε_z，rollout 里的 z 只是 DP 自己的采样，只多了"成功"这一层筛选，信号弱；z 分支有 stage 1 的 teacher 兜底。
- **共享主干。** "只更新 kp 分支"在参数上分不干净，靠 z 缺失 + 混入 stage 1 batch + KL 来约束。要隔离得更彻底，可以在 kp 分支上用 LoRA，或只解冻 kp 的进出投影和最后几层。

**代价与替代。** 每轮都要重新热身 critic。替代方案是让 critic 读 kp̂（DSRL-NA 式：先在动作空间学 Q(s, kp̂)，再蒸馏到噪声空间），DP 更新后 critic 和 replay 仍然有效，代价是要改现有 critic 的输入。第一版用按轮次重置。

**相对 v5 去掉的：** f_ω 想象 BPTT、3 点一致性想象奖励、示范池 α 和监督 β 的调度。

### 3.9 场景流标签

定义（与 EgoWAM 一致）：当前帧 t 上取 N 个查询点（例如 28×40 网格，覆盖全图，**包含执行者**），标签是它们在 [t, t+T] 内的 3D 位置，全部表示在 t 时刻的相机坐标系（重力对齐）。静止点按位移阈值过滤（EgoWAM：机器人 2 mm、人 10 mm）。

| 步骤 | 人视频 | 仿真 |
|---|---|---|
| 相机位姿 | Aria VIO / MegaSaM / DROID-SLAM（和取头部 3 点共用） | FK 的头部相机位姿 |
| 3D 位置 | MegaSaM / UniDepth / MoGe-2 度量深度反投影 | 深度缓冲 |
| 未来轨迹 | Track4World / SpatialTracker v2 / TAPIP3D；或 CoTracker3 + 深度 | 分割缓冲找所属刚体（物体或机器人连杆），p_{t+k} = B_{t+k} · B_t⁻¹ · p |
| 质量控制 | 遮挡点不算损失；静止背景点扣除相机运动后位移应接近 0，残差大的片段丢掉 | 真值 |
| 时间窗 | 人更快，T 短（EgoWAM：1 s） | T 长（EgoWAM：1.5 s） |

运动点远少于静止背景，损失里对运动点加权，或者过采样运动区域。

---

## 4. 风险与待确认

1. **SuperMimic latent 是否适合扩散。** 没有正则，可能有维度饱和、维度高度相关、decoder 对某些方向特别敏感等问题。
   - 退路一：在 g 空间扩散，再送进冻结的 enc_motion。
   - 退路二：重训一个带 KL 或球面约束的接口 latent。
2. **先验缺任务姿态。** BONES-SEED 过滤掉了接触类 clip，噪声空间 RL 可能够不到需要的姿态。按轮次蒸馏只更新 kp 通路；如果瓶颈在 z 分支（任务姿态 → z），需要再放开 z 分支，或改成 RECAP 式的优势条件化直接微调 DP。
3. **场景流能否单独弥合人机差异。** EgoWAM 的结论建立在有遥操作数据兜底的前提上；我们的机器人帧要到 stage 3 才出现（仿真渲染）。关键实验是 stage 2 → stage 3 第一轮的迁移：只在人帧上训练的视觉通路，面对机器人渲染帧时 kp̂ 和 h_f 是否还合理。备选（已讨论、暂不采用）：输入里 mask 掉执行者，并加一个跨执行者共用的当前 3 点 token。
4. **stage 2 没有本体。** 人视频里模型只能从图像判断"手现在在哪"，部署时本体里有这个信息，但 stage 2 没学过去读它（stage 1 学过）。如果迁移不好，考虑给 stage 2 一个由 SLAM + HaMeR 得到的当前 3 点作为本体的替代。
5. **三重视觉域差**：人的真实视频、G1 仿真渲染、G1 真实相机。
6. **扣除相机运动。** G1 走路时头部相机一直在晃，比 EgoWAM 的固定相机重要得多。
7. **按轮次蒸馏的漂移。** kp 通路反复吸收 π_w 的改写，可能偏离 stage 2 学到的人视频先验。蒸馏时混入 stage 2 数据并加 KL。
8. **伪标签噪声。** HaMeR 在遮挡下会抖，直接进入 stage 2 的 kp 标签；做时间平滑、按置信度加权。
9. **负迁移。** mocap 264 h 对 100 条视频，batch 要按来源固定比例、按来源加权；stage 2 混入后，stage 1 的闭环指标不能变差。

**Stage 3 待确认：**

- f_ω 只留在 stage 1（MorphMimic 想象蒸馏），stage 3 不用。
- kp 是 3 点还是 5 点：现有 tracker 是 human/five 版。倾向 DP 生成 3 点，脚的位置保持 mask。

---

## 5. 先做的廉价验证

1. **z\* 的分布**：BONES-SEED 上每一维的直方图、|z| > 0.95 的比例、PCA 有效维数。
2. **decoder 对 z 误差的敏感度**：给 z\* 加 σ 噪声跑闭环，画跟踪误差和摔倒率随 σ 的曲线。
3. **stage 1 主实验**：只给 3 点时，oracle z\*、MLP HumanEncoder、DP encoder 三者对比头/手误差、摔倒率、‖Δz‖；接近任务姿态的 clip 单独统计。
4. **伪标签质量**：100 条视频的头/手轨迹，人工检查 10 条。
5. **开环运动学回放**：重定向后的人视频 3 点 → stage 1 的 DP → decoder，在仿真里看摔倒率和手部误差（对比罐子半径 2.75 cm）。回放结果同时用来构造 SGS 重置库。
6. **跨执行者视觉探针**：冻结 V-JEPA 2.1，在人帧上训练"当前帧 → 未来 3 点"，在 G1 仿真渲染帧上测试；对比加 / 不加场景流辅助回归、场景流包含 / 不包含执行者点。直接检验 §3.3 的主张。

---

## 6. 构建顺序

1. **诊断**：上面的第 1、2 项。
2. **执行接口**：给 SuperMimic 加一个直接从 z 出发的入口 `a = dec(z, o_hist)`。
3. **Stage 1**：移植 MorphMimic 的 student 蒸馏流程，先用 MLP HumanEncoder 跑通，再换成 masked DiT。
4. **视觉通路**：预先算好 V-JEPA 2.1 特征；人视频跑 SLAM、HaMeR、Track4World 得到 3 点和场景流标签；实现 flow query，训练 stage 2。
5. **重置库**：100 条人示范回放（第 5 项验证），按 phase 取起点。
6. **Stage 3**：在 `train_fivepoint.py` 上把 five-point FM 换成 DP（actor 读 DP 条件 tokens、输出 ε_kp），tracker 换成 DiT 的 z 分支 + SuperMimic decoder；can-pnp 导出场景流真值（包含机器人连杆）；外面套一层轮次循环，加一个成功池蒸馏脚本。

---

## 7. 参考

- ω-0：https://gentlefress.github.io/OMEGA-0_page/ （arXiv 2608.06375）
- EgoWAM：https://gatech-rl2.github.io/egowam.github.io/ （arXiv 2607.08436）
- MorphMimic：https://github.com/lzyang2000/morphmimic
- SuperMimic：arXiv 2610.28480
- AnyBody：arXiv 2606.29209；ViT-H：arXiv 2610.32966；DreamVLA：arXiv 2611.10872
- PredActor：arXiv 2609.24840；OneSteer：arXiv 2610.41354；OpenHLM：arXiv 2606.22174
- Ego2Sim2Humanoid：arXiv 2611.19196
