# 人视频模仿：扩散 encoder 驱动冻结的 SuperMimic decoder

> 状态：idea 记录（2026-11-24），**可行性未验证**，架构已讨论多轮并定稿到 v5。
> 架构页：[IDEA_VIDEO_IMITATION_DP.html](IDEA_VIDEO_IMITATION_DP.html)（浏览器直接打开；与 claude.ai 上的 artifact 同步）。
> 与其他文档的关系：独立于 [IDEA_WM_IDM_BRIDGE.md](IDEA_WM_IDM_BRIDGE.md) 的 S1/S0 级联优化。这里复用 SuperMimic 的 tracker 和身体 WM f_ω，但不依赖现有 S1。

---

## 0. 一句话

部署时给 G1 一条人的 ego 示范视频。一个扩散 encoder（下称 DP encoder）读这条示范、机器人当前的相机画面和本体状态，输出 SuperMimic 的 32 维 latent z，交给冻结的 SuperMimic decoder 执行。全程不使用任何机器人遥操作轨迹。

---

## 1. Problem

### 1.1 手上的资源

| 资源 | 内容 |
|---|---|
| SuperMimic tracker | `a = dec(enc_motion(g), o_hist)`。g 是 680 维（10 帧 × 68），z 是 32 维、经 tanh、**没有 KL 正则**（`supermimic/policy/nets.py`） |
| 身体 WM f_ω | 71 维特权状态 + 动作 + ξ → 下一状态，可微，带摔倒 head；不含物体、不含夹爪 |
| BONES-SEED | SuperMimic 的训练集，G1 关节空间，约 13 万条、264 h；过滤时**去掉了含物体接触的 clip** |
| 人的 ego 视频 | 约 100 条，任务相关 |
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

**(c) 人和机器人外观不同。** 手的样子、相机高度都不一样。如果预测目标是像素或 VAE latent，这些差异会全部进入监督信号。

**(d) 示范场景不等于执行场景。** 示范里物体的位置和机器人眼前的不同，不能逐点照抄示范轨迹；人比机器人快，还需要时间对齐。

**(e) 下层的 OOD。** 前倾够物、下蹲这类任务姿态在 BONES-SEED 里很少；抓起物体后的负载动力学 tracker 也没见过。

### 1.3 核心问题

在没有任何配对数据、没有遥操作的条件下，如何让一个以人视频为条件的生成式 encoder，在冻结 tracker 的 latent 空间里生成可执行、并且符合眼前场景的全身动作。

---

## 2. Novelty

1. **以 3 点为枢纽的组合式训练。** ego → 3 点从人视频学，3 点 → z 从 mocap 学，两段的端到端耦合只靠仿真 RL。三份数据两两不配对，组合本身就是要验证的主张。
2. **扩散 encoder 锚定在冻结 tracker 的 latent 上。** z 的标签全部来自 enc_motion，生成结果天然落在 decoder 认识的流形上。stage 1 用 MorphMimic 式的想象蒸馏，把 latent 回归换成 flow matching。
3. **按场景改写的示范模仿。** kp̂ 是生成出来的，不是照抄示范。这和 ViT-H 那种"复现视频里的身体动作"不同。
4. **人 / 机器人视频混合的 RL 后训练。** 成功 rollout 回流成机器人示范和配对监督；f_ω 按 SuperMimic 原有的方式辅助训练。

**骨干不是贡献。** 结构上它就是一个多模态 masked DiT，加上 ω-0 式的 query。审稿人会说"ω-0 加 MaskedMimic"，这在架构层面是对的，贡献要放在上面四点。

### 相关工作

| 工作 | 做了什么 | 和本文的区别 |
|---|---|---|
| ω-0（2608.06375） | VLM + V-JEPA 2.1；motion query 和 video query；action DiT 输出 SONIC latent；Wan encoder 只当未来 latent 的目标 | 用 2,220 条遥操作；语言条件；没有 RL |
| EgoWAM（2607.08436） | 人和机器人共训，只换世界预测目标：Pixel 最差，DINO 在 OOD 上最好，相机稳定的 3D 场景流在 ID 上最好、对人机不对齐最稳 | 双臂机械臂，纯 BC |
| AnyBody（2606.29209） | 5 个关键点随机 mask → 16 维球面 latent → 冻结 decoder，确定性，RL 在 latent 上学残差 | 最接近 stage 1；确定性；没有视频 |
| ViT-H（2610.32966） | 单目视频 → 扩散生成 G1 参考动作 → SONIC 派生的 tracker；用 8k 条网络视频做 DPO | 复现身体动作，不按场景改写任务；运动学参考，不是 latent |
| DreamVLA（2611.10872）、E2S2H | VLA 输出 SONIC token | 数据来自生成动作或 real-to-sim；不以示范视频为条件 |
| MorphMimic（github: lzyang2000/morphmimic） | SuperMimic 框架上的 student 蒸馏：冻结 decoder，只训新 encoder，标签在想象里当场计算 | stage 1 直接沿用它的流程 |
| MaskedMimic、PredActor（2609.24840）、OneSteer（2610.41354） | 稀疏约束 → 全身控制；状态-动作空间的扩散控制器 | 不在冻结 tracker 的 latent 里做 |

---

## 3. Method

### 3.1 总体结构

```text
示范视频 ─► V-JEPA 2.1（视频，冻结）─┐
示范 3 点 ──────────────────────────┤
                                    ▼
                      ┌── 上下文侧（每次重规划算一次）─────────────┐
                      │ video query × Nᵥ → 意图 tokens               │
当前帧 ─► V-JEPA 2.1 ─┤ flow query × N_f → h_f（确定性回归 3D 场景流）│──┐
       （图像，冻结）  └─────────────────────────────────────────────┘  │ 干净条件
本体 + zₜ₋₁ ─────────────────────────────────────────────────────────────┤
                                                                         ▼
                      ┌── 生成侧：masked DiT（flow matching）──────────┐
                      │ ① 去噪 kp tokens → kp̂ + 进度 φ                  │
                      │ ② 固定 kp̂，去噪 z tokens → z[t : t+K]           │
                      └──────────────────────────────────────────────────┘
                                                                         ▼
                                   SuperMimic decoder（冻结）+ obs_hist → a → PD → G1
```

推理路径上只有 V-JEPA 2.1、上下文侧、masked DiT 和 decoder。f_ω 只在训练时用。

### 3.2 视觉组件

视觉模型有两个角色，分开选：

| 位置 | 选择 | 依据 | 消融 |
|---|---|---|---|
| 当前帧输入 | V-JEPA 2.1 图像模式，冻结 | ω-0 用它；把当前帧编码器换成 Wan，成功率 79.1 → 63.6 | 两边都用 DINOv3 |
| 示范视频输入 | V-JEPA 2.1 视频模式，冻结 | 需要时序；和当前帧共享特征空间，便于比对"示范进行到哪一步"和"示范里的物体对应眼前哪一个" | DINOv3 逐帧 + 时序层 |
| 预测目标：执行者 | kp̂（头 + 双手），扩散生成 | 要执行，必须是合法样本；改写方式不止一种 | — |
| 预测目标：效果 | 相机稳定的 3D 场景流，去掉执行者身上的点 | EgoWAM 的结论；与外观无关 | EgoWAM 式只读不写；DINO 未来特征；Wan latent |

### 3.3 上下文侧

- **video query**：可学习 query（Perceiver / Q-Former 式），交叉注意示范 tokens 和示范 3 点，把一段长度不定的示范压缩成 Nᵥ 个意图 tokens。
- **flow query（方案 B）**：读意图 tokens 和当前帧特征，**确定性回归**场景流，隐状态 h_f 作为 DiT 的条件，即"先预测未来，再决定动作"。
  - 选回归而不是生成，是因为 h_f 被当作特征读，条件均值就够；z 和 kp̂ 要被执行，所以必须是生成式。
  - 依据是 ω-0 的消融：去掉 video query，成功率 79.1 → 64.5。
  - 没有场景流标签的 stage 不算损失，并对 h_f 做条件 dropout。
  - 对照方案 A：EgoWAM 式，场景流在 DiT 里扩散、只读不写、部署时丢掉。

### 3.4 生成侧：masked DiT

在普通 DiT 上改三处：

1. **逐 token 的时间步。** kp 和 z 的噪声水平各自独立（Diffusion Forcing / UniDiffuser 式），所以时间步 embedding 按 token 给，不用全局 adaLN。
2. **交叉注意力读条件。** 目标 token 之间做自注意力，再对条件 token 做交叉注意力。条件天然不读目标，KV 能在去噪步之间缓存（RDT、DiT-Policy 式）。
3. **分模态投影。** kp 和 z 各自归一化，各接一个线性层进出，加模态 embedding。

规模：总共 150–400 个 token，8–12 层、宽 512，约 5000 万到 1 亿参数。视频由外部冻结编码器和 query 处理，不进去噪序列。

采样是两步：先只对 kp 去噪，得到 kp̂ 后把它当干净条件固定，再对 z 去噪。一次输出一段 z，长度 K 和每 8 步重规划一次对齐，并以 zₜ₋₁ 为条件，避免模式来回切换。

### 3.5 Mask 与噪声配置

| | Stage 1 · mocap | Stage 2 · 人视频 | Stage 3 · 仿真 RL | 部署 |
|---|---|---|---|---|
| 示范视频 | 空 | 视频 A | 人 / 机器人视频，按 α 混合 | 人 ego 视频 |
| 示范 3 点 | 同一 clip 的 FK，加偏移和时间扭曲 | A 的伪标签 | 示范的伪标签或 FK | 示范的伪标签 |
| 当前帧 | 空 | 视频 B 的帧（≤ t） | G1 相机渲染 | G1 真实相机 |
| h_f | 空 | flow query 输出 | flow query 输出 | flow query 输出 |
| 本体 + zₜ₋₁ | student 访问到的状态 | 空 | 仿真本体 | 真实本体 |
| 来源 token（示范 / 当前） | mocap / mocap | 人 / 人 | 人或机器人 / 机器人 | 人 / 机器人 |
| kp tokens | FK 未来 3 点 | B 的未来 3 点 | rollout 的 FK 真值 | 先采样 |
| z tokens | z\* = enc_motion(g) | 纯噪声，损失权重 0，混入 stage 1 batch | DSRL + 成功自模仿 | 后采样 |
| flow query | 不训练 | Track4World 标签 | 仿真真值，按 β 混合 | 只算 h_f |

规则：

- **条件 dropout 在所有 stage 都做。** 有数据也按概率换成空 token。这样模型不能靠"哪些输入是空的"反推所处的 stage，部署时某一路信号缺失也有对应的处理方式。
- **kp̂ 的 exposure bias。** 训练时 z 看到的是加噪的真值 kp，推理时是自己采出来的 kp̂。按比例换成模型自己的预测，或者让真值噪声的幅度匹配 kp̂ 的实测误差。
- **stage 1 防照抄。** 示范 3 点和目标来自同一条 clip，所以输入加扰动、目标保持干净。

### 3.6 Stage 1：mocap 蒸馏（MorphMimic 式）

- **teacher**：enc_motion + decoder，冻结。
- **student**：DP encoder，输入 3 点（脚几乎总是 mask，手偶尔 mask）、本体和 zₜ₋₁。
- **数据流**：
  - 仿真里一半环境由 student 执行，rollout 进 replay buffer，f_ω 交替重拟合。
  - 从 buffer 采起点，在 f_ω 里用 student 推进想象 rollout，在 student 自己访问到的状态上当场计算 teacher 的 z_motion 和 a_teacher。
- **损失**，在 MorphMimic copy 模式的基础上改：
  - latent 项 `‖z_h − z_motion‖²` 换成以 z_motion 为目标的 **flow matching**；
  - 动作项 `‖dec(·) − a_teacher‖²` 用单步 x0 估计 ẑ₀ 计算，**只加在低噪声的 τ 上**。直接对采样结果做 MSE 会把多峰压垮。
- **成本**：copy 模式每一步都对状态做 detach，想象里的 DP 采样**不需要梯度**，成本接近训练普通 DP。return 模式（加 −J）需要可微采样，先不做。
- **对照**：同一 teacher、同一 decoder，DP encoder 对比 MLP HumanEncoder（MorphMimic 原版，属于 AnyBody 一类）。在只给 3 点的设置下比较跟踪误差和摔倒率，作为 stage 1 的主实验。

注意：MorphMimic 是独立复现，goal 是 670 维，和本仓库 supermimic-reproduced 的 680 维不同，两边的 z 空间不能混用。要移植的是 student 的流程，teacher 和 decoder 用自己的 checkpoint。

### 3.7 Stage 2：人视频（可选，推荐最简版）

- 从同一任务里取不同实例的 (A, B)：A 当示范，B 当"当前"。监督 kp̂ → B 的未来 3 点，场景流 → B 的 3D 场景流。
- z tokens 固定为纯噪声、损失为 0，靠混入 stage 1 的 batch 保持"3 点 → z"不被遗忘。
- 跳过这一步也能跑：直接用示范 3 点当目标。代价是 stage 3 的 RL 要从零学会"按场景改写示范"。

### 3.8 Stage 3：仿真 RL 后训练

- **场景与示范**：闭环跑在 can-pnp，当前观测始终是 G1 相机渲染。示范从示范池采样：一开始只有人视频，成功 rollout 渲染成视频后加入。
  - 示范中机器人视频的比例 α **上限 0.5**，因为部署时示范永远是人视频。
  - 监督 batch 中机器人 rollout 的比例 β 可以更高（约 0.7）。
- **策略**：RL 只训练噪声空间的 π_w（DSRL），z 始终经过 DP 的去噪器生成。DP 用成功 rollout 自模仿慢更新，加一个 KL 拉回 stage 1 的先验。
- **f_ω 的用法与 SuperMimic 相同**：
  - rollout 进 replay buffer，f_ω 交替重拟合；
  - π_w 从 buffer 起点在冻结的 f_ω 里想象 H 步，BPTT 拿一阶梯度，末尾用 critic 自举；
  - 想象奖励 = 头和手与 kp̂ 的一致性 + 存活（摔倒 head）。
- **和 SuperMimic 的两点不同**：
  - 想象里没有新图像，视觉条件固定为起点帧，H 不宜超过一两个重规划周期；
  - DP 去噪链要反传梯度，步数要少。
- **任务奖励**：抓取、放置这类不连续结果 f_ω 推不出来，用仿真奖励。3 点一致性约束手相对物体的位姿和进度 φ，不逐帧约束绝对位置。

### 3.9 场景流标签

定义：当前帧 t 上取 N 个查询点（例如 28×40 网格），标签是它们在 [t, t+T] 内的 3D 位置，全部表示在 t 时刻的相机坐标系（重力对齐），执行者身上的点 mask 掉。

| 步骤 | 人视频 | 仿真 |
|---|---|---|
| 相机位姿 | Aria VIO / MegaSaM / DROID-SLAM（和取头部 3 点共用） | FK 的头部相机位姿 |
| 3D 位置 | MegaSaM / UniDepth / MoGe-2 度量深度反投影 | 深度缓冲 |
| 未来轨迹 | Track4World / SpatialTracker v2 / TAPIP3D；或 CoTracker3 + 深度 | 分割缓冲找所属刚体，p_{t+k} = B_{t+k} · B_t⁻¹ · p |
| 去掉执行者 | HaMeR 或手部分割 | 分割缓冲 |
| 质量控制 | 遮挡点不算损失；静止背景点扣除相机运动后位移应接近 0，残差大的片段丢掉 | 真值 |
| 时间窗 | 人更快，T 短（EgoWAM：1 s） | T 长（EgoWAM：1.5 s），或用 φ 对齐 |

运动点远少于静止背景，损失里对运动点加权，或者过采样运动区域。

---

## 4. 风险与待确认

1. **SuperMimic latent 是否适合扩散。** 没有正则，可能有维度饱和、维度高度相关、decoder 对某些方向特别敏感等问题。
   - 退路一：在 g 空间扩散，再送进冻结的 enc_motion。
   - 退路二：重训一个带 KL 或球面约束的接口 latent。
2. **先验缺任务姿态。** BONES-SEED 过滤掉了接触类 clip，DSRL 可能够不到需要的姿态。出现时改成 RECAP 式的优势条件化，直接微调 DP。
3. **示范与执行的时间对齐。** φ 在 stage 1 有真值，stage 2 有近似真值，stage 3 只能靠奖励间接约束。
4. **三重视觉域差**：人的真实视频、G1 仿真渲染、G1 真实相机。
5. **扣除相机运动。** G1 走路时头部相机一直在晃，比 EgoWAM 的固定相机重要得多。
6. **f_ω 看不到物体。** 在 can-pnp rollout 上重拟合只能学到部分负载效应。
7. **伪标签噪声。** HaMeR 在遮挡下会抖，stage 1 的输入扰动要和实测误差匹配。
8. **负迁移。** mocap 264 h 对 100 条视频，batch 要按来源固定比例、按来源加权；stage 2 混入后，stage 1 的闭环指标不能变差。

---

## 5. 先做的廉价验证

1. **z\* 的分布**：BONES-SEED 上每一维的直方图、|z| > 0.95 的比例、PCA 有效维数。
2. **decoder 对 z 误差的敏感度**：给 z\* 加 σ 噪声跑闭环，画跟踪误差和摔倒率随 σ 的曲线。
3. **stage 1 主实验**：只给 3 点时，oracle z\*、MLP HumanEncoder、DP encoder 三者对比头/手误差、摔倒率、‖Δz‖；接近任务姿态的 clip 单独统计。
4. **伪标签质量**：100 条视频的头/手轨迹，人工检查 10 条。
5. **开环运动学回放**：重定向后的人视频 3 点 → stage 1 的 DP → decoder，在没有物体的仿真里看摔倒率和手部误差（对比罐子半径 2.75 cm）。
6. **视觉探针**：冻结 V-JEPA 2.1，在留出的视频上回归未来 3 点轨迹，和 DINOv3 对比。

---

## 6. 构建顺序

1. **诊断**：上面的第 1、2 项。
2. **执行接口**：给 SuperMimic 加一个直接从 z 出发的入口 `a = dec(z, o_hist)`。
3. **Stage 1**：移植 MorphMimic 的 student 蒸馏流程，先用 MLP HumanEncoder 跑通，再换成 masked DiT。
4. **视觉通路**：预先算好 V-JEPA 2.1 特征；人视频跑 SLAM、HaMeR、Track4World 得到标签；实现 video query 和 flow query，训练 stage 2。
5. **Stage 3**：can-pnp 渲染 G1 相机、导出场景流真值；搭 SuperMimic 训练循环（想象 rollout 里的动作换成 `dec(z, o_hist)`）；接入任务奖励；示范池与 α、β 调度。

---

## 7. 参考

- ω-0：https://gentlefress.github.io/OMEGA-0_page/ （arXiv 2608.06375）
- EgoWAM：https://gatech-rl2.github.io/egowam.github.io/ （arXiv 2607.08436）
- MorphMimic：https://github.com/lzyang2000/morphmimic
- SuperMimic：arXiv 2610.28480
- AnyBody：arXiv 2606.29209；ViT-H：arXiv 2610.32966；DreamVLA：arXiv 2611.10872
- PredActor：arXiv 2609.24840；OneSteer：arXiv 2610.41354；OpenHLM：arXiv 2606.22174
- Ego2Sim2Humanoid：arXiv 2611.19196
