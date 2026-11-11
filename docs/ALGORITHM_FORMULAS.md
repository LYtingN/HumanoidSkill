# 当前算法的公式化描述（S1 + S0 + 两个世界模型）

本文按代码现状（2026-11-09，新物体 WM 接入之后）把用到的算法写成公式，重点是两个世界模型：身体 WM $f_\omega$ 与物体 WM $M_\psi$。
代码入口：`experiments/train_dsrl.py`（只在仿真里的 DSRL）、`experiments/train_v3.py`（仿真 + 想象）。
默认数值见第 10 节；每个公式后标出对应的代码位置。

---

## 0. 总览

```
o_t ──E──► c ──μθ──► w ──D(冻结 DiT)──► r ──reference_plan──► 目标 g_k ──π_S0──► a_k ──仿真/f_ω──► s_{k+1}
 图像+本体   观测token  噪声        参考轨迹 32 帧                   (10 帧预览)    关节动作      身体状态
                                     │                                                         │
                                     └─ 夹爪开度 c_k（绕过 tracker，直接执行）                  └─ FK φ → 五个关键点 K_k
                                                                                                   │
                                                                     物体 WM M_ψ(历史 10 帧, 未来 32 帧关键点 + 夹爪指令) → 未来罐子
                                                                                                   │
                                                                              任务势函数 Ψ → 回报
```

| 层 | 组成 | 是否训练 |
|---|---|---|
| S1 | DINOv3 + DiT（DDIM，$\eta=0$）+ 噪声策略 $\mu_\theta$ | 只训练 $\mu_\theta$（DSRL） |
| S0 | SuperMimic tracker $\pi_{S0}$ | 冻结 |
| 身体 WM | SuperMimic 的 $f_\omega$（随 tracker checkpoint 一起） | 冻结 |
| 物体 WM | $M_\psi$，3 个成员的集成 | 训练 |
| critic | 已删除，待重新设计（见第 7 节） | — |

---

## 1. S1：冻结的 DiT + DSRL 噪声策略

观测编码（冻结）：

$$c = E(o_{t-1:t}) \in \mathbb{R}^{n_\text{tok}\times d},\qquad E = \text{obs\_proj}\big(\text{DINOv3}(\text{img}),\ \text{state}\big)$$

噪声策略（actor，唯一被训练的 S1 部分）：

$$w \sim \pi_\theta(w\mid c) = \mathcal{N}\big(\mu_\theta(c),\ \mathrm{diag}(\sigma^2)\big),\quad \log\sigma \ge -2.5,\quad w\in\mathbb{R}^{32\times 37}$$

$\mu_\theta$ 最后一层零初始化、$\log\sigma=0$，所以初始策略就是 base 策略 $w\sim\mathcal{N}(0,I)$。

$$\log\pi_\theta(w\mid c) = -\sum_i\Big[\tfrac12\big(\tfrac{w_i-\mu_i}{\sigma_i}\big)^2 + \log\sigma_i + \tfrac12\log 2\pi\Big]$$

解码（冻结，确定性）：

$$r = D(c, w) = \text{DDIM}_{\eta=0}(c, w) = \big(q^\text{ref}_{1:32},\ \text{grip}_{1:32},\ R^\text{rel}_{1:32}\big)$$

- $q^\text{ref}_k\in\mathbb{R}^{29}$：身体关节角；$\text{grip}_k\in\mathbb{R}^2$：左右夹爪开度；
- $R^\text{rel}_k$（6D）：第 $k$ 帧根部朝向相对当前根部朝向的旋转。

每 8 个控制步（50 Hz）重规划一次，执行 $r$ 的前 8 帧。
代码：`hskill/s1/noise_policy.py`（`NoisePolicy`、`SteeredDiffusionSource.decode`）。

---

## 2. S0：参考计划与 tracker

**参考计划**（`hskill/s0/controller.py: reference_plan`、`hskill/s0/odometry.py`）。数据里没有根部平移，
根部的世界朝向、高度和速度由当前状态和腿部运动学推出：

$$R^\text{root}_k = R^\text{root}_\text{now}\,R^\text{rel}_k$$

$$z^\text{root}_k = z_\text{ground} - \min_{\text{foot}}\big[R^\text{root}_k\,\text{sole}(q^\text{ref}_k)\big]_z,\qquad
v^\text{root}_{k} = -\frac{\text{sole}_\text{stance}(k{+}1) - \text{sole}_\text{stance}(k)}{\Delta t}$$

（较低的那只脚视为不打滑的支撑脚；$z_\text{ground}$ 用重规划时实测的关节角和根高标定。）

**tracker**（冻结）：第 $k$ 步看到参考第 $k..k{+}9$ 帧组成的目标 $g_k$（680 维）和 10 帧本体观测历史 $o^\text{hist}_k$：

$$a_k = \pi_{S0}(o^\text{hist}_k,\ g_k(r)) \in \mathbb{R}^{29}$$

夹爪不经过 tracker：右夹爪指令 $u_k = \text{grip}_{k,\text{right}}$ 直接执行。

---

## 3. 世界模型 1：身体 WM $f_\omega$（SuperMimic，冻结）

**状态与上下文**（`hskill/s0/body_world_model.py`）：

$$s = [\,p^\text{root}(3),\ q^\text{root}_\text{wxyz}(4),\ q(29),\ v^\text{root}_\text{world}(3),\ \omega^\text{root}_\text{root}(3),\ \dot q(29)\,]\in\mathbb{R}^{71}$$

$$\xi = [\,\text{足底摩擦}(1),\ \text{编码器偏置}(29),\ \text{躯干质心偏移}(3)\,]\in\mathbb{R}^{33}$$

夹爪不在状态里（WM 是在裸 G1 手上训练的，ALOHA 夹爪的重量由重力补偿抵消）。

**一步动力学**（`supermimic/world_model/world_model.py`）：

$$s_{k+1} = f_\omega(s_k, a_k, \xi) = \text{integrate}\Big(s_k,\ \text{MLP}_\omega\big(\text{canon}(s_k),\ a_k,\ \bar\xi\big)\Big)$$

网络预测状态增量，`canon` 是与朝向无关的规范化状态，$\bar\xi$ 是归一化后的上下文。

**终止头**：用相对参考根部的高度误差 $e_h$ 和倾斜误差 $e_\text{tilt}$：

$$\ell_k = \text{MLP}\big([e_h/0.25,\ e_\text{tilt}/0.8,\ |e_h|/0.25,\ |e_\text{tilt}|/0.8]\big),\qquad p^\text{fall}_k = \sigma(\ell_k)$$

**观测头**（解析）：$o_{k+1} = \text{obs}(s_{k+1}, a_k, \xi)$，用来滚动更新 tracker 的观测历史。

**$\Phi$：tracker + $f_\omega$ 闭环推演 $H=32$ 步**（`hskill/executors/imagine.py: imagine_body`）：

$$\Phi(r, s_0, o^\text{hist}_0, \xi):\quad
\begin{cases}
a_k = \pi_{S0}(o^\text{hist}_k,\ g_k(r)) \\
s_{k+1} = f_\omega(s_k, a_k, \xi) \\
o^\text{hist}_{k+1} = [\,o^\text{hist}_k[1:],\ \text{obs}(s_{k+1}, a_k, \xi)\,] \\
C_{k+1} = C_k\,(1-\sigma(\ell_{k+1})),\quad C_0 = 1
\end{cases}
\qquad k = 0..H{-}1$$

输出状态序列 $s_{0:H}$ 和摔倒概率 $1-C_H$（目前只记录，不参与评分）。

**正运动学** $\varphi$（`hskill/s0/kinematics.py`）：$\varphi(s) \to$ 左右夹爪、手腕、相机、躯干的位姿。右夹爪位姿记为

$$h_k = \varphi(s_k)_\text{right gripper} = (p^g_k, q^g_k)\in\mathbb{R}^7$$

**作用**：给定一个候选意图 $r$，不跑仿真就得到"S0 会把手带到哪里"，即实际分支的手部轨迹。

---

## 4. 世界模型 2：物体 WM $M_\psi$

（`hskill/interaction/model.py`）一次调用预测当前时刻 $t$ 之后 $H=32$ 帧的物体，条件是这段时间里关键点怎么动。

### 4.1 输入与输出

**关键点**（`KEYPOINT_BODIES`）：骨盆、左/右脚（ankle roll link）、左/右末端执行器（ALOHA 夹爪基座），每个是位置 + 四元数：

$$K_\tau = \{p_{\tau,i},\ q_{\tau,i}\}_{i=1}^{5}$$

| | 帧 | 内容 |
|---|---|---|
| 历史 | $\tau = t-L+1 .. t$，$L=10$ | 关键点 $K_\tau$；双夹爪实测宽度 $w_\tau\in\mathbb{R}^2$；物体 $o_\tau = [p, q, v]$（加速度由速度差分得到） |
| 条件 | $\tau = t+1 .. t+H$ | 未来关键点 $K_\tau$；双夹爪指令 $u_{\tau-1}\in\mathbb{R}^2$ |
| 输出 | $\tau = t+1 .. t+H$ | 物体 $\hat o_\tau = [p, q, v, a]$；"两指握住"的 logit $\ell_\tau$ |

$$\hat o_{t+1:t+H},\ \ell_{t+1:t+H} = M_\psi\big(K_{t-L+1:t},\ w_{t-L+1:t},\ o_{t-L+1:t},\ \ K_{t+1:t+H},\ u_{t:t+H-1}\big)$$

未来关键点是**输入**，不是预测：同一段历史配不同的未来手，就得到不同的物体未来。这正是第 6 节 $J$ 与 $J^*$ 的区别所在。不输入任何场景物体（箱子、桌面等）。

### 4.2 特征

所有量都表示在 $t$ 时刻骨盆的朝向坐标系里：只取偏航 $R_y$，原点在骨盆正下方，z 保持离地高度。

每个历史帧（62 维）：

$$\Big[\ \underbrace{\tfrac{R_y^\top(p_{\tau,i}-o)}{0.5},\ \text{6D}(R_y^\top R(q_{\tau,i}))}_{5\ \text{个关键点},\ 45},\ \tfrac{w_\tau}{0.065},\ \tfrac{R_y^\top(p^o_\tau - o)}{0.5},\ \text{6D}(R_y^\top R(q^o_\tau)),\ \tfrac{R_y^\top v^o_\tau}{0.5},\ \tfrac{R_y^\top a^o_\tau}{10}\ \Big]$$

每个未来帧（47 维）：5 个关键点（45）+ $u/0.065$（2）。

网络输入是全部帧拼接：$10\times 62 + 32\times 47 = 2124$ 维；MLP 输出 $32\times 13 = 416$ 维。

### 4.3 输出参数化：以"右手刚性握持"为零点

设 $g_\tau = (p^g_\tau, q^g_\tau)$ 是右末端执行器的位姿。先求 $t$ 时刻物体相对右手的位姿：

$$p^\text{rel} = R(q^g_t)^\top(p^o_t - p^g_t),\qquad q^\text{rel} = (q^g_t)^{-1}\otimes q^o_t$$

网络第 $\tau$ 帧的输出记为 $y_\tau\in\mathbb{R}^{13}$（最后一层零初始化）：

$$\begin{aligned}
\hat p_\tau &= p^g_\tau + R(q^g_\tau)\,p^\text{rel} + R_y\,(0.05\,y_{\tau,0:3}) \\
\hat q_\tau &= \text{normalize}\big(\exp(R_y\,0.1\,y_{\tau,3:6}) \otimes q^g_\tau \otimes q^\text{rel}\big) \\
\hat v_\tau &= \tfrac{\hat p_\tau - \hat p_{\tau-1}}{\Delta t} + R_y\,(0.1\,y_{\tau,6:9}) \\
\hat a_\tau &= \tfrac{\hat v_\tau - \hat v_{\tau-1}}{\Delta t} + R_y\,(1.0\,y_{\tau,9:12}) \\
\ell_\tau &= y_{\tau,12}
\end{aligned}$$

$\hat p_t, \hat v_t$ 取历史最后一帧。$y=0$ 时，罐子从 $t$ 开始被右手刚性握住、跟着手走，这是初始化时的行为。

### 4.4 更长的窗口

超过 $H$ 帧时分段串联：上一段的预测物体加上已知的未来关键点和宽度，作为下一段的历史（`rollout_members`）。诊断脚本里 64 帧的回放用的就是这个。

### 4.5 集成（$E=3$）

各成员独立预测 $\hat o^{(e)}$：

- 均值：位置、速度、加速度直接平均；四元数先与成员 0 对齐符号再平均、归一化；
- 不确定度：$u = \|\operatorname{std}_e\,\hat p^{(e)}_{t+H}\|$。

单模型和集成共用同一套接口（单模型即 $E=1$）。

### 4.6 训练

**数据**（`hskill/sim/collect.py`）：仿真回合每步记录五个关键点、双夹爪实测宽度和指令、罐子状态，以及"两指都接触罐子"标签。
同时对每个决策点用 $\Phi$ 推演一个 32 步窗口，记录 $\Phi$ 给出的五个关键点。

**窗口**（`hskill/data/buffer.py`）：按 $0.7^\text{age}$ 选一轮采集，再均匀选一个存活的决策点 $t$，取：
- 历史 $t-9..t$，回合开始之前的帧用第一帧填充；
- 条件 $t+1..t+32$：未来关键点以概率 $\rho=0.5$（`phi_fraction`）用 $\Phi$ 的，否则用仿真的；指令总是仿真记录的；
- 标签：仿真里罐子的 $t..t+32$。加速度由速度差分得到（`object_targets`）。

**损失**（只计仍处于回合内的步）：

$$\mathcal{L}_M = \Big\langle \tfrac{\|\hat p-p\|^2}{0.01^2} + \tfrac{4(1-\langle\hat q,q\rangle^2)}{0.1^2} + \tfrac{\|\hat v-v\|^2}{0.1^2} + \tfrac{\|\hat a-a\|^2}{2^2}\Big\rangle + 0.1\,\big\langle \text{BCE}(\ell, \text{held})\big\rangle$$

集成的每个成员在各自的 batch 上训练。

---

## 5. 奖励：终止奖励 + 势函数 shaping

（`tasks/can_pnp/task_spec.py`）

$$\text{placed} \iff \text{罐子在箱内} \wedge \text{直立}(\cos\ge 0.866) \wedge \text{夹爪张开}(u > 0.03)\ \text{持续}\ 0.2\,\text{s}$$

终止奖励和 shaping 分开：

$$r_k = R_\text{term}(s_{k+1}) + \gamma\,\phi(s_{k+1}) - \phi(s_k)$$

$$R_\text{term} = \begin{cases} 10 & \text{placed（回合终止）} \\ 0 & \text{摔倒 / 掉落}(z<0.2)\ /\ \text{数值发散（回合终止）} \end{cases}\qquad \phi(\text{终止状态}) = 0$$

$$\phi(s) = 0.5\,e^{-d/0.1} + 0.5\,e^{-d/0.5},\qquad d = \|p - p_\text{goal}\|$$

- 失败不给负值：回合终止，之后的所有奖励随之失去，这就是惩罚。
- 超时不算终止（以后由 critic bootstrap）。
- 一条终止的轨迹，shaping 部分加起来恒为 $-\phi(s_0)$，与策略无关，所以 shaping 不改变最优策略，只传递"离目标更近了"的信号。
- 双尺度：0.5 m 那一项让远处也有信号（0.86 m 处 $\phi\approx 0.09$，单用 0.1 m 尺度只有 $2\times10^{-4}$），0.1 m 那一项让靠近目标时仍然精细。

不折扣（$\gamma=1$）时，一个回合或一个 chunk 的奖励之和等于 `TaskTracker.potential` 在末尾的值减去起点的 $\phi$：

$$\textstyle\sum_k r_k = \Psi_\text{end} - \phi(s_0),\qquad \Psi_\text{end} = \begin{cases} 10 & \text{placed} \\ 0 & \text{失败} \\ \phi(s_\text{end}) & \text{否则} \end{cases}$$

仿真 GRPO 的回合回报和想象里的 chunk 回报目前都用这个不折扣的形式。同一组从同一个起点出发，$\phi(s_0)$ 相同，不影响组内排序。

---

## 6. 想象中的打分：实际分支 $J$ 与理想分支 $J^*$

（`hskill/executors/imaginer.py`）从记录的决策点出发：DiT 的观测 token $c$、机器人状态 $s_0$、tracker 观测 $o^\text{hist}_0$、$\xi$、物体 WM 的历史 $\mathcal{H}_t = (K, w, o)_{t-9:t}$、当前物体 $x_0$、箱子中心 $b$（只用于任务判定）、目标点。对候选 $w$：

$$r = D(c,w),\qquad s_{0:H} = \Phi(r, s_0, o^\text{hist}_0, \xi),\qquad u_k = \text{grip}_k\ (\text{双夹爪})$$

**实际分支**：S0 实际带到的关键点

$$K_{t+k} = \varphi_K(s_k),\qquad k = 1..H$$

**理想分支**：意图被完美执行（`ideal_branch="full"`，默认）。全部关节和根部朝向取参考，根部位置从起点的实际根部出发、按参考的腿部里程计积分（`reference_frames`）；五个关键点全部来自参考。状态 $k$ 对应参考第 $k{-}1$ 帧，因为 SuperMimic 用参考第 $k$ 帧评判第 $k$ 步之后的状态：

$$s^*_k = \big(p^\text{root,odo}_{k-1},\ q^\text{root,ref}_{k-1},\ q^\text{ref}_{k-1}\big),\qquad K^*_{t+k} = \varphi_K(s^*_k)$$

可选 `ideal_branch="arms"`：只把 $\Phi$ 身体的 14 个手臂关节换成参考，骨盆和双脚与实际分支相同（只衡量手臂的执行误差）。

全身版本的注意点：根部位置是推算的，推算误差会计入 $J^*-J$；参考在物理上不一定可行；chunk 之间的跳变作用在全部五个点上（诊断脚本的 `smooth_seams` 对五个点都做平滑）。

**两个分支用同一个物体 WM、同一段历史，只差未来的关键点**：

$$\boxed{\;J = \Psi\big(M(\mathcal{H}_t,\ K_{t+1:t+H},\ u)\big) - \Psi_0,\qquad J^* = \Psi\big(M(\mathcal{H}_t,\ K^*_{t+1:t+H},\ u)\big) - \Psi_0\;}$$

- $\Psi_0 = \phi(s_0)$ 是起点的 shaping 势函数；
- 目前 chunk 之后不加任何价值估计（critic 已删除，待重新设计）；代码里 `chunk_returns` 与 `returns` 相同，排序检查用前者；
- 执行损失：$J^* - J$；
- 每个分支各有集成不确定度 $u$、$u^*$。

---

## 7. critic（已删除，待重新设计）

2026-11-09 删除了 progress critic $W$ 和 residual critic $V$（`hskill/learners/critic.py`、`hskill/data/demo_progress.py`）。
删除的原因：两者都是蒙特卡洛回归，代表的不是当前策略的价值；progress critic 在想象中只看本体。
buffer 里给 critic 准备的样本字段（`final`、`time`、`step_frac`、`to_end`、`policy_only`）和 Episodes 的 `end_step` 保留，供新 critic 使用。

---

## 8. 策略更新：GRPO（PPO-clip 形式）

（`hskill/learners/grpo.py`）

### 8.1 组内优势

把同一组 $G$ 个样本的分数 $R_{1:G}$ 标准化，只用保留的成员（keep mask $m$）：

$$\bar R = \frac{\sum_i m_i R_i}{\sum_i m_i},\quad
\sigma_R = \sqrt{\frac{\sum_i m_i (R_i-\bar R)^2}{\sum_i m_i - 1}},\quad
A_i = \begin{cases} \dfrac{R_i - \bar R}{\sigma_R + 10^{-6}} & m_i \wedge \sigma_R > \sigma_\text{min} \wedge \sum m \ge 2 \\ 0 & \text{否则} \end{cases}$$

### 8.2 PPO 损失

对所有存活决策点 $j$：

$$\rho_j = \exp\big(\log\pi_\theta(w_j\mid c_j) - \log\pi_{\theta_\text{old}}(w_j\mid c_j)\big)$$

$$\mathcal{L}_\pi = \frac{1}{N}\sum_j \Big[-\min\big(\rho_j A_j,\ \mathrm{clip}(\rho_j, 1\pm\epsilon) A_j\big) + \beta\,\mathrm{KL}\big(\pi_\theta(\cdot\mid c_j)\,\|\,\mathcal{N}(0,I)\big)\Big]$$

$\epsilon = 0.2$，$\beta = 10^{-3}$；近似 KL 超过 `target_kl` 就提前停止这一轮更新。

### 8.3 仿真 GRPO（DSRL 本身；`train_dsrl.py`，以及 `train_v3.py` 的 `sim_anchor`）

- 每组环境从同一个抓取起点出发；回合回报 $R_e = \Psi_\text{episode end}$；
- 同一回合的所有决策点共享该回合的优势 $A_e$；
- `train_dsrl.py`：每 GPU 32 组 × 8 个环境，全部用策略噪声；评估时另跑 mean 模式（第一轮还跑 base 模式）；
- `train_v3.py` 的 `sim_anchor`：16 个环境一组，其中 25% 的组用 $w\sim\mathcal{N}(0,I)$（base），base 组不参与更新。

### 8.4 想象 GRPO（`train_v3.py: imagine_update`）

- 从 buffer 取 256 个决策点，每个配 16 个候选 $w \sim \pi_\theta(\cdot\mid c)$；
- 打分：

$$R = J + \lambda\,\max(J^*-J,\ 0),\qquad \lambda = \texttt{intent\_weight}\ (\text{默认 } 0)$$

- 保留条件：$m = \text{有限} \wedge u \le 0.03\,\text{m}$（$\lambda>0$ 时还要 $u^*\le 0.03$）；
- 平局阈值：$\sigma_\text{min} = 10^{-3}\,\Psi_0$；
- 每次更新用 `imag_target_kl` = 0.01 提前停止；一个 cycle 内策略相对 cycle 开始时的 KL 超过 `cycle_kl` = 0.05 就停止想象更新。

### 8.5 排序检查与门控

每个 cycle：$G$ 组环境从同一起点出发，共享噪声走若干个决策点，然后每个环境执行自己的候选，完整跑完一个 chunk。比较仿真结果和想象的 $J^\text{chunk}$：

$$\tau = \text{Kendall-}\tau\big(J^\text{chunk}_{1:8},\ \Delta\Psi^\text{sim}_{1:8}\big)\quad(\text{组内平均})$$

- $\tau < 0.2$：本 cycle 跳过想象，下个 cycle 强制采集；
- $\tau < 0.3$：想象更新次数减半；$\tau > 0.6$：加倍（范围 1–32）。

同时记录理想分支在"S0 精确执行"的候选上的误差（`rank/ideal_*`、`rank/gap_*`）。

---

## 9. 一个 cycle（`train_v3.py`）

1. **采集**（每 `collect_every` = 4 个 cycle，或被门控强制）：
   - 仿真跑一轮抓取起点回合；
   - 评估物体 WM 在新数据上的误差（`wm_fresh/*`）；
   - 做一次仿真 GRPO；新数据入 buffer；
   - 成功率比历史最好低超过 0.04 就回滚策略，并把 `cycle_kl` 减半。
2. **拟合**：物体 WM 各成员 200 步。
3. **排序检查**：得到 $\tau$，调整想象更新次数或跳过。
4. **想象更新**：最多 `imag_iters` 次想象 GRPO。

---

## 10. 默认数值

| 类别 | 参数 |
|---|---|
| 时间 | 50 Hz；回合 14 s（700 步）；每 8 步重规划；推演 $H=32$ |
| 噪声策略 | MLP 512；$\log\sigma_0=0$，下限 −2.5；lr $2\times10^{-5}$ |
| PPO | epochs 4，minibatches 8，$\epsilon$ 0.2，$\beta$ $10^{-3}$，target_kl 0.05，grad clip 1.0 |
| 仿真 | DSRL：每 GPU 32 组 × 8；v3：每 GPU 512 环境，组大小 16，base 比例 0.25；罐子 0.1 kg |
| 想象 | 256 起点 × 16 候选；imag_iters 8（1–32）；imag_target_kl 0.01；cycle_kl 0.05 |
| 物体 WM | 历史 $L$ = 10，预测 $H$ = 32；MLP 2124 → 512×3 → 416；lr $3\times10^{-4}$；batch 4096；预热 2 轮采集 + 2000 步，之后每 cycle 200 步；$\rho$ = 0.5；$E$ = 3；$u_\text{max}$ = 3 cm |
| buffer | 8 轮采集，权重 $0.7^\text{age}$ |
| 排序检查 | 每 cycle；组大小 8；共享噪声 0–6 个决策点；$\tau$ 阈值 0.2 / 0.3 / 0.6 |
| 任务 | 成功 10，失败 0；shaping 尺度 (0.1, 0.5) m、权重 (0.5, 0.5)；保持 0.2 s；直立 cos 0.866；松手 0.03 m；掉落高度 0.2 m |

---

## 11. 两个 WM 的对照

| | 身体 WM $f_\omega$ | 物体 WM $M_\psi$ |
|---|---|---|
| 来源 | SuperMimic，随 tracker 一起训练好 | 本项目，在 can_pnp 仿真数据上在线训练 |
| 状态 | 机器人 71 维 | 罐子：历史 10 维（位置、四元数、速度），预测 13 维（再加加速度） |
| 条件 | 关节动作 $a$（来自 $\pi_{S0}$）、上下文 $\xi$ | 10 帧历史（五个关键点、双夹爪宽度、罐子）+ 未来 32 帧的五个关键点和双夹爪指令（无场景物体） |
| 输出 | 状态增量；终止概率；解析观测 | 未来 32 帧罐子的位置、朝向、速度、加速度，以及 held logit（一次输出） |
| 参数化 | 直接预测增量后积分 | 以"右手刚性握持"为零点的残差，速度、加速度用有限差分加修正 |
| 推演 | 与 $\pi_{S0}$ 闭环，得到 $\Phi$ | 开环，未来关键点来自 $\Phi$（实际）、参考（理想）或仿真（训练）；超过 32 帧时分段串联 |
| 在算法里的作用 | 意图 $r$ → S0 实际带到的关键点（实际分支）；身体姿态 | 关键点轨迹 → 罐子结果 → $\Psi$，给 $J$ 和 $J^*$ 打分 |
| 状态 | 冻结 | 训练中；3 个成员，分歧用来过滤候选 |
| 已知局限 | 未建模夹爪和物体对身体的反作用 | 用仿真物体状态做输入（不通用）；没有场景信息，罐子碰桌子、落进箱子时不准；用实际轨迹训练，却被用来评估反事实轨迹 |
