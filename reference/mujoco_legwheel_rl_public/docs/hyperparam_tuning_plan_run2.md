# 调参计划 run2:冻结网络结构,以 reward shaping 为主攻方向

前置阅读:`docs/hyperparam_tuning_plan.md`(总体方法论)、`docs/tuning_run1_results.md`(run1 结果)。
本文档先反思 run1 的设计失误,再给出 run2 计划。

## 1. run1 反思:四个设计失误

### 1.1 网络结构进了搜索空间 —— 与端上部署约束直接冲突

run1 把 `policy_net`/`value_net` 当作普通因子放进 PB 和 TPE,搜索自然选了更大的
policy([64,64]→[128,64])。但策略网络要部署到端上,推理耗时有硬要求,**参数量不许增加**。
这意味着 cand010 的"获胜"里有一部分来自网络加宽,而这部分恰恰是不可部署的。
更麻烦的是:cand010 的增益中,控制参数贡献多少、网络加宽贡献多少,run1 无法分离
(两者在同一搜索空间里联动)。**run1 的直接结论不可部署,只能借用其中的控制参数。**

### 1.2 主指标太窄 —— 半数 reward 分项没人管

主指标 RMSE = sqrt((err2_lin_vel + err2_ang_vel)/ep_len),只覆盖**线速度 + yaw 角速度**
两个跟踪量。其余全部靠 S(θ) sanity check 和存活/能耗硬约束兜底——这些是"不许变坏"的
闸门,不是"要变好"的目标。TPE 对 sinyaw(航向保持)、height、action_rate 没有任何优化
梯度。结果完全印证:12 个分项里仅 4 个明显改善(ang_vel +18.3%、energy −26.8%、
action_rate −15.4%、pitch_velocity −6.3%),**tracking_sinyaw −0.7% 原地踏步**,
height 两项甚至退化 −9%。

### 1.3 reward 权重被 PB 一刀切冻结 —— shaping 旋钮全部缺席

阶段一 PB 把全部 8 个 `reward_config.*` 因子排在 13 名开外并冻结。但这个"不敏感"
结论很可能是**伪结论**:reward 权重改变的是收敛后的**稳态行为平衡点**,不是收敛速度;
PB 在 1e8 短训 + 单 seed + 只测两个水平的条件下,对这类因子系统性失明(它们的影响
还没显现就被 σ_noise=0.047 淹没了)。于是阶段二只剩下优化器旋钮(lr、updates、网络)
和控制增益,**没有任何一个能直接压 action_rate、抬 sinyaw 的旋钮**。
run1 的两个主要症状——action_rate 高企(baseline 822 / cand010 696,绝对值都太高)、
sinyaw 无长进——正是 shaping 缺失的直接后果。

### 1.4 单一标量目标导致"拆东墙补西墙"

总 return 提升(cand010 5176 vs baseline 5028)掩盖了分项间的此消彼彼伏:height 两项
退化换来 ang_vel 和能耗改善。单目标搜索下这是必然——目标里没有的分项就是可牺牲的。
run2 必须把"分项不退化"从纸面规则变成搜索的硬约束。

## 2. run1 → run2 计划变更对比

| 维度 | run1 | run2 | 变更原因 |
|---|---|---|---|
| 网络结构(policy/value) | 进搜索空间(PB 因子 + TPE 离散选择) | **物理删除,冻结为 [64,64] / [512,512,512,512]** | 端上部署对推理耗时有硬要求,参数量不许增加(失误 1.1) |
| reward 权重(`reward_config.*` 数值型) | PB 短训筛查后全部冻结,从未被调 | **11 个全部进搜索空间,是 R1 的主角** | PB"不敏感"是伪结论,shaping 旋钮缺席导致 action_rate 高、sinyaw 不涨(失误 1.3) |
| 主指标 | 归一化跟踪 RMSE(lin_vel + ang_vel 误差) | 不变(口径与 run1 可比) | 保持可比性 |
| 分项约束 | 仅存活 ≥550 步、能耗 ≤ 预算两条 | **新增(达成率/per-step 口径,第 5 节):action_rate 部署(deterministic)口径 ≤0.15/步、sinyaw 达成率 ≥base2+2pp、关键分项达成率退化 ≤2pp**,违反记 WORST | 单目标下旁观分项必然被牺牲(失误 1.2/1.4) |
| 判定标准 | RMSE Δ > 2·σ_pool 即显著 | **帕累托判定:RMSE 显著 + 5 条分项条件全过才算赢** | 杜绝拆东墙补西墙 |
| 搜索起点 | 原始 baseline | **base2 = cand010 增益类参数(lr/updates/kp/notsmoothing)移植回 baseline 网络;动作映射/扭矩限幅按物理·固件对齐固化(不取搜索值)**,先验证(R0)再搜索 | 复用 run1 有效结论,剥离不可部署部分;物理参数不交予搜索(见 4.3) |
| 阶段流程 | 阶段 0 噪声底 → PB 筛选 → TPE+ASHA → 全量确认 | **R0 base2 验证 → R1 TPE+ASHA(免 PB)→ R2 全量确认** | 噪声底复用 run1(σ=0.0472 不重跑);reward 权重类因子对 PB 短训系统性失明,不再用 PB 筛 |
| 搜索因子数 | 29(PB)→ 11(阶段二) | 13(11 个 reward 权重/sigma + kp、notsmoothing 随行) | 聚焦 shaping |
| 初筛方式 | 无原理初筛,直接 PB 实验筛选(29 因子 × 32 runs) | **原理初筛先行(第 4 节,逐项过机制)→ 免 PB 直接 TPE** | PB 对稳态塑形类因子系统性失明;原理初筛零算力成本、可解释、可审查 |
| 指令范围因子(cmd_* 5 个) | 进 PB,阶段一后冻结 | 不进空间(保持 base2 值) | 与行为塑形关系间接,收缩搜索维度 |
| PPO 优化器超参(lr、updates 等) | 进搜索空间 | 固定为 cand010 值(lr 3e-4、updates 8) | run1 已收敛出可靠区间,把预算让给 reward 权重 |
| 能耗/存活约束 | 有 | 沿用(预算值复用 run1 的 0.0423) | 不变 |
| 工程改动 | 新建整条流水线 | 只改 space/metrics/worker/report/tune.py + rewards 的 height 投影(5.1);**obs、网络、部署导出零改动** | 约束评估读已有 reward_terms 即可 |
| 预算 | ~27 卡时(实测) | **~10 卡时(双卡约 5 小时)** | 免 PB、噪声底复用、搜索轮次减少 |
| gate 设置 | gate-1/2/3 | gate-R0(base2 是否成立)/ gate-R1(约束满足率,过紧则放宽一档)/ gate-R2(帕累托 + 部署审查) | R1 新增"搜索空跑"保险 |

## 3. run2 设计原则

1. **部署约束前置**:policy/value 网络结构冻结为 baseline([64,64] / [512,512,512,512]),
   从搜索空间物理删除(不是"冻结因子",是根本不进 space 定义);任何候选的 obs/act 维度、
   单次推理 FLOPs 与 baseline 完全一致,端上耗时零变化。
2. **reward 权重是主角**:11 个数值型 `reward_config.*` 全部进搜索空间(pitch_scale=0
   保持关闭,见第 4 节初筛),这是对 run1 失误 1.3 的直接纠正。
3. **目标函数升级为"主指标 + 分项硬约束"**:RMSE 仍为主指标,但 action_rate、sinyaw
   设绝对/相对门槛,关键分项设不退化容差(纠正 1.2/1.4)。
4. **复用 run1 资产**:噪声底(σ=0.0472)不重跑;cand010 的控制参数经移植验证后作为
   新起点;工程流水线(tune.py 三阶段、双卡调度、断点续搜)不动,只改 space 和目标/约束
   评估代码。

## 4. 全参数原理初筛表

初筛原则:每个参数先回答三个问题——**(a) 它的机制是什么、影响哪个行为/指标;
(b) 调它会不会改变部署接口(观测/动作语义、推理耗时、固件对齐);
(c) 调它是不是在移动评估球门(任务分布/协议口径)**。只有 (b)(c) 都为否、
且机制上与待改进症状(action_rate 高、sinyaw 不涨、分项此消彼长)相关的参数才进搜索。

结论标记:**搜索** = 进 R1 空间;**base2** = 固定为 cand010 移植值(run1 已收敛);
**冻结·部署** = 改动会破坏端上耗时/固件对齐;**冻结·结构** = 改动重定义观测/动作/
奖励语义;**冻结·协议** = 改动等于移动评估球门;**冻结·稳定器** = 非性能旋钮;
**冻结·本轮** = 有价值但不在本轮范围。

### 4.1 PPO/训练超参(12 项)

**half_life_s 的理论分析**(0.5s 是否合理):

数值换算(`ppo.py:76-85`):half_life=0.5s、ctrl_dt=0.01s → γ=0.5^(0.01/0.5)≈**0.9862**,
有效视野 H=1/(1−γ)≈**72 步 ≈ 0.72s**,回报权重 1.5s 后衰减到 1/8。
与任务各时标对照:控制周期 0.01s;姿态失稳时间常数 τ=√(l/g)≈0.1~0.2s
(轮腿倒立摆,等效质心高 0.1~0.3m);平衡纠偏因果视野 ~0.5~1s;指令切换 3s;episode 6s。

- **站得住的部分**:本任务 reward 稠密(每步有跟踪/平滑/能耗项),credit assignment
  不依赖长视野;拉长 γ 的主要代价是 GAE 方差上升、值函数更难学。稠密奖励 + 快动态,
  短半衰期是标准做法,0.5s 与平衡纠偏视野(0.5~1s)量级一致、偏短。
  (run1 PB 显示低水平略好,但那测的是收敛速度,同一类 PB 失明,不作数。)
- **死亡回避问题——已由数据判定(不再是开放问题)**:策略在环真实 rollout
  (4096 envs 带推力,`tools/measure_eval_dynamics.py`,
  图 `tuning/run1/reports/failure_dynamics_v2.png`):
  - 失败前 0.2s 姿态中位数仍正常(≈+0.03),随后急转直下;**100% 的失败从
    穿越中带(-0.3)到触线(-0.6)不超过 0.6s**,中位 0.08s——主导失败模式是
    **推力触发的快失稳**(85% 的失败前 1s 内有推力,随机基线约 33%),
    doomed 集的演化完全落在 0.72s 视野内;
  - **结论:"解决 ~1s 内存放 ≈ 解决长期存活"对本任务成立,不需要全局长视野**;
    存活瓶颈是推力后的快速纠偏能力(鲁棒性/增益),不是视野问题。
    termination_penalty 消融保留,但 half_life 消融优先级降低。
  - ⚠️ 工程教训:batch_context **离线重放跨进程不是 bit-identical**
    (XLA matmul 算法/GPU 非确定性,实测仅 28/212 失败步数复现,其余 ~1.4s
    提前摔倒);凡轨迹级统计必须策略在环重跑,不可用离线重放
    (早期 v1 图 failure_dynamics.png 基于失效重放,作废,以 v2 为准)。
- **可质疑二(平滑性)**:action_rate 是即时项,与 γ 无机制通路;本轮冻结它不耽误治病。
- **结论**:冻结·本轮成立;鲁棒性专项加一条消融——half_life 0.5/1.0/2.0s × 摔倒率,
  用 batch_test 验证"近视导致摔倒"假设。

| 参数 | 现值 → run2 取值 | 原理(机制 → 影响面) | 初筛结论 | 依据 |
|---|---|---|---|---|
| learning_rate | 1e-4 → 3e-4 | 优化步长 → 收敛速度与最终策略质量 | **base2** | run1 PB + TPE 一致收敛到高端,已可靠 |
| num_updates_per_batch | 4 → 8 | 每批样本复用轮数 → 样本效率 vs 过拟合 | **base2** | run1 TPE top 组一致取 8 |
| entropy_cost | 1e-2 → 不变 | 探索熵正则 → 探索/利用平衡 | 冻结·稳定器 | 只影响探索强度,不直接塑形稳态行为;run1 PB 主效应弱 |
| half_life_s | 0.5(γ≈0.986,视野≈0.72s)→ 不变 | 回报半衰期 → credit assignment 视野 | 冻结 | 与平衡纠偏视野量级一致;失败动力学诊断(见上)证实主导失败为推力触发的快失稳(发散中位 0.08s),近视原理上充分,无需长视野 |
| reward_scaling | 0.1 → 不变 | reward 数值缩放 → 与 lr 联动的有效步长 | 冻结·稳定器 | 理论上不改变最优点,与 lr 冗余(调 lr 已覆盖) |
| unroll_length / num_minibatches / batch_size | 75 / 16 / 512 → 不变 | rollout 结构 → num_envs、批次统计、GPU 吞吐 | 冻结·结构 | 三项耦合决定并行度;改动同时改变优化动力学与硬件利用,无法归因 |
| max_grad_norm | 1.0 → 不变 | 梯度裁剪 → 训练稳定性 | 冻结·稳定器 | 保险丝,不是性能旋钮 |
| normalize_observations | True → 不变 | 运行统计归一化 → 观测分布 | 冻结·部署 | 部署侧 extract 已带 running_statistics,必须一致 |
| policy/value 网络结构 | [64,64] / [512,512,512,512] → 不变 | 容量 → 推理 FLOPs | **冻结·部署** | 端上耗时硬约束,从空间物理删除(失误 1.1) |

### 4.2 速度环 vel_pid(轮速→扭矩,11 项)

| 参数 | 现值 → run2 取值 | 原理(机制 → 影响面) | 初筛结论 | 依据 |
|---|---|---|---|---|
| target_scale | 30 → 不变 | 动作→残差轮速映射增益 → 轮速响应幅度 | 冻结·部署(固化) | 物理/固件映射常数,与固件对齐,不交予搜索(见 4.3 分析) |
| kp | 2e-4 → 搜索 [1.2e-4, 2.2e-4] | 残差累加 P 增益 → 力矩响应快慢/抖动 | **搜索**(base2 值 1.7e-4 ±30% 随行) | 与 action_rate、能耗直接相关;权重变化后需重新匹配 |
| notsmoothing | 0.5 → 搜索 [0.18, 0.33] | 目标轮速低通(越小越平滑)→ 轮速指令抖动 | **搜索**(base2 值 0.25 ±30% 随行) | 机制上就是平滑旋钮,与 action_rate 症状对口 |
| torque_limit | 0.0295 → 不变 | 轮电机扭矩限幅 → 物理安全边界 | 冻结·部署(固化) | 电机额定/固件限幅,物理参数不交予搜索 |
| ki / kd | 0 / 0 → 不变 | 积分/微分增益 | 冻结·部署 | 固件侧即纯 P,训练须与固件一致 |
| integral_limit / iters | 1.0 / 4 → 不变 | 积分限幅/每步 PID 迭代数 | 冻结·部署 | 与固件 control.cpp 对齐项 |
| direct_torque / residual_target | False / True → 不变 | 扭矩生成方式/目标语义 | 冻结·结构 | 重定义动作语义,与固件残差模式对齐 |
| target_bias | 0.0 → 不变 | 目标映射偏置 | 冻结·结构 | 0 为对称映射,无非零物理理由 |
| obs_residual_history | True → 不变 | 观测是否含残差历史 | 冻结·部署 | 改变观测维度/语义 |

### 4.3 动作映射(7 项)

**动作映射的目的(机制分析,依据 `env.py:270-295`、`rewards.py:275-279`)**:

1. **归一化映射(本质)**:物理值 ↔ 网络归一化输出 [-1,1] 之间的单位换算层
   (thigh: `qpos_now + a×scale` clip 到 range;轮: `a×target_scale` 残差叠加)。
   scale/bias 就是这本换算字典的刻度。
2. **前向增益整形**:`action_scale.thigh` = 大腿姿态每步最大增量(姿态变化率上限),
   `target_scale` = 轮速指令每步最大残差;决定响应带宽、分辨率与物理抖动幅度。
3. **可行域裁剪**:`action_range.thigh` 限定蹲姿可达集(clip 边界)。

**由"归一化映射"本质得出的关键推论**:scale 的内部变化与网络输出层是
**重参数化关系**——映射变窄,策略学着输出更大的值即可补偿,表达能力不变。
因此映射参数里真正不可被网络补偿的只有三处:

- **限幅边界**(action_range 的 clip、tanh 饱和区):物理可达集,网络补不了;
- **每步变化率上限**(增量模式下 scale × 最大输出):硬带宽边界;
- **归一化空间与物理空间的惩罚错位**:`action_rate`、`energy_cost` 都作用在
  归一化输出上(`rewards.py` 中 `act - last_act`、`jp.abs(action[i])`,未经 scale),
  而真实物理抖动/能耗是 ×scale 后的量。缩 scale 可在惩罚项数值不变的情况下
  降低物理抖动——这是映射内部尺度唯一有独立信息量的效应,本质是钻了
  "奖励定义在归一化空间"的空子,属于可利用但不干净的手段。

**是否应搜索的判断:不搜,固化。**

- 原理理由(主):动作映射是**确定性、由物理/固件定义**的参数——电机额定扭矩、
  关节行程、控制频率、固件映射常数都是先验给定,不是"学出来"的。让搜索动它
  等于让优化器改写物理定义;且 scale 内部变化与网络输出层是重参数化关系
  (见上),搜索它约等于替网络学一个它自己能学的自由度。
- 实证教训:run1 把 action_range.thigh 搜到 [-0.283, 0] 导致蹲姿变浅,
  部署审查反而成了风险项——这就是"搜索动了物理参数"的直接后果。
- **固化取值来源原则:物理规格/固件常数 > 搜索收敛值**。cand010 的
  0.367 / [-0.283,0] / 44.4 本身是搜索产物,不能"追封"为物理常数;
  固化值取固件对齐值(= baseline:0.25 / [-0.5,0] / 30 / 0.0295 / 0.0295),
  即 `docs/deploy_firmware_comparison.md` 的核对项,训练与固件必须一致。
- 残余风险与预案:权重与增益存在补偿关系(action_rate_scale 加重 → 策略变保守
  → 最优 scale 上移),只搜权重一头可能卡次优点。若 gate-R1 确认卡约束且
  实机/固件允许调整映射常数,才以固件对齐值 ±20% 窄范围放回搜索
  (此时固件侧需同步改常数,走部署变更流程,不再是纯训练侧决策)。

| 参数 | 现值 → run2 取值 | 原理(机制 → 影响面) | 初筛结论 | 依据 |
|---|---|---|---|---|
| action_scale.thigh | 0.25 → 不变 | 大腿动作缩放 → 蹲姿调节幅度 | 冻结·部署(固化) | 物理映射参数,取固件对齐值,不交予搜索(见上分析) |
| action_range.thigh | [-0.5, 0] → 不变 | 大腿动作限位 → 最低/最高姿高 | 冻结·部署(固化) | 关节行程/物理可达集;run1 搜到 -0.283 致蹲姿变浅即为教训 |
| action_scale.wheel | 0.0295 → 不变 | 轮动作缩放(PID 模式下被 target_scale 覆盖) | 冻结·部署(固化) | 物理映射参数,取固件对齐值 |
| action_bias.thigh | -1.0 → 不变 | 动作偏置(-1 → 半蹲中位) | 冻结·结构 | 与 scale/range 耦合定义动作中位,改动=重定义语义 |
| action_bias.wheel | 0.0 → 不变 | 轮动作偏置 | 冻结·结构 | 0,无物理理由改 |
| delta_thigh_action / last_act_obs | True / False → 不变 | 增量动作/观测不含 last_act | 冻结·部署 | 与已部署快照(82ba252 时代)对齐,改观测=改部署接口 |

### 4.4 reward 权重与形状(13 项)—— 本轮主角

| 参数 | 现值 → run2 搜索范围 | 原理(机制 → 影响面) | 初筛结论 | 依据 |
|---|---|---|---|---|
| action_rate_scale | -0.5 → 绝对值 [0.05, 4.0](0.1×~8×) | 动作变化率惩罚 → **平滑度** | **搜索** | 主攻"action_rate 太高";机制最直接的旋钮 |
| zero_keep_scale | 10.0 → [3, 30](0.3×~3×) | 零指令保持奖励总闸 → **sinyaw/sinposition 航向与轮位保持** | **搜索** | 主攻"yaw 没有长进";sinyaw 奖励经此加权 |
| sigma | 0.5 → [0.25, 1.0](0.5×~2×) | 跟踪奖励容差带宽(exp(-err/σ))→ 奖励梯度粗细 | ~~搜索~~ **冻结 0.5**(事后决议) | σ 小奖励苛刻易饱和,σ 大无区分度。**教训:σ 兼为评分标准,搜它等于让各组自选裁判松紧,跨组 reward 不可比(run2 cand000/026 即靠 σ≈0.87 灌水);后续轮次一律固定代码默认值,只搜权重** |
| sigma_wheel | 0.5 → [0.25, 1.0](0.5×~2×) | 轮位置跟踪容差 | ~~搜索~~ **冻结 0.5**(事后决议) | 同上,轮位通路专用;冻结原因见 sigma 行 |
| vel_scale | 2.0 → [0.6, 6.0](0.3×~3×) | 线速度+yaw 角速度跟踪权重 → 主任务拉力 | **搜索** | 与 RMSE 主指标对口;run1 曾作锚定项,run2 放开但由主指标约束防漂 |
| height_scale | 2.0 → [0.6, 6.0](0.3×~3×) | 高度跟踪权重 | **搜索** | run1 中 height 两项退化 9%,需给修复手段 |
| roll_scale | 4.0 → [1.2, 12](0.3×~3×) | roll 姿态权重 | **搜索** | batch_test 失败模式全为 pitch/roll 型,姿态权重可调 |
| energy_scale | -1.0 → 绝对值 [0.1, 4.0] | 能耗惩罚 | **搜索** | 与 action_rate 协同(平滑↔省能) |
| pitch_velocity_penalty_scale | -0.5 → 绝对值 [0.05, 2.0] | 俯仰角速度惩罚 → 抗点头 | **搜索** | 失败模式为 pitch 型,直接对口 |
| base_scale | 1.0 → [0.5, 2.0] | 存活保底奖励 → "活着" vs "完成任务"的平衡 | **搜索** | 影响存活与跟踪的取舍点 |
| posture_scale | 1.0 → [0.5, 2.0] | 姿势基础系数 | **搜索** | 姿态项总增益 |
| pitch_scale | 0.0 → 不变 | pitch 姿态代价权重 | 冻结·本轮 | 开启=启用未验证通路,与 pitch_velocity_penalty 功能重叠 |
| termination_penalty_enable/value | False / -10.0 → 不变 | 摔倒瞬间一次性惩罚 → 危险区局部值函数 | 冻结·本轮 | (1) 稀疏信号:训练死亡率仅 ~4%,每 episode 最多触发一次,1e8 短训中信噪比极低,TPE 测不准;(2) 死亡已自带隐式惩罚(损失后续跟踪回报,半衰期 0.5s 下约 0.7s 视野),显式项只是加系数;(3) 与本轮症状(action_rate/sinyaw/分项平衡)无机制通路。它对应的真问题是 batch_test 摔倒率,应在鲁棒性专项(与 C 层/推力一起,以成功率为指标)单独消融 |

### 4.5 指令分布 cmd_*(评估协议,9 项)

| 参数 | 现值 → run2 取值 | 原理(机制 → 影响面) | 初筛结论 | 依据 |
|---|---|---|---|---|
| lin_vel_x 范围 | [-0.2, 0.2] m/s → 不变 | 指令采样分布 → **定义任务本身** | 冻结·协议 | 调它=移动球门,run1/run2 指标立刻不可比;须与实机使用场景一致 |
| ang_vel_yaw 范围 | [-1.5, 1.5] rad/s → 不变 | 同上 | 冻结·协议 | 同上 |
| height_value 范围 | [-0.5, 0] → 不变 | 同上 | 冻结·协议 | 同上 |
| roll_value 范围 | [-0.2, 0.2] → 不变 | 同上 | 冻结·协议 | 同上 |
| cmd_prob | 0.9 → 不变 | 有指令概率 → 任务混合比 | 冻结·协议 | 同上 |
| command_change_interval / cmd_start_delay_s | 300 步(3s)/ 0.2s → 不变 | 指令切换节奏 | 冻结·协议 | 同上 |
| cmd_lin_vel_x / ang_vel 零死区 | 0.000195 / 0.01 → 不变 | 零指令死区 → 零保持奖励的激活边界 | 冻结·部署 | 与固件死区语义对齐 |

### 4.6 情节/仿真结构(8 项)

| 参数 | 现值 → run2 取值 | 原理(机制 → 影响面) | 初筛结论 | 依据 |
|---|---|---|---|---|
| ctrl_dt / sim_dt / action_repeat | 0.01 / 0.0025 / 1 → 不变 | 控制 100Hz/仿真 400Hz | 冻结·部署 | 与固件控制频率对齐 |
| episode_time / episode_length | 6.0s / 600 步 → 不变 | RMSE 分母与评估口径 | 冻结·协议 | 改动使所有历史指标不可比 |
| action_delay_steps | 0 → 不变 | 控制链路延迟建模 | 冻结·部署 | 与部署快照一致;改它需实机标定依据 |
| obs_history_len | 1 → 不变 | 观测历史长度 | 冻结·部署 | 改观测维度 |
| wheel_radius | 0.0195 → 不变 | 轮半径 → 线速度换算 | 冻结·部署 | 物理参数,与实机一致 |

### 4.7 观测/奖励结构开关(6 项)

| 参数 | 现值 → run2 取值 | 原理(机制 → 影响面) | 初筛结论 | 依据 |
|---|---|---|---|---|
| sinyaw / sinyaw_scale | True / 0.5 → 不变 | 航向保持通路开关/误差映射灵敏度 | 冻结·结构 | sinyaw 进观测与奖励两路,调 scale 改变观测语义;yaw 问题由 zero_keep_scale 解决更干净 |
| wheel_sin_scale / wheel_sinpos_start / wheel_pos_nosin | 0.1 / True / False → 不变 | 轮位三角映射结构 | 冻结·结构 | 观测语义 |
| sinpitch_zero | True → 不变 | 零指令 pitch 基准 | 冻结·结构 | 奖励基准定义 |

### 4.8 噪声/域随机化/推力(C 层,~30 项)

| 参数 | 现值 → run2 取值 | 原理(机制 → 影响面) | 初筛结论 | 依据 |
|---|---|---|---|---|
| noise_config.level + 6 scales | 1.0;joint_pos 0.02、joint_vel 1.0、linvel 0.01、gyro 0.02、pitch 0.02、roll 0.02 → 不变 | 传感器噪声水平 → 鲁棒性 | 冻结·部署 | 应与实机噪声实测匹配,不是性能旋钮 |
| push_config.* | interval [1,5]s、magnitude [0,0.5] m/s → 不变 | 随机推力 → 抗扰鲁棒性 | 冻结·协议 | batch_test 的考核项,调它=降低考核标准 |
| randomization 6 开关 + ~20 范围 | 默认值(见 `legwheel/randomization.py`)→ 不变 | 域随机化 → sim2real 差距 | 冻结·本轮 | 改变训练分布而非行为目标;run1 计划已定:部署前单独做 C 层消融 |

### 4.9 初筛结论统计

| 结论 | 数量 | 去向 |
|---|---|---|
| 搜索 | 13(11 reward + kp + notsmoothing) | R1 TPE+ASHA 空间,范围见 4.4/4.2 |
| base2 | 4(lr、updates,以及 kp/notsmoothing 的中心值) | R0 验证后固定 |
| 冻结·部署 | ~20(含固化动作映射/扭矩限幅) | 不动 |
| 冻结·结构 | ~14 | 不动 |
| 冻结·协议 | ~10 | 不动 |
| 冻结·稳定器 | 3 | 不动 |
| 冻结·本轮 | ~32(pitch_scale、termination_penalty、C 层) | 后续单独消融 |

## 5. 评价指标设计:子 reward 理论可达值与达成率口径

run1 的目标/约束用的是**原始 episode 和**(满分 600 的项和门控项混在一起),
不同分项量级不可比、门控项的上限被高估,导致"哪些项差、差多少"的判断失真
(失误 1.2 的另一半)。本节先从原理推导每个子 reward 的可达值,再据此设计指标。

### 5.1 跟踪/姿态项:理论上限与达成率

所有跟踪项公式均为 `exp(-|err|/σ)`(每步满分 1),σ = reward_config.sigma × cmd 半量程
(cmd_scales = [0.2 m/s, 1.5 rad/s, 0.25 rad, 0.2],sigma=0.5)。
episode 上限 = 600 × 激活率;**门控项(零指令时才激活)上限远低于 600**:

注意 sinyaw 的语义:它量的不是绝对 yaw=0,而是 **delta_yaw=0**——`sinyaw_start`
在转向指令激活期间每步跟随当前 yaw(`env.py:406-408`),转向结束时刻锁定为基准,
零指令窗口内只考核相对该基准的漂移。作用是让机器人**不转时保持住方向**
(抗扰保向),转向期间不记分;观测中的 sinyaw 也是同一 diff 量,与固件一致。
因此提升该分项的手段是减小扰动下的航向漂移(与失败诊断的"推后纠偏"同类),
旋钮是 zero_keep_scale,而不是"回正到绝对零位"。
cmd_prob=0.9 + 零死区 → 零 yaw 指令占比 ≈ 10.7% → sinyaw/sinposition 上限 ≈ 64。

| 分项 | 激活条件 | σ | 结构性上限 | baseline 实测 | 达成率 | 隐含平均误差(Jensen 下界) |
|---|---|---|---|---|---|---|
| tracking_lin_vel | 全程 | 0.1 m/s | 600 | 447.0 | 74.5% | ≥0.029 m/s |
| tracking_ang_vel | 全程 | 0.75 rad/s | 600 | 342.2 | **57.0%** | ≥0.42 rad/s |
| tracking_height_jnt | 全程 | 0.125 rad | 600(几何 ~583,见下) | 335.8 | 56.0% | ≥0.073 rad |
| tracking_height_action | 全程 | 0.125 rad | 600(同上) | 341.8 | 57.0% | ≥0.070 |
| tracking_sinyaw | 零 yaw 指令 | 0.1 | **≈64(实测激活 60.9 步)** | 47.2 | **85.0%(实测)** | ≥0.016 rad |
| tracking_wheels_sinposition | 零指令 | 0.5 | **≈64(实测激活 57.8 步)** | 30.9 | **60.9%(实测)** | ≥0.25 |
| pitch_cost | 全程 | 0.1 | 600 | 432.2 | 72.0% | ≥0.033 rad(≈1.9°) |
| roll_cost | 全程 | 0.1 | 600 | 469.6 | 78.3% | ≥0.025 rad(≈1.4°) |

两个解读修正(达成率已经策略在环实测,`tools/measure_eval_dynamics.py`):

1. "sinyaw 没有长进"需重估:其上限 ≈64 而非 600(实测激活 60.9 步/episode,
   与解析 63.6 一致),baseline 激活步达成率已 **85.0%**——折算物理量即零指令
   窗口内航向漂移仅 ~0.016 rad(≈1°,delta_yaw 口径),保向能力本就不差,
   头部空间仅 ~15pp;R1 对它的合理目标是 +2~3pp,不是翻倍。
2. 真正低达成项是 **ang_vel(57%)、height 两项(56~57%)、sinposition(60.9%)**;
   ang_vel 与主指标 RMSE 直接相关,是拉升空间最大的跟踪项。
3. **height 的上限受 roll 冲突压缩**(策略在环实测,
   `tools/analyze_roll_height_conflict.py`,图 `tuning/run1/reports/roll_height_conflict.png`):
   roll 靠 thigh 差动实现(耦合 k≈0.433 rad/unit-sinroll),与 height 共用
   thigh 行程 [-0.5, 0];14.8% 的步指令对在几何上不可同时满足 →
   height 几何上限 0.971(≈583/600);行为层面策略按权重比
   (roll_scale 4 : height_scale 2)**牺牲 height 保 roll**——roll 激活窗口
   height_jnt 均值 0.633 vs 非激活 0.728(差 ~9.5pp),而 roll_cost 几乎不受
   height 极端指令影响(0.842 vs 0.844)。因此 height 的现实可达上限
   ≈ 0.65×600 ≈ 390,当前 336 的差距只有 ~9pp,且本质是权重比的有意取舍——
   R1 搜索 height_scale / roll_scale 正是在调这个平衡点;
   `reward_config.do_deconflict`(冲突时降权,现关闭)是现成的结构性解法,
   列入 R1 后单独消融。

   **已实现的重设计(合作式投影,2026-08-27)**:(θ_L, θ_R) 正交分解为
   mean(高度)/ split(roll 差动)两个自由度,冲突只剩"指令对超出可行域"
   一种。reward 侧把 height 参考值投影到当前 roll 指令的可行区间:
   `h_ref = clip(h_cmd, -0.5+δ, 0-δ)`,δ = `roll_thigh_coupling`·|cmd_roll|
   (新增配置 `reward_config.height_feasible_projection` 默认开、
   `roll_thigh_coupling=0.433` 实测值)——roll 拿足 split,height 在剩余
   可行域内做到最好,不可行部分不再考核,二者不再互相牺牲。
   **注意:这改变了 reward 语义,base2 与 run1 baseline 的 reward 分项
   不再同口径(RMSE 主指标不受影响)**;roll 激活窗口的
   height_jnt 达成率在新口径下预期从 0.633 明显回升。

### 5.2 惩罚项:per-step 物理量口径

episode 和没有物理可读性,除以 episode_length 后才是可设定目标的物理量:

| 分项 | per-step 物理含义 | baseline | cand010 | 参考目标(gate 可调) |
|---|---|---|---|---|
| action_rate | mean\|Δa\| + 二阶差分(归一化动作,4 维均值) | 1.37 | 1.16 | **部署口径 ≤0.15**(见下方口径修正;原 ≤0.5 目标系 stochastic 口径,已废止) |
| pitch_velocity_penalty | \|pitch 角速度\| (rad/s) | 0.62 | 0.58 | ≤0.4 |
| energy_cost | 混合代理量(关节速度+轮动作) | 0.036 | 0.027 | 沿用能耗预算约束 |

baseline 的 action_rate 每步 1.37,意味着策略平均每步动作变化接近归一化量程的
70%,量化印证了"action_rate 太高"的直觉;0.5 对应每步变化 ~15% 量程,
是"平滑但不失响应"的参考点。

**口径修正(2026-08-27 实测,`tools/analyze_action_rate.py`,
baseproj-s0 末态 ckpt 策略在环分解)**:上表的 action_rate 是
**stochastic 采样口径**(brax `ppo.train` 的 `deterministic_eval=False`
默认值,训练期所有 eval 指标都含探索噪声)。实测分解:

| 口径 | action_rate/步 | 占比 |
|---|---|---|
| 训练报表(stochastic 采样) | 1.358 | 100% |
| 部署口径(deterministic 均值动作,含噪声+push+域随机化) | 0.317 | 23% |
| 干净环境(deterministic,无噪声无 push) | 0.027 | 2% |

即报表值的 **77% 是策略 σ 的采样噪声**(entropy_cost=1e-2 维持 σ 不退火),
与部署行为无关;部署口径的 0.317 里又有 **91% 是对观测噪声+push 的高增益
反应**(干净环境仅 0.027)——真机传感器噪声不会消失,这 0.3/步(折算大腿
目标 ~1.25 rad/s @50Hz 摆动)会真实出现,仍需压,但性质是"降噪增益"。

**对约束设计的影响**:若约束沿用 stochastic 报表口径,搜索只要让策略熵
塌缩(σ→0)就能"达标",哪怕确定性策略照样抖——约束失效且误导搜索。
因此约束必须改用**部署口径**(见 5.3 修正)。同理,RMSE 与存活率的报表
绝对值也含探索噪声:所有 trial 同口径对相对比较无碍,但与 batch_test
(deterministic)的绝对值不可直接比。

### 5.3 指标设计结论

1. **主指标不变**:归一化跟踪 RMSE(物理可解释、与 run1 可比)。
2. **报告与约束口径改为达成率/per-step**:
   - 跟踪/姿态项:达成率 = 分项和 ÷ 结构性上限(600 或实测激活步数);
   - 惩罚项:per-step 值 = 分项和 ÷ avg_episode_length。
3. **R1 硬约束按新口径重写**(替换第 6 节 R1 中的原始值约束):
   - **action_rate 用部署口径**(deterministic,见 5.2 口径修正):
     尾部训完后由 worker 用确定性策略加测,per-step ≤ **0.15**
     (当前部署口径 0.317,约束目标约为减半);stochastic 报表口径保留在
     仪表板观察,不作判定;
   - tracking_sinyaw 达成率 ≥ base2 达成率 + 2pp(头部空间小,不设高门槛);
   - 关键分项(lin_vel、ang_vel、pitch、roll)达成率退化 ≤ 2pp;
   - ang_vel / sinposition 不设下限改进要求,但纳入报告仪表板观察。
4. **工程配套**(metrics-only,不动 obs/reward 通路,部署零影响):
   rewards.py 增加 `track/active_steps_sinyaw`、`track/active_steps_sinpos` 计数,
   使门控项的达成率用**实测激活步数**做分母,替代 ≈64 的估算;
   worker 尾部增加确定性 eval 加测(复用 `tools/analyze_action_rate.py`
   的 rollout 口径:deterministic + 域随机化 + push + 噪声)。

## 6. run2 流程

### 阶段 R0:新起点 base2 的建立与验证(前置,约 0.5 卡时)

把 cand010 的**优化器与速度环增益**移植到 baseline 网络 + 固件对齐的映射,得到 **base2**:

- 沿用 cand010(增益类,搜索收敛可信):`learning_rate=3e-4`、
  `num_updates_per_batch=8`、`vel_pid.kp=1.7e-4`、`vel_pid.notsmoothing=0.25`
- 固化·物理/固件对齐(**不取 cand010 搜索值**,见 4.3 分析):
  `vel_pid.target_scale=30`、`vel_pid.torque_limit=0.0295`、
  `action_scale.thigh=0.25`、`action_range.thigh=[-0.5, 0]`、`action_scale.wheel=0.0295`
  —— 即全部回到 baseline/固件值
- 改回 baseline:policy [64,64]、value [512,512,512,512]

即 base2 与 baseline 的差异只剩 4 个增益(lr、updates、kp、notsmoothing);
R0 验证的问题相应收窄为"**这 4 个增益移植到原网络/原映射后是否仍然成立**"。

跑 2 seeds @1e8,对照 baseline 同长度(run1 已有,可直接比):
RMSE 应不低于 baseline 太多(预期 0.60~0.70,弱于 cand010 属正常——
cand010 的增益里本就有网络容量和映射漂移的贡献);若反而差于 baseline,
说明增益移植不成立,回退为纯 baseline 起点并在报告中如实记录。

### 阶段 R1:reward shaping 搜索(TPE + ASHA,约 25 trials @1e8)

搜索空间(全部 log 域,以 base2 为锚):

| 组 | 因子 | 范围 | 说明 |
|---|---|---|---|
| 跟踪权重 | reward_config.vel_scale / height_scale / roll_scale | 0.3×~3× | 各项跟踪拉力 |
| 零保持 | reward_config.zero_keep_scale | 0.3×~3× | sinyaw / sinposition 航向与轮位保持的总闸 |
| 惩罚 | reward_config.action_rate_scale | 0.1×~8×(绝对值) | **主攻 action_rate**,现值 −0.5,历史上试过 −100 的注释说明方向早有直觉 |
| 惩罚 | reward_config.energy_scale / pitch_velocity_penalty_scale | 0.1×~4×(绝对值) | 能耗与俯仰速度 |
| 形状 | reward_config.sigma / sigma_wheel | 0.5×~2× | 跟踪奖励的容差带宽度 |
| 基础 | reward_config.base_scale / posture_scale | 0.5×~2× | 保底项 |

(11 个数值型权重全进搜索;pitch_scale=0 固定不开,见第 4 节初筛。如 R1 发现某权重卡边界,gate 时可人工扩界。)

可选随行因子(窄范围,防止权重变化后控制增益失配):`vel_pid.kp`、`vel_pid.notsmoothing`
在 base2 值 ±30% 内随行。

**目标与约束(相对 run1 的核心改动;口径按第 5 节达成率/per-step 设计)**:

- 主指标:归一化跟踪 RMSE(同 run1,口径不变,保证与 run1 可比)
- 硬约束(任一不满足记 WORST,不参与 TPE 建模排序):
  - `action_rate` per-step(**部署口径**:deterministic 策略加测,见 5.2 口径修正)
    ≤ 0.15(**绝对物理门槛**,回应"action_rate 都太高",当前部署口径 0.317)
  - `tracking_sinyaw` 达成率 ≥ base2 达成率 + 2pp(头部空间仅 ~26pp,见 5.1)
  - 存活 ≥ 550 步、能耗/步 ≤ 预算(沿用 run1)
  - 关键分项(lin_vel、ang_vel、pitch_cost、roll_cost)达成率退化 ≤ 2pp(**不许拆东墙**)
- ASHA 剪枝、双卡 flock 调度、SQLite 断点续搜:沿用 run1 实现

约束评估只需读各 trial 已有的 `reward_terms.json`(run1 已直出)+ 5.3 新增的
激活步数计数,改 `legwheel/tuning/metrics.py` 的判定函数和 `worker.py` 的
WORST 逻辑即可;obs 通路零改动。(2026-08-27 修订:rewards.py 有改动——
height 参考值可行域投影,见 5.1 修正点 3,部署侧不受影响。)

### 阶段 R2:全量确认与帕累托判定(4e8 × 3 seeds)

top-2 候选 + base2 对照,各 4e8 步 × 3 seeds。通过条件(全部满足才算赢):

1. RMSE ≤ base2 − 2·σ_pool(统计显著,同 run1 判据)
2. action_rate per-step(部署口径)均值 ≤ 0.15 且逐 seed 均 ≤ 0.20
3. tracking_sinyaw 达成率均值 ≥ base2 + 2pp
4. 其余关键分项逐 seed 达成率退化 ≤ 2pp
5. batch_test(4096 envs 带推力)成功率 ≥ base2

产出:与 run1 同款的对比报告(参数表、reward 分项表、训练曲线,复用
`tools/compare_confirm.py`;**分项训练过程曲线为必备件**,run1 报告曾只给
终值柱状图,run2 起每个分项的 mean±std 曲线必须随报告产出)。

### gate 设置

- **gate-R0**:人工确认 base2 成立(看 2 seeds 短训对照)
- **gate-R1**:审查约束满足率——若 25 个 trial 里满足全部硬约束的不足 3 个,
  说明约束过紧或权重空间不够。两档处置,按症状选择:
  (a) action_rate 压不下或 tracking 被卡,且实机/固件允许调整映射常数 →
  启动动作映射放开预案(`target_scale`、`action_scale.thigh` 以固件对齐值 ±20%
  窄范围放回搜索,固件侧需同步改常数、走部署变更流程,见 4.3);
  (b) 约束整体过紧 → 人工放宽一档(如 action_rate 门槛 0.6→0.75)。
  不要让搜索空跑

  **gate-R1 决议记录(2026-08-28,第一轮 R1 后)**:采用 (b) 档 + 软约束改造。
  第一轮实测(7 完成 trial,19 剪枝):0 trial 全过硬约束,action_rate 部署口径
  全部 ≥0.28(ar_scale 加到 −0.975 时为 0.28 但存活崩到 394;−3.6 的极端值
  早期即被剪枝)——0.15 门槛在本权重空间不可达;且硬约束记 WORST 使完成
  trial 目标值全为 10.0,TPE 完全失明。决议:
  1. action_rate 部署门槛 0.15 → **0.30(均值)/ 0.35(逐 seed)**(可达区间
     0.28~0.35,仍比 base2 的 0.37 低约 20%);
  2. R1 搜索目标改**软约束**:objective = 真实 RMSE + Σ 归一化违反罚
     (ar/能耗超上限按 value/threshold−1;存活每差 50 步罚 1;达成率每差
     2pp 罚 1),只有死亡/发散记 WORST;R2 最终判定仍用硬门槛;
  3. 其余约束(sinyaw +2pp、关键分项退化 ≤2pp、能耗预算、存活 ≥550)不变;
  4. 工程修正记录:`base_scale`/`posture_scale` 为代码未引用的死参数
     (run1 计划修订㊻已剔除),不进 R1 空间——搜索因子实为 11 个
     (9 个活 reward 权重 + kp/notsmoothing 随行)。
  第一轮数据归档于 `tuning/run2/archive_r1a/`。

  **gate-R1 第二次决议(2026-08-28,第二轮后)**:软约束生效(TPE 有梯度、
  采样向高 ar_scale 集中),但两轮 13 个完成 trial 显示 ar 部署口径在存活
  健康(≥500 步)前提下的可达前沿 ≈ 0.34(唯一 <0.30 的 trial 存活崩到
  394);高 ar_scale 区被 ASHA 早期误剪。决议:
  1. ar 门槛 0.30 → **0.35(均值)/ 0.40(逐 seed)**,对齐前沿、仍略低于
     base2 的 0.37;到 0.15~0.28 必须动映射/固件((a) 档),本轮不走;
  2. 同 study 续搜 25 trials,ASHA min_resource 4 → 6,减少重罚区误剪;
  3. R2 以 0.35/0.40 口径判定,报告如实说明取舍。
- **gate-R2**:帕累托判定 + 部署审查(网络已与 baseline 一致,天然满足端上耗时要求;
  仍需按 `docs/deploy_firmware_comparison.md` 核对固件对齐项)

## 7. 预算估计(双 RTX PRO 6000 并行)

实测:1e8 步 ≈ 11 min/卡,4e8 步 ≈ 42 min/卡。

| 阶段 | 规模 | 卡时 |
|---|---|---|
| R0 验证 | 2 × 1e8 | ~0.4 |
| R1 搜索 | 25 trials,ASHA 剪枝后约 18 × 1e8 当量 | ~3.3 |
| R2 确认 | (2 候选 + 1 对照) × 3 seeds × 4e8 | ~6.3 |
| 合计 | | **~10 卡时 ≈ 双卡 5 小时** |

## 8. 工程改动清单(预计)

- `legwheel/tuning/space.py`:run2 空间定义(reward 权重为主,网络因子删除)
- `legwheel/tuning/metrics.py`:分项硬约束评估(读 reward_terms,判定函数)
- `legwheel/tuning/worker.py`:约束不满足记 WORST 并标注 violated_constraints
- `legwheel/tuning/report.py`:报告中输出约束满足情况与帕累托判定表
- `scripts/tune.py`:`run2` 子命令/配置入口
- `legwheel/rewards.py`:height 参考值可行域投影(已实现,见 5.1 修正点 3)
  + metrics 增加 `track/active_steps_sinyaw|sinpos` 计数(门控项达成率实测分母)
- `legwheel/env.py`:仅 metrics 初始化补键(激活计数),
  obs/动力学零改动
- 网络、部署导出:**零改动**(这是本计划的硬要求)
