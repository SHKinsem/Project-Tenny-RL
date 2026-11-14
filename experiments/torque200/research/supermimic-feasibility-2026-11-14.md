# SuperMimic 对 当前原参考轮腿机器人 原轮腿全扭矩 200 Hz 的可迁移性

日期：2026-11-14。性质：第一手源码调研与工程判断，未安装依赖、未训练、未改控制代码，也未查看当前训练状态。本文用 research skill 整理；这是当前任务的委派子任务，不再分派。

## 结论

可以借鉴“用仿真交互训练可微动力学模型，再通过模型反传优化策略”的方法；不能把官方 G1 项目或权重直接接到 当前原参考轮腿机器人 的四路扭矩接口。官方任务是参考运动跟踪，动作是低增益 PD 的关节位置目标，控制频率 50 Hz；当前原参考轮腿机器人 的目标是原轮腿、四路绝对扭矩、200 Hz。动作语义、参考数据、奖励、状态表示和时间尺度都需要适配。现有证据不能保证它比当前 PPO 更快学会平衡，也不能保证官方默认配置适合 RTX 2080 SUPER 8 GB。[官方任务配置](https://github.com/HybridRobotics/supermimic/blob/abc124c1bd5976341034668af3d12c087f2f1f55/src/supermimic/tasks/env_cfgs.py)、[基础动作/时序配置](https://github.com/HybridRobotics/mjlab_research/blob/dd51a9fcc98791a2f0b018b3c7f71a3eefd5ecdd/src/mjlab/tasks/tracking/tracking_env_cfg.py)、[论文](https://arxiv.org/abs/2610.28480)

部署时不必运行 world model。官方 ONNX 导出只封装所选任务的 encoder、共享 decoder 和归一化；motion 导出额外携带参考运动查表，velocity 导出为观测到动作的网络。但默认训练的 actor_obs 是 rep_state，并非自动满足实机可测观测契约；需要明确选择和验证 proprio、历史顺序、动作缩放等。[导出实现](https://github.com/HybridRobotics/supermimic/blob/abc124c1bd5976341034668af3d12c087f2f1f55/src/supermimic/runners/exporter.py)、[默认 runner](https://github.com/HybridRobotics/supermimic/blob/abc124c1bd5976341034668af3d12c087f2f1f55/src/supermimic/runners/supermimic_runner.py)

## 本项目输入事实与证据边界

下列信息来自本次主任务提供的项目上下文，本文没有重新检查训练状态：JAX/Brax + MJX；四路绝对扭矩尺度 [1, 1, 0.0295, 0.0295] Nm；actor 24 维、critic 51 维；policy 200 Hz、physics 400 Hz；actor 64×2、critic 512×4；128 个环境、PPO、保留原奖励，尚未稳定平衡。这些是迁移分析前提，不是 SuperMimic 作者验证过的配置。

源码快照为 SuperMimic main 的 abc124c1bd5976341034668af3d12c087f2f1f55，配套 mjlab_research main 的 dd51a9fcc98791a2f0b018b3c7f71a3eefd5ecdd。两仓分别读取，并未证明这两个提交配对兼容。下面区分当前 main 的事实、论文方法和本文工程推论；不把最新 main 配置当作论文实验的精确复现配置。[SuperMimic 树](https://api.github.com/repos/HybridRobotics/supermimic/git/trees/abc124c1bd5976341034668af3d12c087f2f1f55)、[mjlab_research 树](https://api.github.com/repos/HybridRobotics/mjlab_research/git/trees/dd51a9fcc98791a2f0b018b3c7f71a3eefd5ecdd)

## 动作：位置设定值，经 PD 产生扭矩

源码链路如下：

1. SuperMimic 的 gencoord_env_cfg 继承 unitree_g1_flat_tracking_env_cfg，未替换其动作类型。
2. tracking_env_cfg 使用 JointPositionActionCfg，use_default_offset=True；G1 配置把 scale 换成 G1_ACTION_SCALE。
3. G1 执行器是 BuiltinPositionActuatorCfg，具有 stiffness、damping、effort_limit；G1_ACTION_SCALE 按 0.25 × effort_limit / stiffness 计算。
4. Dynamics 内部也将动作处理成 action × scale + offset，再归一化作为网络输入。

所以策略输出对应关节位置设定值 q_sp = αa + q₀；最终电机扭矩由位置执行器产生，不是四路直接 Nm 命令。论文附录所说的低增益 PD / torque proxy 不等同于取消 PD。[SuperMimic 环境](https://github.com/HybridRobotics/supermimic/blob/abc124c1bd5976341034668af3d12c087f2f1f55/src/supermimic/tasks/env_cfgs.py)、[G1 环境](https://github.com/HybridRobotics/mjlab_research/blob/dd51a9fcc98791a2f0b018b3c7f71a3eefd5ecdd/src/mjlab/tasks/tracking/config/g1/env_cfgs.py)、[G1 执行器和缩放](https://github.com/HybridRobotics/mjlab_research/blob/dd51a9fcc98791a2f0b018b3c7f71a3eefd5ecdd/src/mjlab/asset_zoo/robots/unitree_g1/g1_constants.py)、[Dynamics](https://github.com/HybridRobotics/supermimic/blob/abc124c1bd5976341034668af3d12c087f2f1f55/src/supermimic/models.py)

迁移推论：world model 的输入原则上可以换成真实执行扭矩，不要求数学上必须使用 PD 位置目标；但必须使用 当前原参考轮腿机器人 的动力学交互数据重新训练，并确保 actor、仿真、replay、归一化、动作饱和/限幅与部署使用同一套扭矩语义。不能把现有 G1 的 action scale 或位置偏置直接保留。

## World model 学什么、策略怎样训练

当前 Dynamics 是普通 MLP：每个隐藏层 Linear → LayerNorm → GELU。输入是归一化状态表示、归一化处理后动作、sim_ctx；输出由 representation.apply_prediction 解释并积分。默认 world model 隐藏层为 2048×3。[模型](https://github.com/HybridRobotics/supermimic/blob/abc124c1bd5976341034668af3d12c087f2f1f55/src/supermimic/models.py)、[默认网络大小](https://github.com/HybridRobotics/supermimic/blob/abc124c1bd5976341034668af3d12c087f2f1f55/src/supermimic/runners/supermimic_runner.py)

| 表示 | 状态输入 | 网络预测与解析更新 |
|---|---|---|
| GenCoord | 根高度、机体系重力方向、根线/角速度、关节位置/速度；另保留根位置和四元数辅助状态 | 预测 Δ根线速度、Δ根角速度、Δ关节速度，再解析积分关节位置和根姿态，保持运动学一致 |
| MaxCoord | 各刚体局部位置、6D 旋转、线/角速度，根高度和重力方向 | 预测各刚体线/角速度增量，解析积分位置和旋转、变换坐标系；再用解析 IK 恢复关节状态 |

两种表示都不是从像素学习的视觉模型，也不是直接把 actor 的 24 维观测当作完整动力学状态。MaxCoord 的 IK 和刚体映射增加了轮腿迁移工作；对本项目先研究 GenCoord 式低维结构化模型更自然，这是工程判断。[表示完整实现](https://github.com/HybridRobotics/supermimic/blob/abc124c1bd5976341034668af3d12c087f2f1f55/src/supermimic/representations.py)

sim_ctx 从每个环境读取摩擦、COM 偏置、编码器偏置的随机化参数；它们是仿真中已知的条件，不要求部署 actor 获得这些真值。SuperMimic 环境仅保留 foot_friction、base_com、encoder_bias 三类事件，去掉原跟踪环境的随机速度推扰。论文附录解释未观测突变会增加模型拟合难度；不能据此推断“不受推扰训练也一定有足够抗扰性”。[sim_context.py](https://github.com/HybridRobotics/supermimic/blob/abc124c1bd5976341034668af3d12c087f2f1f55/src/supermimic/sim_context.py)、[事件筛选](https://github.com/HybridRobotics/supermimic/blob/abc124c1bd5976341034668af3d12c087f2f1f55/src/supermimic/tasks/env_cfgs.py)、[论文附录](https://arxiv.org/pdf/2610.28480)

训练由两个相互交替的部分组成：

- 物理仿真执行当前策略并加入探索噪声，把完整状态、动作、上一步动作、参考运动时间、终止/截断标记、sim_ctx 写入 replay；有历史观测时也存相应观测。
- WorldModelLearner 从 replay 抽取连续片段，做自回归多步监督预测；对积分后的根姿态、关节或刚体通道用按 nominal error 加权的平方误差。
- Policy learner 从 replay 抽取起点，在可微 world model 中展开轨迹，冻结动力学模型参数但保留动作到未来状态的梯度，通过 BPTT 更新 actor。当前默认是 actor-critic 变体：加入末端 value bootstrap，并以 GAE targets 拟合 critic；另有无 critic 的 BPTT 选项。
- 当前 main 的优化器是隐藏 Linear 权重用 Muon、其余参数用 fused capturable Adam，并有 torch.compile、CUDA graph 捕获。这是当前代码事实，不将其追溯为原论文所有实验的配置。

来源：[runner](https://github.com/HybridRobotics/supermimic/blob/abc124c1bd5976341034668af3d12c087f2f1f55/src/supermimic/runners/supermimic_runner.py)、[world model learner](https://github.com/HybridRobotics/supermimic/blob/abc124c1bd5976341034668af3d12c087f2f1f55/src/supermimic/learners/wm_learner.py)、[policy learner](https://github.com/HybridRobotics/supermimic/blob/abc124c1bd5976341034668af3d12c087f2f1f55/src/supermimic/learners/policy_learner.py)、[optimizer/CUDA graph](https://github.com/HybridRobotics/supermimic/blob/abc124c1bd5976341034668af3d12c087f2f1f55/src/supermimic/learners/learner_utils.py)

## 数据需求和任务差别

官方训练入口默认加载 W&B motion registry 的 lafan_run1_subject5，也可提供本地 motion_file。参考运动提供关节位置/速度、根/刚体运动目标；world model 的动力学训练标签则来自物理仿真交互，两种数据不能混同。不存在“只下载人类视频就得到本机器人扭矩控制器”的源码路径。[train.py](https://github.com/HybridRobotics/supermimic/blob/abc124c1bd5976341034668af3d12c087f2f1f55/src/supermimic/scripts/train.py)、[goal 窗口及表示](https://github.com/HybridRobotics/supermimic/blob/abc124c1bd5976341034668af3d12c087f2f1f55/src/supermimic/representations.py)

当前 main 的 actor 是多任务目标 encoder + 共享 decoder，latent_dim=32，默认网络层 2048,1024,512,256。motion 与 velocity 两种目标可共用 decoder；默认 slot_mix 是 motion=1、velocity=0。velocity 训练目标从参考运动提取，其奖励仍是 motion tracking，并不是独立的“无需示范纯速度任务”。[任务定义](https://github.com/HybridRobotics/supermimic/blob/abc124c1bd5976341034668af3d12c087f2f1f55/src/supermimic/tasks/task_spec.py)、[runner](https://github.com/HybridRobotics/supermimic/blob/abc124c1bd5976341034668af3d12c087f2f1f55/src/supermimic/runners/supermimic_runner.py)

迁移推论：如果目标只是站立与轮式速度控制，不一定需要 LAFAN1 或人形 motion encoder；可以保留自家奖励并研究模型学习/BPTT。但这已是借鉴算法后重构任务，而非官方 G1 示例换 XML 即可。若原奖励包括不可微离散分支，需要单独处理训练梯度；不能假设 MJX 可微就消除了接触、终止或奖励分支问题。

## 50 Hz 与 200 Hz

官方基础环境 physics timestep=0.005 s、decimation=4，因此 policy=50 Hz。当前 policy horizon=32，对应 0.64 s；照搬到 当前原参考轮腿机器人 200 Hz 后仅覆盖 0.16 s。这是算术推论，不能据此直接把 horizon 固定为 128：更长 BPTT 增加显存和计算，也累积模型误差，需用真实时间跨度与误差共同选择。[仿真时序](https://github.com/HybridRobotics/mjlab_research/blob/dd51a9fcc98791a2f0b018b3c7f71a3eefd5ecdd/src/mjlab/tasks/tracking/tracking_env_cfg.py)、[BPTT horizon](https://github.com/HybridRobotics/supermimic/blob/abc124c1bd5976341034668af3d12c087f2f1f55/src/supermimic/learners/policy_learner.py)

当前 reference goal window 是 10 帧、stride=5，即 50 Hz 下约 0.9 s 前瞻；200 Hz 按相同帧数仅约 0.225 s。模型的解析积分还依赖 dt 与 decimation；当前 当前原参考轮腿机器人 200/400 Hz 是 decimation=2，不能复用官方积分时间常数。折扣、history、噪声、action-rate penalty 及 replay 覆盖秒数同样需要按时间单位审查。[时间窗口和积分系数](https://github.com/HybridRobotics/supermimic/blob/abc124c1bd5976341034668af3d12c087f2f1f55/src/supermimic/representations.py)

## RTX 2080 SUPER 8 GB 与依赖

当前 main 的原始默认规模：

| 项目 | 默认值 |
|---|---:|
| 物理环境数 | 4096 |
| 每环境 replay 长度 | 1000 |
| world model MLP | 2048×3 |
| world model 更新 | 10 batches × 2048，horizon=8 |
| actor imagined batch / horizon | 4096 / 32 |
| actor encoder/decoder 隐藏层 | 2048,1024,512,256 |
| critic 隐藏层 | 256,128 |

来源：[入口](https://github.com/HybridRobotics/supermimic/blob/abc124c1bd5976341034668af3d12c087f2f1f55/src/supermimic/scripts/train.py)、[runner defaults](https://github.com/HybridRobotics/supermimic/blob/abc124c1bd5976341034668af3d12c087f2f1f55/src/supermimic/runners/supermimic_runner.py)、[WM defaults](https://github.com/HybridRobotics/supermimic/blob/abc124c1bd5976341034668af3d12c087f2f1f55/src/supermimic/learners/wm_learner.py)、[policy defaults](https://github.com/HybridRobotics/supermimic/blob/abc124c1bd5976341034668af3d12c087f2f1f55/src/supermimic/learners/policy_learner.py)

这些不是面向 8 GB 显卡的默认配置。作为量级估算，4096×2048 的单个 FP32 激活张量约 32 MiB；3 层×32 步对应约 3 GiB 的这种原始激活体量，尚不包括反传所需其他中间量、actor、梯度、优化器、replay、仿真状态和 CUDA graph。此估算不是准确峰值测量，也不是已实测 OOM；它说明减少真实环境数到 128 并不会自动降低独立设置的 imagination batch=4096。小机器人可缩小表示和网络，但缩小后的收敛速度必须重新验证。论文 H100 级训练速度不能外推到 2080 SUPER。本文未运行显存或速度基准。

依赖是 PyTorch + mjlab/MuJoCo-Warp，非当前 JAX/Brax/MJX 路线。SuperMimic pyproject 要 Python >=3.13,<3.14，mjlab 指向 sibling editable path。明显存在版本漂移：

- 此 SuperMimic 的 uv.lock 记录 mjlab 1.5.0、torch 2.10.0、MuJoCo 3.10.0、mujoco-warp 3.10.0.1、Warp 1.14.0、rsl-rl-lib 5.4.0。
- 此次读取的 mjlab_research main pyproject 已是 mjlab 1.6.0、torch>=2.14.0、MuJoCo/mujoco-warp~=3.16.0、Warp>=1.15.0、rsl-rl-lib 5.5.2。

因此 README 的两个裸 clone 命令并不构成固定可复现环境；若未来实施，应先找到已知兼容的两仓提交和 CUDA/PyTorch wheel，单独隔离环境。本文没有证明任何特定 wheel 与本机驱动/显卡可用，也没有安装验证。[SuperMimic 依赖](https://github.com/HybridRobotics/supermimic/blob/abc124c1bd5976341034668af3d12c087f2f1f55/pyproject.toml)、[锁文件](https://github.com/HybridRobotics/supermimic/blob/abc124c1bd5976341034668af3d12c087f2f1f55/uv.lock)、[mjlab_research 依赖](https://github.com/HybridRobotics/mjlab_research/blob/dd51a9fcc98791a2f0b018b3c7f71a3eefd5ecdd/pyproject.toml)

## 建议的验证顺序（尚未实施）

1. 先为现有原轮腿四扭矩 200 Hz 建立更容易诊断的平衡 baseline，保留原实验作为对照。明确成功率、连续站立时长、倾角、饱和占比和扰动恢复等可重复指标。仅靠换算法不能定位奖励或动作约束问题。
2. 从当前 MJX 仿真采集实际执行的扭矩、完整状态、必要上下文和 episode 边界，训练小型结构化速度增量 world model。按完整 rollout/seed 划分训练验证集，比较单步与多步误差；特别覆盖低速平衡、轮地接触、饱和及跌倒边界。
3. 只有模型在与控制有关的时间跨度内足够可靠，再做小规模 BPTT 或模型辅助方案，与 PPO 比较相同仿真步预算和相同墙钟/显存预算。使用真实仿真评估，防止策略只学会利用模型误差。
4. 若保留全扭矩部署目标，可考虑让原先跑通的控制器/策略作为训练教师：记录仿真最终执行器扭矩，行为克隆初始化四扭矩 actor，再以 RL 改进。记录的是实际 torque，而不是旧位置/速度指令；检查教师扭矩是否落在当前四路尺度内、200 Hz 重采样、延迟，以及旧控制器内部状态造成的部分可观测性。部署可仅保留扭矩 actor。此为工程候选，未验证其适用性；教师加奖励、直接力矩策略的相关第一手概念来源分别为 [TGRL](https://proceedings.mlr.press/v202/shenfeld23a.html) 与 [Path Integral Guided Policy Search](https://research.google/pubs/path-integral-guided-policy-search/)，不表示这些论文验证了 当前原参考轮腿机器人。

选择建议：保留现有 MJX 作为数据和评估源，先验证小 world model 的预测价值，比整体迁移 G1 训练栈更容易隔离风险。此建议不是授权开工；当前任务只调研。

## 邻近方向与未决事项

主任务另发现后续预印本 [BlinkMimic](https://arxiv.org/abs/2611.16422) 与[作者代码](https://github.com/fuyichen0725/blinkmimic)，可进一步关注 failure-directed resets 与结构化速度预测思路。它不是 SuperMimic 官方分支；其人形运动跟踪 GPU 速度数据不能视为本轮腿实机全扭矩证据。本文未深入审计该仓库。

仍未确定：当前原参考轮腿机器人 在现扭矩约束下可达的稳定域；现奖励/终止与学习瓶颈；world model 对轮地滑移和饱和的预测精度；200 Hz 所需有效 imagination 时长；8 GB 上可用 batch/网络；数据覆盖与教师质量；两仓的可复现版本配对。任何成功率或训练时间承诺都应等这些验证完成后再作判断。
