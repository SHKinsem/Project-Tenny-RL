# 本轮BC起点短训的奖励/PPO源码审计

2026-11-16。**本文件只适用于 `aligned-limit-ablation-20261116` 六次从BC bootstrap重新开始的短训。它不代表用户比较前已有的成熟模型。** 尚未确定那个模型的checkpoint、critic训练状态、探索std和原优化设置，因此下面“随机未训练critic”等事实不能外推到它。当前应先确认成熟模型身份，再讨论较小LR微调；本子任务没有启动训练或回放。

## 结论先行

源码能够确定两类事实，但不能据此选择唯一原因：

1. **当前原奖励与原地平衡判据有明确差异。** 直接pitch项权重为0；真实机身XY不进入奖励；“位置”项是轮关节角均值。启动/恢复/长期平衡仍受速度、横滚、高度、俯仰角速度和终止后的未来回报损失间接约束，因此不能说原奖励完全不关心平衡，也不能仅凭缺项就宣称它造成了本轮退化。
2. **本轮BC起点的PPO也存在需要辨别的更新侧风险。** 该BC主要拟合动作均值，critic明确未训练；16步unroll与短GAE意味着初期目标依赖这个critic。冻结BC anchor只是共同均值参数化，没有BC回归损失或相对BC的KL约束，不能防止更新后离开原动作。风险成立，因果尚未确定。

固定初态的合格时间下降是该评价条件下的策略退化，但不自动等于PPO实现出错。若共同窗口原奖励增加而合格时间下降，支持目标取舍不符；若两者都下降，说明这些条件下连原任务表现也退化，应进一步查更新、价值估计及分布偏移。后者也不能独自证明总体随机训练目标变差。

## 证据优先级

- 实际执行归档：actual-experiment-source.tar.gz（本机原始资料，未入此归档），SHA256 `768dba854238c024076f83e76be80eb92f51677d160a18fbca2623d8d083347d`，已与完成核验中的归档SHA相等。
- 其中 `native/src/cpp/legwheel_task.cpp` 是本轮真实原生实现；`experiments/aligned_limit_ablation/ppo.py` 是本轮实际PPO包装；`experiments/reward_ablation/train.py` 是已修改并真实执行的训练入口。本次按tarfile只读读取，没有用公开目录的新版本替代它们。
- 基础 `legwheel/rewards.py`、`legwheel/env.py`、`hybrid/env.py`、`force_teacher/env.py` 和安装的Brax `train.py`仍从实际运行源读取，并先逐个核对本轮manifest保护SHA，全部一致。
- GAE loss、分布、wrapper、normalizer等安装依赖读取的是当前同一venv文件；其SHA写入JSON，但这些文件未被旧归档/保护清单逐一冻结，应保留这个证据边界。
- BC事实优先取受保护bootstrap `verified.json`，SHA `5dc3e7e48b976abc7420885bff5763b30f14f2dcb723d949b196e8340fdbe1d7`，已核对。其源快照 `source-1-pretrain.py` 的SHA也与该凭据files条目相同。不把当前后来修改的pretrain.py当作当年执行的完整证据。

机器可读信息见[source-audit.json](source-audit.json)。本子任务只生成源码审计文件，没有重复主代理的40case分项数值复算。

## 原生奖励分项如何解释

实际归档 `native/src/cpp/legwheel_task.cpp:593–637`：所有 `reward/*` 指标先写入**未乘权重的原始值**，再加权组成总奖励。当前控制步 `.01s` 与 `reference_dt=.01s` 相等，外部时长倍率为1。PPO另将总奖励乘0.1，这个缩放不包含在原生return里。

| 原始指标 | 本轮有效权重 | 公式/含义 |
| --- | ---: | --- |
| tracking_lin_vel | +2 | `exp(-abs(local_vx-command_vx)/(sigma*command_scale_vx))`，分母约0.1m/s |
| tracking_ang_vel | +2 | `exp(-abs(gyro_z-command_yaw)/(sigma*command_scale_yaw))`，分母0.75rad/s |
| tracking_wheels_sinposition | +10 | 零vx指令门内，`exp(-abs((qL+qR)/2-position_start)/0.5)`；是轮角，不是真实XY |
| tracking_sinyaw | +10 | 零yaw门内，`exp(-abs(sinyaw-sinyaw_start)/(sigma*command_scale_roll))`，分母约0.1 |
| pitch_cost | **0** | 实际算出的是正的指数跟踪分数；名字cost并不代表已减入reward |
| roll_cost | +4 | `exp(-abs(sinroll-sin(command_roll))/(sigma*command_scale_roll))`；本轮command_roll=0 |
| tracking_height_jnt | +2 | `exp(-abs(mean(hipL-height_ref,hipR-height_ref))/(sigma*command_scale_height))`，分母0.125rad |
| energy_cost | -1 | 4子步×4电机的`mean((actual_torque/reference_scale)^2)` |
| action_rate | -0.5 | 4子步×4电机的一/二阶归一化扭矩差；如下 |
| pitch_velocity_penalty | -0.5 | `abs(gyro_y)` |
| termination_penalty | 0 | 配置值-10存在，但开关false，不生效 |

本轮command全0，相关enable为true；`sinpitch_zero=true`设定参考/观测含义，**不会把 `pitch_scale=0` 自动打开**。直接俯仰新增项和真实XY新增项在baseline里也都是0。`base_scale`、`posture_scale`虽然保留在配置中，此实际加权路径没有使用它们；`do_deconflict=false`。

因此可按上表对 `native_first_episode_metrics` 的累计原值加权，重构native return（允许浮点求和顺序误差）。不能把所有正的 `reward/*` 累加，也不能把 `pitch_cost` 的原始累计当作真实pitch奖励贡献。最高正分是每控制步30，再扣代价；没有固定“存活奖励”字段，但多个跟踪项在准静态状态仍可持续给正分。

俯仰相关的直接权重为0不等于角度不受任何约束：环境在原生 `sinpitch<-0.6` 或 `>0.85`、`abs(sinroll)>0.6` 或几何/有限值保护失效时终止。真实XY漂移不是终止条件。这些阈值也不等于评价的10°/5cm/0.1m/s合格带宽。

### 子步归一化和统计阶段

实际归档 `experiments/torque_limit_ablation/env.py:38–91`、native `legwheel_task.cpp:424–458`：两组的扭矩归一化都固定 `[1,1,.0295,.0295]`。令每物理子步 `u=tau_actual/reference_scale`，`reference_dt/sim_dt=4`，则变化原值为

`mean_substeps,actuators(4*abs(u_t-u_(t-1)) + 16*abs(u_t+u_(t-2)-2*u_(t-1)))`。

它衡量400Hz实际电机扭矩变化，不是100Hz策略动作变化，也不是末子步ctrl差。发送均值/前拍观测仍按基线物理尺度归一化，±2组没有隐含减轻相同物理力矩的代价。

Native `xmat/sensordata`保留MuJoCo `mj_step`的派生阶段；post-integration qpos/qvel并不等于同一时间的传感器。现NPZ只有末子步ctrl，不能精确补回四个子步的effort/rate。主代理若按窗口分析，可用原Evaluator的prefix累计指标相减；不要把单个末子步当作完整400Hz积分。Native首回合累计包含终止步，后续reset不计入该回合指标（归档runtime.cpp:283–300）。

## PPO目标、截断和初始化

实际PPO包装归档 `experiments/aligned_limit_ablation/ppo.py:67–100` 仍明确使用：lr3e-5、entropy=.001、固定EMA归一化、传入完整actor/critic、无动力学随机化。它只将网络factory指向共同anchor头，没有添加BC惩罚。配置与实际train入口匹配。

- 每次2,097,152环境步，512环境、512 batch、16步unroll、1 minibatch/1 update；单次unroll是0.16秒。30秒是外层回合上限，不是30秒Monte Carlo学习目标。
- γ=.9862327044933592，折扣半衰期0.5秒；λ=.9025，`γλ`迹权重半衰期约0.05952秒。`γ^16≈.80107`、`(γλ)^16≈.15518`。不能把半衰期描述成硬截止，但它们说明较远回报和critic估计如何进入当前更新。
- 本机受保护Brax train.py传标准PPO loss；当前安装loss.py:184–275用16步末 `next_observation` 的V做bootstrap，真termination切断未来价值，GAE目标/优势stop-gradient，优势默认标准化。
- EpisodeWrapper把到3000步的超时标记为truncation，若同时环境失败则仍是termination。GAE对truncation transition及反向递推置mask；本轮 `bootstrap_on_timeout` 未显式传入，实际默认false，所以没有额外的timeout reward-bootstrap修正。不要把它笼统说成“超时和倒地完全相同”，也不要说已做完整终端状态bootstrap正确性证明。当前40条评价全部提前失败，30秒评价截断并不是这些已观察终止的直接原因。
- Actor loss用概率比裁剪区间[.7,1.3]；这不是物理动作变化或KL的硬上限。Critic默认 `vf_loss_coefficient=.5`，实际MSE乘`.5*coefficient`。再加`-.001*entropy`；Adam固定lr3e-5、全局梯度范数裁剪1。KL只记录，默认无自适应KL调lr或相对BC KL约束。
- 按当前控制流推论，每批由当前actor采样，随后只有1 minibatch×1 update且normalizer固定，唯一SGD梯度计算时新旧actor仍相同、概率比通常约1。概率比裁剪因而不限制这一步更新之后的策略偏移；不能把ε=.3当成实际trust-region。这是源码推论，不是本子任务测得的clipping统计，也不单独证明退化根因。
- 采样和loss都用同一个已替换的policy.apply；target log-prob针对保存的pre-cap raw_action计算。共同头只重参数化Gaussian均值，不是采样后再变换随机变量，不需要另补head Jacobian。容量裁剪位于环境中，使用pre-cap请求的likelihood正确。
- 训练入口保存最后参数，原地合格带宽不是更新验收门或回滚条件。`on_policy`每段做的是共同6秒确定性native评价，不参与SGD选优。

### BC到底预训练了什么

受保护 `verified.json.bc` 明确记录：动作拟合目标是每路归一化 `force/Fscale`、`wheel_torque/.0295` 的均方误差，critic是 **Random untrained value network, fitted observation normalization only**。候选使用72条episode、13,404样本、2048次拟合；这不是拟合该原奖励的价值函数，也不是PPO训练成熟的critic。

同一个凭据的旧候选bc段有 `initial_exploration_pre_tanh_std=.15`，但后续 `provenance.exploration_reset.pre_tanh_std=.01`、`identity.exploration_pre_tanh_std=.01`。**本轮最终初始化应引用.01**；旧.15不能当作本轮实际初始σ。重置仅将std-head kernel置0、bias设为inverse-softplus目标，确定性均值保持；PPO期间σ仍可正常学习。本源码子任务没有另行加载或评价checkpoint的std，结论来自已核对SHA的最终reset凭据。

完整参数digest与六次0步恢复验证一致；normalizer mean/std/variance漂移为0。计数器可以增加，但normalize只用mean/std。冻结anchor是factory closure，不在optimizer参数树里；它随输入状态计算固定BC均值，并不构成“靠近BC”的损失约束。两组确定性初始物理轨迹相同；随机只在同状态/同key的pre-cap请求上对齐，约5.4%右腿初始随机请求超过低组容量，这属于处理差异。

这些关于critic/std的事实只属于本轮BC bootstrap，**不能用于用户已有成熟模型**。

## 训练reset有没有隐藏随机化

核验过SHA的实际 `legwheel/env.py:178–201` 定义 `reset(rng, random=False)`。代码虽然包含yaw±π、pitch±π/8、roll±π/18采样，但默认false时选择模型qpos0，并用默认零速度创建data。受保护Brax train.py:408/415/423只调用 `env.reset(key_envs)`，未传random=True；Playground/Vmap/Episode wrapper同样只透传rng，自动重置默认恢复缓存first data/obs。Hybrid环境会在fresh步骤清理相关历史和参考。

所以，**没有发现本轮PPO从大范围随机姿态开始、而评价只用nominal的错配**。随机键仍参与动作采样和无效推力间隔等记录，但noise0/pushFalse/cmd_prob0/randomization_fn=None使这些不会改变默认物理reset。评价的nominal覆盖训练默认reset；固定±.01/±.02rad是额外工况，不属于训练初态分布，报告时应区分。

## 最小辨别诊断（建议，未执行）

1. 优先确认用户所指成熟checkpoint及其控制模式、完整actor/critic/normalizer、std与原LR；不要继续把这六次BC起点短训当作其成熟模型的续训表现。
2. 对本轮已有数据，用相同首段/共同存活窗口比较真实原地合格和上述加权原奖励，再分解effort/rate、位置代理、速度等实际贡献。共同窗口能减少提前失败造成的累计回报偏差；原生无折扣窗口分数仍不是完整PPO随机折扣目标。
3. 只有需要进一步解释这个BC起点实验时，再在同一冻结轨迹上比较初始critic与折扣回报/TD残差/优势方向，结合相同状态的actor漂移。若另获授权，可做小预算actor冻结、仅critic拟合或单次actor更新诊断；当前不自动增加训练。

本次只写本目录审计产物。没有修改生产/冻结源码，没有新训练或回放，也未对成熟模型作性能或初始化断言。
