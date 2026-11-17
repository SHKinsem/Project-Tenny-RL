# 成熟模型真实位置微调：独立只读审查

审查对象：`experiments/mature_balance_finetune/train.py`、`matrix.py`、`report.py`，复用的真实位置奖励适配，以及本轮`prepare.py`、`verify.py`、实际旧force PPO副本和冻结合同。未修改训练/冻结源码，未执行训练、GPU工作负载或额外测试。本审查针对成熟模型起点的六次新训练，不引用此前BC起点实验作为本轮对照。

## 结论

未发现需要中断当前六次矩阵的训练接入、奖励单位或评价口径错误。最终结果仍须等待六次运行完成、统一120秒评价及`verify.py`的实际通过；代码审查不能替代这些执行证据。

## 已核对的关键路径

1. **实际优化器学习率**：旧部署`force_teacher.ppo.train_student`内部确实把LR强制写成3e-5。本轮`train_with_ppo`绕过该覆盖，保留原`make_ppo_networks`、固定EMA恢复、Brax wrapper和无随机化接线，直接把manifest中的3e-5或3e-6传给`train.train`。`observed_learning_rate`核对实际训练metrics中的learning_rate；运行末尾要求出现观测值，独立verify还复核每次记录。浮点32的3.000000106e-6在规定容差内，不是被覆盖为3e-5。

2. **完整成熟初始化**：实际checkpoint来自预选成熟来源；prepare保护原树摘要，训练读取后规范化仅零值std_eps的类型，并核对完整数组摘要。0步回调再次校验恢复后的actor、critic和normalizer整体；即使默认restore_value_fn行为变化，critic未恢复也会被该守卫拦下。后续每个回调都要求normalizer均值/std/方差差异为0，最终完整checkpoint重载须精确一致。

3. **原动作头和物理映射**：使用原成熟metadata构造原force网络工厂和native policy template，拒绝leg_anchor或policy_leg_force_scales对齐变体。奖励适配只覆盖reset参考和奖励，不覆盖物理step、动作或观测；真实位置组的新增posture权重为0，原其它奖励保留。部署配置完整JSON需与冻结配置相同，初始descriptor亦保留原件供字节核对。

4. **两组只改LR**：prepare从同一ppo字典生成high_lr/low_lr，奖励均为同一`ARMS.position`；verify明确比较两组PPO除learning_rate外完全相同，并核对实际保存配置、相同完整初始化、实际完成步数、所有九个回调步数。每个seed两组均新建优化器，不把旧运行当重复；执行顺序已冻结。

5. **30秒训练包装**：manifest和实际Brax参数中的episode_length为3000，控制周期0.01秒、action_repeat=1，故外层训练回合确为30秒。原StudentEnv配置中的旧6秒字段保留以维持其它源码/观测语义，不控制这里的外层time limit。每次2,097,152步、num_evals=9、每batch8192步，故每262,144步回调一次、起点加八次更新共九点，不发生ceil导致的不等预算。

6. **奖励时间和单位**：训练位置项取post-integration的free-root世界XY减固定reset原点；posture权重0。报告把原native base reward与位置追加项分开，追加项按`10*expm1(-(distance/.05)^2)*dt/.01`逐post-step累积，与训练端原`dt/reference_dt`缩放一致；base reward已是逐步量，不再重复乘dt。目标回报只加一次。每个回调验证base return与native首回合return吻合，独立verify按轨迹再算所有35个最终工况。

7. **阶段与初态口径**：回调0表示恢复后的完整成熟模型，不是训练后模型；后续descriptor/NPZ/evaluation均绑定实际新增训练步数，文件存在则拒绝覆盖。姿态/位置采用统一post-step轨迹，原始NPZ只保留首回合。每回合XY原点固定；当前无随机化且auto-reset恢复同一物理初态，参考保持不随身体或动态观测平衡角更新。

8. **共同30/120秒判定**：不更新成熟模型与六个最终策略共用五初态。30秒与120秒窗口分开，RMS/均值用时间加权，末10秒指标在观察不足30秒时为未知。任务终止优先于恰好达到时限；任一主初态未完成某时域，则该seed对应漂移比较记未知，不能因早倒位移小获利。按seed先平均主初态，再做三个配对差；预留初态单列，完整种子数在表格/图中显示。

9. **失败与保护守卫**：拒绝非有限参数、非零normalizer漂移、回调缺失/重复、未完成预算、LR缺失/偏离、非精确重载以及原模型/冻结文件变化。非有限标量训练metrics被单独记录，独立verify要求不存在，不能仅凭run.status为completed就宣称最终已验证。runtime-freeze覆盖执行代码和输入JSON；prepare另归档执行源码，保护原成熟checkpoint与主要部署文件。

## 需保留的解释边界

- 两组都加入真实位置项，只比较该目标下的学习率。相对不更新成熟参考的变化同时包含新增奖励和参数更新效应，不能声称已独立证明位置奖励的因果收益。合同和报告已正确保留此限制。
- 起点是在25份同指纹历史checkpoint的主三个工况中筛选出的低漂移成熟模型；该固定起点的微调效果不代表所有历史策略。预留工况没有参与起点选择，但五个固定初态仍不等于随机鲁棒性或实机验证。
- 训练seed是重复单位，不能把五个确定性初态当作五次独立训练重复。只有三个配对seed，不宜使用统计显著或泛化保证的表述。
- `elapsed_seconds`从输入核验前计时，包含恢复/JIT、30秒原生过程评价、保存和核验；它不是独占GPU优化耗时，不能用其差异推断学习率组的纯训练性能。
- `matrix.py`显式处理子进程非零退出，但`subprocess.TimeoutExpired`未转写matrix.status为failed，超时时磁盘状态可能停留training，而主进程实际会异常退出。当前无超时证据，不影响已经按实际步数完成的运行；如发生超时，应由主任务识别退出并记录失败，不应改变冻结源码或把该条当完成结果。

## 最终验收入口

`verify.py`已准备核对六次实际LR/完整初始化/相同步数/九回调、0步共同轨迹、最终checkpoint到native动作重建、执行源码及原输入保护，并独立重算35工况的奖励与时间/位移指标。应以该脚本和实际120秒结果为最终证据；本文件仅为静态独立审查。
