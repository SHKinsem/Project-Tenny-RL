# 弱轮角位置奖励：独立只读审查

对象为当前`experiments/wheel_hold_ablation`及本轮部署源码副本、prepare/evaluate脚本与冻结合同。只写本记录，没有修改训练/冻结源、重跑训练或执行额外GPU任务。以下为静态独立审查；正式三次新训练、统一120秒结果与最终独立数据核验仍以实际完成记录为准。

## 结论

未发现需要中断当前冻结矩阵的奖励项选择、共同评价回报会计或基线复用错误。当前实现保留权重1的轮角项，未将它删除，也未把共享航向项一起降权。

## 关键核对

1. **唯一奖励变化正确**。部署`legwheel/rewards.py`先对轮角、航向分别应用原命令门控，再组成`zero_keep=10*(rawWheel+rawYaw)`，其中`metrics['reward/tracking_wheels_sinposition']`是已门控的未加权轮角项。`WheelHoldEnv._get_reward`在父类原奖励及XY追加项后，只把`zero_keep`加上`(1-10)*rawWheel`，得到`1*rawWheel+10*rawYaw`。未修改共享`zero_keep_scale`，也未改其它reward键；权重10分支原样返回父类字典，避免不必要的运算变化。

2. **原物理、观测和XY任务保持**。新类只扩展初始化校验及奖励，不覆盖`step`或`_get_obs`。继承的真实XY项仍来自`RewardExperimentEnv('position')`，权重10/尺度.05m，直接姿态追加权重0；reset原点固定。构造与训练入口同时要求原轮角/航向项启用、共享权重10，实际环境完整配置与原manifest一致。原成熟动作头和力/扭矩映射不由本改动更换。

3. **实际训练目标和共同评价严格区分**。native descriptor保持原base10任务，故native `base_return`仍含轮角10和航向10。共同评价值为`base_return + true_position_addition_return`；弱组实际训练目标再加`(1-10)*native_raw_wheel_score_sum*control_dt/reference_dt`。代码使用native首回合原始分项累计，而非用末子步ctrl或不同时间阶段的姿态重建。当前dt=.01、reference_dt=.01，比例为1；PPO统一reward_scaling尚未应用在这些记录值上，文档已明确。XY追加项只计一次。

4. **报告没有把共同目标冒充弱组目标**。旧baseline的`objective_return`仅规范化为相同base10+XY共同值；弱组分别保留`common_xy_objective_return`、`wheel_position_adjustment_return`和`training_objective_return`。缺原始轮角证据时实际弱目标保留未知，不从旧字段猜测。最终evaluate脚本为全部35个工况传入native首回合分项以计算实际目标；报告明确它与共同base10回报不同，回报不独立排名物理平衡效果。

5. **复用旧weight10基线有匹配证据**。prepare从上一轮high_lr manifest继承设置，逐seed检查已完成的实际步数、完整成熟初始化、实际LR、有效PPO/环境/XY奖励以及descriptor身份，保存原checkpoint树和记录的保护哈希。新训练的优化器接线、LR核验及保护helper又与上一轮执行代码作AST匹配。只有三个weight1 seed新训练，旧三次明确标为reused并保留其原时间/来源，未伪装本轮重新跑过；最终仍将旧策略与新策略一起做同一120秒评价。

6. **完整起点和预算守卫保留**。新入口把完整成熟normalizer/actor/critic传入同一force网络工厂和Brax，0步完整数组摘要必须一致；normalizer统计后续必须零漂移。实际LR须由优化器metrics确认3e-5，wrapper外层回合为3000步，动作重复1，100Hz即30秒。2,097,152步与九次262,144步间隔回调逐点核验，失败不能冒充少预算完成。

7. **新weak各阶段可恢复完整参数**。0步和每次策略回调均先保存独立`checkpoint-step-N`中的normalizer、actor和critic，立即加载核对完整数组摘要，写入路径、树哈希与重载结果；descriptor绑定对应checkpoint。最终另存完整checkpoint。这里的完整指策略/价值/归一化参数，不含Adam状态、环境状态或PRNG执行历史；重新续训会新建优化器，不是位级连续恢复。

8. **物理结论仍采用共同口径**。成熟不更新参考、旧weight10和新weight1最终策略共用五初态；主三初态先在seed内平均、再按三个训练seed配对，预留两初态单列。30/120秒分开；提前失败会使该seed对应的完整时域漂移比较未知，不能凭早倒的小位移获胜。报告公开复用来源、有限种子/固定初态边界，未把仿真弱轮角选择写成已解决实机IMU漂移或轮滑。

## 需保留的边界

- **全阶段完整保存仅能保证本轮新weak运行**。复用的旧weight10训练当时保存过哪些完整参数应按旧产物事实报告；其历史中间推理descriptor不能据此追认成完整critic checkpoint。新weak九个完整参数点可供后续明确选择，未保存Adam的边界需保持。
- 旧baseline与新weak的进程时间不宜解释为纯训练性能对照：本轮新weak增加每阶段完整checkpoint保存/加载核验，原生过程评价和I/O均包含在elapsed中。固定步数保证更新预算一致，时间记录主要用于成本追溯。
- 已知门禁与GPU smoke证明的是限定状态/短训链路，而不是最终改善。需等三次新训练完成，再核对新旧实际配置/步数/完整初始化、全部新阶段重载、native原始轮角分项与实际目标会计，以及统一35工况指标。
- 两组只比较轮角位置权重10与1。它们都使用真实XY奖励；相对不更新参考的变化不能独立归因为XY奖励。本轮没有额外IMU漂移、轮滑或噪声模型，也没有把仿真world XY加入actor观测。

审查不建议为上述解释边界修改当前冻结训练源；最终归档和结论应保留这些事实范围。
