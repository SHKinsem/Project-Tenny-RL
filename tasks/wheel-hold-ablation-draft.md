# 成熟策略的弱轮角约束对照

用户最新要求保留较弱轮角位置奖励，而非完全删除，以考虑实机IMU漂移。默认轮角项权重由10减到1；真实XY项仍为10、尺度5厘米，学习率固定3e-5。以刚找回的成熟081557/000002801664完整normalizer/actor/critic为共同起点，不从事后挑选的仅推理descriptor或BC初始化开始。

## 唯一训练差异

只改tracking_wheels_sinposition这一项的贡献。原共享zero_keep_scale也控制航向，不能整体调整它；航向权重和其余原奖励保持。真实XY原点reset固定，不增减actor观测，不加入IMU噪声/漂移模型。训练外层30秒、100/400Hz、原±1N·m/Fs1/轮扭矩尺度、PPO五设置、normalizer、γ/λ/entropy等沿用上一轮high_lr。

baseline为上一轮weight10的high_lr三个seed；仅当新适配weight10行为/源/完整起点与配置核验等价后复用。weak_wheel权重1新训三个配对seed，各2097152步；仅新增6291456步，不重跑已匹配baseline。两组均完整重建优化器。0和每262144步记录五初态30秒物理指标及原生奖励分项。新weak保留独立descriptor和完整actor/critic/normalizer checkpoint以便后续续训（不含Adam历史）；旧复用组历史中间产物仍仅descriptor。

## 评价与解释

新旧两组最终策略与不更新参考统一五初态×120秒；主要三初态先在seed内平均再配对，±.02rad预留单列。30/120秒真实XY RMS、末10秒均值、1厘米/5厘米联合合格时间、启动峰值、姿态/速度及终止一起报告；早倒不能凭小位移获胜。

原生评价器仍保留原wheel10回报作为共同参考。实际weak目标精确计算为：common_base10 + true_XY_addition + (1-10)*raw_wheel_score_sum*dt/reference_dt。不能把共同回报误称weak实际目标，也不能用末子步ctrl重建奖励。记录每项原始累计和实际训练目标，供网页复用。

核对actor是否使用绝对XY以及轮角/IMU相关输入，区分reward与部署观测。弱轮角项只是仿真设计选择，不证明解决IMU或轮滑误差；本轮不同时改变观测噪声条件。固定初态/三seed不能证明实机鲁棒性。

CPU针对验证90秒、等价/固定策略核对120秒、必要GPU smoke约1分钟、三次正式新训练预计6～8分钟。保护生产、旧源和旧模型，保持已有冻结目录不变，不部署、不提交推送、不自动增加预算。具体机器状态在忽略的tasks/results/wheel-hold-ablation-20261116/PROGRESS.md。
