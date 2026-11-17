# 轮角位置奖励10→1：成熟模型配对结果

2026-11-16。**保留弱轮角约束后，本轮真实位置保持改善：120秒末段位移的训练seed中位数由13.500降至6.499毫米，3/3配对seed改善。** 原地附近的累计时间也增加，全部最终回放保住本次规定的平衡带。训练过程仍会波动，尚未验证IMU漂移或轮滑鲁棒性。

## 按用户最新偏好实施

用户将“删除轮角位置项”改为保留更小奖励，考虑实机IMU漂移。本轮因此测试**权重10与1**，没有运行删除组。真实XY项仍为10、尺度5厘米，固定学习率3e-5；偏航仍10，原奖励其余项不动。特别地，原代码的zero_keep_scale同时加权轮角和偏航，本轮只从zero_keep贡献中减去9×raw wheel score，未全局修改共享系数。

两组共同从找回的成熟run `20261115T081557.290279Z_bd32fd40/checkpoints/000002801664` 完整恢复normalizer/actor/critic；原±1N·m/Fs1/轮扭矩动作映射、30秒训练回合、100/400Hz、512env/batch512、unroll16、1 minibatch/1 update、gamma/lambda/entropy等保持。没有引入噪声、动作头变换或重新BC，优化器均重建。

原weight10组复用上一轮high_lr的三个配对seed，已核对有效PPO/完整起点/源码与模型身份，并通过新适配weight10逐项等价门禁。weak_wheel组新训三个seed，各2,097,152步，新增6,291,456步；比较整体包含旧6,291,456步。复用不是本轮重新训练，报告保留原run与时间证据。

## 统一120秒结果

原模型不更新作为参考，新旧各三个最终策略，共7策略×5固定初态。**35/35首回合均120秒time_limit，剔除首1秒后的5厘米联合合格时间均119秒。** 联合条件同时要求俯仰误差≤10°、真实XY位移在距离带内、平面速度≤.1m/s；未把延迟终止或早倒前的小位移当改善。

以下三个主要初态（标称/±.01rad）先在每个seed内平均，再取三个训练seed的中位数；参考为同一固定策略，没有额外训练seed。XY为相对reset的真实机身世界平面距离模长。

| 条件 | 120秒末10秒平均XY | 120秒XY RMS | 最大XY位移 | 1cm内累计合格时间 |
| --- | ---: | ---: | ---: | ---: |
| 成熟模型，不更新 | 17.388 mm | 17.187 mm | 19.469 mm | 1.827 s |
| 轮角权重10，复用 | 13.500 mm | 13.493 mm | 31.563 mm | 2.000 s |
| 轮角权重1，新训 | 6.499 mm | 6.779 mm | 21.591 mm | 117.280 s |

1厘米时间是剔除首1秒后的**累计**时间，不是保证全程或最长连续时间。最大位移先按每工况取最大，再在seed内平均。

| seed | 权重10末段XY | 权重1末段XY | 配对差：1减10 |
| --- | ---: | ---: | ---: |
| 0 | 13.500 mm | 6.499 mm | -7.002 mm |
| 1 | 13.704 mm | 5.320 mm | -8.384 mm |
| 2 | 11.124 mm | 8.807 mm | -2.317 mm |

末段配对中位差-7.002mm，3/3改善；组中位数约减52%。XY RMS配对中位差-6.714mm，最大位移-9.972mm，平面速度RMS-.001825m/s，三个seed均同方向。1厘米累计时间的**配对增量中位数115.930秒**，不是直接相减两组中位数得到的115.280秒。

两个预留±.02rad初态同向：末段XY组中位13.497→6.482mm。仍有个别工况超过1厘米：weak的15条回放中13条超过一半观察时间在1厘米联合带内；seed2的pitch-minus仅38.21秒，heldout-minus为0秒，后者末段约11.384mm。不能称全部扰动工况都保持1厘米内。

代价：俯仰RMS配对中位略增0.0393°（各seed约+.028～+.040°），仍在原联合带内；新弱组最大位移虽均低于weight10组，但有2/3 seed仍高于未更新成熟参考。因此结论是本轮位置保持改善，不是所有维度全面优于原模型。

## 曲线仍有退化，保留完整快照

每262144步评估五初态并保存策略。24个非0步weak阶段中，6个阶段未能让三个主初态全部撑满30秒；最终均恢复。因此弱轮角项没有让训练单调稳定，不能只看最终或总奖励选择策略。

从18个主初态均完成30秒的weak阶段中，事后按三主初态XY RMS最小选择seed0、第786432步。此探索性候选另做五初态120秒，全部撑满：主/预留末10秒平均XY为**2.402/3.357毫米**，120秒RMS为2.985/3.769mm。独立末段复算误差4.34e-19。[候选曲线](selected-snapshot/comparison.png)已实际查看。它是事后挑选的单策略，不混入上述预先定义的最终步数配对统计，也不是额外独立seed。

本轮新weak每个阶段都保留完整normalizer/actor/critic，候选checkpoint为：

`$HOME/projects/tenny-wheel-hold-ablation-20261116/matrix/seed-0/weak_wheel/checkpoint-step-786432`

其SHA为`50fad8920bdc31a4127c23e3d282defea6efbe62d947200361c2a5c129c6fd26`。重载后重新导出与保存策略动作误差0，可作为后续继续训练起点；不含Adam历史，继续时重建优化器。旧复用baseline的历史中间文件仍只有descriptor，不能追认为完整critic checkpoint。[候选参数核验](selected-snapshot/verification.json)与[来源摘要](selected-snapshot/summary.json)已保存。

## 实机IMU顾虑与观测边界

已沿实际部署StudentEnv→Hybrid→LegWheelRobot和force_teacher网络核对：**24D actor没有世界XY，也没有真实机身local-vx。** 它接收command、gyro、yaw/pitch/roll误差特征、关节位置（轮槽是轮共模角误差）、真实关节速度、轮速×半径和前拍归一化发送扭矩。机身local-vx只在51D privileged critic；critic也没有直接拼世界XY。

真实XY在本实验中只用于仿真奖励和评价，没有因增加该奖励而成为部署策略输入。轮角观测一直保留，降低奖励权重不等于删除编码器反馈。但这也不会凭空获得绝对定位能力，不能由无观测噪声的仿真推断已解决IMU偏置、角度估计漂移、轮滑或轮径误差。本轮为单项权重对照，没有同时追加这些误差。源副本和SHA在deployed-source，结构化边界在[observation-boundary.json](observation-boundary.json)。

## 奖励会计与核验

原生评估器保留原base10供共同评价。实际weak训练目标严格为`common_base10 + true_XY_addition + (1-10)*native_raw_wheel_score_sum*dt/reference_dt`；原组的最后一项为0。本控制步dt/ref=.01/.01=1，PPO内部reward_scaling仍.1。报告区分common_xy_objective_return、wheel_position_adjustment_return和training_objective_return；两组训练目标不同，不能直接用总奖励跨组排名，更不能把native共同base当weak实际目标。

- 16个真实保存状态的weight10全项等价、weight1其他项不变及单轮角调整误差均0；reset物理/actor观测一致，动力学及观测方法继承同函数。该门禁不是完整JAX/native轨迹等价测试。
- 三个新run完整初始digest仍为`7d7a680147538e923c2df91e47886a04202856af9e3c08aa5d9f72dac3f1a28a`，normalizer漂移0；实际LR2.999999924e-5固定生效。27个完整阶段checkpoint有限/重载/摘要与对应descriptor身份通过；最终参数保存及推理导出核验通过。
- 旧baseline15条及不更新参考5条的同120秒NPZ全部字段与旧评价逐值一致。新0步与原120秒轨迹前30秒的物理/观测/动作/奖励前缀一致，只有末步正常30秒timeout done标记不同。
- 56份原文件、4个旧checkpoint树、30个新冻结文件与25个旧冻结文件保持。训练wiring四helper AST与旧组相同，源归档SHA`79f9c1cce1c156e9663fc433f1bd7b891ca2b948c03f04b800591be4b137773e`。
- 35case从原NPZ独立重算位置/速度/姿态/奖励和目标会计，综合最大浮点误差8.73e-10；独立CSV对700项统计核对与JSON一致。实际查看30/120秒、过程图、同seed渲染和候选图。
- 训练器6项CPU测试、报告10项通过，主体.015/.429秒，完整入口.83/1.28秒。门禁主体4.94秒；首次辅助脚本reset step计数类型处理错误已局部修正，不涉及模型代码。

## 时间、产物与状态

仅新三次正式矩阵墙钟342.946秒（5.72分钟），各136.536/98.053/94.851秒，包含JIT/训练/阶段评价/保存，不是纯GPU吞吐。8192步smoke56.950秒另计，权重不进入正式矩阵。最终独立核验主体12.623秒；历史基线的337.743秒是旧三run进程时间之和，不是本轮新增训练或同一个矩阵墙钟。准备、审批、源核对、补充回放和报告整理另计。

正式会话37568与候选回放86398均退出0。未改生产、默认奖励/观测设置或旧模型，无提交推送，没有新增训练扫描。

入口：HTML报告（本机原始资料，未入此归档）、[配对JSON](comparison/paired-summary.json)、[过程曲线](comparison/progress.png)、35条轨迹（本机原始资料，未入此归档）、[完整核验](comparison/verification.json)、[独立统计复核](statistics-review.md)、[同seed实际快照](comparison/replay-seed0.png)。新完整最终参数在独立WSL根的`matrix/seed-N/weak_wheel/checkpoint`，新阶段参数为同目录`checkpoint-step-N`；后续浏览器可复用experiments/wheel_hold_ablation/report.py和本结果JSON。
