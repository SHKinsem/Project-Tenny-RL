# 物理初始化对齐的容量实验

两组使用同一物理请求坐标与冻结BC均值锚点，执行容量分别为±1/±2N·m及对应Fs1/Fs2。原奖励、观测扭矩单位、30秒/100Hz/400Hz和PPO预算固定。条件与限制见[实验合同](../../tasks/aligned-limit-ablation-draft.md)。两组均重新训练，旧基线不复用。

- `contract`：共同Fs2请求、容量配置、锚点比例和descriptor变换。
- `networks.make_ppo_networks`：保留原可训练参数/分布与wheel head；两腿均值使用冻结BC锚点补偿。锚点通过显式checkpoint或`TENNY_ALIGNED_ANCHOR_CHECKPOINT`加载，不进入参数树。
- `env.AlignedLimitEnv`：先将同一动作映射为共同腿力请求，再按每组容量裁剪；扭矩能耗、变化率和前拍观测仍用原物理单位。
- `export.policy_template`：将`TENNY_ALIGNED_ANCHOR_DESCRIPTOR`中的不可变BC头附到每次导出。`policy_from_brax`只替换当前可训练网络数组，锚点保留。
- `native_patch.patch_sources`：绑定四份原版源码，为独立副本加入相同锚点与容量逻辑，旧无新增字段路径保持原样；不写生产评估器。
- `matrix.run`：六个独立进程、三个配对seed、两组都新训；每次校验冻结输入。`report.paired_summary`/`write_report`：无训练依赖的40case配对计算和HTML/PNG/CSV/JSON。

必须区分两项对齐：确定性0步实际闭环轨迹一致；同状态同随机数的pre-cap物理请求一致。随机请求超过低组容量时的实际裁剪差异属于处理效应。此共同头改变了旧训练的参数化，不能把旧±1训练拿来替代新基线，也不声称本轮随机探索分布与旧策略相同。

初始化补偿不是固定把输出减半：mutable均值可偏离冻结锚点并使用更大范围，必须用探针证明。后续网页可直接读取同一结果JSON和纯汇总；本实验模型绑定独立原生实现与完整冻结源码，不直接导入生产。

裸normalizer/actor/critic checkpoint不足以单独恢复完整策略语义，必须同时携带冻结BC anchor及新head；原生确定性descriptor已嵌入anchor。续训不能把新checkpoint自身误设为锚点。实验状态和真实冻结源保持独立，不从目录维度猜测兼容性。

针对测试：`python -X utf8 tools/test_cpu.py discover -s tests -p 'test_aligned_limit_*.py'`。运行状态、准确机器路径与真实冻结源见忽略的`tasks/results/aligned-limit-ablation-20261116/PROGRESS.md`；继续前先读状态，勿重跑已有矩阵。
