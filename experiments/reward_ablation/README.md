# 姿态与真实位移奖励对照

独立的四组×三种子等步数实验，不接生产训练管理器或网页。保留原物理、24维actor输入、动作、奖励各项与6秒训练回合；只比较新增俯仰/真实世界XY位移惩罚。预先冻结设计见项目`tasks/reward-ablation-draft.md`。

| 组别 | 新增俯仰项 | 新增真实位移项 |
| --- | ---: | ---: |
| baseline | 0 | 0 |
| posture | 权重4 | 0 |
| position | 0 | 权重10 |
| both | 权重4 | 权重10 |

新增项为`-weight * (1-exp(-normalized_error²))`，角尺度10°、位移尺度5cm。正确处0，偏离为负，无新增正的存活常数。原轮角代理保持奖励和侧倾奖励不动。俯仰和世界XY参考在reset固定，Brax auto-reset恢复相同初态时保持；不跟随身体或旧观测适配器的动态参考更新。

## 分工与入口

- `rewards.py`：无训练库依赖的冻结四组系数、标准库参考和数组后端实现。
- `legwheel_env.py`：在既有已部署StudentEnv上只扩展reset参考及奖励；不覆盖物理step、动作或观测。
- `train.py`：读取显式manifest、相同完整初始化，调用已部署force PPO，验证0步恢复与固定归一化，保存最终模型和进度。
- `matrix.py`：按冻结顺序在独立子进程串行运行，记录代码/manifest身份，检查实际训练步数。
- `evaluation.py`：用统一SI轨迹计算原地/姿态/速度同时合格的持续时间。
- `report.py`：只读完整轨迹，按训练seed做配对描述，生成JSON/CSV/HTML/PNG。

在具备既有Brax/JAX GPU、MuJoCo和native_eval的WSL ext4工作目录中，显式提供实验代码、部署源码与原生Python模块路径；复用已有依赖，不在Windows挂载目录训练：

```bash
python -m experiments.reward_ablation.matrix \
  --manifest /path/to/inputs/manifest.json --output-dir /path/to/new-matrix

# 一个冻结矩阵成员；不会自动选取/覆盖production latest。
python -m experiments.reward_ablation \
  --manifest /path/to/inputs/manifest.json --arm posture --seed 0 \
  --output-dir /path/to/new-run
```

manifest包含`ppo`实际参数、保存的`env_config`、完整BC初始化checkpoint及数组哈希、四组系数、配对seed、执行顺序、统一native descriptor及精确初态。训练入口从部署配置重建环境，完整JSON或记录系数与实际固定ARMS不一致就拒绝；当前不支持只改manifest来自定义奖励。所有组使用同一actor/critic/normalizer，优化器重新初始化。仅将`std_eps=0`标量统一成Brax实际使用的float32以消除保存的float64零的类型差异，数值/网络/统计不变，随后0步完整数组摘要必须完全匹配。

默认本轮预算为每次2,097,152环境步，512环境、batch512、unroll16、1 minibatch/1 update，3个训练seed，共12次。记录实际步数，失败不当作少预算完成；输出目录须新建或空，失败记录保留。训练内按相同524,288步间隔在统一原任务上评估，训练奖励不参与跨组排名；所有最终策略都在同一30秒原任务上评价。

## 结果语义与网页复用

每次保存`effective-config.json`、`status.json`、`progress.jsonl`、训练loss日志、最终native descriptor与完整checkpoint。新策略使用独立身份和checkpoint哈希，初始化历史放在`initializer_provenance`，不沿用旧导出身份；浮点参数含NaN/Inf时拒绝。冻结原BC另作参考。状态用于后续任务视图，评估结果沿用`experiments/balance_diagnostics`的schema v1，新增`standing`结果；未来网页可直接消费同一数据，任务执行仍应与状态轮询分开。

有效站立时间排除初始1秒：仅当真实位置距reset≤5cm、俯仰相对固定参考≤10°、平面速度≤0.1m/s同时满足，才累积真实采样区间；同时记录最长连续段、总存活及30秒截断。死亡后不再累积，但不补造位置/姿态。带宽是本实验预先声明的描述口径，并非实机验收标准。

每个训练seed内先对标称/±.01rad固定初态汇总，再比较3个配对训练seed；±.02rad为额外固定初态。评估初态不冒充独立训练重复，不声称统计显著或随机鲁棒性。仅好于退化baseline但不及未训练BC应称减缓退化。不同训练reward、不同存活时长的全程reward/s、早倒产生的小位移都不能独立排名。

本轮保持6秒训练回合，未同时改变折扣、探索、学习率或actor观测；单次权重与短训预算未改善，不能排除其他权重或更长训练。世界位置只作为奖励/评价真值，实机位置反馈与可观测性须另行验证。

必要CPU验证：`python -X utf8 tools/test_cpu.py discover -s tests -p 'test_reward_ablation_*.py'`；模型/训练入口还应在既有WSL环境验证初始化一致、奖励与物理隔离以及回合参考固定。无需全仓回归或修改生产环境。
