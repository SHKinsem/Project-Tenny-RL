# 接入现有 Brax 训练

`experiments/torque200/timed_console_train.py` 保留原训练生命周期，按作用域替换 `acting.Evaluator`。原 PPO、奖励、策略头和优化器逻辑继续由原模块负责。当前 `leg_force_bc` 默认选择 C++ 评估，其他训练模式保留 GPU 评估。

运行进程的 `TENNY_EVAL_BACKEND` 可设为 `auto`、`native` 或 `gpu`；默认 `auto`。`native` 用于尚未接入的训练模式时明确报错。包目录由 `TENNY_NATIVE_EVAL_PYTHONPATH` 指定，默认独立构建目录的 `python/`。选择了 C++ 却缺少模块或完整指标能力时，必须修复构建，不会静默改成 GPU。

## 参数与 reset

`native_eval/brax.py` 保留 Brax 的构造/评估指标接口，每次评估都读取当前训练的 normalizer 和 actor；不复制 critic。`brax_policy.py` 负责动态 Dense 层、float32 数组和确定性均值头，任务模板仅提供结构和输出变换。不能用初始 checkpoint 的导出权重替代后续 actor。

`legwheel.py` 从本轮保存的任务配置、策略合同和网络结构建立模板，沿用独立导出的严格检查。每次 reset 使用原评估环境和相同的 PRNG 分裂顺序，再一次性转移物理状态、命令及参考锚点。完整 rollout、终止/截断、首回合 mask 和自动重置均在 C++ 内完成。

## 指标与停止

任务每控制步提供原始 metrics，核心只累计每个 world 的首回合，包含终止步。`episode_metrics['reward']` 与 `returns` 相同。Brax 适配核对全部字段集，保留 `eval/episode_*` 的 mean/std、`per_step` 处理和 `aggregate_episodes=False` 数组语义；缺失指标、非有限数值或不完整评估不能作为成功评估发布。

取消保持原训练的协作边界：在评估前、reset 后及完整原生调用后检查停止请求。原生 API 另支持原子 `cancel()`；被取消的部分回合不会发布为完整评估指标。原调度的评估间隔、剩余预算预留及保存/恢复流程保持原口径。

## 耗时和身份

每轮报告记录实际 backend、扩展/Python适配哈希、当前 actor 哈希及原生调用次数。`evaluation_actor_sha256` 随 actor 更新而变化，用于排查误用旧模型。

计时日志在外层 `ppo.evaluate_first` / `ppo.evaluate` 中细分独占阶段：

| 阶段 | 工作 |
| --- | --- |
| `evaluation.params_to_host` | 当前 actor/normalizer 的同步、转移和布局转换 |
| `evaluation.reset_to_host` | 原环境 reset、首次 reset JIT 和初态转移 |
| `evaluation.native_prepare` | 本次模型、任务和策略建立 |
| `evaluation.native_rollout` | 原生固定工作量、统计及绑定返回 |

阶段是独占时间。比较完整评估占用时，将外层残余与这些子阶段相加；不能只拿 `native_rollout` 与旧 GPU 完整调用比较，也不能重复加外层包含子阶段前的时间。

独立接口/任务验收见[验收流程](validation.md)。接入后的同起点短训练结果、整体吞吐和数值边界见[训练对照记录](../TRAINING-INTEGRATION-RESULTS.md)。

继续拆解采样/PPO内部开销时，按[训练分阶段诊断](../../training_profile/README.md)运行独立profiler；其设备时间和采集开销不直接并入本页的生产墙钟日志。
