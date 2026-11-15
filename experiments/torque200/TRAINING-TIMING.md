# 训练耗时记录

完整网页控制台的新训练和续训默认经 `timed_console_train.py` 调用原训练入口，每个 run 生成独立日志。控制台服务更新并重启后生效；正在运行的旧进程保持原样，不会补造其历史记录。直接运行旧 `console_train.py` 仍保留旧行为。

## 看什么文件

- `timing.jsonl`：追加式时间线。训练尚未结束、取消或失败时仍可读取；不会随学习曲线的窗口裁剪。
- `timing-summary.txt`：按总耗时排序的可读表，包含次数、均值、最小/最大值和占进程墙钟比例。
- `timing-summary.json`：同一汇总的结构化版本，另含当前未闭合段、计时器开销、归账残差和结束状态。
- `status.json`、`report.json`：记录上述文件路径与计时口径。report另附写入前的`timing_summary`快照，完整最终统计以独立汇总为准。原 `metrics.jsonl` 的训练/评估指标继续保留，可按环境步数对照。

运行中约每10秒发布一次汇总；正常结束、停止保存、取消和失败时在 `finally` 中发布最终汇总。不要只依赖最后一个汇总文件：它可能比最新时间线稍旧。

从追加日志重建并打印汇总，无需安装JAX或加载模型：

```bash
python experiments/torque200/timing_log.py <run目录>
```

在已部署的WSL中，也可使用：

```bash
python "$HOME/projects/tenny-torque200/src/timing_log.py" \
  "$HOME/projects/tenny-torque200/artifacts/console/leg_force_bc/<run_id>"
```

命令只读日志，支持运行未结束和最后一行被意外截断的情况；只累计截至最后一条完整事件观测到的时间，不推测进程此后是否还在运行。旧运行缺少该日志时明确报错。

## 口径

每条 `segment` 事件包含 `run_id`、`mode`、`phase`、`wall_start_seconds`、`wall_end_seconds`、`duration_seconds`、`steps_start`、`steps_end`、`outcome` 与补充 `metadata`。同一步数可能有训练快照、评估和写盘等多条记录；步数是环境步数，不是物理子步数。

时间轴从计时器建立开始，单调递增。教师准备完成后原有训练预算会重新起算，但这条时间轴不会重置，因此准备和正式训练可以一起核算。锁等待、代码导入前的极短Python启动、控制台父进程的启动/源码同步不属于同一个完整测量范围：本日志记录训练子进程内、计时器建立之后的过程。

采用独占计时：进入子步骤时暂停父步骤，退出后恢复。汇总可以相加，不应再把父步骤的包围时间重复加上。`segments` 是分段数量；父步骤恢复后也形成新段，所以它不等于PPO更新次数或评估次数。每次真实操作的边界可结合时间线和原指标日志查看。

| 阶段 | 含义 |
| --- | --- |
| `preparation.runtime_lock`、`preparation.*` | 运行锁、混合动作预检或教师采集/拟合/闭环门禁；已有阶段发布点直接进入时间线 |
| `setup.imports`、`setup.model_and_environment` | 依赖导入、模型/配置建立等准备 |
| `setup.reset_first`、`setup.reset_and_networks`、`setup.networks` | 首次reset与网络建立；首次reset包含其JIT，并等待实际结果完成 |
| `preparation.restore`、`artifact.load` | 检查点恢复与加载；metadata中的stage帮助区分初始化和最终重载 |
| `ppo.initialize` | Brax训练入口初始化，直到初始参数回调；可能包含JIT与设备工作 |
| `ppo.first_epoch` | 第一段采样与PPO更新，包含首次训练JIT |
| `ppo.train` | 后续采样与PPO更新的组合，至完成参数回调；不是单独的GPU kernel耗时 |
| `ppo.evaluate_first`、`ppo.evaluate` | 真实评估；首次包含评估JIT，跳过的评估不会冒充真实评估 |
| `callback.snapshot_and_checks` | 参数已完成之后的CPU快照、归一化检查与回调；预览和IO另行扣除 |
| `callback.preview`、`callback.progress` | 按需预览与进度回调，内部IO不重复统计 |
| `io.status`、`io.metrics`、`io.progress`、`io.config`、`io.report` | 状态、原始指标、曲线窗口、配置与报告写入 |
| `save.checkpoint`、`save.cpu_params`、`save.export` | 检查点、CPU参数快照及策略导出 |
| `verify.parameters`、`save.finalize` | 参数一致性核对与剩余最终导出/推理验证 |
| `lifecycle.*` | 生命周期衔接、返回、完成、失败或取消 |

JAX异步执行，不能把函数返回前的Python提交时间视为GPU实际执行时间。这里只在原本需要把完整参数取回CPU的安全回调边界，先等待参数完成，再计CPU快照；首次设置reset也等待其结果，原Brax评估本身已有完成等待。不在每个物理子步或每次SGD中强制同步。采样和SGD在已编译循环内，当前合并记录；纯编译时间也没有伪造拆分。需要更细GPU划分时再单独采集性能追踪。

`recorder_overhead_seconds` 单独累计计时、序列化、追加日志与周期汇总的开销，并从业务分段扣除。最终日志/汇总发布存在无法在同一事件内自描述其完整写入耗时的问题，故最终结果明确给出 `measurement_cutoff_seconds`，将最终发布尾部排除；以 `reporting_tail_note` 为准。运行中的最后一小段记录开销也可能尚待下一事件确认，残差会如实保留。

## 兼容与验证边界

计时使用独立入口包装现有公开函数和回调，保留其签名、参数及返回值，不改变PPO配置、评估频率、控制模式或停止保存协议。旧入口及物理/策略源文件保持原字节，避免仅因可观测性改动使模型指纹和准备缓存失效。进程管理同时识别旧、新两种入口名。

静态与假时钟测试可以证明归账、异常、取消和完成等待的调用顺序；它们不能证明真实GPU训练开销或策略效果。需要真实新运行后再用日志判断瓶颈，不把测试样例或旧运行重建成新运行证据。
