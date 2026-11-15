# 采样与 PPO 分阶段诊断

`brax_instrumentation.py` 仅添加 `jax.named_scope`，保留原 `pmap`/JIT 与算法表达式；`trace_summary.py` 计算真实 GPU 事件的区间并集。任务边界、优势/梯度及优化器标签独立于机器人尺寸。

当前运行适配支持已有 `leg_force_bc` 确定性站立配置的采样/PPO诊断：训练动作仍按原随机策略生成，关闭的是独立评估与检查点写出。临时参数更新只留在隔离进程内。它不会启动控制台正式训练或替换现有模型。

## 运行

在 WSL 从本目录调用，源码会复制到 ext4 独立目录：

```bash
bash profile-wsl.sh --help
bash profile-wsl.sh --run-dir "$VERIFIED_RUN" --epochs 4
```

不指定 run 时读取空闲控制台的最近已验证模型。构建/运行环境复用已有 GPU venv；入口无需安装新的 profiler 包。默认工作目录是 `~/projects/tenny-ppo-profile`，可用 `TENNY_PROFILE_WORKDIR` 改为另一个 ext4 路径，解释器由 `TENNY_PYTHON` 覆盖。

启动器设置明确的命令、GPU会话和编译截止时间，持运行锁，并记录实际参数。参数含义以 `--help` 为准。存在正式训练时拒绝启动。采集会影响运行速度；不能把 profiler 内墙钟直接当生产墙钟。

到达截止时间后按SIGINT、SIGTERM、SIGKILL顺序停止独立进程组，另有最多25秒退出宽限；实际退出码和墙钟写入`launch.json`。当前修正版已通过CPU回归，完整GPU采集的数值及阶段覆盖门槛尚未通过，详见结果记录。

严格参数门槛与“能否分析设备热点”分别记录。GPU物理/归约可能存在数值不确定性；同一 executable 的重复对照、原版/标记版误差及未分类比例都必须保留。详情见[计时合同](MEASUREMENT.md)和[实测结果](RESULTS.md)。

## 比较并行环境数量

使用独立入口，统一PPO批量后对比128/256/512环境；吞吐来自无profiler样本，搬运另用单epoch诊断：

```bash
bash scaling-wsl.sh --help
bash scaling-wsl.sh --run-dir "$VERIFIED_RUN" --envs 128 256 512 --batch-size 512
```

默认2个预热、8个稳态测量、1个采集和1个恢复epoch；每组8192控制步/epoch及1次更新，分别在独立进程中加载同一模型。工作目录由`TENNY_SCALING_WORKDIR`指定，默认独立ext4目录。总墙钟默认480秒，编译/单组另设上限，超时不追加轮次。

公平性、数据驻留、内存峰值与传输口径见[对照合同](SCALING.md)，结果见[规模对照记录](SCALING-RESULTS.md)。统一batch与生产batch不同，不能将本对照的倍数直接当作生产整体或收敛收益。

只用CPU重放已完成矩阵：

```bash
"$TENNY_PYTHON" -B report_scaling.py "$MATRIX_DIR" --output "$SUMMARY_JSON"
```

## 按需阅读

| 任务 | 入口 |
| --- | --- |
| 理解阶段、计时与验收边界 | [MEASUREMENT.md](MEASUREMENT.md) |
| 修改阶段标记/兼容新的 Brax | `brax_instrumentation.py`、`tests/test_instrumentation.py` |
| 重放 trace、修正 GPU 归因 | `trace_summary.py --help`、`tests/test_trace_summary.py` |
| 了解本机数据和优化优先级 | [RESULTS.md](RESULTS.md) |
| 修改或运行并行规模对照 | [SCALING.md](SCALING.md)、`run_scaling.py`、`launch_scaling.py` |

CPU 回归：

```bash
"$TENNY_PYTHON" -B -m unittest discover -s tests -v
```

如果没有设置 `TENNY_PYTHON`，使用同一个 GPU venv 的解释器路径。测试不执行 GPU 训练；本机源快照不随仓库分发，对应可选用例会跳过。
