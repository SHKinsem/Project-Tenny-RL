# 原生 MuJoCo 评估

完整评估循环在 C++ 内执行，Python 负责配置、模型导出和读取结果。通用执行器管理环境、控制周期、首回合统计、重置、取消和诊断；任务适配器负责观测、动作、奖励与终止。

当前支持确定性无状态 MLP，以及轮腿和单摆两种任务适配。轮腿支持虚拟腿力、直接扭矩、匹配包络；单摆使用不同的模型维数与观测。环境数、网络形状、评估时长和整数物理子步由输入决定。支持范围以运行时 `capabilities` 和各任务合同为准。

## 开始使用

在 WSL 中从本目录运行，构建过程把源码复制到独立 ext4 工作目录：

```bash
bash build-wsl.sh
WORK="${NATIVE_EVAL_WORKDIR:-$HOME/projects/tenny-native-eval}"
PY="${TENNY_PYTHON:-$HOME/projects/tenny-reference/.venv-gpu/bin/python}"
export PYTHONPATH="$WORK/python${PYTHONPATH:+:$PYTHONPATH}"
"$PY" -m native_eval capabilities
"$PY" -m native_eval --help
```

已验证的混合动作 run 可用 `export-run` 导出描述，再用 `evaluate` 评估。两者的参数由子命令 `--help` 说明。Python 调用：

```python
from native_eval import Evaluator

engine = Evaluator("model.json")
result = engine.evaluate(initial_states, num_envs=8, steps=600)
```

`initial_states` 是调用者提供的物理状态及任务参考锚点。省略时使用 XML 默认状态和适配器的参考初始化；这不等于训练环境的随机 reset。公平比较必须复用同一组初始状态。

## 按需阅读

| 要做什么 | 阅读入口 |
| --- | --- |
| 使用、修改接口或结果语义 | [接口合同](INTERFACE.md) |
| 构建、换机器、排查导入失败 | [构建说明](docs/build.md) |
| 接入新任务或策略 | [适配说明](docs/task-adapters.md) |
| 导出或修改轮腿任务参数 | [轮腿配置合同](LEGWHEEL-CONFIG.md) |
| 修改逻辑后验证、复测性能 | [验收流程](docs/validation.md) |
| 查看本机实际误差、耗时和限制 | [验证记录](VALIDATION.md) |
| 接入或排查现有训练评估 | [Brax训练接入](docs/brax-training.md) |

本模块提供独立 Python/CLI 入口。现有控制台的 `leg_force_bc` 训练已接入 C++ 评估；GPU 继续负责训练，其他模式保留原评估后端。实际训练收益与边界见[短训练对照](TRAINING-INTEGRATION-RESULTS.md)。
