# 验收流程

先按[构建说明](build.md)构建，再在 WSL 设置以下变量。所有输出保存在独立工作目录，不覆盖模型产物。

```bash
WORK="${NATIVE_EVAL_WORKDIR:-$HOME/projects/tenny-native-eval}"
PY="${TENNY_PYTHON:-$HOME/projects/tenny-reference/.venv-gpu/bin/python}"
"$PY" -m pip install --target "$WORK/build-deps" -r "$WORK/src/requirements-test.txt"
export PYTHONPATH="$WORK/python:$WORK/build-deps${PYTHONPATH:+:$PYTHONPATH}"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PY" -m pytest -q -p no:cacheprovider "$WORK/src/tests"
```

测试用独立 MuJoCo C API 验证物理、raw derived-stage `xmat` 和奖励；覆盖不同网络尺寸、激活、环境数、控制子步、首回合 mask、reset、seed、取消/GIL 与并发拒绝。策略与任务的严格参数测试用于防止悄悄丢弃配置。

## 冻结模型的等价性与耗时

先用 `native_eval export-run --help` 导出已验证 run。公平比较输入来自[原 CPU/GPU 对照](../../torque200/CPU-GPU-EVALUATION.md)的本机 `tasks/results/cpu-gpu-eval/result/`：其中有 contract、共同观测、相同 reset、GPU/CPU 轨迹与计时。它们不随源码入库；换机器时需要显式提供新的冻结参考产物。

```bash
"$PY" "$WORK/src/tests/validate_saved_model.py" \
  --descriptor "$DESCRIPTOR" --reference-dir "$REFERENCE_DIR" --output-dir "$OUTPUT_DIR"
```

脚本核对模型/源码身份，沿用测量前确定的九项门槛；记录模型建立、首轮含 trace、三次无 trace 的完整评估调用。报告使用历史基准时会明确标注，不冒充本轮重新测过 GPU。不得放宽阈值来消除失败。

轮腿第二种控制语义及不同子步/环境数的独立 C API 对照：

```bash
"$PY" "$WORK/src/tests/validate_legwheel_modes.py" \
  --descriptor "$DESCRIPTOR" --initial "$REFERENCE_DIR/initial.npz"
```

需要另存一般评估结果时，用 `native_eval evaluate --help`。trace 单独写 NPZ；其保存开销不混入评估耗时。未启用 trace 的后续调用用于比较中位耗时。

核对 checkpoint、导出、XML及运行源码未被修改，并记录控制台状态。评估加速不直接等于 PPO 加速；首次 JIT、参数传输和训练集成成本应另测。支持范围和真实结果只更新[验证记录](../VALIDATION.md)，避免在多个入口复制会漂移的数字。
