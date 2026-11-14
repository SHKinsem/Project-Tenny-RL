# 设备与原 MJX 吞吐检查

`device_benchmark.py` 是独立诊断脚本；未修改原模型、奖励、噪声或推力配置。`env.reset(key, False)` 沿用原版非随机初始姿态，每个环境有独立 RNG。固定零动作，无策略网络、无 PPO 更新、无自动 episode reset，因此结果仅为原环境推进吞吐，不代表平衡效果或真实训练速度。

脚本强制设置 `JAX_PLATFORMS=cpu/cuda`，检查实际设备与矩阵乘结果所在设备；CUDA 不可用时必须失败，不回退 CPU。矩阵乘结果与精确参考对照。批量 reset/step 使用 `vmap`，100步用 `lax.scan`，每次用 `block_until_ready` 等待计算完成。分别记录 reset/rollout 的 lower+compile 时间、首次执行/预热时间，随后相同初始状态重复3次并统计中位吞吐。若持久编译缓存命中，compile 时间可能是加载缓存时间。

每步检查 qpos、qvel、观测、reward、done 均有限。done 指标是被标记终止的 transition 数，不是摔倒次数（没有自动 reset）。JAX `memory_stats` 是分配器信息，不是显卡总显存峰值；主代理应另行 nvidia-smi 采样。

默认32环境×100步；可先用 `--batch-size 2 --steps 20` 验证，再用相同32×100分别测CPU/GPU。使用未存在的JSON输出路径，避免覆盖旧报告。

WSL示例（先将脚本复制到ext4，具体GPU环境路径由主代理确定）：

```bash
ROOT=/home/jaccob/projects/tenny-reference
CPU="$ROOT/mujoco_esp32_robot_deploy_public/.venv/bin/python"
"$CPU" "$ROOT/device_benchmark.py" --repo "$ROOT/mujoco_legwheel_rl_public" \
  --expected-backend cpu --batch-size 32 --steps 100 --output "$ROOT/artifacts/bench-cpu.json"
"$GPU_PYTHON" "$ROOT/device_benchmark.py" --repo "$ROOT/mujoco_legwheel_rl_public" \
  --expected-backend cuda --batch-size 32 --steps 100 --output "$ROOT/artifacts/bench-gpu.json"
```

该脚本尚未运行；上述是执行方式，不是已通过结论。
