# 原项目训练与导出复现（静态核查，2026-11-14）

范围：WSL 仿真、PPO 训练、checkpoint 恢复与导出。用户明确排除 ESP32 编译。

## 已核实的源码事实

- README 的 30 维观测、PID 参数已落后于当前代码；`default_config()` 实际 `last_act_obs=False`、26 维 actor 观测、4 维动作、`delta_thigh_action=True`。应以当前 config 和模型为准。
- 原 `curriculum_learning()` 使用 actor `[64,64]`、critic `[512,512,512,512]`、`reward_scaling=0.1`、`learning_rate=1e-4`、半衰期0.5秒的折扣。
- 默认 batch_size=512、num_minibatches=16，构成8192并发环境。不要直接执行无参数 `train.sh` 在8GB WSL内存上验证。
- ZIP 有导出的策略数组，未见完整训练 checkpoint。预训练策略演示可用数组；训练 smoke 从随机初始化开始，不能称为原训练 checkpoint 续训。
- `get_ppo_config` 可覆盖已存在的PPO字段，但没有 `num_eval_envs`。独立 harness 在构造配置后显式加 `num_eval_envs=2`，由原 `train_ppo` 按 Brax 签名传递。

## 独立 harness

`tasks/training_smoke.py` 只修改运行时训练规模；保留环境、奖励、噪声、推力、域随机化、600步 episode 与原网络尺寸。调用原训练、原导出和部署仓 NumPy 推理代码，输出 checkpoint、配置、进度、动作对照及最大误差。不改作者仓库。

WSL 示例（将脚本复制到 ext4；输出目录必须尚不存在）：

```bash
BASE=/home/jaccob/projects/tenny-reference
RL="$BASE/mujoco_legwheel_rl_public"
FW="$BASE/mujoco_esp32_robot_deploy_public"
JAX_PLATFORMS=cpu "$FW/.venv/bin/python" "$BASE/training_smoke.py" \
  --repo "$RL" --deploy-root "$FW" --output "$BASE/artifacts/train-smoke"
```

再次运行同一命令，改为新 output，并传 `--restore <首轮报告中的checkpoint绝对路径>` 可验证原 Brax 恢复训练入口。仅加载 checkpoint 做动作对照与真实恢复训练应分别汇报。

默认训练步数16；Brax实际进度以输出报告为准。只证明训练/保存/恢复/导出的流程，不证明策略收敛。CPU JIT 可能仍耗时；128/8192级别的默认并发未使用。

导出器保留小数点后6位，因此与训练端允许小量量化误差。脚本对8个原环境 reset 观测比较，恢复阈值1e-6，导出动作阈值1e-3；失败时保留 report.json，不掩盖误差。该验证覆盖 NumPy 部署推理，并未验证实际 ESP32 浮点执行。

## 依赖建议

复用仿真 venv，先使用作者 jax/jaxlib 0.9.2、MuJoCo/MJX 3.12.0、playground 0.2.0，再加入训练所需 Brax 0.14.2、Orbax、mediapy。作者 requirements 声明 flax0.12.9 与 JAX0.9.2存在元数据冲突；主代理当前解析得到 flax0.12.6，应先验证该可解析组合与 pip check，而不是预先强装冲突版本。完整实际版本应以安装后的 freeze 和运行结果记录。

本文件是静态检查记录，未宣称训练已执行或通过；执行结果由主代理补充到任务记录。
