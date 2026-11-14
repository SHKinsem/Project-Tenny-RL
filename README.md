# Project Tenny RL

> 🚧 **进行中（Work in progress）**：项目仍在开发，目录结构、脚本和结果随时可能变动。

Tenny 轮腿机器人的强化学习与仿真实验：基于 MuJoCo MJX + Brax PPO 训练平衡/速度跟踪策略，并配有网页预览。

## 目录

| 路径 | 说明 |
|---|---|
| `experiments/torque200/` | 全扭矩接口 200 Hz 独立实验（环境回归测试 + 短训练流水线），详见其中的 `README.md` |
| `reference/` | 参考项目的副本（见下） |
| `tasks/` | 任务计划与记录 |
| `CONTROL-MODES.md` | 控制模式说明 |
| `*.ps1` / `*.cmd` | Windows 下的启动、训练、测试脚本（训练在 WSL 中执行） |

## 参考项目

本仓库参考了作者「星际浪子」的两个 MIT 协议开源项目，副本位于 `reference/`，各自保留原 LICENSE：

- `reference/mujoco_legwheel_rl_public`：腿轮机器人（LegWheelRobot）的 MJX + Brax PPO 强化学习项目
- `reference/mujoco_esp32_robot_deploy_public`：WL Pro Robot 轮腿机器人 ESP32 固件与主机仿真

`experiments/torque200` 由上述环境改造而来，**不是原项目**；对原环境的修改仅在本仓库内。

## 依赖

Python、JAX、Brax、MuJoCo（MJX）；GPU 训练在 WSL2 中运行。具体版本见各子项目的 `requirements*.txt`。
