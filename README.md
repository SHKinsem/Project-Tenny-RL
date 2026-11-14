# Project Tenny RL

> 🚧 **进行中（Work in progress）**：项目仍在开发，目录结构、脚本和结果随时可能变动。

Tenny 轮腿机器人的强化学习与仿真实验：基于 MuJoCo MJX + Brax PPO 训练平衡/速度跟踪策略，并配有网页预览。

## 快速开始

```bash
python tenny.py            # 打开网页控制台，点按钮运行任务（Python 3.11+，无需安装依赖）
python tenny.py list       # 终端里列出所有任务
python tenny.py run test-torque200 steps=16   # 终端里运行任务
```

控制台地址形如 `http://tenny.localhost:8800/`（`*.localhost` 由浏览器直接解析到本机，无需改 hosts）。
端口从 8800 起试；Windows 的 Hyper-V/WSL 有时会保留整段端口，此时自动改用系统分配的空闲端口，以终端打印为准。

任务在 `tenny/tasks.toml` 里登记，网页和命令行共用。Windows 下任务经 `wsl.exe` 在 WSL 中执行，
Linux 下直接用 bash；macOS 不支持训练类任务。

## 目录

| 路径 | 说明 |
|---|---|
| `tenny.py`、`tenny/` | 统一入口（网页控制台 + CLI）与任务清单 `tenny/tasks.toml` |
| `experiments/torque200/` | 全扭矩接口 200 Hz 独立实验（环境回归测试 + 短训练流水线），详见其中的 `README.md` |
| `reference/` | 参考项目的副本（见下） |
| `docs/` | 文档（控制模式 `docs/CONTROL-MODES.md` 等） |
| `tasks/` | 过程记录与辅助脚本 |

## 已知限制

- 训练依赖 GPU + JAX，只能在 Linux/WSL 中运行。
- 训练输出（`experiments/torque200/results/`、`tasks/results/`）不入库，只保存在本地。
- 训练所需的 WSL 环境（`~/projects/tenny-reference` 等）需按 `tasks/REPRODUCTION.md` 自行搭建。

## 参考项目

本仓库参考了作者「星际浪子」的两个 MIT 协议开源项目，副本位于 `reference/`，各自保留原 LICENSE：

- `reference/mujoco_legwheel_rl_public`：腿轮机器人（LegWheelRobot）的 MJX + Brax PPO 强化学习项目（[原项目](https://gitee.com/starimpact_admin/mujoco_legwheel_rl_public)）
- `reference/mujoco_esp32_robot_deploy_public`：WL Pro Robot 轮腿机器人 ESP32 固件与主机仿真（[原项目](https://gitee.com/starimpact_admin/mujoco_esp32_robot_deploy_public)）

`experiments/torque200` 由上述环境改造而来，**不是原项目**；对原环境的修改仅在本仓库内。

## 依赖

Python、JAX、Brax、MuJoCo（MJX）；GPU 训练在 WSL2 中运行。具体版本见各子项目的 `requirements*.txt`。
