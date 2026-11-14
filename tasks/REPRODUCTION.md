# 原项目本机复现结果

日期：2026-11-14。范围：原版轮腿模型、预训练策略仿真、PPO 流程与导出；不含 ESP32 编译/烧录，也未迁移 Tenny 机构。

## 使用

当前控制台：http://172.21.75.211:8765/ 。WSL 重启后 IP 可能改变，以启动终端显示为准。MuJoCo 画面在独立的 `LegWheel Web + Sim` 窗口，网页是控制面板。

关闭旧仿真后，在项目目录用 PowerShell 执行 `pwsh -File .\start-simulation.ps1`。终端会打印控制地址。网页开关 ON 后策略出力并重置机器人；摇杆控制移动，松手归零；高度和横滚滑条会保持。窗口键盘：W/S 前后、A/D 转向、Q/E 高度、B/N 横滚、SPACE 重置。退出可关闭仿真窗口或在启动终端 Ctrl+C。不要同时启动两个实例占用8765端口。

本机 Windows→WSL localhost:8765 实测拒绝连接；WSL 网卡 IP 地址访问成功，因此启动脚本仅绑定 WSL 自身 IP。未修改 Windows/WSL 网络配置。页面保留作者的 HTTP fallback；主机服务提供 HTTP 控制接口。

## 路径和环境

- 原压缩包：`download/robot_repositories_source.zip`，SHA256 `4C10F258EC9B1AC333C854B122EE4E6038169A23BD097006AC8A7F43C227F29A`。
- Windows 解压参考：`reference/`；原源码未改。
- WSL 工作副本：`~/projects/tenny-reference/`。
- 独立 Python：`~/projects/tenny-reference/mujoco_esp32_robot_deploy_public/.venv/bin/python`。
- Python 3.12.3；JAX/JAXLIB 0.9.2；MuJoCo/MJX 3.12.0；playground 0.2.0；Brax 0.14.2；Flax 0.12.6；Orbax 0.12.10。完整版本见 `results/requirements-resolved.txt`；`pip check` 通过。
- 安装使用仿真仓原 requirements-sim.txt。Flax 由解析器选取兼容版本0.12.6，未强装训练仓声明冲突的0.12.9。
- 首轮复现仿真和训练均用 CPU。后续已完成独立 JAX GPU 环境与实际验证，见 `GPU-TEST.md`。
- 当前 Codex 默认 Windows 沙箱无法访问 WSL；通过经审批的沙箱外执行运行。

## 实测结果

1. 部署仓57项测试全部通过，包括闭环速度/转向测试。
2. 作者 `sim_verify_hold_obs.py`：6项运动学探针通过；6秒/600步闭环全部通过，0失败、0摔倒重置。最大观测/锚点误差小于4.77e-7。
3. 原导出策略已在图形窗口中运行并站立；Windows HTTP 开启控制成功，网页显示 ON 和实际姿态。
4. 首轮 PPO：2训练环境、2评估环境、16环境步；保留原模型、奖励、域随机化和网络宽度。完成训练、Orbax 保存、加载和原导出器导出，用部署仓 NumPy 策略对照8组观测。加载误差0，导出最大动作误差1.7121434211730957e-5，耗时约63.65秒。
5. 第二轮从首轮checkpoint加载参数继续训练16步，再次保存、加载、导出成功。加载误差0，导出最大动作误差1.436471939086914e-5，耗时约31.85秒。

两轮报告分别在 `results/train-smoke-20261114-143038/` 与 `results/train-smoke-20261114-143234/`。实际checkpoint保留在 WSL 同名 `artifacts/` 目录。Brax 此流程的步数重新从0计，不应将其称为完整优化器状态无缝续训。

## 复现命令

在 Windows PowerShell 可用 `wsl -d Ubuntu --exec bash <脚本的/mnt/c/...路径>` 调用以下脚本：

- `tasks/bootstrap-wsl.sh`：首次复制源码并安装独立仿真环境。
- `tasks/verify-sim-wsl.sh`：依赖检查、57项测试、600步闭环核对。
- `tasks/run-training-wsl.sh`：新建带时间戳的短训练输出目录。
- `tasks/run-training-wsl.sh --restore ~/projects/tenny-reference/artifacts/train-smoke-20261114-143038/checkpoints/000000000016`：加载参数继续训练。
- `tasks/collect-results-wsl.sh`：将诊断报告复制回 Windows。

独立诊断脚本 `tasks/training_smoke.py` 不改作者源码。初次运行曾因配置包含 Path 而无法JSON序列化，已改用 ConfigDict.to_json_best_effort 后完成两轮验证。

## 验证边界

短训练仅证明流程跑通，不证明新策略学会平衡或达到作者效果。当前交互演示使用压缩包内作者已有的策略数组，未被短训练权重覆盖。原完整训练 checkpoint 不在压缩包中。未做长时间收敛训练或真机验证；后续有限GPU性能评估见 `GPU-TEST.md`。
