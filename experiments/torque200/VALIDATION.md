# 全扭矩 200 Hz 最小改版验收

日期：2026-11-14（香港时间）。最终完整运行：`smoke-20261114-154038`。
`test-torque200-training.ps1` 正常退出，退出码 0；终端打印 `TORQUE200_ALL_CHECKS_OK`。

## 已验证结果

| 项目 | 结果 |
|---|---|
| 环境回归 | 18/18 通过 |
| 模型 | 原参考机器人；4 路绝对扭矩；无位置/速度伺服 |
| 时序 | 控制周期 5 ms，物理步 2.5 ms，每次动作 2 子步 |
| 观测 | Actor 24 维，Critic 51 维 |
| 训练设备 | `cuda:0`，参数树平台 `gpu` |
| 第一轮 | 2 训练环境、2 评估环境、16 累计环境步 |
| 第二轮 | 从第一轮 checkpoint 加载参数，再训练 16 步 |
| 参数完整性 | 每轮全部 25 个叶节点有限且保存前后精确一致，最大误差 0 |
| 训练 loss | entropy、policy、total、value loss 全部有限 |
| 第一轮导出最大动作误差 | `1.741945743560791e-5` |
| 第二轮导出最大动作误差 | `1.3709068298339844e-5` |
| 导出验收阈值 | `1e-3`，8 组观测包含动态状态 |
| 继续训练后的网络最大参数变化 | `0.00020013749599456787`，确认实际更新 |
| 原版本保留 | Windows 358 项、WSL 354 项文件指纹未变 |

最终两轮诊断分别约 52.67 秒、53.23 秒，包含评估、保存、加载、导出与动作对照。
已有编译缓存，不能把该总耗时当作正式 PPO 吞吐或预计收敛时间。

## 产物

- 运行日志与报告：[results/smoke-20261114-154038/](results/smoke-20261114-154038/)
- 第一轮报告：[first/report.json](results/smoke-20261114-154038/first/report.json)
- 第二轮报告：[resumed/report.json](results/smoke-20261114-154038/resumed/report.json)
- 参数验收：[parameter-verification.json](results/smoke-20261114-154038/parameter-verification.json)
- 环境测试：[environment-tests.log](results/smoke-20261114-154038/environment-tests.log)
- 原 WSL 指纹核对：[original-preserved.log](results/smoke-20261114-154038/original-preserved.log)
- 原 Windows 指纹核对：[original-windows-preserved.json](original-windows-preserved.json)

两轮各自目录内有 `env_config.json`、`ppo_config.json`、`progress.json`、
`params_cpu.pkl`、`smoke_params.cpp.txt` 和 `policy_metadata.json`。

实际 Orbax checkpoint 位于 WSL：

```text
~/projects/tenny-torque200/artifacts/smoke-20261114-154038/first/checkpoints/000000000016
~/projects/tenny-torque200/artifacts/smoke-20261114-154038/resumed/checkpoints/000000000016
```

## 验证边界

这是短训练流水线验收，不代表学会平衡、奖励收敛、长期稳定或真机可用。
腿部初始扭矩尺度为 1 N·m，轮为 0.0295 N·m，未做控制效果调优。
原腿关节无显式机械限位的模型设置保持不变，长期角度范围仍待评估。
仅验证 NumPy 导出推理，没有进行 ESP32 编译或 C++ 实机执行。
恢复是加载网络及归一化参数继续训练，不包含完整优化器状态无缝恢复。

参数核对显式兼容原 Orbax 加载器将 UInt64 计数恢复为 `{hi, lo}` 字典的表现；
两个计数字段均逐一比较，未跳过计数、Critic 或探索尺度。
CUDA 驱动版本格式提示与 PjRt 缓存兼容提示仍可见，但本次实际 GPU 训练与验收均通过。

原版演示继续使用原 `start-simulation.ps1`，不加载这些短训练权重。
新入口和具体变更见 [README.md](README.md)。

## 独立实时窗口追加验收

运行记录：[preview-20261114-155322/preview-report.json](results/preview-20261114-155322/preview-report.json)。
启动 `start-torque200-preview.ps1 -Seconds 8`，退出码0。
加载上述第二轮checkpoint；CPU后端，200Hz仿真控制，8.016秒显示238帧（约29.7fps）。
220步物理/推理计算耗时约0.250秒，渲染累计约3.451秒；包含终止后刻意停留0.7秒，
不能把总步数除墙钟时间当作训练吞吐。预热后该演示可按真实时间推进；并行训练性能未测。
查看器加载模型与导出模型初始动作最大误差3.994e-6。
10次相同确定性零指令回合均在22步/0.110秒触发终止，9次自动重置。
这些是同一初态的重复播放，不是独立成功率测试。确认模型尚未学会平衡。
截图已查看，机器人、扭矩与终止状态正常显示；不开展新训练，不改原版。


## 128环境定时训练准备与面板验证

2026-11-14：实时曲线的3项轻量测试通过（缺失评估不补造、空运行不展示旧记录、重复步数拒绝）。网页确认默认suspended=true、frames=0；新运行启动后旧曲线清空、显示倒计时，随后读取本轮稀疏评估和连续loss。保存/导出启动前验证：25个参数叶重载误差0，导出动作误差约5.14e-6，无新增训练。唯一15分钟运行training-20261114-085906，香港16:59:06开始、计划17:14:06停止，最终结果待写。


## 完整网页训练控制台（2026-11-14）

新增console_train.py/training_control.py，旧timed_train与reference保留。runner 10项标准库测试、manager 7项真实假子进程测试通过（并发启动、重连、停止保存、死亡进程、模式隔离）。网页实际验证原版GPU：90秒定时在101.36秒计算边界停止，0.50秒收尾，9600环境步；从该模型不限时恢复，手动停止在86400步，59.47秒停止、0.37秒收尾。两次保存全参数误差0，导出动作误差6.58445e-6 / 2.29478e-6；第二轮4项loss有限。第一轮在捕获更新参数后触发定时停止，未追加该区间loss，报告如实为空。所有原版配置沿用正式PPO，仅batch缩至8保持16minibatches得到128环境。

刷新网页后仍可控制同一任务；运行中禁用重复开始与模式切换；到时等待状态、时长输入恢复、无限时隐藏时长已验收修正。结果在results/console-control/summary.json。无持续训练留给用户，无旧15分钟任务监控。

## 打开画面时同步策略（2026-11-14）

- 仅关闭→打开触发一次按需请求，重复开启不更新，关闭取消未完成请求；模型加载后重置展示回合。
- 原版与全扭矩真实保存权重各32个观测，与Brax确定性推理最大动作误差分别3.65e-7和2.09e-7；参数包28612/28088字节。CPU轻量发布测得0.82/0.59ms，仅代表此次参数包生成，不含GPU等待或完整训练影响。
- 20项runner/controller/snapshot测试通过。独立8771端口真实网页+仿真渲染验收：请求步数9800→11200，收到JPEG；观看期间不更新；关闭后帧计数停止；取消后保持关闭。该验收使用真实已保存权重和CPU策略发布进程，不执行PPO，不等同真实GPU训练回调全链路性能测试。
- Terra网页点按确认新文案、来源/步数、旧训练兼容提示，曲线和用户训练继续。当前这轮在更新前启动，没有强行中断；新请求接口从下一轮训练生效。
- 证据：results/preview-on-open/numerical-check.json、http-check.json、deployment.json。
