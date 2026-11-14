# GPU 本机验证（2026-11-14）

## 结论

WSL 的 RTX 2080 SUPER 8GB 已可被 JAX CUDA 后端实际使用。矩阵计算、原模型批量推进、CPU checkpoint 加载后继续 GPU 短训练、保存/加载/导出均通过。原作者预训练策略的600步GPU闭环核对全部通过，0失败、0摔倒重置。

## 可复用环境与命令

- GPU Python：`~/projects/tenny-reference/.venv-gpu/bin/python`。
- JAX/JAXLIB/CUDA12 plugin/PJRT：均为0.9.2；其余基础包保持先前通过的CPU版本。完整依赖见 `results/gpu/requirements-gpu-resolved.txt`，pip check通过。
- 显卡：RTX 2080 SUPER，8192MiB；驱动610.74；Ubuntu Python3.12.3。
- GPU测试设置：`JAX_PLATFORMS=cuda`、`CUDA_VISIBLE_DEVICES=0`、`XLA_PYTHON_CLIENT_PREALLOCATE=false`、`XLA_PYTHON_CLIENT_MEM_FRACTION=0.35`。实测JAX分配器limit为3006316492字节（约2.8GiB）；这不是整张卡的总显存限制。
- 独立环境安装：`tasks/setup-gpu-wsl.sh`。采用项目推荐的uv与阿里云PyPI镜像，基础包通过完整版本约束固定。首次CUDA运行库下载约3GB，缓存已保留。

在 Windows 项目目录启动默认2环境、16步测试：

```powershell
pwsh -File .\test-gpu-training.ps1
```

之后需要另测更大训练批次，可显式指定参数，例如 `-BatchSize 64 -Steps 512 -EvalEnvs 4`；这组更大PPO参数本次尚未测试。

同规模基准通过 `tasks/benchmark-device-wsl.sh cpu|cuda --batch-size 32|256 --steps 100` 执行。原策略闭环用 `tasks/verify-gpu-sim-wsl.sh`。这些脚本由 `wsl -d Ubuntu --exec bash <脚本的/mnt/c/...路径>` 调用；Codex需要沙箱外执行权限。

现有交互仿真使用原CPU环境；本次GPU测试另起进程，未覆盖已有演示策略。

## 实际设备与训练

- 矩阵计算结果设备为 `cuda:0`，数值校验通过。
- 训练完成后的参数树平台明确为 `gpu`，不是CPU回退。
- 从CPU报告 `train-smoke-20261114-143038` 的checkpoint加载参数，在GPU上继续16环境步，2训练环境、2评估环境。
- checkpoint重新加载动作误差0；导出为原格式后，NumPy部署推理对8组观测的最大动作误差为 **1.315772533416748e-5**（阈值1e-3）。
- 首次GPU这次完整诊断约178.79秒，包含编译、评估和导出。不能把这次小规模总耗时当作正式训练吞吐。
- 原模型、观测、奖励、域随机化和网络宽度保持原样。没有ESP32编译/烧录。

训练证据：`results/gpu/train-gpu-20261114-145047/report.json`，对应checkpoint保留于WSL `artifacts/`。

## 预热后模型推进吞吐

每组相同原模型、固定零动作、每环境100步；`vmap`并行、`lax.scan`推进，明确同步等待设备完成；预热后重复3次取耗时中位数。检查状态/奖励/观测有限值。没有策略推理、PPO梯度更新或自动episode重置，不能将加速比直接当作端到端训练加速比。

| 并行环境 | CPU环境步/秒 | GPU环境步/秒 | GPU/CPU |
|---|---:|---:|---:|
| 32 | 3,097.91 | 4,812.75 | 1.55× |
| 256 | 4,203.39 | 27,488.75 | 6.54× |

编译耗时单独记录：CPU约12–14秒，GPU约22–25秒（仅rollout lower/compile，不含初始化/预热）。这是正常桌面使用中的有限基准，没有控制其他应用负载，不是硬件极限性能。

## 显存与日志

- GPU短训练期间，nvidia-smi每秒采样的整卡显存峰值 **4026MiB**；起点3414MiB。包含Windows桌面和其他应用，差值不是精确进程显存。
- 256环境GPU基准整卡峰值3961MiB；JAX分配器峰值在用约88.6MiB、峰值池约134MiB。二者统计范围不同。
- 本次未发生显存不足。
- 启动存在 `Could not get kernel mode driver version ... Version does not match the format X.Y.Z` 日志。上述实际CUDA计算、训练与导出均通过；本次未修改驱动或尝试消除此提示，不推断其对其他工作负载的影响。

汇总：`results/gpu/summary.json`；原始JSON与显存CSV均在同目录。

GPU闭环：6项运动学探针通过；600步观测核对全部通过，0失败、0摔倒重置，最大观测/锚点误差4.76837158203125e-7。此检查使用CUDA物理状态和作者原NumPy部署策略，未替换为短训练策略。证据：`results/gpu/gpu-sim-verify.json`。

## 使用边界

本次证明GPU流程可运行、批量MJX推进获得加速；没有执行长期收敛训练，也没有测256环境完整PPO训练吞吐。默认8192环境训练仍未验证。建议下一阶段单独测试适合本机的PPO并行规模与训练速度，再决定长训练配置。

配置依据：[JAX安装说明](https://github.com/jax-ml/jax/blob/main/docs/installation.md)、[JAX显存分配说明](https://github.com/jax-ml/jax/blob/main/docs/201/gpu-memory.md)。
