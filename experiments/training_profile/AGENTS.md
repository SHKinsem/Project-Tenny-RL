# 采样与 PPO 诊断

本模块给已安装 Brax 的原 JIT 加阶段元数据，并离线分析 GPU trace；任务参数与模型来自调用者指定的已验证 run。

- 先读[README.md](README.md)定位运行/重放入口；本机结论与未通过门槛见[RESULTS.md](RESULTS.md)。
- 修改阶段边界时读[计时合同](MEASUREMENT.md)，再改 `brax_instrumentation.py`；必须保留剥除标签后 AST 相等和作用域恢复检查。
- 修改 trace 归因时读同一合同，再改 `trace_summary.py`；设备/主机时间、跨 stream 重叠、未知和混合类别不能混算。
- 比较并行环境数时读[SCALING.md](SCALING.md)，使用独立`scaling-wsl.sh`入口；结果见[SCALING-RESULTS.md](SCALING-RESULTS.md)。保持统一PPO工作量，吞吐与profiler分开计量。
- 调用 GPU 前核对输入模型、生产空闲和预算。使用有截止时间的启动器，先复用已有采集做 CPU 重放；不要为修分析器重复跑 GPU。

使用依赖已有 WSL 训练环境。诊断不保存训练模型；生产源码、检查点及安装包文件须做前后身份核对。数据在 WSL 独立工作目录，仓库只保留入口、合同、测试和结果说明。
