# 腿执行能力配对实验

独立比较±1与±2N·m腿输出限制，同公式重标定虚拟腿力范围，固定30秒回合、原奖励与观测的扭矩物理单位。详细条件见[实验合同](../../tasks/torque-limit-ablation-draft.md)。生产默认参数、服务和旧模型不由本模块修改。

- `contract.actuator_contract`：执行上限/F范围与固定归一化尺度分开，拒绝本轮未计划的上限。
- `env.TorqueLimitEnv`：在原奖励实验环境上只扩大执行器/力范围；控制循环保留原能耗、变化率、前拍扭矩观测尺度。
- `native_patch.patch_legwheel_source`：对严格绑定的源码生成隔离副本补丁，允许可选归一化分母；不写生产文件。没有该补丁的原生评估器不能读取本实验的新字段。
- `report.paired_summary` / `write_report`：从版本1诊断JSON生成配对汇总和报告，不加载训练运行时。两组各自0步BC参考分别保留，最终差与相对各自起点的增量分开。

同一BC网络/normalizer数组不意味着相同初始闭环行为：扩大F范围会使相同动作产生更大腿力。该差异必须在报告中显示；不能通过修改学习率、预算、初始化或奖励去掩盖。

后续网页可读取`paired-summary.json`/`results.json`并复用纯报告计算。限幅裁剪聚合与末子步幅值统计应保持各自的采样说明。模型绑定冻结实验环境、XML及独立原生适配，不直接作为生产兼容模型导入。

针对验证：`python -X utf8 tools/test_cpu.py discover -s tests -p 'test_torque_limit_*.py'`。本轮执行脚本、真实源码冻结和输入/模型路径存于忽略的`tasks/results/torque-limit-ablation-20261116/`，重现先读其中状态，不重启已有矩阵。
