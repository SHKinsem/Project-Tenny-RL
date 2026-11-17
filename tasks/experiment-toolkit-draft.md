# 实验通用工具与网页体验映射

用户希望把最近对照实验提炼为可复用工具，并核对这些实际操作是否能在现有网页找到、是否方便、是否值得做成体验功能。

## 本轮交付

新增独立标准库包`experiments/experiment_tools`，不修改历史训练/评估源码、冻结目录、模型、生产服务，不启动新GPU训练。

- 输入冻结与核对：文件/目录SHA库存，明确它不是机器人模型兼容性指纹。
- 有预算的顺序任务运行器：明确argv、超时、结果合同、日志与持久终态，拒绝覆盖历史；不默认运行已有训练。
- 通用配对统计：case先在seed内汇总，再做治疗减基线的配对差；缺失/未完成时域不补0，不按早倒的小位移排名。
- 快照筛选：按明确指标与完整时域筛选；保留step/seed、是否有已验证完整参数及证据，不把descriptor当可续训checkpoint。
- 一个CLI与Python函数接口，统一JSON，便于后续网页调用。

两个已有真实来源作为adapter验证：成熟模型学习率对照、弱轮角权重对照。通用统计不知道机器人几何/奖励公式/网络维度；物理测量继续由已有LegWheel adapter负责。历史报告只读消费，不重写其算法或既有结果。

## 数据接口

`paired_study` v1包含conditions、comparison的baseline/treatment、seeds、partitions（case IDs）、带unit/direction的metrics、records与可选reference_records；每record有values与逐指标eligible。None/未完成为unknown，完整case矩阵缺行或重复拒绝。snapshot记录是阶段聚合，事后筛选不当作额外独立seed。

`experiment_jobs` v1包含明确的jobs argv、各job超时、总预算和可选frozen_inputs/expected_result合同。dry-run不执行。运行器只负责声明合同，不代替模型兼容性/实际物理/PPO初始化验证。

## 网页检查

按用户已有偏好由Terra只读查看实际8095界面，截图后评估：历史模型续训、参数、奖励配置、配对批量实验、阶段快照/较好模型回退、固定初态长评估/对照报告、模型完整性与输入冻结信息。禁止启动/停止/暂停/保存/导入真实训练或更改全局设置。不把独立实验目录中的能力当作已上线网页功能。

输出功能映射与优先级；本轮不部署新网页功能。只有实际截图能作视觉审查证据，旧文档与代码用于补充能力说明；无法访问则明确限制。

## 验证预算与边界

CPU针对测试90秒；使用临时CPU子进程验证超时/失败/结果异常/冻结变化；用两份真实小型summary核对旧配对数字和候选选择，无需读取273MB轨迹或运行MuJoCo。记录主体与环境耗时，不全仓重跑，不追加训练或外部发布。结果在忽略的`tasks/results/experiment-toolkit-20261117/`。
