# 弱轮角位置约束对照

从同一成熟模型出发，轮角位置项由10减到1；真实XY奖励10、航向奖励10、学习率3e-5和其他条件保持。合同见[对照说明](../../tasks/wheel-hold-ablation-draft.md)。旧weight10模型仅在等价门禁通过后复用，新组独立训练。`env`只改单项贡献，`train`区分共同评价回报与实际训练目标，`report`为可复用纯指标和离线报告。

修改或继续前读忽略的`tasks/results/wheel-hold-ablation-20261116/PROGRESS.md`确认已有运行状态。每阶段保存完整参数与独立descriptor；恢复优化器重新建立。奖励使用仿真真实XY不等于actor接收世界坐标；当前actor仍无绝对XY观测，实机IMU/轮滑鲁棒性需要另行验证。

完整阶段参数保存针对本轮新weak组；旧复用baseline的历史中间descriptor不包含完整critic。事后选出的较好阶段与最终步数配对统计分开报告，不作为额外训练seed。
