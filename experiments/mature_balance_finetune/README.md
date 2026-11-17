# 成熟平衡策略的位置微调

从已通过120秒固定初态回放的历史腿力模型继续，保留完整actor、critic、normalizer与原动作映射；实验条件见[合同](../../tasks/mature-balance-finetune-draft.md)。真实XY位置项复用既有奖励适配器，两组只改变固定学习率。`train`在实际Brax入口绑定学习率，`matrix`按冻结输入顺序运行，`report`为无训练依赖的物理指标和可复用报告。

修改、恢复或继续前，先读忽略结果目录`tasks/results/mature-balance-finetune-20261116/PROGRESS.md`确认真实起点和运行状态。各阶段descriptor均独立保存；完整checkpoint不包含Adam历史，恢复时优化器重新建立。模型继续采用原腿力动作头，不使用对齐容量实验的冻结BC锚点。

中间descriptor只保存确定性推理语义；完整actor/critic/normalizer参数在最终checkpoint中。中间策略可回放，不能将它当作该时刻的完整PPO续训checkpoint。事后挑选的较好中间策略须与预先定义的最终步数配对比较分开报告。
