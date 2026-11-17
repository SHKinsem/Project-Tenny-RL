# 训练回合时限对照

仅比较原奖励下600与3000个控制步的训练回合上限。冻结条件和判读规则见[实验合同](../../tasks/horizon-ablation-draft.md)；控制层级另见[频率核对](../../docs/control-frequency-review.md)。不修改生产默认参数或服务。

- `contract.horizon_manifest`：从冻结的100Hz原奖励配置深拷贝，只改变外层PPO训练时限；拒绝启用依赖内部回合长度的终止惩罚。
- `report.paired_summary(bundle)`：读取版本1平衡诊断JSON，返回汇总、逐工况行和绘图序列。不加载模型或训练框架，按三个训练seed配对，固定初态只在seed内平均。
- `report.write_horizon_report(bundle, output_dir)`：从已保存轨迹生成HTML、完整/前10秒PNG、CSV与配对JSON。需要Matplotlib，数据计算本身不依赖绘图。

后续浏览器接入可复用`paired_summary`和已有`balance_diagnostics`轨迹/指标合同；本模块没有新增服务或网页控制。6秒组复用必须标记，并在训练记录之外核对先前评价绑定的checkpoint和descriptor哈希。累计合格、最长连续段、环境终止前时长分别保留，不能用更晚终止替代原地平衡。

针对验证：`python -X utf8 tools/test_cpu.py discover -s tests -p 'test_horizon_ablation_*.py'`。实际本机数据及模型路径保存在忽略的`tasks/results/horizon-ablation-20261116/`，不作为模块默认值。
