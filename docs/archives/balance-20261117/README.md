# 原地平衡实验公开证据归档

归档日期：2026-11-17。这里保存十组已完成本机调查/实验的脱敏报告、统计、训练曲线、有效配置、审计脚本和关键图。总览见[实验综述](../../reports/2026-11-balance-experiments.md)，下一会话入口见[网页交接](../../../tasks/experiment-web-handoff.md)。

## 阅读入口

| 记录 | 要回答的问题 |
|---|---|
| [固定策略诊断](balance-diagnostics-20261116/FINDINGS.md) | 6秒内存活是否掩盖后续失稳，轮角与真实位移是否一致 |
| [奖励对照](reward-ablation-20261116/FINDINGS.md) | BC起点下增加姿态/真实位置项能否改善 |
| [6/30秒回合](horizon-ablation-20261116/FINDINGS.md) | 单独延长回合是否有效，频率结论边界 |
| [限幅来源](torque-limit-audit-20261116/FINDINGS.md) | ±1/±24来源、饱和测量频率和实际占用 |
| [±1/±2与重标定](torque-limit-ablation-20261116/FINDINGS.md) | 动作能力变化与初始化物理变化的区别 |
| [物理初始化对齐](aligned-limit-ablation-20261116/FINDINGS.md) | 对齐后容量对照结果 |
| [更新退化审计](training-regression-audit-20261116/FINDINGS.md) | 奖励/更新退化证据，以及成熟平衡模型找回 |
| [成熟模型学习率微调](mature-balance-finetune-20261116/FINDINGS.md) | 3e-5/3e-6配对、奖励冲突和中间候选 |
| [弱轮角奖励](wheel-hold-ablation-20261116/FINDINGS.md) | 轮角权重10→1带来的真实位置改善、代价和完整候选 |
| [通用工具验证](experiment-toolkit-20261117/FINDINGS.md) | 通用统计/运行/筛选接口与历史结果等价性 |

图保留在各报告相对目录；GitHub 可直接查看 PNG。JSON/CSV 保留指标单位和每seed数据，JSONL保留已有过程记录；没有在归档时新增训练或物理测量。各 FINDINGS 中“未提交推送”等文字描述报告形成时的历史状态，当前发布以 Git 历史为准。

## 范围与完整性

[manifest.json](manifest.json)逐项记录427份选中文件的原始相对路径、大小、原始SHA、公开副本SHA与是否转换；归档有效载荷约15.18 MB，另有索引/清单。168份大数据或冗余产物约1.270 GB仍只在本机。清单中的原路径相对项目根，不含本机用户名。

公开副本把仓库/用户目录替换成 `${REPO}`、`${USERPROFILE}`、`${WINDOWS_HOME}` 或 `$HOME`；JSON/JSONL重新序列化。原始报告内部的descriptor/checkpoint/source SHA仍指向**原始字节**，不能拿它们当公开脱敏文件的SHA。核对公开副本用清单的 `archive_sha256`，追溯本机原件用 `source_sha256`。数值/布尔/未知值未改。

这是一份结论和证据归档，**不是完整模型或原始轨迹备份**。未上传大型 `results.json`、NPZ、模型descriptor/完整参数二进制、压缩运行时包和部分临时文件；逐文件省略范围在清单中。此归档不能单独恢复PPO训练。原件仍在忽略的 `tasks/results/<记录名>/`，完整checkpoint在本机独立WSL实验目录。

原报告指向未纳入文件的Markdown链接已改成“本机原始资料”说明，避免伪装成有效下载链接。JSON中的历史文件引用仅作来源说明；`${...}`路径不保证可直接执行或打开。网页只读接入可消费小型summary/CSV，不需要先读取数百MB轨迹。

## 历史脚本与继续训练

同目录的 `.py/.sh/.ps1` 是脱敏后的历史审计/执行源副本，保存当时做法；路径占位符不会自动变成当前环境，不应批量执行。日常复用优先使用[实验工具](../../../experiments/experiment_tools/README.md)和各 `experiments/*` 的正式接口。

weak seed0/786432有原记录时完整参数重载验证；较早high_lr seed1/1048576仅有确定性descriptor。真正续训必须回到本机完整模型，实时检查路径、源码、动作/观测和参数兼容性。跨进程续训恢复网络/归一化、Adam重建。事后挑选的单策略不算额外seed，也不证明鲁棒收敛。

网页入口检查只读进行，未开始训练或部署。会话内截图可见，但截图落盘受Edge桥接未连接阻断，因此归档没有网页PNG，也不声称完整视觉或无障碍验收。
