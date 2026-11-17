# 固定策略平衡诊断

比较固定策略的首回合存活、姿态与真实机身漂移。使用现有原生评估器，不启动训练或访问工作台。运行入口只依赖显式传入的descriptor和初态；指标、绘图与模型执行分离。

```bash
# 在已有MuJoCo、NumPy、Matplotlib的WSL环境中，复用已构建native_eval。
export PYTHONPATH="$NATIVE_EVAL_WORKDIR/python:$PROJECT_ROOT"
python -m experiments.balance_diagnostics evaluate \
  --manifest /path/to/manifest.json --output-dir /path/to/new-results --seconds 30

# 只重新展示已有结果，不加载策略、不运行物理。
python -m experiments.balance_diagnostics report \
  /path/to/results.json --output-dir /path/to/report
```

执行目录必须为新建或空目录，避免覆盖已有证据。版本1以6秒为固定参考，最长评估时域可设6～120秒。工具对每策略分别做6秒和长时域回放，要求首回合共同前缀的观测、动作、物理状态、命令和奖励逐点相同。原终止条件保留，自动reset后的样本排除。

## 输入与输出

输入manifest示例：

```json
{
  "schema_version": 1,
  "seed": 0,
  "scope": "Fixed nominal and explicitly listed perturbed resets only",
  "policies": [
    {"id": "initial", "label": "初始化", "descriptor": "initial.json"},
    {"id": "final", "label": "最终模型", "descriptor": "final.json"}
  ],
  "initial_states": [
    {"id": "nominal", "label": "标称初态", "state": {
      "qpos": [], "qvel": [], "command": [0, 0, 0, 0],
      "extra": {"position_start": [0], "sinyaw_start": [0], "sin_balance_zero": [0]}
    }}
  ]
}
```

`qpos/qvel`须为完整实际模型状态，示例空数组需要替换。descriptor格式遵循[原生接口](../native_eval/INTERFACE.md)和[轮腿合同](../native_eval/LEGWHEEL-CONFIG.md)。所有策略必须使用相同XML、任务与控制周期；源/策略/初态身份随结果保存。模型导出与初始化解析属于调用者，不猜测训练目录或将不兼容模型强行加载。

输出为`results.json`、首/长回放原始NPZ、`summary.csv`、中文`report.html`、完整时域与放大时间段PNG。JSON版本1包括：

- `policies`：descriptor和策略哈希、来源元数据。
- `initial_states`及其SHA256：各策略共用的实际物理状态、任务参考锚点。
- `parity`：6秒与长回放的首回合共同前缀误差及原6秒回报。
- `cases[]`：策略/初态身份、统一SI轨迹`trace`、纯计算结果`analysis`。

新采集的case还可含`native_first_episode_metrics`，为原生任务首回合各步指标的累加值。须按具体指标定义解释；例如当前hybrid裁剪指标先对子步取max，零和可以证明无裁剪，非零和不是裁剪时间占比。旧结果没有该字段时保持未知，不补零。

`trace`以`t_s=0`真实reset起始，随后每行对应一次控制步后的状态。包含机身世界位置/速度、ZYX俯仰/侧倾、俯仰角速度、带符号轮角推算距离、末物理子步电机命令、步奖励和终止。位置以各自reset为参考；俯仰误差使用保存的`sin_balance_zero`恢复参考角。姿态由post-integration的qpos四元数计算，避免直接把MuJoCo原生raw xmat的派生阶段当作同时间的qpos。

`analysis.windows.first_6s/full_episode`含实际观察时间、位移/路径长/姿态/速度RMS、末1秒表现、单位时间回报和命令接近限幅比例。提前失败不补零、不把其奖励与更长回合累计奖励直接比大小。RMS使用实际时间间隔的梯形积分，窗口使用最后可用采样点，不插值。阈值事件默认5cm、俯仰误差10°、平面速度0.1m/s，持续0.2秒才确认；这些是描述性阈值，不能当作已验证的成功标准或根因。

## 模块与后续网页接入

| 模块 | 职责 |
| --- | --- |
| `runner.evaluate_manifest` | 冻结输入的原生执行和前缀对齐，写结果 |
| `legwheel` | 按任务命名解析模型布局，转成统一物理量；当前只支持free-root轮腿 |
| `metrics.analyze_trace` | 无第三方库、无I/O的窗口/事件计算 |
| `report.write_report` | 从结果生成图像/HTML/CSV，Matplotlib懒加载 |
| `actuators.summarize_actuator_intervals` | 无第三方库的逐执行器区间统计，区分未裁剪请求、发送命令、实际输出与归一化动作范围 |

执行器统计调用方必须标注采样语义：完整物理子步、周期内保持值、周期末代表样本不能混用。函数用实际区间权重计算窗口、90/95/99%占用和最长段；末子步样本加权不补回未观察到的子步。无观测窗口的比例和峰值保留未知。来源与一次实际审计见[限幅核对](../../docs/torque-limit-review.md)；未新增生产API。

未来网页可读取同一`results.json`，按policy/initial-state id选择曲线，按`t_s`联动预览和事件标记。应在独立评估任务中运行runner，通过已有任务控制提供状态/取消；避免把30秒回放和序列化放进高频state轮询，或修改正在训练的模型。新增机器人只替换trace适配，不把关节索引塞进通用指标/展示。本轮未新增生产API、网页入口或训练回调。

## 测量边界

- 机身世界位移与轮关节角推算量分开；后者不包括机身转角补偿、完整里程积分或打滑估计。图中使用两者的模长，带符号原始量在JSON。
- `actuator_command_nm`是末物理子步ctrl，不是步内平均力矩；`command_at_limit_fraction`是按控制周期加权的末子步命令样本占比，不是真实物理子步饱和时间比例，也不证明发生了裁剪或电机能力不足。
- 整回合奖励/秒仍受存活时长及访问状态不同影响，不能独立排名策略；比较奖励取舍应使用相同的共同存活窗口。
- 同一策略在固定小扰动上的结果不代表随机初态、外推、域随机化或实机鲁棒性。先后越界只能描述轨迹，不能单独建立因果。
- 本工具更改评估上限，不改训练的6秒时序合同、奖励或checkpoint。

针对性测试：`python -X utf8 tools/test_cpu.py discover -s tests -p 'test_balance_diagnostics_*.py'`。在报告字段、物理适配或窗口算法变化时运行；无需为查看已有结果重跑训练或全仓测试。
