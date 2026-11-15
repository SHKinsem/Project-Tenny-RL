# Checkpoint 累计属性与续训进度 v1

2026-11-15 更新。累计合同已接入[生产工作台](workbench-production.md)的读取、真实保存包装层、精确历史续训及完整手动快照。早期合成验证与30秒独立续训属于历史验收；当前控制边界见[控制、保存与加载](workbench-controls.md)。原控制/物理/策略源指纹保持独立。

## 累计与预算

checkpoint 累计量 = 该运行固定的父模型累计量 + 该运行在实际捕获点的新增量。秒数和环境步数分开计算，不能按频率从步数推算训练耗时。一次运行里的每个快照都相对相同父模型计算。

| 模拟阶段 | 历史秒 | 本轮秒 | 累计秒 | 本轮预算 | 本轮比例 | 累计比例 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 载入父模型 | 600 | 0 | 600 | 600 | 0% | 50% |
| 新增一分钟 | 600 | 60 | 660 | 600 | 10% | 55% |
| 暂停，报告计数不变 | 600 | 60 | 660 | 600 | 10% | 55% |
| 最终保存另耗15秒 | 600 | 60 | 660 | 600 | 10% | 55% |

累计目标始终是历史600 + 本轮预算600 = 1200。若本轮先后保存60秒、120秒快照，它们是660秒、720秒，不能把660秒再加到第二个快照中。从660秒模型开启下一运行，再新增60秒，累计是720秒；下一轮追加预算另算。

明确的新训练历史为0。旧模型未知历史用null，不能因“新建了运行目录”就当作从零训练。历史秒数未知时仍可显示本轮比例；历史步数若已知，步数累计可独立保留。预算0表示不限时，百分比、累计目标、剩余预算均为null，已知累计量仍返回。

读取不会按当前时间外推进度。累计采用既有报告的训练预算计时：优先training_stop_elapsed_seconds，或从报告elapsed中分离finalization_seconds；无法分离时保持未知。原版预算含初始化/JIT/评估；hybrid排除准备后仍含初始化/JIT/评估。clockPolicyId必须一致，不能直接拼接不同政策的秒数。超预算保留实际量，剩余预算最低0，显示层可自行限制进度条长度。

控制协议1还从本轮预算及累计中扣除实际暂停区间。暂停意图需到安全回调边界才生效，等待期间不能提前声称计时已冻结。手动快照记录真实捕获点，不能把请求时刻或HTTP接受时刻当作捕获量。

## 两份版本化记录

### 运行起点：workbench-run.json

位置：`console/<mode>/<run>/workbench-run.json`。它保存一层父模型属性快照，不递归查询父目录。

| 字段 | 含义 |
| --- | --- |
| schemaVersion | 整数1；未知版本拒绝 |
| id | 该运行起点记录的持久UUID |
| runId / mode / controlHz | 当前运行身份、模式与控制频率 |
| clockPolicyId | 明确的计时政策标识，例如本例console-original-v1 |
| origin | new、checkpoint或legacy |
| parentCheckpoint | checkpoint来源需附完整元数据；new/legacy为null |

status.json须显式发布`workbench_context_id`，与起点记录id相同。身份/模式/频率/政策错误或未绑定的文件不会用于计算累计；只返回警告与未知历史。若状态带clock_policy_id/control_hz，亦须一致。报告与当前状态的起点身份不一致时，拒绝该累计来源；旧报告不能把最终计数借给新起点。

### 保存属性：checkpoint同级的 .workbench.json

例：权重目录`checkpoints/000000100000`的旁路元数据是`checkpoints/000000100000.workbench.json`。普通文件checkpoint也使用同样的追加后缀规则。旁路文件不会被当作额外模型，不放进权重目录，避免改变现有权重目录的哈希。

| 字段 | 含义 |
| --- | --- |
| schemaVersion / id | 版本1与持久checkpoint UUID，不按排序或路径生成 |
| createdByRunId | 实际产生此模型的运行身份，读取时需匹配所属运行 |
| mode / controlHz / clockPolicyId | 模式、控制频率与计时政策 |
| parentCheckpointId | 固定父模型身份；没有可记录父身份时为null |
| baseTrainedSeconds / baseTrainedSteps | 该运行固定的历史起点，分别允许null |
| capturedSeconds / capturedSteps | 本次实际捕获点的本轮量，不是保存请求时刻或预算 |
| trainedSeconds / trainedSteps | 基数加捕获量；任一未知则该维累计未知 |

秒数为非负有限JSON数；步数为非负整数，最大2^53−1，避免浏览器丢失精度。所有可空字段须显式存在。构造与读取均检查总量关系、模式/频率/政策、UUID和自引用。秒数校验允许两个浮点ULP以内的序列化误差；步数总量精确相等。未知历史不能被合法化为伪造零。

只读候选的metadataStatus=valid仅表示旁路合同与身份检查通过；该读取器不核验权重或恢复组件，候选verification仍为unverified、canResume仍为false。可续训起点由`checkpoint-options`的实际验证结果提供，开始续训时按精确路径再次检查；保存和导入已有完整产物验证流程。UUID不分配C编号或显示别名，也不提供全局身份注册/去重；短别名命名域及更细权重绑定仍待后续。

## 模块与读取结果

[checkpoint_metadata.py](../experiments/torque200/workbench_api/checkpoint_metadata.py)提供make_run_context、capture_checkpoint、validate_run_context、validate_checkpoint、calculate_progress。它不创建文件、不启动进程，不读取系统时钟来推进训练。创建时可传UUID供重试复用；不传时生成新UUID，调用者应保留创建结果的身份。

```python
context = make_run_context(
    run_id="next-run", mode="original", clock_policy_id="console-original-v1",
    origin="checkpoint", parent=validated_checkpoint,
)
saved = capture_checkpoint(context, elapsed_seconds=60, steps=10000)
progress = calculate_progress(context, elapsed_seconds=60, steps=10000, duration_seconds=600)
```

GET state的run与progress新增contextId、sourceCheckpointId、clockPolicyId、baseTrainedSeconds/baseTrainedSteps、trainedSeconds/trainedSteps、targetTrainedSeconds、remainingSeconds、segmentPercent与cumulativePercent。GET checkpoints读取有效旁路记录后提供持久id、父身份、历史量、捕获量与累计量。数据文件变化纳入revision；GET仍无写盘副作用。损坏/缺失元数据不影响旧候选的浏览，但累计属性保持未知。

## 早期可执行例子与验证

在仓库根执行，例子强制使用可自动清理的临时目录，不接收生产路径：

```powershell
python -B -m experiments.torque200.workbench_api.examples.checkpoint_resume
python -B -m unittest discover -s experiments/torque200/tests -p 'test_workbench_*.py' -v
```

[示例脚本](../experiments/torque200/workbench_api/examples/checkpoint_resume.py)实际构造JSON文件，重新创建reader读取，再保存/继续下一轮。它输出600→660→720秒和100000→110000→120000步；暂停和保存阶段的计时不新增到累计。浮点百分比可能保留IEEE尾数，展示层负责格式化。

累计模块早期批次合并62/62通过：20个纯元数据案例、9个文件级累计案例、26个已有读取案例、7个HTTP案例。该批次仅使用临时合成产物，HTTP只访问自有随机回环端口，结束后关闭服务。后续真实训练、手动快照、续训和UI接线验收另见[生产整合](workbench-production.md)；当前测试总数不由这一历史批次定义。

## 已接通行为与保留边界

- 已接通安全保存旁路：权重成功后发布周期记录，最终报告使用实际训练截止量；失败保留权重与旧manifest并披露，不透传旧成功ID。重试复用身份。
- 新运行发布固定起点及workbench_context_id，状态/报告沿用同一身份；历史起点来自所选精确模型的可信旁路记录。
- 真暂停/同进程继续、独立完整快照、已验证历史模型续训、目录选择与完整导入均已接通。保存完成由持久回执确认，HTTP接受只表示排队。
- 跨进程续训恢复网络与观测归一化，优化器重新建立；同进程暂停则保留优化器，不能混用两种恢复语义。
- 旧模型迁移仍需可靠历史依据，未知保持null。完整历史迁移、短别名命名域、更细权重绑定及跨机器人完整语义合同仍属后续工作。
