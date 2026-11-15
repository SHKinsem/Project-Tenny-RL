# Checkpoint 累计与续训进度草稿

2026-11-15。仅 workbench-api Worktree、临时合成产物；不改原树、训练器、管理器、启动器、生产或 UI 接线。HTTP 保持 GET/HEAD。

## C-01：先固定累计语义

- 累计 = 固定父模型累计 + 本轮实际捕获的新增量。多个中途快照相对相同起点计算，不能逐次相加。
- 时长与步数独立可空。未知不填0、不按目录名或预算推算。只有明确new起点为0；legacy起点未知。
- 预算为本轮追加时长。累计目标=历史时长+本轮预算；历史600、预算600：初始50%，新增60后55%。不限时不算百分比，已知累计量仍保留。
- 不按读取时间外推；暂停和最终保存保留最后报告的计数，沿用既有训练/保存计时分离。分段与累计比例分开，超预算保留实际量。

## C-02：版本化数据合同

- 标准库 checkpoint_metadata.py 构造/验证run context、checkpoint metadata、计算进度，返回JSON dict，无磁盘/进程操作。
- run/workbench-run.json：schemaVersion=1；id(UUID)、runId/mode/controlHz/clockPolicyId、origin(new/checkpoint/legacy)、parentCheckpoint（完整一层元数据或null）。状态须显式发布workbench_context_id与其id相同才采用，拒绝错配的旧sidecar。
- checkpoint旁路文件为同级`<名称>.workbench.json`，不改权重目录哈希。schemaVersion=1；id、createdByRunId、mode/controlHz/clockPolicyId、parentCheckpointId、baseTrainedSeconds/baseTrainedSteps、capturedSeconds/capturedSteps、trainedSeconds/trainedSteps。总量须与基数/捕获量一致，拒绝自引用。
- 模式/控制频率/计时政策一致；秒数非负有限，步数非负整数且不超过JS安全整数。持久UUID与显示短别名分开，暂不分配短别名。合同有效不等于权重可恢复。
- 入口：make_run_context(*,run_id,mode,clock_policy_id,origin='new',parent=None,context_id=None)；capture_checkpoint(context,*,elapsed_seconds,steps,checkpoint_id=None)；validate_run_context(value,*,run_id=None,mode=None)；validate_checkpoint(value,*,mode=None)；calculate_progress(context,*,elapsed_seconds,steps,duration_seconds)。MetadataError含code/message。
- calculate_progress返回本轮/累计百分比、历史/累计秒步、targetTrainedSeconds、remainingSeconds、sourceCheckpointId/contextId/clockPolicyId、warnings。context=None表示未知历史，不从旧报表私有trained字段推断。

## C-03：接入与验证

- state读取已绑定context得到累计量/进度；checkpoints校验sidecar的mode/run身份，sidecar不作为候选。持久id/累计可读，verification仍unverified、canResume仍false。
- 缺失/损坏/未知版本/错配/不兼容均警告并保持未知，不读取外部引用。元数据纳入revision，GET不得写盘。
- tempfile覆盖序列化→读取→进度→再次保存/恢复；附独立可执行示例。
- 验收600+600、600→660→下一轮、多快照不重复、暂停、保存计时、不限时、新训练、未知/部分历史、数值/身份/模式/政策错误、分页revision、只读性。
- 真实保存器接线、历史迁移、短别名、实际训练暂停/恢复、UI连接仍为待办。

## C-04：验收结果与确定边界

- 已完成5函数纯模块、state/checkpoints读取、HTTP贯通和临时文件重载示例；合并62/62通过，无训练/模型加载。
- 持久身份拒绝nil UUID，缺父身份用null。浮点百分比不强行round，55%可有IEEE尾数；秒数总量校验仅容许两个ULP的表示误差，步数严格相等。
- 当前状态和报告属于不同起点时拒绝累计；有效新起点不从无绑定的旧报告借用计数。元数据schema通过仍不等于实际权重已校验。
- 只由fixture写临时文件；实际保存器需先完成权重持久化，再原子发布元数据，并在重试时复用身份。上述写协议与生产/UI接线仍待后续。
