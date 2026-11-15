# 工作台只读接口草稿

日期：2026-11-15。位置：独立 `workbench-api` Worktree，基线 `2ec12cf`。原工作区只读；不连接生产、不启动训练、不接入 UI 原型。

## A-01：基线核对

- 当前有 original、torque200 和六个 hybrid 模式；没有原工作区未提交的 teacher 扩展，不能复制或宣称已支持。
- 两套 handler 提供 GET `/state`、`/learning-data`、`/stream.mjpg` 和 POST `/training`、`/control`。training 只有 start/stop；control 的 pause/unpause 暂停预览。
- TrainingController 构造会 mkdir；status/runs/latest 持锁并可能修复 active.json。只读模块不得调用它。
- progress.json 只保留 2000 个步数合并后的点；metrics.jsonl 保留每次回调（含重复步数）。优先读取 JSONL，不把窗口当完整原始历史。
- 原版计时包括初始化/JIT/评估；hybrid 准备完成后重置时钟，之后仍包括初始化/JIT/评估。最终保存另计，读取优先 training_stop_elapsed_seconds。累计未知不能填零。

## A-02：首批合同（先记录，再实施）

- 独立标准库模块 experiments/torque200/workbench_api，不导入训练管理器、ML 或仿真。GET `/api/workbench/v1/state`、`checkpoints`、`metrics`、`capabilities`；只提供 GET/HEAD，拒绝控制方法。单独回环服务器，不修改现有 handler、启动器、UI 或部署配置。
- WorkbenchReader(artifacts) 公开 state(mode)、checkpoints(mode, offset, limit, revision)、metrics(mode, run_id, offset, limit, revision)、capabilities()。返回 JSON DTO；失败使用含 code/status/message 的受控异常。
- 公共字段 schemaVersion、revision、readAtMs、source、warnings。磁盘修改时间与生产者时间分开；readAtMs 不是训练/评估产生时间。多文件仅 per-file 一致性，不宣称事务快照。
- run ID 是现有运行目录引用；checkpoint referenceId 是相对位置的确定性引用，重命名会变化。持久 checkpoint ID/短别名尚未写入，返回 null，不按排序生成 C 编号。候选产物不等于核验通过或可续训。
- 状态返回 reportedPhase/stage、liveness=unverified、active=null；不把磁盘 training 记录宣称为正在运行，不按读取时钟外推进度。保留本轮与累计字段，累计未知为 null。
- 指标按文件顺序分页，保留重复/乱序步数、零/负/微小原值和缺口，不取绝对值、不平滑、不降采样。JSONL 缺失才回退 progress.json 并标明窗口局限。无事件时间戳返回 null。revision 校验防止分页混用版本。
- 限制读取字节、单行、记录数和目录扫描；拒绝越界/链接逃逸及非法查询；损坏/半写入记录显式降级，不泄漏绝对路径。无产物时不创建目录或锁文件。
- 能力明确：只读查询可用；真正训练暂停/继续、后台保存、指定模型恢复、累计迁移、目录桥接和新 UI 连接均未接通。

## A-03：验证与后续

- 测试只创建临时合成产物，HTTP 仅用自己的随机回环端口；覆盖 10000 点、重复步数、异常值、半行、旧窗口、分页版本冲突、缺失历史、路径逃逸与拒绝写入。
- 读取前后文件清单与内容不变；缺失根目录仍不存在；导入不加载训练/仿真依赖。
- 下一写阶段先定 checkpoint manifest/持久 ID、累计与恢复组件，再补 runner 的真实暂停/快照协议。UI 继续使用模拟数据。

## A-04：验收后明确的读取边界

- 同一来源的默认选择与显式 runId 使用同一指标 revision；切换到另一运行仍冲突。Windows 目录 mtime 存在延迟更新，选择证据采用目录名称/类型，实际来源文件仍以 mtime 和内容 hash 检测变更。
- 有限原始计数相除仍可能溢出，派生比例返回 null 与 warning；不改变原始计数。固定文件名被非普通文件替代时在 open 前拒绝，并使用可用的非阻塞标志，避免 FIFO 卡住读取。
- JSONL 按 LF 有界逐行解析；合法的末行无换行保留值并标警告，损坏末行保留 gap，不抛弃整页。full 只指当前文件完整读取，producerHistoryComplete 仍为 null。
- 32 项合成数据/随机回环集成测试通过；生产接线、真实模型验证、累计历史写入仍不在本批。

## A-05：累计元数据后续批次

2026-11-15按用户继续授权，已在同Worktree补checkpoint属性合同、运行起点绑定、累计秒/步与续训比例，仅用合成数据验证。详见checkpoint-progress-draft.md与docs/checkpoint-progress.md；合并62/62通过。真实保存器发布、历史迁移、权重验证和UI/生产接线继续待办。
