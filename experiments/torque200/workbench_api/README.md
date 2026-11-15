# 工作台只读接口 v1

独立标准库模块，读取显式指定的 `artifacts/console/<mode>/<run>`。不创建产物目录、锁文件或状态文件，不实例化TrainingController，不加载模型、不导入JAX/MuJoCo。现已通过workbench_web接入[生产工作台](../../../docs/workbench-production.md)的只读路由；写入在独立persistence包装层，控制沿用旧协议。ui-prototype仍独立模拟。

现状、缺口与后续待办见[后端对照表](../../../docs/workbench-backend.md)，实施前草稿见[workbench-api-draft.md](../../../tasks/workbench-api-draft.md)。模式包括original、torque200、六个hybrid及两种teacher，定义以[control_modes.py](../control_modes.py)为准。模块同时支持仓库命名空间及WSL顶层workbench_api导入。

## 验证

在仓库根目录执行，Python 3.11+，无需额外依赖：

```powershell
python -B -m unittest discover -s experiments/torque200/tests -p 'test_workbench_*.py' -v
```

测试创建并清理临时合成产物，HTTP 使用自己的随机 IPv4 回环端口。不需要生产地址、真实 checkpoint、训练进程或 GPU。

手动检查空目录或自行准备的隔离产物：

```powershell
python -B -m experiments.torque200.workbench_api --artifacts 'C:/Temp/tenny-workbench-fixture' --mode original --port 0
```

服务器只绑定 `127.0.0.1`，默认端口 0，由系统分配并打印实际地址。上述目录不存在时返回空数据，仍不创建目录。使用 Ctrl+C 退出。这里的路径必须由调用者明确指定，不会自动寻找正在运行的训练。

## HTTP 合同

仅 GET/HEAD；控制方法返回 405 和 `Allow: GET, HEAD`。所有响应为 JSON、`Cache-Control: no-store`，不开放跨域读取。

| 路径（前缀 `/api/workbench/v1`） | 参数 | 返回主体 |
| --- | --- | --- |
| `/state` | mode | available、mode、run、progress |
| `/checkpoints` | mode、offset、limit、revision | available、mode、items、pagination |
| `/metrics` | mode、runId、offset、limit、revision | available、mode、runId、items、definitions、pagination、rawSource、historyCompleteness |
| `/capabilities` | 无 | 当前只读能力与未接通能力 |
| `/descriptor` | mode | 当前run身份、保存的机器人/任务/观测/动作/计时与PPO描述；未知为null |

生产集成另提供`/catalog`，模式名称/频率由后端适配描述，前端不维护硬编码列表。`capabilities.legacyControls`仅在集成层说明旧协议的实际能力，本版本化命名空间继续只读。

mode 默认是服务器启动时指定的模式（默认 original）。分页 offset 从 0 开始；checkpoints 默认 limit=100、上限 1000，metrics 默认 500、上限 2000。未知、重复、空查询参数及非法模式/范围返回 400。`runId` 是所选模式下的直接子目录名，不接受路径。

所有成功响应包含：

| 字段 | 语义 |
| --- | --- |
| schemaVersion | 整数 1 |
| revision | 当前数据来源的确定性版本，与读取时间分开 |
| readAtMs | 本次读取时间；不是心跳或评估产生时间 |
| source.kind / consistency | artifact_files 或 static_contract；文件读取只具备 per_file 一致性 |
| source.files | 相对产物路径、modifiedAtMs、recordedAtMs、sizeBytes；不返回系统绝对路径 |
| warnings | 有 code/message、可带相对 path 的降级原因 |

分页包含 offset、limit、total、nextOffset（结束为 null）。第一页取得 revision 后，后续页传同一 revision；来源发生变化返回 409，调用者重新读取第一页，不将不同版本拼到一起。revision 不代表多个文件具有原子事务一致性。

错误格式：`{"schemaVersion":1,"error":{"code":"…","message":"…"}}`。未知路由或指定运行不存在为 404；源超过资源限制为 413。损坏/不可读取产物通过 warnings 和 available 状态体现，不把异常冒充为真实零值。

## 状态与进度

run.id 是现有运行目录引用，mode 是选择范围。reportedPhase/reportedStage 是磁盘最后记录；`active: null`、`liveness: "unverified"` 明确未检查进程存活。文件很新也不能证明进程仍运行，过旧也不能直接改写成 interrupted。不调用旧管理器的状态修复。

steps、elapsedSeconds、durationSeconds 是本轮量。预算 0 表示不限时，不计算比例。elapsedSeconds 优先使用 training_stop_elapsed_seconds，避免将最终保存耗时算入训练进度；其次用报告的 finalization_seconds 分离。保存/终态缺少两者时无法确认预算用时，返回 null 和 warning；其他阶段保留报告计时，不按当前时间外推。hybrid 明确报告训练时钟尚未开始时，新增训练用时为 0，准备用时另列。派生比例超出有限数值范围时保持 null。

有效且绑定的workbench-run.json可提供历史起点、计时政策与父模型身份，返回baseTrainedSeconds/baseTrainedSteps、trainedSeconds/trainedSteps、targetTrainedSeconds与cumulativePercent。历史600秒、本轮预算600秒从50%开始，新增60秒为55%。旧产物缺乏可信累计元数据时仍为null。segmentPercent只表示本轮预算比例；不因接口读取而继续增长。计时政策与格式见[累计合同](../../../docs/checkpoint-progress.md)。此批不改变运行器。

## checkpoint 候选

items描述候选目录或受限引用：referenceId、relativePath、runId、mode、kind。referenceId由相对位置确定，重命名会改变，不能当作持久模型身份。若同级`<名称>.workbench.json`有效，返回持久id、父身份、捕获量与累计量，metadataStatus=valid；缺失或错误则累计/id保持null。alias仍null，不按排序生成C编号。元数据格式见[累计合同](../../../docs/checkpoint-progress.md)。

`verification: "unverified"`、`canResume: false` 表示本模块未加载或核验模型。已有旧报告不等于本次实际核验，也不解锁续训。此接口不导入、不恢复、不打开目录，不修改模型文件。

## 原始曲线

优先 metrics.jsonl，保留文件顺序与每次回调，包括重复或乱序步数。仅当 JSONL 缺失时回退 progress.json；后者已由运行器合并重复步数并截取窗口，不能还原全部原始回调。

每个 items 元素包含 sequence、steps、elapsedSeconds、recordedAtMs、values、gap。sequence 是来源文件的记录位置，不能跨运行当业务 ID。values 包含 episode_seconds、episode_reward、value_loss、policy_loss；definitions 提供单位与 `stepsScope: "segment"`。episode_seconds 仅在 ctrl_dt 与所选模式一致时由平均回合步数换算，其余保留原值。

无效/缺失值为 null；零、负值和微小值保持原样。坏记录保留 gap，末尾未完成记录给出警告。没有生产者时间戳则 recordedAtMs 为 null，不用读取时间、文件修改时间或本轮 elapsedSeconds 伪造评估时刻。图表应在 gap 处断开，坐标、平滑与降采样由 UI 决定。

historyCompleteness 的 full 仅表示完整读取了当前 JSONL 文件，不能证明生产者已保存全部回调；window 表示旧 progress 窗口，unknown 表示无法确认。historyScope 固定为 source_file，producerHistoryComplete 为 null，rawSource 标出实际文件。不同运行不自动拼接，也不自动生成累计横轴。

## 读取限制

JSON 单文件 4 MiB、JSONL 32 MiB、单行 256 KiB、最多 100000 条记录；每目录最多 2048 条、每模式最多 256 个运行、最多 4096 个 checkpoint 候选；每个响应累计读取最多 64 MiB。超过限制显式报错，不静默截成“完整历史”。大历史索引留待后续。

只读取受限根内的固定文件与直接子目录；不接受客户端绝对路径，不跟随逃逸链接。路径、权限与格式问题只返回相对来源和受控错误。多文件状态可能来自相邻的写入时刻，接入真实状态前仍需运行器发布统一 revision/manifest。

## 后续待办

- 元数据构造/校验、累计读取与生产包装层原子发布已接通；旧历史完整迁移、显示别名注册、权重绑定细化与恢复组件声明仍待补充。报告只在实际manifest发布成功时提供workbench_checkpoint_id。
- runner 实现真正训练暂停/继续与后台快照，区分请求点、捕获点、完成状态；不要映射到预览暂停或停止保存。
- 增加真实心跳/状态一致性、参数 schema、目录桥接与多环境预览，再单独验收 UI 接线及部署。
