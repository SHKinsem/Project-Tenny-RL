# 工作台后端：当前能力与接口边界

2026-11-15 更新。当前已完成真实工作台接线、控制扩展与生产发布，入口及验收见[生产整合](workbench-production.md)。`workbench-ui` 使用真实接口；独立 `ui-prototype` 保留模拟数据。本文末尾保留明确标注的首批只读合同。

## 当前入口与能力

原版/全扭矩由 [web_preview.py](../experiments/torque200/web_preview.py) 提供，hybrid 由 [hybrid_web.py](../experiments/torque200/hybrid_web.py) 提供；工作台适配层统一数据、控制与多窗入口。

| 入口 | 当前行为 | 能力边界 |
| --- | --- | --- |
| GET `/state` | 预览状态、监督器真实活动状态与已核验最新模型 | 新鲜度决定写入资格；产物中的 reportedPhase 不代表进程存活 |
| GET `/learning-data` | 当前 progress 窗口与 measured 采样标记，旧模式保留证据回退 | 各指标按实际采样独立显示；历史原值由 v1 metrics 分页读取 |
| GET `/stream.mjpg?viewport=<id>&generation=<n>` | 至多四个独立 CPU 预览；每次开启同步一次，观看期间固定版本 | 代次绑定，旧流在代次变化后结束；真实四窗验收约每窗 2 FPS |
| POST `/training` | start、stop、pause、resume、save；支持精确 checkpoint 路径及参数覆盖 | 暂停/手动保存要求活动进程声明控制协议 1；旧运行不热升级 |
| POST `/control` | 保留单窗开关、仅预览暂停/继续、重置与模式切换 | 预览 pause 不作用于训练；训练暂停使用 `/training` |
| GET `/api/workbench/v1/checkpoints` | 受限候选及旁路 UUID、历史量、捕获量与累计量 | 元数据有效不等于权重可恢复 |
| GET `/api/workbench/v1/checkpoint-options` | 已验证的最终模型、手动快照与可用导入模型 | 选择精确路径；开始续训时再次验证，跨进程优化器重建 |
| GET `/api/workbench/v1/parameters` | 并行/PPO/评估的 11 项字段 schema、有效值与锁定状态 | 只提交编辑项，完整组合校验，训练中锁定 |
| GET/POST `/api/workbench/v1/viewports` | 至多四窗的创建、开启、暂停、重置、关闭与释放 | 独立 CPU 仿真不代表全部并行训练环境 |
| GET/POST `/api/workbench/v1/storage`，POST `/api/workbench/v1/import` | 后续保存根登记与完整模型导入 | 活动训练保存位置不可变；导入暂存验证后原子发布 |
| GET/POST `/api/local/directories` | Windows 本机原生目录选择、打开与取消 | 桥接核对目标发行版及后端授权；不执行拼接命令 |
| GET `/api/workbench/v1/capabilities` | 集成服务的 legacyControls 按实际运行和桥接状态声明能力 | 未连接或旧运行缺控制协议时不授权相应写入 |
| GET `/api/workbench/v1/catalog`、`descriptor` | 模式目录、保存的机器人/任务/动作/观测/运行描述 | 未记录字段为 null，不从维数推断跨机器人兼容 |

训练暂停保留同一个进程、网络与优化器；暂停时间不计入训练预算。手动保存使用安全回调中的真实参数，生成独立完整快照，不停止训练。HTTP 接受表示排队，完成以请求 ID 对应的状态回执为准。精确状态及字段见[控制、保存与加载](workbench-controls.md)。

[training_control.py](../experiments/torque200/training_control.py) 负责进程、命令和模型验证；其构造及查询保留既有目录创建、加锁和活动记录修复行为。独立产物读取模块不实例化该管理器。高频读取复用可失效的验证缓存，启动、精确选择、导入与模型预览仍做严格验证。

[console_train.py](../experiments/torque200/console_train.py) 的 progress.json 是显示窗口，metrics.jsonl 保留每次回调及重复步数。步数默认为本轮环境步；累计量由绑定的运行起点和 checkpoint 旁路记录提供。原版预算含初始化/JIT/评估；hybrid 先排除准备；最终保存另计。控制协议另扣除真实暂停区间。

## 需求对照

| 需求 | 已实现 | 保留边界或后续工作 |
| --- | --- | --- |
| BE-01 状态与新鲜度 | 监督器与产物 DTO 分离，mode/run 关联，独立心跳/辅助通道，重连状态及读取缓存 | 生产者时间戳与更完整事件合同仍需补齐 |
| BE-02 训练命令 | 真暂停/原位继续、独立快照、停止保存、请求 ID 与回执 | 旧运行不热升级；跨进程续训不恢复优化器 |
| BE-03 checkpoint 与累计 | 固定父基数、实际捕获量、原子旁路、精确历史选择与导入 | 旧历史未知保持 null；完整历史迁移、短别名命名域及更细权重绑定后续处理 |
| BE-04 准备流程 | 阶段显示、真实/不定进度、准备缓存与首帧反馈 | 更完整阶段凭据、事件与失效说明仍是后续合同 |
| BE-05 参数集中调节 | schema、草稿、字段及组合校验，实际 Brax 与记录一致 | 网络结构、动作和归一化等身份契约不作普通参数编辑 |
| BE-06 固定版本/多窗预览 | 四个独立 CPU 实例、每次开启同步、代次取消与资源释放 | 帧/遥测生产时间与更细关联仍需完善；帧率受实际资源约束 |
| BE-07 曲线原值与历史 | 原值分页、重复/缺口/单位、版本冲突及独立采样 | 大历史索引、显式累计轴与跨运行比较仍待后续 |
| BE-08 目录与导入 | 本机桥接、登记保存根、受控选择与完整导入校验 | 不迁移活动运行；任意新机器人不能绕过兼容合同导入 |
| BE-09 历史比较 | 保留各运行来源，不跨 run 拼接 | 条件差异、可比性与备注仍待后续 |
| BE-10 独立评估 | 保留训练内原生评估和导出数值核验 | 独立评估方案/预算、分布与行为指标仍待后续 |
| BE-11 原始表/导出 | 返回原始点，不把降采样结果当原始记录 | 范围导出、下载引用与完整性仍待后续 |
| BE-12 配置预设 | 保留自动默认及已验证的配置继承 | 可编辑预设库及迁移仍待后续 |
| BE-13 资源信息 | 真实运行计时、步数及预览帧率 | 完整设备来源、资源采样与吞吐语义仍待后续 |

布局、折叠、拖拽、字体与渐进披露由 UI 负责。新增机器人仍需完整 robot/task/action/policy/runtime 语义合同及实际第二实现，不能只替换模型文件或对齐数组长度。

## 只读模块与历史首批合同

独立读取模块见 [workbench_api/README.md](../experiments/torque200/workbench_api/README.md)。它可单独绑定回环端口测试；集成服务另接训练控制、参数、视窗和存储能力。独立只读 router 的能力声明仅描述其自身。

以下为最早只读批次的入口，当前仍保留：

| GET/HEAD 路径 | 查询 |
| --- | --- |
| `/api/workbench/v1/state` | mode |
| `/api/workbench/v1/checkpoints` | mode、offset、limit、revision |
| `/api/workbench/v1/metrics` | mode、runId、offset、limit、revision |
| `/api/workbench/v1/capabilities` | 独立读取模块无需控制状态 |

数据返回 schemaVersion、revision、读取时间、来源及 warnings。读取时间不是评估产生时间，多文件读取不具备事务一致性；辅助资源失败与监督器连接状态分别处理。

历史首批读取验收为 32/32：26 个读取案例与 6 个随机回环端口 HTTP 案例，仅使用临时合成产物。累计扩展历史批次合并为 62/62，验证固定父基数、未知历史、计时分离与旁路身份。这些是早期模块验收记录；当前真实控制、GPU 及浏览器接线验收见[生产整合](workbench-production.md)。

累计例子 600 秒历史加 600 秒预算从 50% 开始，新增 60 秒为 55%。真实保存与 UI 已接通；缺乏可信历史时保持 null，不用目录名、读盘时间或本轮步数倒推。格式见[累计合同](checkpoint-progress.md)。
