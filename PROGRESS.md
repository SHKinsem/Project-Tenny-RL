# 工作台交接

更新于 2026-11-15。UI、后端功能补齐及频繁断线修复此前已完成本机验收并独立发布。本轮用户要求先整合所有未合并改动、删除已合并本地分支，再审查过度防御与性能；本轮不推送、不部署、不启动或重启训练。

## 当前成果

- 黑白灰三栏工作台：可调整宽度、独立折叠、渐进披露曲线、固定策略预览和底部累计进度。
- 真实接口支持同进程暂停/继续、手动快照、精确历史 checkpoint 续训、参数编辑、最多四个独立预览、本机目录选择/打开及完整模型导入。
- 未知加载进度使用不定加载条。状态心跳与辅助读取分离，最多两条 API 连接；短暂失败禁写并自动退避重连，保留最后数据及健康画面，提示集中在导航。
- 高频读取按依赖文件签名缓存验证；训练、导入、选定模型等动作仍执行完整验证。保留 512 默认/历史配置继承、原生评估、准备阶段缓存和计时。

## 验证与边界

频繁断线修复的工作台回归 153 项（Windows 因 stat 能力跳过 2 项），Linux 缓存 12 项与生命周期 8 项、Node 43 项及隔离 HTTP 验收通过。正式页面连续 45 次读取无失败，中位 43ms、最大 119ms；首次完整验证仍可能数秒，有限采样不能保证未来所有连接都无故障。

功能补齐阶段完成一次隔离 GPU 验收：45 秒预算，实际训练 41.0702 秒、新增 40960 步；暂停期间计数冻结，暂停中保存后同进程继续，最后协作停止并保存。原用户训练按自己的预算自然完成，验收产物没有替换生产 latest。

101 个已部署基线文件中 96 个字节不变，5 个外围接点有意变更。模型源码指纹和 runtime defaults 哈希保持原值。详细范围、真实四窗及原生目录验收的限制见 [控制合同](docs/workbench-controls.md)与 [tasks/todo.md](tasks/todo.md) 的 review。

跨进程续训恢复网络与归一化，优化器重新初始化；同进程暂停保留优化器。旧模型缺乏可靠历史时累计量仍为 null。真实四窗约每窗 2 FPS。完整 robot/task/action/policy/runtime 语义合同和 Tenny 机构第二实现属于后续工作。

## 接手入口

- [生产整合与回退](docs/workbench-production.md)
- [控制、保存、目录、加载与重连](docs/workbench-controls.md)
- [后端现状及后续需求](docs/workbench-backend.md)
- [设计约定与可运行示例](docs/workbench-design.md)
- [累计元数据合同](docs/checkpoint-progress.md)

精确部署 manifest、进程、端口、备份、验收原始日志及机器路径仅保存在忽略的 `tasks/results/workbench-production/`、`workbench-completion/` 与 `workbench-reconnect/`，以及部署端的 `workbench-current.json`。执行任何后续部署前重新读取当前记录和运行状态，不能把历史 PID 当作现况。

原工作区的实验改动与工作台已部署基线已在本地逐项合并并完成CPU回归；合并后的仓库不能直接视为已部署版本。后续部署必须按差异审阅，不能整目录覆盖。不要终止承载正式桥接的会话，也不要为更新网页或补协议能力重启用户训练。

## 本轮本地整合与审查

- 工作台分支基线为 `fdbb77d`，原目录待合并实验已保存为 `f42a2c4`，逐文件原始字节备份位于忽略的 `tasks/results/workspace-integration-20261115/`。
- 已完成合并提交 `23ba11d`：保留配置化腿力实验和工作台控制/缓存/累计语义，补齐两者交汇的归一化快照恢复与测试fixture。CPU主测试327项（16跳过）、工作台155项（2跳过）、前端43项、原型45项均无失败；199个Python文件和24个shell脚本语法通过。没有运行本轮GPU训练、部署或推送。
- 已确认两分支均为main祖先后删除本地 `codex/workbench-integration` 与 `feat/unified-entry`；原工作树已解除Git登记并清空。326个忽略文件先归档并逐个SHA256核验；Windows仍占用最后空目录，用户已选择暂时保留。远端分支未变。
- 最终代码审查确认3项P2：教师缓存身份绑定无关训练参数、多窗显示预算拖慢仿真时钟、暂停仍持续重绘编码。随后用户授权的本地修复已完成：阶段依赖缓存、每视窗墙钟播放/有界补步和暂停JPEG复用均通过针对性与相关HTTP回归，测试命令累计约13秒；没有追加GPU训练、推送或部署。实际行为及验证局限以 `tasks/todo.md` 最新修复review、`experiments/torque200/FORCE-TEACHER.md`及`docs/workbench-controls.md`本地修复段为准。生产仍为此前部署版本。

## 统一入口历史交接（2026-11-14）

**Status:** 已完成（2026-11-14）。`feat/unified-entry` 已快进合并到 `main` 并推送（`46ed78c..bd034fd`）。SDD 账本已删除。

## 遗留（可选，未做）
- `tasks/restart-*-wsl.sh`：写死 PID 的一次性脚本，未登记到 tasks.toml，可删。
- `experiments/torque200/VALIDATION.md` 里指向 `results/` 的链接在公开仓库中失效（results 不入库）。
- 缺少钉住 `mode=="selected"` 解析逻辑的单测。
- `web-preview-wsl.sh` 用 `--host 0.0.0.0`（原有行为；WSL mirrored 网络下可能暴露到局域网）。
- 想要固定端口：管理员执行 `netsh int ipv4 add excludedportrange protocol=tcp startport=8800 numberofports=1`（未验证保留后能否绑定，需先试）。
