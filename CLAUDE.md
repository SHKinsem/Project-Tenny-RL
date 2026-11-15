# Project Tenny — 项目规则（给接手的 Agent）

## 规则
- 本仓库是**公开**的（GitHub: SHKinsem/Project-Tenny-RL）：已跟踪文件里不得出现本机用户名、绝对路径、密钥。WSL 家目录用 `$HOME` / `Path.home()` / `~`，仓库根用脚本自身位置或环境变量 `TENNY_ROOT` 推导。
- `tenny/` 只用 Python 标准库（3.11+）；命令一律以 argv 列表执行，不经 shell；本地服务只监听 `127.0.0.1`，只执行 `tenny/tasks.toml` 登记的任务。
- `*.sh` 必须是 LF 换行（`.gitattributes` 固定）；改写脚本按字节操作，别让编辑器把它们变成 CRLF。
- 训练输出目录（`experiments/torque200/results/`、`tasks/results/`）不入库（已在 `.gitignore`）；不要 `git add -A`，用 `git add <文件>` 或 `git add -u`。
- `reference/` 是第三方项目副本，保留原 LICENSE，不改其内容。
- 合并到 `main`、推送、改写 Git 历史前必须先问用户。
- 对用户汇报用中文。

## 入口与验证
- 统一入口：`python tenny.py`（网页）/ `list` / `run <任务> name=value`。设计与计划：`docs/superpowers/specs/`、`docs/superpowers/plans/`。
- CPU、回环HTTP及shell验证先读[验证指南](docs/validation.md)，使用仓库测试入口与已验证的运行方式；测试数量和平台跳过以当次结果为准。
- 训练/仿真类任务**只能在作者本机跑**：需要 WSL/Linux + GPU，以及按 `tasks/REPRODUCTION.md` 搭好的 `~/projects/tenny-reference`、`~/projects/tenny-torque200`。云端/新克隆里没有这些环境，也没有 `results/` 历史产物——不要声称跑过训练，改动训练脚本时用 `bash -n` 和代码审查验证，并在汇报里写明“未实机验证”。

## 已知待办（可选）
- `tasks/restart-*-wsl.sh`：写死 PID 的一次性脚本，未登记到 tasks.toml，可删。
- `experiments/torque200/VALIDATION.md` 里指向 `results/` 的链接在公开仓库中失效（results 不入库）。
- 缺少钉住 `runner.start` 中 `mode=="selected"` 解析逻辑的单测。
- `experiments/torque200/web-preview-wsl.sh` 用 `--host 0.0.0.0`（原有行为；WSL mirrored 网络下可能暴露到局域网）。
