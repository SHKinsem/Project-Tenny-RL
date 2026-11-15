# 统一入口（tenny.py）— Progress

**Status:** 已完成（2026-11-14）。`feat/unified-entry` 已快进合并到 `main` 并推送（`46ed78c..bd034fd`）。SDD 账本已删除。

## 遗留（可选，未做）
- `tasks/restart-*-wsl.sh`：写死 PID 的一次性脚本，未登记到 tasks.toml，可删。
- `experiments/torque200/VALIDATION.md` 里指向 `results/` 的链接在公开仓库中失效（results 不入库）。
- 缺少钉住 `mode=="selected"` 解析逻辑的单测。
- `web-preview-wsl.sh` 用 `--host 0.0.0.0`（原有行为；WSL mirrored 网络下可能暴露到局域网）。
- 想要固定端口：管理员执行 `netsh int ipv4 add excludedportrange protocol=tcp startport=8800 numberofports=1`（未验证保留后能否绑定，需先试）。
