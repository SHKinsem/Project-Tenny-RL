# 统一入口与目录结构设计

## 目标
- 根目录只保留入口与文档，脚本按用途归类。
- 一个 Python 入口 `tenny.py`：网页控制台（点按钮运行任务）+ CLI，共用同一份任务清单。
- 能在 Windows、Linux 上使用同一入口；仅用 Python 标准库（3.9+）。

## 非目标
- 不重写训练代码；不让训练原生运行在 Windows/macOS（依赖 GPU+JAX，仍在 Linux/WSL 中执行）。
- 不替换现有训练控制台（`experiments/torque200/web_preview.py`），入口页只提供跳转。

## 目录
```
tenny.py                 唯一入口
tenny/                   入口实现
  tasks.toml             任务清单：id、name、group、needs、cmd.{windows,posix}
  runner.py              按系统选命令、起子进程、收集日志、支持停止
  server.py              127.0.0.1 HTTP 服务（http.server）
  ui/index.html          控制台页面
scripts/{train,preview,benchmark,verify}/   原根目录 12 个脚本归类
experiments/torque200/   实验包（legwheel/ tests/ assets/ 与其 *-wsl.sh、*.py）
docs/                    CONTROL-MODES.md 等
tasks/                   保持不动
```

## 行为
- `python tenny.py`：启动服务并打开浏览器；按 group 展示按钮，点击运行，页面轮询日志，带"停止"。
- `python tenny.py list` / `run <id>`：终端等价用法。
- 只执行 `tasks.toml` 登记的任务，不接受任意命令；服务仅监听 127.0.0.1。
- `needs: wsl` 的任务：Windows 下经 `wsl.exe` 调用，Linux 直接 `bash`；不满足时提示而不是失败。

## 迁移
- `git mv` 保留历史；修正脚本互相引用路径，去掉写死的 `C:\Users\xusen\...`，改为相对路径。
- 根目录旧脚本删除（不保留兼容壳）；README 更新为入口说明。

## 验证
1. `python tenny.py list` 列出全部任务。
2. 网页能启动，按钮触发任务，日志可见，可停止。
3. 跑一个短任务（环境回归测试）确认搬迁后路径正确。
