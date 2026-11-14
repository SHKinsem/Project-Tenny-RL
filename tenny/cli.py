"""命令行：list / run / web（默认 web）。"""
import argparse
import sys
import time
import webbrowser

from . import registry, runner, server


def _list(tasks):
    group = None
    for t in sorted(tasks.values(), key=lambda t: t["group"]):
        if t["group"] != group:
            group = t["group"]
            print(f"\n[{group}]")
        ps = " ".join(f"{p['name']}={p.get('default')}" for p in t.get("params", []))
        print(f"  {t['id']:<18} {t['name']}  {ps}")
    return 0


def _run(tasks, task_id, kv):
    try:
        given = dict(item.split("=", 1) for item in kv)
        job = runner.start(task_id, given, tasks)
    except (ValueError, registry.TaskError) as e:  # ValueError: 不是 name=value 形式
        print(f"错误: {e}", file=sys.stderr)
        return 2
    if job.url:
        print(f"地址: {job.url}")
    seen = 0
    try:
        while True:
            snap = job.snapshot(seen)
            for line in snap["lines"]:
                print(line)
            seen = snap["next"]
            if snap["status"] != "running":
                return 0 if snap["status"] == "ok" else (snap["returncode"] or 1)
            time.sleep(0.2)
    except KeyboardInterrupt:
        job.stop()
        return 130


def _web(tasks, port, open_browser):
    # 端口 0 兜底：Windows 的 Hyper-V/WSL 会动态保留整段端口（bind 报 WinError 10013）
    for p in [*range(port, port + 10), 0]:
        try:
            srv = server.make_server(tasks, port=p)
            break
        except OSError as e:
            err = e
    else:
        print(f"无法监听端口 {port}-{port + 9} 或任意端口: {err}", file=sys.stderr)
        return 1
    url = f"http://tenny.localhost:{srv.server_address[1]}/"
    print(f"Project Tenny 控制台: {url}  (Ctrl-C 退出)", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        runner.stop_all()
    return 0


def main(argv=None, tasks=None):
    ap = argparse.ArgumentParser(prog="tenny")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("list", help="列出任务")
    r = sub.add_parser("run", help="运行一个任务")
    r.add_argument("task")
    r.add_argument("params", nargs="*", help="name=value")
    w = sub.add_parser("web", help="启动网页控制台（默认）")
    w.add_argument("--port", type=int, default=8800)
    w.add_argument("--no-browser", action="store_true")
    a = ap.parse_args(argv)
    tasks = tasks if tasks is not None else registry.load()
    if a.cmd == "list":
        return _list(tasks)
    if a.cmd == "run":
        return _run(tasks, a.task, a.params)
    return _web(tasks, getattr(a, "port", 8800), not getattr(a, "no_browser", False))
