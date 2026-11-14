# 统一入口 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用一个 Python 入口 `tenny.py`（网页控制台 + CLI 共用同一份任务清单）取代根目录 10 个启动脚本。

**Architecture:** `tenny/tasks.toml` 登记任务；`registry.py` 校验参数并生成 argv；`runner.py` 按系统选命令（Windows 经 `wsl.exe`，Linux 直接 `bash`）、管理后台任务与日志；`server.py` 提供 127.0.0.1 HTTP API 和单页 UI；`cli.py` 提供 `list` / `run` / `web`。

**Tech Stack:** Python 3.11+ 标准库（`tomllib`、`http.server`、`subprocess`、`unittest`），原生 JS 单页，无第三方依赖。

**Spec:** `docs/superpowers/specs/2026-11-14-unified-entry-design.md`

## Global Constraints

- 仅用 Python 标准库（3.11+，因 `tomllib`）；不新增任何依赖。
- 服务仅监听 `127.0.0.1`；只执行 `tasks.toml` 登记的任务，不接受任意命令。
- 命令一律以 argv 列表执行，**不经 shell**。
- 根目录旧 `.ps1` / `.cmd` 删除，不保留兼容壳；不新建 `scripts/` 目录。
- `*-wsl.sh` 留在 `experiments/torque200/` 与 `tasks/` 原处。
- 现有训练控制台（`experiments/torque200/web_preview.py`）不改，入口页只显示它的链接。
- WSL 发行版默认 `Ubuntu`，可用环境变量 `TENNY_WSL_DISTRO` 覆盖。
- 仓库是**公开**的：新文件里不得出现用户名、绝对路径、密钥。

## Review Focus

- 未知任务 id → 404，不崩溃、不执行任何命令。
- 参数越界 / 类型错 / 未登记参数（如 `steps=abc`、`steps=0`、`evil=1`）→ 400，且不起进程。
- 同一任务连点两次 → 第二次 409，不起第二个进程。
- 跨站请求：非 `127.0.0.1`/`localhost` 的 Host（DNS rebinding）或非 `application/json` 的 POST → 拒绝。
- 任务输出极大（数万行）→ 内存有上限，页面仍能拉到最新行；停止后进程树被杀。

---

## File Structure

| 文件 | 职责 |
|---|---|
| `tenny.py` | 3 行入口，调用 `tenny.cli.main` |
| `tenny/__init__.py` | 空，使 `tenny` 成为包 |
| `tenny/tasks.toml` | 任务清单（唯一事实来源） |
| `tenny/registry.py` | 加载清单、参数校验、生成 argv；异常类型 |
| `tenny/runner.py` | 平台命令构造、`Job`（子进程+日志环形缓冲+停止）、任务注册表 |
| `tenny/server.py` | HTTP API + 静态 UI，Host/Content-Type 校验 |
| `tenny/cli.py` | `list` / `run` / `web` 子命令 |
| `tenny/ui/index.html` | 控制台页面 |
| `tests/test_tenny.py` | 标准库 `unittest`，覆盖 registry / runner / server |
| `.gitattributes` | 固定 `*.sh` 为 LF |

运行测试统一命令：`python -m unittest discover -s tests -v`（在仓库根目录）。

---

### Task 1: 任务清单加载与参数校验（registry）

**Files:**
- Create: `tenny/__init__.py`、`tenny/registry.py`、`tests/test_tenny.py`
- Create: `tenny/tasks.toml`（先放一个占位任务，Task 4 填满）

**Interfaces:**
- Produces:
  - `registry.ROOT: Path`（仓库根）、`registry.TOML: Path`
  - `class TaskError(ValueError)`；子类 `UnknownTask`、`Busy`
  - `load(path=TOML) -> dict[str, dict]`（id → 任务字典）
  - `resolve_params(task, given: dict|None) -> dict`（校验并补默认值）
  - `build_args(task, values: dict) -> list[str]`
- 任务字典字段：`id, name, group, needs("wsl"|缺省), script|cmd(二选一), args[], params[], url`
- 参数字典字段：`name, type(int|float|str|choice|flag), default, min, max, choices, flag, label`

- [ ] **Step 1: 新建分支**

```bash
git checkout -b feat/unified-entry
```

- [ ] **Step 2: 写失败的测试**

`tests/test_tenny.py`：

```python
import sys, time, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tenny import registry  # noqa: E402

PY = sys.executable
TASKS = {
    "echo": {"id": "echo", "name": "echo", "group": "t",
             "cmd": [PY, "-u", "-c", "import sys;print('hi',sys.argv[1])"],
             "args": ["{word}"],
             "params": [{"name": "word", "type": "str", "default": "x"}]},
    "sleep": {"id": "sleep", "name": "sleep", "group": "t",
              "cmd": [PY, "-u", "-c", "import time;print('go',flush=True);time.sleep(60)"]},
    "noisy": {"id": "noisy", "name": "noisy", "group": "t",
              "cmd": [PY, "-u", "-c", "[print(i) for i in range(6000)]"]},
    "num": {"id": "num", "name": "num", "group": "t",
            "cmd": [PY, "-c", "pass"], "args": ["{n}"],
            "params": [{"name": "n", "type": "int", "min": 1, "max": 5, "default": 2},
                       {"name": "mode", "type": "choice", "choices": ["a", "b"], "default": "a"},
                       {"name": "fast", "type": "flag", "flag": "--fast", "default": False}]},
}


class RegistryTest(unittest.TestCase):
    def test_defaults_and_coercion(self):
        v = registry.resolve_params(TASKS["num"], {"n": "3", "fast": True})
        self.assertEqual(v, {"n": 3, "mode": "a", "fast": True})

    def test_bad_values_rejected(self):
        for bad in ({"n": "abc"}, {"n": 0}, {"n": 6}, {"mode": "z"}, {"evil": "1"}):
            with self.assertRaises(registry.TaskError, msg=bad):
                registry.resolve_params(TASKS["num"], bad)

    def test_build_args(self):
        v = registry.resolve_params(TASKS["num"], {"n": 4, "fast": True})
        self.assertEqual(registry.build_args(TASKS["num"], v), ["4", "--fast"])
        v = registry.resolve_params(TASKS["num"], {})
        self.assertEqual(registry.build_args(TASKS["num"], v), ["2"])

    def test_shell_metacharacters_stay_one_arg(self):
        v = registry.resolve_params(TASKS["echo"], {"word": "a; rm -rf /"})
        self.assertEqual(registry.build_args(TASKS["echo"], v), ["a; rm -rf /"])

    def test_real_toml_is_valid(self):
        tasks = registry.load()
        for t in tasks.values():
            self.assertTrue(t["name"] and t["group"], t["id"])
            if "script" in t:
                self.assertTrue((registry.ROOT / t["script"]).is_file(), t["script"])
            registry.resolve_params(t, {})  # 默认值必须自身合法


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: 运行，确认失败**

Run: `python -m unittest discover -s tests -v`
Expected: `ModuleNotFoundError: No module named 'tenny'`（或 ImportError）

- [ ] **Step 4: 实现**

`tenny/__init__.py`：空文件。

`tenny/tasks.toml`（占位，Task 4 替换；脚本路径指向仓库里已存在的文件）：

```toml
[[task]]
id = "throughput"
name = "吞吐探测"
group = "测试"
needs = "wsl"
script = "experiments/torque200/throughput-wsl.sh"
```

`tenny/registry.py`：

```python
"""任务清单：加载 tasks.toml、校验参数、生成 argv。"""
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOML = Path(__file__).with_name("tasks.toml")


class TaskError(ValueError):
    pass


class UnknownTask(TaskError):
    pass


class Busy(TaskError):
    pass


def load(path=TOML):
    with open(path, "rb") as f:
        tasks = tomllib.load(f)["task"]
    out = {}
    for t in tasks:
        if t["id"] in out:
            raise TaskError(f"重复的任务 id: {t['id']}")
        if ("cmd" in t) == ("script" in t):
            raise TaskError(f"{t['id']}: cmd 与 script 必须二选一")
        out[t["id"]] = t
    return out


def _coerce(p, raw):
    name, kind = p["name"], p["type"]
    if kind in ("int", "float"):
        try:
            v = int(raw) if kind == "int" else float(raw)
        except (TypeError, ValueError):
            raise TaskError(f"{name}: 需要{kind}")
        if v < p.get("min", v) or v > p.get("max", v):
            raise TaskError(f"{name}: 超出范围 [{p.get('min')}, {p.get('max')}]")
        return v
    if kind == "choice":
        if str(raw) not in p["choices"]:
            raise TaskError(f"{name}: 必须是 {p['choices']} 之一")
        return str(raw)
    if kind == "flag":
        return raw in (True, "true", "1", 1, "on")
    if kind == "str":
        return str(raw)
    raise TaskError(f"{name}: 未知参数类型 {kind}")


def resolve_params(task, given):
    given = given or {}
    params = task.get("params", [])
    unknown = set(given) - {p["name"] for p in params}
    if unknown:
        raise TaskError(f"未登记的参数: {sorted(unknown)}")
    return {p["name"]: _coerce(p, given.get(p["name"], p.get("default"))) for p in params}


def build_args(task, values):
    out = [a.format(**values) for a in task.get("args", [])]
    out += [p["flag"] for p in task.get("params", []) if p["type"] == "flag" and values[p["name"]]]
    return out
```

- [ ] **Step 5: 运行，确认通过**

Run: `python -m unittest discover -s tests -v`
Expected: 5 个测试全部 OK（`test_real_toml_is_valid` 需 `throughput-wsl.sh` 存在，已存在）

- [ ] **Step 6: Commit**

```bash
git add tenny tests
git commit -m "feat(tenny): task registry with param validation"
```

---

### Task 2: 运行器（平台命令、后台任务、停止）

**Files:**
- Create: `tenny/runner.py`
- Modify: `tests/test_tenny.py`（追加 `RunnerTest`）

**Interfaces:**
- Consumes: `registry.TaskError/UnknownTask/Busy/ROOT/resolve_params/build_args`
- Produces:
  - `runner.start(task_id, given, tasks) -> Job`
  - `runner.get(job_id) -> Job | None`
  - `runner.stop_all() -> None`（停止所有运行中任务并清空，供退出和测试清理）
  - `Job.id, Job.task_id, Job.url, Job.status`（`"running"|"ok"|"failed"|"stopped"`）、`Job.returncode`
  - `Job.snapshot(since: int) -> dict`：`{"id","task","status","returncode","url","lines":[str],"next":int}`；`since/next` 是**绝对**行号
  - `Job.stop() -> None`
  - 常量 `MAX_LINES = 5000`

- [ ] **Step 1: 写失败的测试**（追加到 `tests/test_tenny.py`，放在 `if __name__` 之前；顶部补 `from tenny import runner`）

```python
from tenny import runner  # noqa: E402


def wait(job, timeout=15):
    end = time.time() + timeout
    while job.status == "running" and time.time() < end:
        time.sleep(0.05)


class RunnerTest(unittest.TestCase):
    def tearDown(self):
        runner.stop_all()

    def test_runs_and_captures_output(self):
        job = runner.start("echo", {"word": "tenny"}, TASKS)
        wait(job)
        self.assertEqual(job.status, "ok")
        self.assertEqual(job.snapshot(0)["lines"], ["hi tenny"])

    def test_unknown_task(self):
        with self.assertRaises(registry.UnknownTask):
            runner.start("nope", {}, TASKS)

    def test_bad_param_starts_nothing(self):
        with self.assertRaises(registry.TaskError):
            runner.start("num", {"n": 99}, TASKS)
        self.assertEqual(runner._jobs, {})

    def test_duplicate_run_rejected(self):
        job = runner.start("sleep", {}, TASKS)
        with self.assertRaises(registry.Busy):
            runner.start("sleep", {}, TASKS)
        self.assertEqual(len(runner._jobs), 1)
        job.stop()

    def test_stop_kills_process(self):
        job = runner.start("sleep", {}, TASKS)
        end = time.time() + 10
        while "go" not in job.snapshot(0)["lines"] and time.time() < end:
            time.sleep(0.05)
        job.stop()
        wait(job)
        self.assertEqual(job.status, "stopped")
        self.assertIsNotNone(job.returncode)

    def test_log_is_capped_and_cursor_is_absolute(self):
        job = runner.start("noisy", {}, TASKS)
        wait(job)
        snap = job.snapshot(0)
        self.assertEqual(snap["next"], 6000)
        self.assertEqual(len(snap["lines"]), runner.MAX_LINES)
        self.assertEqual(snap["lines"][-1], "5999")
        self.assertEqual(job.snapshot(5990)["lines"][0], "5990")
        self.assertEqual(job.snapshot(6000)["lines"], [])

    def test_needs_wsl_missing(self):
        task = {"id": "w", "name": "w", "group": "t", "needs": "wsl", "script": "x.sh"}
        orig = runner.wsl_available
        runner.wsl_available = lambda: False
        try:
            with self.assertRaises(registry.TaskError):
                runner.start("w", {}, {"w": task})
        finally:
            runner.wsl_available = orig

    def test_command_for_script(self):
        task = {"id": "s", "name": "s", "group": "t", "needs": "wsl",
                "script": "tasks/start-sim-wsl.sh", "args": ["--x", "{n}"],
                "params": [{"name": "n", "type": "int", "default": 1}]}
        orig = runner.to_linux_path
        runner.to_linux_path = lambda p: "/linux/" + p.name
        try:
            cmd = runner.build_command(task, {"n": 1})
        finally:
            runner.to_linux_path = orig
        tail = ["/bin/bash", "/linux/start-sim-wsl.sh", "--x", "1"]
        self.assertEqual(cmd[-4:], tail)
        if runner.IS_WIN:
            self.assertEqual(cmd[:4], ["wsl.exe", "-d", runner.DISTRO, "--exec"])
```

- [ ] **Step 2: 运行，确认失败**

Run: `python -m unittest tests.test_tenny.RunnerTest -v`
Expected: `ImportError: cannot import name 'runner'`

- [ ] **Step 3: 实现 `tenny/runner.py`**

```python
"""运行器：按系统构造命令，管理后台任务及其日志。"""
import os
import shutil
import signal
import subprocess
import sys
import threading
import uuid
from pathlib import Path

from . import registry

IS_WIN = os.name == "nt"
DISTRO = os.environ.get("TENNY_WSL_DISTRO", "Ubuntu")
MAX_LINES = 5000
# WSL 侧保存"选中模式"的位置；与 experiments/torque200 的训练控制台约定一致
SELECTED_MODE_FILE = "/home/jaccob/projects/tenny-torque200/artifacts/selected-mode.json"
_SELECTED_CODE = ('import json,sys;from pathlib import Path;p=Path(sys.argv[1]);'
                  'print(json.loads(p.read_text())["mode"] if p.exists() else "original")')

_jobs = {}
_lock = threading.Lock()


def wsl_available():
    return IS_WIN and shutil.which("wsl.exe") is not None


def _wsl(*args):
    return subprocess.run(["wsl.exe", "-d", DISTRO, "--exec", *args],
                          capture_output=True, text=True, encoding="utf-8", errors="replace")


def check_available(task):
    if task.get("needs") != "wsl":
        return
    if sys.platform == "darwin":
        raise registry.TaskError("该任务依赖 Linux/WSL，macOS 不支持")
    if IS_WIN and not wsl_available():
        raise registry.TaskError("该任务需要 WSL，但没找到 wsl.exe")


def to_linux_path(path: Path) -> str:
    if not IS_WIN:
        return str(path)
    r = _wsl("wslpath", "-a", str(path))
    if r.returncode:
        raise registry.TaskError(f"wslpath 失败: {r.stdout}{r.stderr}".strip())
    return r.stdout.strip()


def read_selected_mode():
    if not IS_WIN:
        return "original"  # ponytail: Linux 下无 WSL 约定路径，回退原版；需要时再读 SELECTED_MODE_FILE
    r = _wsl("/usr/bin/python3", "-c", _SELECTED_CODE, SELECTED_MODE_FILE)
    mode = r.stdout.strip()
    if r.returncode or mode not in ("original", "torque200"):
        raise registry.TaskError("读不到选中的模式，未启动任务")
    return mode


def wsl_ip():
    if not IS_WIN:
        return "127.0.0.1"
    out = _wsl("hostname", "-I").stdout.split()
    return out[0] if out else "127.0.0.1"


def build_command(task, values):
    args = registry.build_args(task, values)
    if "cmd" in task:
        return [*task["cmd"], *args]
    script = to_linux_path(registry.ROOT / task["script"])
    if IS_WIN:
        return ["wsl.exe", "-d", DISTRO, "--exec", "/bin/bash", script, *args]
    return ["/bin/bash", script, *args]


class Job:
    def __init__(self, task, argv, url=None):
        self.id = uuid.uuid4().hex[:8]
        self.task_id = task["id"]
        self.url = url
        self.returncode = None
        self.stopped = False
        self._lines, self._dropped = [], 0
        self._lock = threading.Lock()
        kw = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if IS_WIN
              else {"start_new_session": True})
        self.proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     cwd=registry.ROOT, **kw)
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        for raw in self.proc.stdout:
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            with self._lock:
                self._lines.append(line)
                if len(self._lines) > MAX_LINES:
                    del self._lines[0]
                    self._dropped += 1
        self.returncode = self.proc.wait()

    @property
    def status(self):
        if self.returncode is None:
            return "running"
        if self.stopped:
            return "stopped"
        return "ok" if self.returncode == 0 else "failed"

    def snapshot(self, since=0):
        with self._lock:
            start = max(since - self._dropped, 0)
            return {"id": self.id, "task": self.task_id, "status": self.status,
                    "returncode": self.returncode, "url": self.url,
                    "lines": self._lines[start:], "next": self._dropped + len(self._lines)}

    def stop(self):
        if self.returncode is not None:
            return
        self.stopped = True
        if IS_WIN:
            subprocess.run(["taskkill", "/PID", str(self.proc.pid), "/T", "/F"], capture_output=True)
        else:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


def start(task_id, given, tasks):
    task = tasks.get(task_id)
    if task is None:
        raise registry.UnknownTask(f"没有这个任务: {task_id}")
    check_available(task)
    values = registry.resolve_params(task, given)
    if values.get("mode") == "selected":
        values["mode"] = read_selected_mode()
    with _lock:
        if any(j.task_id == task_id and j.status == "running" for j in _jobs.values()):
            raise registry.Busy(f"{task_id} 已在运行")
        url = task["url"].format(wsl_ip=wsl_ip(), **values) if "url" in task else None
        job = Job(task, build_command(task, values), url)
        _jobs[job.id] = job
    return job


def get(job_id):
    return _jobs.get(job_id)


def stop_all():
    for j in list(_jobs.values()):
        j.stop()
    for j in list(_jobs.values()):
        try:
            j.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    _jobs.clear()
```

注意：`url` 里 `{wsl_ip}` 以外的占位符取自 `values`，所以 `format(wsl_ip=..., **values)` 对没有该占位符的 url 也安全；`wsl_ip()` 仅在有 `url` 的任务里才被调用。

- [ ] **Step 4: 运行，确认通过**

Run: `python -m unittest discover -s tests -v`
Expected: Registry 5 个 + Runner 8 个全部 OK。若 `test_stop_kills_process` 在 Windows 失败，先查 `taskkill` 是否可用，不要放宽断言。

- [ ] **Step 5: Commit**

```bash
git add tenny/runner.py tests/test_tenny.py
git commit -m "feat(tenny): cross-platform runner with job log buffer and stop"
```

---

### Task 3: HTTP 服务、CLI、网页界面

**Files:**
- Create: `tenny.py`、`tenny/server.py`、`tenny/cli.py`、`tenny/ui/index.html`
- Modify: `tests/test_tenny.py`（追加 `ServerTest`、`CliTest`）

**Interfaces:**
- Consumes: `registry.load/UnknownTask/Busy/TaskError`、`runner.start/get/stop_all`、`Job.snapshot/stop`
- Produces:
  - `server.make_server(tasks, host="127.0.0.1", port=8800) -> ThreadingHTTPServer`（`port=0` 取空闲端口）
  - HTTP：`GET /`、`GET /api/tasks`、`GET /api/jobs/<id>?since=N`、`POST /api/run/<id>`（body=参数 JSON）、`POST /api/stop/<id>`
  - 错误统一 `{"error": str}`；404 未知任务/作业，409 重复运行，400 参数错/环境不满足/Host 或 Content-Type 非法
  - `cli.main(argv=None) -> int`

- [ ] **Step 1: 写失败的测试**（追加；顶部补 `import http.client, json, threading, io, contextlib` 以及 `from tenny import server, cli`）

```python
import contextlib, http.client, io, json, threading  # noqa: E402
from tenny import cli, server  # noqa: E402


class ServerTest(unittest.TestCase):
    def setUp(self):
        self.srv = server.make_server(TASKS, port=0)
        self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def tearDown(self):
        runner.stop_all()
        self.srv.shutdown()
        self.srv.server_close()

    def call(self, method, path, body=None, host=None, ctype="application/json"):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        headers = {"Content-Type": ctype}
        if host:
            headers["Host"] = host
        c.request(method, path, json.dumps(body) if body is not None else None, headers)
        r = c.getresponse()
        data = r.read()
        c.close()
        try:
            return r.status, json.loads(data)
        except ValueError:
            return r.status, data

    def test_index_and_task_list(self):
        status, html = self.call("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn(b"Project Tenny", html)
        status, tasks = self.call("GET", "/api/tasks")
        self.assertEqual({t["id"] for t in tasks}, set(TASKS))
        self.assertNotIn("cmd", tasks[0])
        self.assertNotIn("script", tasks[0])

    def test_run_poll_finish(self):
        status, j = self.call("POST", "/api/run/echo", {"word": "web"})
        self.assertEqual(status, 200)
        job = runner.get(j["id"])
        wait(job)
        status, snap = self.call("GET", f"/api/jobs/{j['id']}?since=0")
        self.assertEqual((snap["status"], snap["lines"]), ("ok", ["hi web"]))

    def test_error_codes(self):
        self.assertEqual(self.call("POST", "/api/run/nope", {})[0], 404)
        self.assertEqual(self.call("GET", "/api/jobs/zzzz")[0], 404)
        self.assertEqual(self.call("POST", "/api/run/num", {"n": 99})[0], 400)
        self.assertEqual(self.call("POST", "/api/run/num", {"evil": 1})[0], 400)
        self.assertEqual(self.call("POST", "/api/run/sleep", {})[0], 200)
        self.assertEqual(self.call("POST", "/api/run/sleep", {})[0], 409)

    def test_rejects_foreign_host_and_form_posts(self):
        self.assertEqual(self.call("GET", "/api/tasks", host="evil.example")[0], 400)
        self.assertEqual(self.call("POST", "/api/run/echo", {}, ctype="text/plain")[0], 400)
        self.assertEqual(runner._jobs, {})

    def test_stop_endpoint(self):
        _, j = self.call("POST", "/api/run/sleep", {})
        self.assertEqual(self.call("POST", f"/api/stop/{j['id']}", {})[0], 200)
        wait(runner.get(j["id"]))
        self.assertEqual(runner.get(j["id"]).status, "stopped")


class CliTest(unittest.TestCase):
    def tearDown(self):
        runner.stop_all()

    def test_list_and_run(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(cli.main(["list"], TASKS), 0)
            self.assertEqual(cli.main(["run", "echo", "word=cli"], TASKS), 0)
        self.assertIn("echo", out.getvalue())
        self.assertIn("hi cli", out.getvalue())

    def test_run_failure_and_unknown(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertNotEqual(cli.main(["run", "nope"], TASKS), 0)
            self.assertNotEqual(cli.main(["run", "num", "n=99"], TASKS), 0)
```

- [ ] **Step 2: 运行，确认失败**

Run: `python -m unittest tests.test_tenny.ServerTest tests.test_tenny.CliTest -v`
Expected: `ImportError: cannot import name 'cli'`

- [ ] **Step 3: 实现服务端 `tenny/server.py`**

```python
"""本地 HTTP 服务：任务 API + 单页 UI。只监听本机，只跑清单里的任务。"""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qsl

from . import registry, runner

UI = Path(__file__).with_name("ui") / "index.html"
LOCAL_HOSTS = {"127.0.0.1", "localhost"}


def public(task):
    keys = ("id", "name", "group", "needs", "params", "url")
    return {k: task[k] for k in keys if k in task}


def make_server(tasks, host="127.0.0.1", port=8800):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", f"{ctype}; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _guard(self):
            """DNS rebinding 防护：Host 必须是本机。"""
            if self.headers.get("Host", "").rsplit(":", 1)[0] not in LOCAL_HOSTS:
                self._send(400, {"error": "bad host"})
                return False
            return True

        def do_GET(self):
            if not self._guard():
                return
            path, _, qs = self.path.partition("?")
            if path == "/":
                return self._send(200, UI.read_bytes(), "text/html")
            if path == "/api/tasks":
                return self._send(200, [public(t) for t in tasks.values()])
            if path.startswith("/api/jobs/"):
                job = runner.get(path.rsplit("/", 1)[1])
                if job is None:
                    return self._send(404, {"error": "no such job"})
                try:
                    since = int(dict(parse_qsl(qs)).get("since", 0))
                except ValueError:
                    since = 0
                return self._send(200, job.snapshot(since))
            self._send(404, {"error": "not found"})

        def do_POST(self):
            if not self._guard():
                return
            # 非 JSON 的 POST（如跨站表单）直接拒绝；fetch 发 JSON 会触发 CORS 预检，我们不应答
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                return self._send(400, {"error": "content-type must be application/json"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            except ValueError:
                return self._send(400, {"error": "bad json"})
            kind, _, ident = self.path.removeprefix("/api/").partition("/")
            try:
                if kind == "run":
                    return self._send(200, runner.start(ident, body, tasks).snapshot(0))
                if kind == "stop":
                    job = runner.get(ident)
                    if job is None:
                        return self._send(404, {"error": "no such job"})
                    job.stop()
                    return self._send(200, job.snapshot(0))
            except registry.UnknownTask as e:
                return self._send(404, {"error": str(e)})
            except registry.Busy as e:
                return self._send(409, {"error": str(e)})
            except registry.TaskError as e:
                return self._send(400, {"error": str(e)})
            self._send(404, {"error": "not found"})

    return ThreadingHTTPServer((host, port), Handler)
```

- [ ] **Step 4: 实现 `tenny/cli.py` 与 `tenny.py`**

`tenny/cli.py`：

```python
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
    for p in range(port, port + 10):
        try:
            srv = server.make_server(tasks, port=p)
            break
        except OSError:
            continue
    else:
        print(f"端口 {port}-{port + 9} 都被占用", file=sys.stderr)
        return 1
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    print(f"Project Tenny 控制台: {url}  (Ctrl-C 退出)")
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
```

`tenny.py`：

```python
import sys

from tenny.cli import main

sys.exit(main())
```

- [ ] **Step 5: 实现 `tenny/ui/index.html`**

```html
<!doctype html>
<html lang="zh">
<meta charset="utf-8">
<title>Project Tenny</title>
<style>
:root{--bg:#0f1f17;--fg:#e8e4d0;--gold:#d4af37;--card:#16301f}
body{font:14px/1.5 system-ui,sans-serif;background:var(--bg);color:var(--fg);margin:0;padding:24px}
h1{color:var(--gold);font-weight:500;margin:0 0 4px}
h2{color:var(--gold);font-size:13px;letter-spacing:.1em;margin:28px 0 8px}
.task{background:var(--card);padding:10px 14px;margin:6px 0;border-radius:6px;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
.task b{min-width:11em}
input,select{background:#0b160f;color:var(--fg);border:1px solid #345;border-radius:3px;padding:2px 6px}
input:not([type=checkbox]){width:7em}
button{background:var(--gold);border:0;border-radius:4px;padding:4px 14px;cursor:pointer}
button.stop{background:#a33;color:#fff}
pre{background:#08110b;padding:12px;border-radius:6px;height:42vh;overflow:auto;white-space:pre-wrap;margin:0}
a{color:var(--gold)}
</style>
<h1>Project Tenny</h1>
<div id="tasks"></div>
<h2>日志 <span id="cur"></span></h2>
<pre id="log">选择上方任务并点击“运行”。</pre>
<script>
const $ = s => document.querySelector(s);
let job = null, since = 0, timer = null;

async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} :
    {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || r.status);
  return j;
}

function field(p) {
  if (p.type === 'choice')
    return `<select name="${p.name}">${p.choices.map(c => `<option ${c == p.default ? 'selected' : ''}>${c}</option>`).join('')}</select>`;
  if (p.type === 'flag') return `<input type="checkbox" name="${p.name}" ${p.default ? 'checked' : ''}>`;
  return `<input name="${p.name}" value="${p.default ?? ''}">`;
}

async function init() {
  const tasks = await api('/api/tasks'), groups = {};
  tasks.forEach(t => (groups[t.group] ??= []).push(t));
  $('#tasks').innerHTML = Object.entries(groups).map(([g, ts]) => `<h2>${g}</h2>` + ts.map(t => `
    <div class="task" data-id="${t.id}"><b>${t.name}</b>
      ${(t.params || []).map(p => `<label>${p.label ?? p.name} ${field(p)}</label>`).join('')}
      <button onclick="run('${t.id}')">运行</button></div>`).join('')).join('');
}

function collect(id) {
  const o = {};
  document.querySelectorAll(`[data-id="${id}"] [name]`).forEach(e =>
    o[e.name] = e.type === 'checkbox' ? e.checked : e.value);
  return o;
}

async function run(id) {
  try { show(await api('/api/run/' + id, collect(id))); } catch (e) { alert(e.message); }
}

async function stopJob() { if (job) await api('/api/stop/' + job.id, {}); }

function show(j) {
  job = j; since = 0; $('#log').textContent = '';
  clearInterval(timer); timer = setInterval(poll, 700); poll();
}

async function poll() {
  const j = await api(`/api/jobs/${job.id}?since=${since}`);
  const pre = $('#log'), atEnd = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 20;
  pre.textContent += j.lines.map(l => l + '\n').join('');
  since = j.next;
  if (atEnd) pre.scrollTop = pre.scrollHeight;
  $('#cur').innerHTML = `${j.task} · ${j.status}`
    + (j.url ? ` · <a target="_blank" href="${j.url}">${j.url}</a>` : '')
    + (j.status === 'running' ? ' <button class="stop" onclick="stopJob()">停止</button>' : '');
  if (j.status !== 'running') clearInterval(timer);
}

init();
</script>
</html>
```

- [ ] **Step 6: 运行，确认通过**

Run: `python -m unittest discover -s tests -v`
Expected: Registry 5 + Runner 8 + Server 5 + Cli 2 全部 OK。

- [ ] **Step 7: 手动冒烟（用占位任务）**

Run: `python tenny.py list` → 应看到 `throughput` 一行。
Run: `python tenny.py web --no-browser`，另开终端 `curl http://127.0.0.1:8800/api/tasks` → JSON；Ctrl-C 退出。

- [ ] **Step 8: Commit**

```bash
git add tenny.py tenny tests
git commit -m "feat(tenny): http server, cli and web console"
```

---

### Task 4: 填入真实任务清单，修复启动器里的写死路径

**Files:**
- Modify: `tenny/tasks.toml`
- Modify（一次性脚本改写，脚本本身不提交）：`experiments/torque200/*-wsl.sh`（9 个）、`tasks/*-wsl.sh`（7 个：`benchmark-device`、`bootstrap`、`collect-gpu-results`、`collect-results`、`run-gpu`、`run-training`、`verify-gpu-sim`）
- Create: `.gitattributes`

**Interfaces:**
- Consumes: Task 1 的任务字典 schema
- Produces: 8 个任务 id：`web`、`train-timed`、`preview`、`sim`、`test-torque200`、`test-original`、`test-performance`、`test-throughput`

- [ ] **Step 1: 写入真实 `tenny/tasks.toml`**（整体替换占位内容）

```toml
# 任务清单：tenny.py 的唯一事实来源。
# script 为 bash 启动器（Windows 经 wsl.exe 调用）；args 中 {name} 取自 params。

[[task]]
id = "web"
name = "训练控制台"
group = "训练"
needs = "wsl"
script = "experiments/torque200/web-preview-wsl.sh"
args = ["--port", "{port}", "--mode", "{mode}"]
url = "http://{wsl_ip}:{port}/"
params = [
  { name = "port", type = "int", default = 8770, min = 1024, max = 65535 },
  { name = "mode", type = "choice", choices = ["selected", "original", "torque200"], default = "selected", label = "模式" },
]

[[task]]
id = "train-timed"
name = "定时训练（全扭矩）"
group = "训练"
needs = "wsl"
script = "experiments/torque200/timed-train-wsl.sh"
args = ["--duration", "{duration}"]
params = [
  { name = "duration", type = "int", default = 900, min = 1, max = 900, label = "秒" },
  { name = "verify_only", type = "flag", flag = "--verify-only", default = false, label = "仅验证" },
  { name = "self_test", type = "flag", flag = "--self-test", default = false, label = "自检" },
]

[[task]]
id = "preview"
name = "策略预览（全扭矩）"
group = "预览与仿真"
needs = "wsl"
script = "experiments/torque200/preview-wsl.sh"
args = ["--seconds", "{seconds}"]
params = [{ name = "seconds", type = "float", default = 0, min = 0, max = 86400, label = "秒(0=不限)" }]

[[task]]
id = "sim"
name = "原版仿真"
group = "预览与仿真"
needs = "wsl"
script = "tasks/start-sim-wsl.sh"

[[task]]
id = "test-torque200"
name = "全扭矩回归测试"
group = "测试"
needs = "wsl"
script = "experiments/torque200/run-wsl.sh"
args = ["--batch-size", "{batch_size}", "--steps", "{steps}", "--eval-envs", "{eval_envs}"]
params = [
  { name = "batch_size", type = "int", default = 2, min = 1, max = 1024 },
  { name = "steps", type = "int", default = 16, min = 1, max = 10000000 },
  { name = "eval_envs", type = "int", default = 2, min = 1, max = 128 },
]

[[task]]
id = "test-original"
name = "原版 GPU 测试"
group = "测试"
needs = "wsl"
script = "tasks/run-gpu-wsl.sh"
args = ["--batch-size", "{batch_size}", "--steps", "{steps}", "--eval-envs", "{eval_envs}"]
params = [
  { name = "batch_size", type = "int", default = 2, min = 1, max = 1024 },
  { name = "steps", type = "int", default = 16, min = 1, max = 10000000 },
  { name = "eval_envs", type = "int", default = 2, min = 1, max = 128 },
]

[[task]]
id = "test-performance"
name = "性能对比"
group = "测试"
needs = "wsl"
script = "experiments/torque200/performance-wsl.sh"
args = ["--web-url", "{web_url}"]
params = [{ name = "web_url", type = "str", default = "http://127.0.0.1:8770" }]

[[task]]
id = "test-throughput"
name = "吞吐探测"
group = "测试"
needs = "wsl"
script = "experiments/torque200/throughput-wsl.sh"
```

- [ ] **Step 2: 运行测试，确认清单合法**

Run: `python -m unittest tests.test_tenny.RegistryTest.test_real_toml_is_valid -v`
Expected: OK（每个 `script` 文件存在，默认参数合法）。若 `start-sim-wsl.sh` 不存在，先 `git ls-files tasks | grep sim` 找真实文件名并修正清单，不要改测试。

- [ ] **Step 3: 新建 `.gitattributes`**

```
*.sh text eol=lf
```

- [ ] **Step 4: 一次性改写脚本（不提交）**

保存为仓库外的临时文件（如 `$env:TEMP\fix_paths.py`），在仓库根目录运行 `python $env:TEMP\fix_paths.py`：

```python
import re
from pathlib import Path

PAT = re.compile(rb"'/mnt/c/Users/xusen/OneDrive - HKUST Connect/Workshop/Projects/Project Tenny(/[^']*)?'")
files = ([(f, "../..") for f in Path("experiments/torque200").glob("*-wsl.sh")]
         + [(f, "..") for f in Path("tasks").glob("*-wsl.sh")])
for f, up in files:
    data = f.read_bytes()
    if not PAT.search(data):
        continue
    data = PAT.sub(lambda m: b'"$WINROOT' + (m.group(1) or b"") + b'"', data)
    head, _, rest = data.partition(b"\n")
    line = b'WINROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/' + up.encode() + b'" && pwd)"\n'
    f.write_bytes(head + b"\n" + line + rest)
    print("fixed", f)
left = [f for f, _ in files if b"Users/xusen" in f.read_bytes()]
assert not left, left
```

预期输出 `fixed ...` 共 16 行（experiments 9 + tasks 7，仅含写死路径的文件）；`assert` 不报错。`WINROOT` 行插在 shebang 之后，位于 `set -euo pipefail` 之前，不影响其语义。

- [ ] **Step 5: 语法检查所有被改脚本**

Run（PowerShell）：

```powershell
git diff --name-only -- '*.sh' | ForEach-Object { $p = (wsl.exe -d Ubuntu --exec wslpath -a (Resolve-Path $_).Path).Trim(); wsl.exe -d Ubuntu --exec bash -n $p; "$_ -> $LASTEXITCODE" }
```
Expected: 每行 `-> 0`。另用 `git diff -- experiments/torque200/run-wsl.sh` 肉眼确认只改了 `WIN=` 行并多了 `WINROOT=` 行。

- [ ] **Step 6: 用真实任务跑一次端到端（短）**

Run: `python tenny.py run test-torque200 steps=16`
Expected: 输出末尾含 `TORQUE200_ALL_CHECKS_OK`，退出码 0；`experiments/torque200/results/` 下出现新的 `smoke-*` 目录。若失败，先读日志里的路径错误，修脚本，不要回退清单。

- [ ] **Step 7: Commit**

```bash
git add tenny/tasks.toml .gitattributes experiments tasks
git commit -m "feat(tenny): real task list; derive launcher paths from script location"
```

---

### Task 5: 删除旧脚本、整理文档、整体验证

**Files:**
- Delete: 根目录 `start-robot-web.ps1`、`start-simulation.ps1`、`start-torque200-preview.ps1`、`start-torque200-training.ps1`、`start-torque200-web.ps1`、`test-gpu-training.ps1`、`test-robot-training.ps1`、`test-torque200-performance.ps1`、`test-torque200-throughput.ps1`、`test-torque200-training.ps1`、`verify-selected-training.cmd`、`tasks/resolve-robot-mode.ps1`
- Move: `CONTROL-MODES.md` → `docs/CONTROL-MODES.md`
- Modify: `README.md`、`docs/CONTROL-MODES.md`、`experiments/torque200/README.md`

- [ ] **Step 1: 删除旧脚本并移动文档**

```bash
git rm start-robot-web.ps1 start-simulation.ps1 start-torque200-preview.ps1 start-torque200-training.ps1 start-torque200-web.ps1 test-gpu-training.ps1 test-robot-training.ps1 test-torque200-performance.ps1 test-torque200-throughput.ps1 test-torque200-training.ps1 verify-selected-training.cmd tasks/resolve-robot-mode.ps1
git mv CONTROL-MODES.md docs/CONTROL-MODES.md
```

- [ ] **Step 2: 找出并更新仍引用旧脚本的文档**

Run: `git grep -n -E "(start|test|verify)-[a-z0-9-]+\.(ps1|cmd)|resolve-robot-mode|CONTROL-MODES" -- README.md docs experiments/torque200/README.md experiments/torque200/VALIDATION.md`

对用户可见的说明逐处替换：

| 旧 | 新 |
|---|---|
| `start-robot-web.ps1` | `python tenny.py run web`（或网页里点"训练控制台"） |
| `pwsh -File .\test-torque200-training.ps1` | `python tenny.py run test-torque200` |
| `start-torque200-training.ps1` | `python tenny.py run train-timed` |
| `start-torque200-preview.ps1` | `python tenny.py run preview` |
| `test-torque200-performance.ps1` / `-throughput.ps1` | `python tenny.py run test-performance` / `test-throughput` |
| `test-gpu-training.ps1` | `python tenny.py run test-original` |
| `start-simulation.ps1` | `python tenny.py run sim` |

`tasks/` 下的历史笔记（`todo.md`、`REPRODUCTION.md` 等）属于过程记录，**不改**。

- [ ] **Step 3: 重写 `README.md` 的目录与用法部分**

把"目录"表替换为：

```markdown
## 快速开始

```bash
python tenny.py            # 打开网页控制台，点按钮运行任务（Python 3.11+，无需安装依赖）
python tenny.py list       # 终端里列出所有任务
python tenny.py run test-torque200 steps=16   # 终端里运行任务
```

任务在 `tenny/tasks.toml` 里登记，网页和命令行共用。Windows 下任务经 `wsl.exe` 在 WSL 中执行，
Linux 下直接用 bash；macOS 不支持训练类任务。

## 目录

| 路径 | 说明 |
|---|---|
| `tenny.py`、`tenny/` | 统一入口（网页控制台 + CLI）与任务清单 `tenny/tasks.toml` |
| `experiments/torque200/` | 全扭矩 200 Hz 独立实验，详见其 `README.md` |
| `reference/` | 参考项目的副本（见下） |
| `docs/` | 文档（控制模式等） |
| `tasks/` | 过程记录与辅助脚本 |

## 已知限制

- 训练依赖 GPU + JAX，只能在 Linux/WSL 中运行。
- WSL 侧环境目录（`/home/jaccob/projects/...`）仍写死在启动器里，换机器需要改。
- `experiments/torque200/scripts/` 与 `tasks/` 下的若干一次性 Python 脚本仍含本机路径。
```

- [ ] **Step 4: 全量测试 + 清单自检**

Run: `python -m unittest discover -s tests -v` → 全部 OK。
Run: `python tenny.py list` → 4 个分组、8 个任务。
Run: `git grep -n "Users/xusen" -- tenny tests README.md docs/CONTROL-MODES.md` → 无输出（spec/plan 文档除外）。

- [ ] **Step 5: 浏览器端到端验证**

Run: `python tenny.py web --no-browser`（后台），用 Playwright 打开 `http://127.0.0.1:8800/`：
1. 页面按"训练 / 预览与仿真 / 测试"三组显示 8 个任务，参数输入框带默认值。
2. 点 `吞吐探测` 的"运行"→ 日志区出现输出，状态由 `running` 变 `ok`。
3. 点 `全扭矩回归测试` 运行后立即点"停止"→ 状态变 `stopped`；再运行同一任务能再次启动。
4. 运行中再点同一任务→ 弹出"已在运行"。
5. 在 WSL 中 `pgrep -af python` 确认停止后无残留训练进程；若有残留，记录为已知限制并写进 README"已知限制"（`taskkill` 杀不到 WSL 内部子进程时的上限）。

- [ ] **Step 6: Commit 并合并**

```bash
git add -A
git commit -m "refactor: replace root scripts with tenny entry; tidy docs"
git push -u origin feat/unified-entry
```
推送后请用户确认，再 `git checkout main && git merge --ff-only feat/unified-entry && git push`。

---

## Self-Review

- **Spec 覆盖：** 入口/CLI/网页（T3）、任务清单与跨平台命令（T1/T2/T4）、安全（T2/T3 测试）、路径推导+`.gitattributes`（T4）、删除旧脚本+文档（T5）、验证三步（T3 Step 7 / T4 Step 6 / T5 Step 5）。`CONTROL-MODES.md` 移动在 T5。
- **占位符扫描：** 无 TBD；所有代码步骤含完整代码。
- **类型一致：** `snapshot(since)` 返回键 `lines/next/status/url/task/id/returncode` 在 runner、server、cli、UI、测试中一致；`TaskError/UnknownTask/Busy` 在 registry 定义、runner 抛出、server 映射 404/409/400；`MAX_LINES`、`stop_all`、`wsl_available`、`to_linux_path`、`build_command` 名称在 T2 定义并被测试按同名使用。
- **Review Focus 对应：** 未知任务（`test_unknown_task`/`test_error_codes`）、参数越界与未登记（`test_bad_values_rejected`/`test_bad_param_starts_nothing`）、重复运行（`test_duplicate_run_rejected`）、跨站（`test_rejects_foreign_host_and_form_posts`）、大输出与停止（`test_log_is_capped_and_cursor_is_absolute`/`test_stop_kills_process`）。
- **已知遗留：** 7 个一次性 Python 脚本里的本机路径、WSL 侧 `/home/jaccob` 路径，均写入 README"已知限制"。
