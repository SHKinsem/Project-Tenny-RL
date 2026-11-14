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
KILL_GRACE = 5  # seconds to wait before SIGKILL after SIGTERM on POSIX
# WSL 侧保存"选中模式"的位置（~ 由 WSL 内的 python 展开）；与 experiments/torque200 的训练控制台约定一致
SELECTED_MODE_FILE = "~/projects/tenny-torque200/artifacts/selected-mode.json"
_SELECTED_CODE = ('import json,sys;from pathlib import Path;p=Path(sys.argv[1]).expanduser();'
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
        try:
            for raw in self.proc.stdout:
                line = raw.decode("utf-8", "replace").rstrip("\r\n")
                with self._lock:
                    self._lines.append(line)
                    if len(self._lines) > MAX_LINES:
                        del self._lines[0]
                        self._dropped += 1
        finally:
            self.proc.stdout.close()
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
            # Escalate to SIGKILL if process ignores SIGTERM
            def _escalate():
                if self.returncode is None:
                    try:
                        os.killpg(self.proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            threading.Timer(KILL_GRACE, _escalate).start()


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
        try:
            job = Job(task, build_command(task, values), url)
        except OSError as e:
            raise registry.TaskError(f"无法启动 {task_id}: {e}")
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
