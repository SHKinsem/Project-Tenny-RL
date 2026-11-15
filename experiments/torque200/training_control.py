"""Persistent, single-job Linux training supervisor (standard library only).

The supervisor owns active.json; the detached runner owns its status.json.
Stopping is cooperative: no signal is sent while a checkpoint is being saved.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any
import uuid
from control_modes import MODES as CONTROL_MODES, mode_spec

try:
    import fcntl
except ImportError:  # Allow the web CLI and tests to import on Windows.
    fcntl = None


class ControlError(RuntimeError):
    """The requested control operation conflicts with the current job."""


MODES = frozenset(CONTROL_MODES)
TERMINAL = frozenset({"completed", "stopped", "cancelled", "failed", "interrupted"})
RUNNER_NAMES = frozenset({"console_train.py", "timed_console_train.py"})


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _write(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _mode(mode: str) -> str:
    if not isinstance(mode, str) or mode not in MODES:
        raise ValueError("Unknown control mode")
    return mode


def resolve_experiment_config(mode, value=None, *, restore=None):
    """Resolve before creating a run; restored weights retain their saved semantics."""
    if mode != "leg_force_bc":
        if value is not None:
            raise ValueError("experiment_config is only supported for leg_force_bc")
        return None
    from force_teacher.options import normalize_options, options_from_metadata
    requested = normalize_options(value) if value is not None else None
    if restore is not None:
        path = Path(restore).resolve().parent.parent / "policy_metadata.json"
        metadata = json.loads(path.read_text(encoding="utf8"))
        saved = options_from_metadata(metadata)
        if requested is not None and requested != saved:
            raise ValueError("Checkpoint experiment_config differs; changed experiments require a new run")
        return saved
    return requested if requested is not None else normalize_options()


def _within(path: str | Path, root: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    if not candidate.is_relative_to(root.resolve()) or candidate == root.resolve():
        raise ValueError("artifact path must remain inside its run directory")
    return candidate


class TrainingController:
    def __init__(self, artifacts: Path, original_root: Path, torque_root: Path,
                 script: Path | None = None, python_executable: str | None = None):
        self.root = (Path(artifacts).resolve() / "console").resolve()
        self.original_root = Path(original_root).resolve()
        self.torque_root = Path(torque_root).resolve()
        self.script = Path(script or Path(__file__).with_name("timed_console_train.py")).resolve()
        self.python_executable = str(python_executable or sys.executable)
        self.root.mkdir(parents=True, exist_ok=True)
        self._children: dict[int, subprocess.Popen] = {}
        self._latest_cache: dict[str, tuple[Any, Any]] = {}
        self._recent_cache: dict[str, tuple[int, Path | None]] = {}
        self._hybrid_validation_cache: dict[tuple[str, str], tuple] = {}

    @contextmanager
    def _locked(self):
        if fcntl is None:
            raise ControlError("Training control requires Linux / WSL")
        with (self.root / "manager.lock").open("a+b") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def _run_path(self, record: dict) -> Path:
        mode = _mode(record.get("mode"))
        run = _within(record["run_dir"], self.root / mode)
        if run.parent != self.root / mode:
            raise ValueError("run_dir must be an immediate child of its mode directory")
        return run

    @staticmethod
    def _process(pid: int) -> tuple[list[str], str] | None:
        try:
            proc = Path("/proc") / str(pid)
            stat = (proc / "stat").read_text()
            fields = stat[stat.rfind(")") + 2:].split()
            if fields[0] in {"Z", "X"}:
                return None
            argv = (proc / "cmdline").read_bytes().decode(errors="replace").split("\0")
            return argv, fields[19]  # Linux stat field 22: process start time.
        except (OSError, IndexError, ValueError):
            return None

    def _alive(self, record: dict) -> bool:
        pid = record.get("pid")
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
            return False
        child = self._children.get(pid)
        if child is not None and child.poll() is not None:
            self._children.pop(pid, None)
            return False
        process = self._process(pid)
        if not process:
            # A web exec/restart can lose its Popen map but retain child PIDs.
            try:
                os.waitpid(pid, os.WNOHANG)
            except (ChildProcessError, OSError):
                pass
            return False
        argv, started = process
        try:
            run = self._run_path(record)
            script = str(Path(record.get("script", str(self.script))).resolve())
            # Exact argv boundaries defeat substring/prefix matches and reused PIDs.
            run_index = argv.index("--run-dir")
            return (script in argv and Path(script).name in RUNNER_NAMES
                    and argv[run_index + 1] == str(run)
                    and record.get("process_start") in {None, started})
        except (ValueError, KeyError, IndexError):
            return False

    def _current(self) -> tuple[dict, dict, bool]:
        record = _read(self.root / "active.json")
        if not record:
            return {}, {}, False
        run = self._run_path(record)
        runner = _read(run / "status.json")
        alive = self._alive(record)
        if not alive and not record.get("ended_at"):
            phase = runner.get("phase", "interrupted")
            if phase not in TERMINAL:
                phase = "interrupted"
            record.update(phase=phase, ended_at=datetime.now(timezone.utc).isoformat())
            if phase == "interrupted":
                record["error"] = "Training process exited without a terminal status; existing artifacts were preserved."
            _write(self.root / "active.json", record)
            self._latest_cache.clear()
        return record, runner, alive

    def _validated(self, run: Path, mode: str) -> dict | None:
        try:
            run = _within(run, self.root / mode)
            report = _read(run / "report.json")
            state = _read(run / "status.json")
            if report.get("mode") != mode or state.get("phase") not in {"completed", "stopped"}:
                return None
            if report.get("all_parameters_finite") is not True:
                return None
            error = report.get("parameter_tree_max_abs_error")
            export_error = report.get("export_max_abs_error")
            tolerance = report.get("export_atol")
            if isinstance(error, bool) or error != 0:
                return None
            if any(isinstance(x, bool) or not isinstance(x, (int, float))
                   or not math.isfinite(x) or x < 0 for x in (export_error, tolerance)):
                return None
            if export_error > tolerance:
                return None
            checkpoint = _within(report["checkpoint"], run)
            export = _within(report["export_path"], run)
            if not checkpoint.exists() or not export.exists():
                return None
            if mode_spec(mode)["teacher"]:
                if mode == "leg_force_bc":
                    from force_teacher.contract import verified_report
                else:
                    from wheel_teacher.contract import verified_report
                verified_report(run, report, self.original_root)
                if mode == "leg_force_bc":
                    from force_teacher.options import options_from_metadata
                    report = {**report, "experiment_config": options_from_metadata(
                        _read(run / "policy_metadata.json"))}
            elif mode_spec(mode)["hybrid"]:
                from hybrid.managed_training import verified_report, verification_signature
                signature = verification_signature(run)
                key = (str(run), mode)
                if self._hybrid_validation_cache.get(key) != signature:
                    verified_report(run, mode, report)
                    self._hybrid_validation_cache[key] = signature
            return {**report, "mode": mode, "run_dir": str(run),
                    "run_id": run.name, "report_path": str(run / "report.json"),
                    "checkpoint": str(checkpoint), "export_path": str(export), "verified": True}
        except (ValueError, KeyError, TypeError, OSError):
            return None

    def _runs(self, mode: str, active: dict | None = None) -> list[dict]:
        directory = self.root / mode
        if not directory.exists():
            return []
        results = []
        for run in sorted(directory.iterdir(), key=lambda p: p.name, reverse=True):
            if not run.is_dir() or (active and str(run) == active.get("run_dir")):
                continue
            validated = self._validated(run, mode)
            if validated:
                results.append(validated)
        return results

    def _latest(self, mode: str, record: dict, alive: bool) -> dict | None:
        try:
            directory_mtime = (self.root / mode).stat().st_mtime_ns
        except FileNotFoundError:
            directory_mtime = None
        # New runs change the directory mtime; active exit changes the record.
        # No full directory scan is needed for each one-second dashboard poll.
        fingerprint = (directory_mtime, record.get("run_dir"), record.get("ended_at"), alive)
        # Hybrid contracts and artifact hashes must remain valid after source/file edits.
        # These checks happen only when selecting a saved model, not in training callbacks.
        if mode_spec(mode)["hybrid"] or mode == "wheel_torque_bc":
            cached = None
        else:
            cached = self._latest_cache.get(mode)
        if cached and cached[0] == fingerprint:
            return cached[1]
        valid = self._runs(mode, record if alive else None)
        result = valid[0] if valid else None
        self._latest_cache[mode] = (fingerprint, result)
        return result

    def _recent(self, mode: str) -> Path | None:
        directory = self.root / mode
        try:
            mtime = directory.stat().st_mtime_ns
        except FileNotFoundError:
            return None
        cached = self._recent_cache.get(mode)
        if cached and cached[0] == mtime:
            return cached[1]
        candidates = [p for p in directory.iterdir() if p.is_dir() and not p.is_symlink()]
        result = max(candidates, key=lambda p: p.name) if candidates else None
        self._recent_cache[mode] = (mtime, result)
        return result

    def _status(self, mode: str | None = None) -> dict:
        record, runner, alive = self._current()
        selected = mode or record.get("mode") or "original"
        latest = self._latest(selected, record, alive)
        if record and (mode is None or record["mode"] == mode):
            state = {**record, **runner}
            # Runner data cannot redefine manager-owned identity or liveness.
            state.update({key: record.get(key) for key in ("mode", "run_dir", "pid")})
            if selected == "leg_force_bc":
                state["experiment_config"] = record.get("experiment_config")
            if not alive:
                state["phase"] = record.get("phase", "interrupted")
                if record.get("error"):
                    state["error"] = record["error"]
            elif (self._run_path(record) / "stop.request").exists() and state.get("phase") not in {"saving", "exporting", "verifying"}:
                state["phase"] = "stopping"
        else:
            state = {"mode": selected, "phase": "idle"}
            previous = self._recent(selected)
            if previous is not None:
                state.update(_read(previous / "request.json"))
                state.update(_read(previous / "status.json"))
                state.update(mode=selected, run_dir=str(previous))
                if state.get("phase") not in TERMINAL:
                    state.update(phase="interrupted", error="Training process is no longer active; existing artifacts were preserved.")
        if alive and state.get("started_at"):
            try:
                started = datetime.fromisoformat(state["started_at"].replace("Z", "+00:00"))
                state["elapsed_seconds"] = max(0.0, (datetime.now(timezone.utc) - started).total_seconds())
            except (ValueError, TypeError):
                pass
        state.update(active=alive, active_mode=record.get("mode") if alive else None,
                     can_resume=latest is not None, latest_checkpoint=latest.get("checkpoint") if latest else None,
                     latest=latest)
        return state

    def status(self, mode: str | None = None) -> dict:
        if mode is not None:
            _mode(mode)
        with self._locked():
            return self._status(mode)

    def latest(self, mode: str) -> dict | None:
        _mode(mode)
        with self._locked():
            record, _, alive = self._current()
            return self._latest(mode, record, alive)

    def runs(self, mode: str) -> list[dict]:
        _mode(mode)
        with self._locked():
            record, _, alive = self._current()
            return self._runs(mode, record if alive else None)

    def start(self, mode: str, duration_seconds: float = 0, resume: bool = False,
              force_revalidate: bool = False, experiment_config=None) -> dict:
        _mode(mode)
        # Strict type/range validation precedes locks, run directories and children.
        requested_options = resolve_experiment_config(mode, experiment_config)
        if (isinstance(duration_seconds, bool) or not isinstance(duration_seconds, (float, int))
                or not math.isfinite(duration_seconds) or not 0 <= duration_seconds <= 7 * 86400):
            raise ValueError("duration_seconds must be 0 (unlimited) or at most seven days")
        if not isinstance(resume, bool):
            raise ValueError("resume must be a boolean")
        if not isinstance(force_revalidate, bool):
            raise ValueError("force_revalidate must be a boolean")
        if not self.script.is_file() or self.script.name not in RUNNER_NAMES:
            raise ValueError("training script must be an existing console_train.py or timed_console_train.py")
        with self._locked():
            record, _, alive = self._current()
            if alive:
                raise ControlError("A training process is still active; wait for saving and exit before starting another")
            self._latest_cache.clear()
            restore = self._latest(mode, record, False) if resume else None
            if resume and not restore:
                raise ControlError("No verified checkpoint is available for this mode")
            options = (resolve_experiment_config(mode, experiment_config,
                       restore=restore["checkpoint"]) if restore else requested_options)
            seed = options["seed"] if options is not None else 0
            if (self.root / mode).is_symlink():
                raise ValueError("mode directory cannot be a symbolic link")
            run = self.root / mode / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "_" + uuid.uuid4().hex[:8])
            run.mkdir(parents=True)
            request = {"mode": mode, "duration_seconds": duration_seconds, "resume": resume,
                       "force_revalidate": force_revalidate,
                       "restore": restore["checkpoint"] if restore else None, "seed": seed, "preview_protocol": 1,
                       "created_at": datetime.now(timezone.utc).isoformat(), "run_dir": str(run)}
            if options is not None:
                request["experiment_config"] = options
            _write(run / "request.json", request)
            _write(run / "status.json", {**request, "phase": "preparing"})
            argv = [self.python_executable, str(self.script), "--mode", mode, "--run-dir", str(run),
                    "--original-root", str(self.original_root), "--torque-root", str(self.torque_root),
                    "--duration-seconds", str(duration_seconds), "--seed", str(seed)]
            if restore:
                argv += ["--restore", restore["checkpoint"]]
            if force_revalidate:
                argv += ["--force-revalidate"]
            env = os.environ.copy()
            env.pop("LD_LIBRARY_PATH", None)
            env.update(JAX_PLATFORMS="cuda", CUDA_VISIBLE_DEVICES="0", XLA_PYTHON_CLIENT_PREALLOCATE="false",
                       XLA_PYTHON_CLIENT_MEM_FRACTION="0.35", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1",
                       JAX_COMPILATION_CACHE_DIR=str(self.torque_root / "jax-cache"), PYTHONUNBUFFERED="1")
            if mode_spec(mode)["hybrid"]:
                from hybrid.managed_training import runtime_root
                from hybrid.readiness import prepare_environment
                # The web preview itself is CPU-only; training uses the GPU.
                # Start from the inherited settings so explicit GPU selection and
                # memory limits survive, using exactly the CLI normalization.
                source = os.environ.copy()
                source["JAX_PLATFORMS"] = "cuda"
                env = prepare_environment(runtime_root(), source)
            record = {**request, "phase": "preparing", "script": str(self.script)}
            try:
                with (run / "train.log").open("ab", buffering=0) as log:
                    child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                             cwd=self.torque_root, env=env, start_new_session=True, close_fds=True)
                self._children[child.pid] = child
                process = self._process(child.pid)
                record.update(pid=child.pid, process_start=process[1] if process else None)
                _write(self.root / "active.json", record)
            except OSError as error:
                record.update(phase="failed", error=str(error), ended_at=datetime.now(timezone.utc).isoformat())
                _write(self.root / "active.json", record)
                raise ControlError("Could not launch training: " + str(error)) from error
            self._latest_cache.clear()
            return self._status(mode)

    def request_preview(self, mode: str) -> dict:
        _mode(mode)
        with self._locked():
            record, runner, alive = self._current()
            if alive:
                if record.get("mode") != mode:
                    raise ControlError("当前训练模式与展示模式不同。")
                if runner.get("preview_protocol") != 1:
                    raise ControlError("这一轮训练在功能更新前启动，暂不支持按需同步；下次开始或继续训练即可使用。当前训练继续运行。")
                run = self._run_path(record)
                request = {"request_id": uuid.uuid4().hex, "mode": mode,
                           "run_id": run.name, "run_dir": str(run), "kind": "live"}
                _write(run / "preview.request.json", request)
                return request
            latest = self._latest(mode, record, alive)
            return {"request_id": uuid.uuid4().hex, "mode": mode,
                    "kind": "saved", "model": latest}

    def cancel_preview(self, request: dict | None):
        if not request or request.get("kind") != "live":
            return
        with self._locked():
            run = self._run_path(request)
            path = run / "preview.request.json"
            if _read(path).get("request_id") == request.get("request_id"):
                path.unlink(missing_ok=True)

    def stop(self) -> dict:
        with self._locked():
            record, _, alive = self._current()
            if alive:
                stop = self._run_path(record) / "stop.request"
                if not stop.exists():
                    _write(stop, {"requested_at": datetime.now(timezone.utc).isoformat()})
            return self._status()
