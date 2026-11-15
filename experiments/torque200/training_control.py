"""Persistent, single-job Linux training supervisor (standard library only).

The supervisor owns active.json; the detached runner owns its status.json.
Stopping is cooperative: no signal is sent while a checkpoint is being saved.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
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
from workbench_parameters import parameter_schema, resolve_parameters, validate_parameters, FIELDS
from workbench_storage import WorkbenchStorage
from workbench_validation import Dependency, filesystem_signature, source_signature
from workbench_api.checkpoint_metadata import MetadataError

try:
    import fcntl
except ImportError:  # Allow the web CLI and tests to import on Windows.
    fcntl = None


class ControlError(RuntimeError):
    """The requested control operation conflicts with the current job."""


MODES = frozenset(CONTROL_MODES)
TERMINAL = frozenset({"completed", "stopped", "cancelled", "failed", "interrupted"})
RUNNER_NAMES = frozenset({"console_train.py", "timed_console_train.py"})
MAX_CACHED_VALIDATIONS = 256
MAX_CACHED_SIGNATURE_ROWS = 200_000


def _signature_rows(signature):
    return sum(len(rows) for group in signature for rows in group)


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
        self.artifacts = Path(artifacts).resolve()
        self.storage = WorkbenchStorage(self.artifacts)
        self.root = (self.artifacts / "console").resolve()
        self.original_root = Path(original_root).resolve()
        self.torque_root = Path(torque_root).resolve()
        self.script = Path(script or Path(__file__).with_name("timed_console_train.py")).resolve()
        self.python_executable = str(python_executable or sys.executable)
        self.root.mkdir(parents=True, exist_ok=True)
        self._children: dict[int, subprocess.Popen] = {}
        self._validation_cache: dict[tuple[str, str], tuple[Any, Any]] = {}
        self._recent_cache: dict[str, tuple[int, Path | None]] = {}

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
        return self.storage.resolve_run(record["run_dir"], mode)

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
            self._validation_cache.clear()
        return record, runner, alive

    def _validated(self, run: Path, mode: str, *, fresh=True, sources=None) -> dict | None:
        try:
            run = self.storage.resolve_run(run, mode)
        except (ValueError, OSError):
            return None
        # Actions always validate bytes again, even if filesystem timestamps or
        # the historical hybrid signature happen to be unchanged.
        if fresh:
            self._validation_cache.pop((str(run), mode), None)
            return self._validate_artifacts(run, mode)
        sources = source_signature(mode, self.original_root) if sources is None else sources
        key = (str(run), mode)
        for attempt in range(2):
            artifacts = filesystem_signature((Dependency(run, recursive=True),))
            signature = (sources, artifacts)
            cached = self._validation_cache.get(key)
            cacheable = sources is not None and artifacts is not None
            if cacheable and cached is not None and cached[0] == signature:
                return deepcopy(cached[1])
            validated = self._validate_artifacts(run, mode)
            after_artifacts = filesystem_signature((Dependency(run, recursive=True),)) if cacheable else None
            after_sources = source_signature(mode, self.original_root) if cacheable else None
            # A single retry handles an atomic publication (and deferred NTFS
            # directory timestamps); never cache or return a torn validation.
            if cacheable and (artifacts != after_artifacts or sources != after_sources):
                self._validation_cache.pop(key, None)
                sources = after_sources
                continue
            self._validation_cache.pop(key, None)
            if cacheable:
                cost = _signature_rows(signature)
                used = sum(_signature_rows(item[0]) for item in self._validation_cache.values())
                while self._validation_cache and (len(self._validation_cache) >= MAX_CACHED_VALIDATIONS
                                                  or used + cost > MAX_CACHED_SIGNATURE_ROWS):
                    oldest = next(iter(self._validation_cache))
                    used -= _signature_rows(self._validation_cache[oldest][0])
                    self._validation_cache.pop(oldest)
                if cost <= MAX_CACHED_SIGNATURE_ROWS:
                    self._validation_cache[key] = (signature, deepcopy(validated))
            return validated
        return None

    def _validate_artifacts(self, run: Path, mode: str) -> dict | None:
        """Validate bytes/contracts; caller independently authorizes the root."""
        try:
            run = Path(run).resolve()
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
            elif mode_spec(mode)["hybrid"]:
                from hybrid.managed_training import verified_report
                verified_report(run, mode, report)
            return {**report, "mode": mode, "run_dir": str(run),
                    "run_id": run.name, "report_path": str(run / "report.json"),
                    "checkpoint": str(checkpoint), "export_path": str(export), "verified": True}
        except (ValueError, KeyError, TypeError, OSError):
            return None

    def _runs(self, mode: str, active: dict | None = None, *, fresh=False) -> list[dict]:
        results = []
        candidates = []
        for root in self.storage.registered_roots():
            directory = root / mode
            if directory.is_symlink() or not directory.is_dir():
                continue
            entries = list(directory.iterdir())
            if len(entries) > 2048:
                raise ControlError("Checkpoint directory exceeds the scan limit")
            candidates.extend(entries)
        sources = source_signature(mode, self.original_root) if not fresh else None
        for run in sorted(candidates, key=lambda p: p.name, reverse=True):
            if run.is_symlink() or not run.is_dir() or (active and str(run) == active.get("run_dir")):
                continue
            validated = self._validated(run, mode, fresh=fresh, sources=sources)
            if validated:
                results.append(validated)
        results.sort(key=lambda item: Path(item["report_path"]).stat().st_mtime_ns, reverse=True)
        return results

    def _latest(self, mode: str, record: dict, alive: bool, *, fresh=False) -> dict | None:
        # Scan lightweight signatures so content edits, late final reports and
        # independent manual runs all invalidate the relevant read cache.
        valid = self._runs(mode, record if alive else None, fresh=fresh)
        return valid[0] if valid else None

    def _recent(self, mode: str) -> Path | None:
        directories = [root / mode for root in self.storage.registered_roots() if (root / mode).is_dir()]
        mtime = tuple((str(path), path.stat().st_mtime_ns) for path in directories)
        cached = self._recent_cache.get(mode)
        if cached and cached[0] == mtime:
            return cached[1]
        candidates = [p for directory in directories for p in directory.iterdir() if p.is_dir() and not p.is_symlink()]
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
                state.update(_read(previous / "status.json"))
                state.update(mode=selected, run_dir=str(previous))
                if state.get("phase") not in TERMINAL:
                    state.update(phase="interrupted", error="Training process is no longer active; existing artifacts were preserved.")
        if alive and state.get("started_at") and state.get("workbench_control_protocol") != 1:
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

    def _select_checkpoint(self, mode, checkpoint, *, fresh=True):
        path = Path(checkpoint)
        if not path.is_absolute() or path.is_symlink():
            raise ValueError("checkpoint must be an absolute, registered checkpoint directory")
        path = path.resolve()
        # Existing legacy reports can use a nonnumeric checkpoint directory.
        # The report, not a filename heuristic, identifies the validated model.
        candidates = [path.parent.parent, path.parent]
        for run in candidates:
            validated = self._validated(run, mode, fresh=fresh)
            if validated and Path(validated["checkpoint"]) == path:
                record, _, alive = self._current()
                if alive and validated["run_dir"] == record.get("run_dir"):
                    raise ControlError("Active training artifacts cannot be selected before verification")
                return validated
        raise ControlError("所选 checkpoint 未通过当前模式、源码和产物验证。")

    @staticmethod
    def _checkpoint_option(report):
        checkpoint = Path(report["checkpoint"])
        metadata = _read(checkpoint.with_name(checkpoint.name + ".workbench.json"))
        try:
            from workbench_api.checkpoint_metadata import validate_checkpoint
            metadata = validate_checkpoint(metadata, mode=report["mode"])
            if metadata["createdByRunId"] != report["run_id"]:
                metadata = {}
        except (ValueError, KeyError, TypeError, MetadataError):
            metadata = {}
        identifier = metadata.get("id")
        return {"id": identifier or str(checkpoint), "path": str(checkpoint),
                "label": "CP·" + identifier[:8].upper() if identifier else checkpoint.name,
                "runId": report["run_id"], "steps": report.get("actual_timesteps", report.get("steps")),
                "trainedSeconds": metadata.get("trainedSeconds"), "trainedSteps": metadata.get("trainedSteps"),
                "compatible": True, "reason": None, "optimizerRestored": False}

    def checkpoint_options(self, mode, limit=50):
        _mode(mode)
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        with self._locked():
            record, _, alive = self._current()
            return {"schemaVersion": 1, "mode": mode,
                    "checkpoints": [self._checkpoint_option(item) for item in self._runs(mode, record if alive else None)[:limit]],
                    "resumeSemantics": {"optimizerRestored": False}}

    def parameters(self, mode, checkpoint=None):
        _mode(mode)
        with self._locked():
            record, _, alive = self._current()
            selected = self._select_checkpoint(mode, checkpoint, fresh=False)["checkpoint"] if checkpoint else None
            result = parameter_schema(mode, checkpoint=selected, locked=alive)
            if alive and record.get("mode") == mode:
                actual = _read(self._run_path(record) / "ppo_config.json")
                values = {field["key"]: actual[field["key"]] for field in FIELDS if field["key"] in actual}
                if values:
                    validate_parameters(values, base=result["effective"])
                    result["effective"].update(values)
                    result["fields"] = [{**field, "value": result["effective"][field["key"]]} for field in result["fields"]]
                    result["source"] = "active_run"
            return result

    def capabilities(self, mode):
        state = self.status(mode)
        active = state.get("active") is True
        controllable = (active and state.get("active_mode") == mode
                        and state.get("workbench_control_protocol") == 1
                        and state.get("phase") not in TERMINAL | {"saving", "exporting", "verifying", "stopping"})
        return {"startTraining": not active, "resumeLatest": not active and bool(state.get("can_resume")),
                "stopAndSave": active, "preview": True, "switchMode": not active,
                "pauseTraining": controllable, "saveSnapshot": controllable,
                "resumeSelected": not active, "parameterWrites": not active,
                "multipleViewports": True, "directoryBridge": True}

    def storage_state(self):
        state = self.status()
        return {**self.storage.describe(), "schemaVersion": 1, "saveRoot": str(self.storage.save_root()),
                "defaultRoot": str(self.root), "roots": [str(root) for root in self.storage.registered_roots()],
                "locked": state.get("active") is True}

    def configure_storage(self, path):
        with self._locked():
            if self._current()[2]:
                raise ControlError("训练期间不能更改 checkpoint 保存目录。")
            self.storage.configure(path)
            self._validation_cache.clear()
            self._recent_cache.clear()
        return self.storage_state()

    def resolve_directory(self, raw):
        if not isinstance(raw, str) or not raw or len(raw) > 4096:
            raise ValueError("path must be an existing registered directory")
        path = Path(raw)
        if not path.is_absolute() or path.is_symlink() or not path.is_dir():
            raise ValueError("path must be an existing registered directory")
        path = path.resolve()
        for root in self.storage.registered_roots():
            try:
                relative = path.relative_to(root)
            except ValueError:
                continue
            if not relative.parts:
                return {"ok": True, "path": str(path)}
            if relative.parts[0] not in MODES:
                continue
            if len(relative.parts) == 1:
                return {"ok": True, "path": str(path)}
            try:
                run = self.storage.resolve_run(root / relative.parts[0] / relative.parts[1], relative.parts[0])
            except (ValueError, OSError):
                continue
            if path == run or (len(relative.parts) == 4 and relative.parts[2] == "checkpoints"
                               and relative.parts[3].isdigit() and not (run / "checkpoints").is_symlink()):
                return {"ok": True, "path": str(path)}
        raise ValueError("Directory is not a registered checkpoint or save location")

    def import_checkpoint(self, path, mode):
        _mode(mode)
        if not isinstance(path, str) or not path or len(path) > 4096:
            raise ValueError("path must identify a complete saved checkpoint directory")
        with self._locked():
            # Import is separate IO, but source validation must not race a new
            # launch into partial model paths. It never touches active artifacts.
            record, _, alive = self._current()
            if alive:
                active_run = self._run_path(record)
                candidate = Path(path).resolve()
                if candidate == active_run or candidate.is_relative_to(active_run):
                    raise ControlError("不能导入正在写入的训练目录；请使用已完成的 checkpoint。")
            result = self.storage.import_checkpoint(path, mode, self._validate_artifacts)
            self._validation_cache.clear()
            self._recent_cache.clear()
        return result

    @staticmethod
    def _request_id(value):
        if value is None:
            return str(uuid.uuid4())
        try:
            parsed = uuid.UUID(value)
        except (ValueError, TypeError, AttributeError):
            raise ValueError("request_id must be a canonical UUID") from None
        if str(parsed) != value or not parsed.int:
            raise ValueError("request_id must be a canonical UUID")
        return value

    def _control(self, action, request_id=None):
        identifier = self._request_id(request_id)
        with self._locked():
            record, runner, alive = self._current()
            if not record:
                raise ControlError("No active training process")
            run = self._run_path(record)
            for folder in ("workbench-control-requests", "workbench-save-requests", "workbench-save-results"):
                directory = run / folder
                if directory.is_symlink() or (directory / (identifier + ".json")).is_symlink():
                    raise ValueError("Control request paths cannot be symbolic links")
            previous_control = _read(run / "workbench-control-requests" / (identifier + ".json"))
            previous_save = (_read(run / "workbench-save-requests" / (identifier + ".json"))
                             or _read(run / "workbench-save-results" / (identifier + ".json")))
            if (previous_control and previous_control.get("action") != action) or (previous_save and action != "save"):
                raise ValueError("request_id was already used for another action")
            if action == "save":
                receipt = _read(run / "workbench-save-results" / (identifier + ".json"))
                if receipt.get("requestId") == identifier:
                    return {"ok": True, "requestId": identifier, "action": action, "accepted": True,
                            "result": receipt, "training": self._status()}
            else:
                receipt = _read(run / "workbench-control-requests" / (identifier + ".json"))
                if receipt:
                    if receipt.get("action") != action:
                        raise ValueError("request_id was already used for another action")
                    return {"ok": True, "requestId": identifier, "action": action, "accepted": True,
                            "training": self._status()}
            if not alive:
                raise ControlError("No active training process")
            if runner.get("workbench_control_protocol") != 1:
                raise ControlError("本轮在功能更新前启动；暂停和手动保存从下一轮生效，当前训练继续运行。")
            if runner.get("phase") in TERMINAL | {"saving", "exporting", "verifying", "stopping"} or (run / "stop.request").exists():
                raise ControlError("Training is finishing; wait for the final verified checkpoint")
            request = {"schemaVersion": 1, "runId": run.name, "requestId": identifier,
                       "action": action, "createdAt": datetime.now(timezone.utc).isoformat()}
            if action == "save":
                queue = run / "workbench-save-requests"
                queue.mkdir(exist_ok=True)
                target = queue / (identifier + ".json")
                existing = _read(target)
                if existing and existing != request and existing.get("action") != action:
                    raise ValueError("request_id was already used for another action")
                if not existing:
                    if sum(1 for _ in queue.glob("*.json")) >= 32:
                        raise ControlError("Too many pending checkpoint requests")
                    _write(target, request)
            else:
                history = run / "workbench-control-requests"
                history.mkdir(exist_ok=True)
                if sum(1 for _ in history.glob("*.json")) >= 4096:
                    raise ControlError("Training control request history is full")
                _write(history / (identifier + ".json"), request)
                _write(run / "workbench-control.json", request)
            return {"ok": True, "requestId": identifier, "action": action, "accepted": True,
                    "training": self._status()}

    def pause(self, request_id=None):
        return self._control("pause", request_id)

    def resume(self, request_id=None):
        return self._control("resume", request_id)

    def save(self, request_id=None):
        return self._control("save", request_id)

    def start(self, mode: str, duration_seconds: float = 0, resume: bool = False,
              force_revalidate: bool = False, checkpoint: str | None = None,
              parameters: dict | None = None) -> dict:
        _mode(mode)
        if (isinstance(duration_seconds, bool) or not isinstance(duration_seconds, (float, int))
                or not math.isfinite(duration_seconds) or not 0 <= duration_seconds <= 7 * 86400):
            raise ValueError("duration_seconds must be 0 (unlimited) or at most seven days")
        if not isinstance(resume, bool):
            raise ValueError("resume must be a boolean")
        if not isinstance(force_revalidate, bool):
            raise ValueError("force_revalidate must be a boolean")
        if checkpoint is not None and (not isinstance(checkpoint, str) or not checkpoint or len(checkpoint) > 4096):
            raise ValueError("checkpoint must be an exact checkpoint path")
        if not self.script.is_file() or self.script.name not in RUNNER_NAMES:
            raise ValueError("training script must be an existing console_train.py or timed_console_train.py")
        with self._locked():
            record, _, alive = self._current()
            if alive:
                raise ControlError("A training process is still active; wait for saving and exit before starting another")
            restore = self._select_checkpoint(mode, checkpoint) if checkpoint else (self._latest(mode, record, False, fresh=True) if resume else None)
            if (resume or checkpoint) and not restore:
                raise ControlError("No verified checkpoint is available for this mode")
            effective, parameter_edits = resolve_parameters(mode, restore["checkpoint"] if restore else None, parameters)
            save_root = self.storage.save_root()
            if (save_root / mode).is_symlink():
                raise ValueError("mode directory cannot be a symbolic link")
            run = save_root / mode / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "_" + uuid.uuid4().hex[:8])
            run.mkdir(parents=True)
            request = {"mode": mode, "duration_seconds": duration_seconds, "resume": bool(resume or checkpoint),
                       "force_revalidate": force_revalidate,
                       "restore": restore["checkpoint"] if restore else None, "seed": 0, "preview_protocol": 1,
                       "parameters": parameter_edits, "requested_effective_parameters": effective,
                       "created_at": datetime.now(timezone.utc).isoformat(), "run_dir": str(run)}
            _write(run / "request.json", request)
            _write(run / "status.json", {**request, "phase": "preparing"})
            argv = [self.python_executable, str(self.script), "--mode", mode, "--run-dir", str(run),
                    "--original-root", str(self.original_root), "--torque-root", str(self.torque_root),
                    "--duration-seconds", str(duration_seconds), "--seed", "0"]
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
            if parameter_edits:
                env["TENNY_WORKBENCH_PARAMETERS"] = json.dumps(parameter_edits, allow_nan=False)
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
            self._validation_cache.clear()
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
            latest = self._latest(mode, record, alive, fresh=True)
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
