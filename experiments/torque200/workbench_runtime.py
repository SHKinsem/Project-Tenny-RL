"""Runner-local safe-boundary pause and queued manual checkpoint publication."""

from collections import OrderedDict
import copy
from datetime import datetime, timezone
import math
from pathlib import Path
import time
import traceback
from uuid import UUID

if __package__:
    from .workbench_persistence import WorkbenchPersistence, PersistenceError, _atomic_json, _inside, _read_json
else:
    from workbench_persistence import WorkbenchPersistence, PersistenceError, _atomic_json, _inside, _read_json


PROTOCOL = 1
MAX_PENDING = 32


def _request(value, run_id, action=None):
    if not isinstance(value, dict) or type(value.get("schemaVersion")) is not int or value["schemaVersion"] != PROTOCOL:
        raise ValueError("Unsupported control protocol")
    if value.get("runId") != run_id:
        raise ValueError("Control run identity mismatch")
    identity = value.get("requestId")
    if not isinstance(identity, str) or str(UUID(identity)) != identity or UUID(identity).int == 0:
        raise ValueError("Control request identity is invalid")
    if value.get("action") not in ((action,) if action else ("pause", "resume")):
        raise ValueError("Invalid control action")
    _created_at(value.get("createdAt"))
    return value


def _created_at(value):
    if isinstance(value, str) and len(value) <= 80:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if result.tzinfo is None:
            raise ValueError("Control time requires an explicit timezone")
        return result.astimezone(timezone.utc)
    if not isinstance(value, bool) and isinstance(value, (int, float)):
        try:
            if math.isfinite(value) and 0 <= value <= 253402300799000:
                return datetime.fromtimestamp(value / 1000, timezone.utc)
        except (OverflowError, OSError):
            pass
    raise ValueError("Control time is invalid")


class ManualSnapshotPublisher:
    """Give each request its own complete, run-shaped, immutable snapshot artifact."""

    def __init__(self, monitor, build):
        self.monitor, self.build = monitor, build

    def __call__(self, request, *, steps, factory, params, elapsed_seconds, stage):
        monitor = self.monitor
        mode_root = monitor.args.run_dir.parent
        stamp = _created_at(request["createdAt"]).strftime("%Y%m%dT%H%M%S.%fZ")
        name = stamp + "_" + UUID(request["requestId"]).hex + "_snapshot"
        run = _inside(mode_root / name, mode_root)
        if run.exists():
            report = _read_json(run / "report.json", run)
            status = _read_json(run / "status.json", run)
            if (report.get("workbench_save_request_id") != request["requestId"] or
                    report.get("source_run_id") != monitor.args.run_dir.name or
                    report.get("completed") is not True or status.get("phase") != "completed" or
                    not report.get("workbench_checkpoint_id")):
                raise ValueError("Existing snapshot request is not a completed matching artifact")
            return self._result(report)
        run.mkdir(exist_ok=False)
        basic = dict(mode=monitor.args.mode, run_id=name, source_run_id=monitor.args.run_dir.name,
                     workbench_save_request_id=request["requestId"], phase="saving", stage="manual_checkpoint",
                     completed=False, steps=steps, actual_timesteps=steps, duration_seconds=0,
                     elapsed_seconds=elapsed_seconds, training_stop_elapsed_seconds=elapsed_seconds,
                     stop_reason="manual_snapshot", training_clock_started=True)
        _atomic_json(run / "status.json", basic, run)
        try:
            stage("configuration")
            for filename in ("env_config.json", "ppo_config.json", "policy_metadata.json"):
                source = _inside(monitor.args.run_dir / filename, monitor.args.run_dir)
                try:
                    value = _read_json(source, monitor.args.run_dir)
                except FileNotFoundError:
                    if filename == "policy_metadata.json":
                        continue
                    raise
                _atomic_json(run / filename, value, run)
            _atomic_json(run / "request.json", dict(
                mode=monitor.args.mode, source_run_id=monitor.args.run_dir.name, save_request_id=request["requestId"],
                parameters=copy.deepcopy(getattr(monitor, "_workbench_parameters", {})),
                duration_seconds=0, snapshot=True), run)
            report = self.build(run, steps=steps, factory=factory, params=params, elapsed_seconds=elapsed_seconds, stage=stage)
            checkpoint = _inside(Path(report["checkpoint"]), run)
            export = _inside(Path(report["export_path"]), run)
            if checkpoint.parent != run / "checkpoints" or not checkpoint.is_dir() or not export.is_file():
                raise ValueError("Snapshot artifacts are incomplete or outside their run")
            if (report.get("all_parameters_finite") is not True or report.get("parameter_tree_max_abs_error") != 0 or
                    type(report.get("export_max_abs_error")) not in (int, float) or
                    not math.isfinite(report["export_max_abs_error"]) or report["export_max_abs_error"] > report.get("export_atol", -1)):
                raise ValueError("Snapshot verification did not pass")
            publisher = WorkbenchPersistence(run, monitor.args.mode, restore=monitor.args.restore)
            original_context = monitor._workbench.context
            if (publisher.context is None or original_context is None or
                    publisher.context["origin"] != original_context["origin"] or
                    publisher.context["parentCheckpoint"] != original_context["parentCheckpoint"]):
                raise ValueError("Snapshot cumulative baseline does not match its source run")
            report = publisher.publish_report({**basic, **report, "phase": "completed", "stage": "verified", "completed": True})
            if not report.get("workbench_checkpoint_id"):
                raise ValueError("Snapshot metadata publication failed")
            _atomic_json(run / "report.json", report, run)
            _atomic_json(run / "status.json", report, run)
            return self._result(report)
        except Exception:
            _atomic_json(run / "status.json", dict(basic, phase="failed", stage="manual_snapshot_failed"), run)
            raise

    @staticmethod
    def _result(report):
        return {"checkpoint": report["checkpoint"], "checkpointId": report["workbench_checkpoint_id"],
                "snapshotRunId": report["run_id"], "capturedSteps": report["actual_timesteps"],
                "capturedElapsedSeconds": report["training_stop_elapsed_seconds"]}


class WorkbenchRuntime:
    """Control intent is separate from a save queue; all writes stay under this run."""

    def __init__(self, monitor, *, clock=time.monotonic, sleep=time.sleep, poll_seconds=0.1):
        self.monitor, self.clock, self.sleep = monitor, clock, sleep
        self.run_dir = monitor._workbench.run_dir
        self.run_id = self.run_dir.name
        self.poll_seconds = poll_seconds
        self.paused = False
        self.desired_paused = False
        self._pause_start = None
        self._paused_seconds = 0.0
        self._last_poll = -math.inf
        self._intent_id = None
        self.control_result = None
        self.results = OrderedDict()
        self._attempted = set()
        self.warnings = []
        self.snapshot_writer = None

    def excluded_seconds(self):
        return self._paused_seconds + (max(0.0, self.clock() - self._pause_start) if self._pause_start is not None else 0.0)

    def reset_budget_clock(self):
        self._paused_seconds = 0.0
        self._pause_start = self.clock() if self.paused else None

    def state_fields(self):
        return {"workbench_control_protocol": PROTOCOL, "paused": self.paused,
                "pause_requested": self.desired_paused and not self.paused,
                "paused_seconds": self.excluded_seconds(),
                "workbench_control_result": copy.deepcopy(self.control_result),
                "workbench_save_results": copy.deepcopy(list(self.results.values())),
                "workbench_control_warnings": copy.deepcopy(self.warnings)}

    def _warning(self, code, message):
        value = {"code": code, "message": message}
        if value not in self.warnings and len(self.warnings) < 16:
            self.warnings.append(value)

    def poll_intent(self, *, force=False):
        now = self.clock()
        if not force and now - self._last_poll < self.poll_seconds:
            return False
        self._last_poll = now
        try:
            value = _request(_read_json(self.run_dir / "workbench-control.json", self.run_dir), self.run_id)
        except FileNotFoundError:
            return False
        except (ValueError, OSError, PersistenceError):
            self._warning("invalid_control_intent", "Control intent was invalid or unreadable and was ignored.")
            return False
        if value["requestId"] == self._intent_id:
            return False
        self._intent_id = value["requestId"]
        self.desired_paused = value["action"] == "pause"
        applied = self.desired_paused == self.paused
        self.control_result = {"requestId": value["requestId"], "action": value["action"],
                               "status": "applied" if applied else "pending"}
        return True

    def _queue(self):
        paths = []
        try:
            directory = _inside(self.run_dir / "workbench-save-requests", self.run_dir)
            if not directory.exists():
                return []
            for index, path in enumerate(directory.iterdir()):
                if index >= MAX_PENDING:
                    self._warning("save_queue_limit", "Save queue exceeds its pending-request limit.")
                    break
                if path.suffix != ".json":
                    continue
                try:
                    value = _request(_read_json(path, self.run_dir), self.run_id, "save")
                    if path.stem != value["requestId"]:
                        raise ValueError("Save filename identity mismatch")
                except (ValueError, OSError, PersistenceError):
                    self._warning("invalid_save_request", "An invalid or linked save request was ignored.")
                    continue
                paths.append((path, value))
        except (ValueError, OSError, PersistenceError):
            self._warning("save_queue_unavailable", "Save queue is unavailable or unsafe and was ignored.")
        return sorted(paths, key=lambda item: (str(item[1]["createdAt"]), item[1]["requestId"]))

    def _remember(self, result):
        self.results[result["requestId"]] = result
        self.results.move_to_end(result["requestId"])
        while len(self.results) > MAX_PENDING:
            self.results.popitem(last=False)

    def _receipt(self, path, result):
        directory = _inside(self.run_dir / "workbench-save-results", self.run_dir)
        directory.mkdir(exist_ok=True)
        _atomic_json(directory / (result["requestId"] + ".json"), result, self.run_dir)
        _inside(path, self.run_dir).unlink()

    def _save(self, path, request, *, steps, factory, params, elapsed_seconds):
        identity = request["requestId"]
        result_path = self.run_dir / "workbench-save-results" / (identity + ".json")
        try:
            existing = _read_json(result_path, self.run_dir)
        except FileNotFoundError:
            existing = None
        except (OSError, ValueError, PersistenceError):
            existing = None
        if isinstance(existing, dict) and existing.get("requestId") == identity and existing.get("runId") == self.run_id and existing.get("status") in ("completed", "failed"):
            self._remember(existing)
            _inside(path, self.run_dir).unlink()
            self._attempted.add(identity)
            return
        if identity in self._attempted:
            return
        self._attempted.add(identity)
        result = {"requestId": identity, "runId": self.run_id, "action": "save", "status": "saving", "stage": "checkpoint"}
        self._remember(result)
        self.monitor.publish()

        def stage(value):
            result["stage"] = value
            self.monitor.publish()

        try:
            if self.snapshot_writer is None:
                raise ValueError("Snapshot writer is not initialized")
            fields = self.snapshot_writer(request, steps=steps, factory=factory, params=params,
                                          elapsed_seconds=elapsed_seconds, stage=stage)
            result.update(fields, status="completed", stage="verified")
        except Exception:
            traceback.print_exc()
            result.update(status="failed", error={"code": "snapshot_failed", "message": "Checkpoint save or verification failed."})
        try:
            self._receipt(path, result)
        except (OSError, ValueError, PersistenceError):
            self._warning("save_receipt_unavailable", "Save result is visible but its durable receipt could not be published.")
        self.monitor.publish()

    def _drain(self, *, steps, factory, params, elapsed_seconds):
        for path, request in self._queue():
            self.monitor.check_stop()
            self._save(path, request, steps=steps, factory=factory, params=params, elapsed_seconds=elapsed_seconds)

    def boundary(self, steps, factory, params):
        self.monitor.check_stop()
        self.poll_intent(force=True)
        previous = (self.monitor.state.get("phase"), self.monitor.state.get("stage"))
        captured_elapsed = self.monitor.elapsed
        try:
            while True:
                if self.desired_paused and not self.paused:
                    self.paused = True
                    self._pause_start = self.clock()
                    if self.control_result:
                        self.control_result["status"] = "applied"
                    self.monitor.timing.set_phase("workbench.paused", steps=steps)
                    self.monitor.publish(phase="paused", stage="workbench_paused")
                self._drain(steps=steps, factory=factory, params=params, elapsed_seconds=captured_elapsed)
                self.poll_intent(force=True)
                if not self.desired_paused:
                    if self.control_result:
                        self.control_result["status"] = "applied"
                    break
                if not self.paused:
                    continue
                self.monitor.check_stop()
                self.sleep(self.poll_seconds)
                self.monitor.publish()
        finally:
            if self.paused:
                self._paused_seconds += max(0.0, self.clock() - self._pause_start)
                self._pause_start = None
                self.paused = False
                if self.monitor.state.get("phase") == "paused":
                    self.monitor.publish(phase=previous[0], stage=previous[1])
                else:
                    self.desired_paused = False
                    self.monitor.publish()
            elif self.control_result:
                self.monitor.publish()

    def finish(self):
        self.desired_paused = False
        if self.control_result and self.control_result.get("status") == "pending":
            self.control_result.update(status="failed", error={"code": "run_finished", "message": "Training finished before this intent reached a safe boundary."})
        for path, request in self._queue():
            if request["requestId"] in self._attempted:
                result = self.results.get(request["requestId"])
                if result and result.get("status") in ("completed", "failed"):
                    try:
                        self._receipt(path, result)
                    except (OSError, ValueError, PersistenceError):
                        self._warning("save_receipt_unavailable", "A terminal save receipt could not be published.")
                continue
            result = {"requestId": request["requestId"], "runId": self.run_id, "action": "save", "status": "failed",
                      "error": {"code": "run_finished", "message": "Training finished before this request reached a safe boundary."}}
            self._remember(result)
            try:
                self._receipt(path, result)
            except (OSError, ValueError, PersistenceError):
                self._warning("save_receipt_unavailable", "A terminal save receipt could not be published.")
        self.monitor.publish()
