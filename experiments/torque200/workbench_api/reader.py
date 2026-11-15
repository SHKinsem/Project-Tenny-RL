"""Bounded, read-only DTOs over console artifacts. Never establishes liveness."""

from __future__ import annotations

from datetime import datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import time

from .mode_catalog import MODES, mode_spec
from .checkpoint_metadata import MetadataError, calculate_progress, validate_checkpoint, validate_run_context
try:
    from ..workbench_storage import WorkbenchStorage
except ImportError:  # Existing WSL top-level package deployment.
    from workbench_storage import WorkbenchStorage


MAX_JSON_BYTES = 4 * 1024 * 1024
MAX_JSONL_BYTES = 32 * 1024 * 1024
MAX_LINE_BYTES = 256 * 1024
MAX_RESPONSE_READ_BYTES = 64 * 1024 * 1024
MAX_RECORDS = 100_000
MAX_DIRECTORY_ENTRIES = 2048
MAX_RUNS = 256
MAX_CHECKPOINTS = 4096
MAX_METRICS_LIMIT = 2000
MAX_CHECKPOINT_LIMIT = 1000
MAX_PROGRESS_RECORDS = 2000
CHECKPOINT_METADATA_SUFFIX = ".workbench.json"
MAX_SAFE_STEPS = 2**53 - 1
_RUN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}\Z")
_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


class WorkbenchError(Exception):
    """A deliberately path-free, transport-ready application error."""

    def __init__(self, code: str, status: int, message: str):
        super().__init__(message)
        self.code, self.status, self.message = code, status, message


def _too_large():
    return WorkbenchError("resource_too_large", 413, "Artifact query exceeds its bounded read limit.")


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        return value if math.isfinite(value) else None
    except OverflowError:
        return None


def _amount(value):
    value = _number(value)
    return value if value is not None and value >= 0 else None


def _count(value):
    return value if type(value) is int and value >= 0 else None


def _recorded_at(value):
    if not isinstance(value, dict):
        return None
    for key in ("recordedAtMs", "recorded_at_ms", "timestamp_ms"):
        amount = _amount(value.get(key))
        if amount is not None:
            return int(amount)
    for key in ("recorded_at", "updated_at", "timestamp"):
        text = value.get(key)
        if isinstance(text, str) and len(text) <= 80:
            try:
                stamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
                if stamp.tzinfo is not None:
                    result = stamp.timestamp() * 1000
                    if math.isfinite(result) and result >= 0:
                        return int(result)
            except (ValueError, OverflowError, OSError):
                pass
    return None


def _reject_constant(_value):
    raise ValueError("Non-finite JSON constant")


def _json(data):
    return json.loads(data.decode("utf-8-sig"), parse_constant=_reject_constant)


def _label(value):
    return value if isinstance(value, str) and _LABEL.fullmatch(value) else None


def _is_link(path):
    return path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())


class _Snapshot:
    def __init__(self, reader, scope):
        self.reader, self.scope = reader, scope
        self.read_at = int(time.time() * 1000)
        self.files = {}
        self.fingerprints = {}
        self.warnings = []
        self.warning_keys = {}
        self.read_bytes = 0

    def relative(self, path):
        for index, root in enumerate(self.reader.roots):
            try:
                relative = path.relative_to(root).as_posix()
                return relative if index == 0 else f"storage/{index}/{relative}"
            except ValueError:
                continue
        raise ValueError("Path is outside registered artifact roots")

    def warn(self, code, message, path=None):
        relative = self.relative(path) if path is not None else None
        key = (code, relative)
        if key in self.warning_keys:
            self.warning_keys[key]["count"] += 1
            return
        if len(self.warnings) >= 64:
            return
        item = {"code": code, "message": message, "count": 1}
        if relative is not None:
            item["path"] = relative
        self.warning_keys[key] = item
        self.warnings.append(item)

    def safe(self, path):
        try:
            root = next((root for root in self.reader.roots if path.is_relative_to(root)), None)
            if root is None:
                return False
            relative = path.relative_to(root)
            current = root
            if _is_link(current):
                return False
            for part in relative.parts:
                current = current / part
                if _is_link(current):
                    return False
            return path.resolve().is_relative_to(root)
        except (ValueError, OSError, RuntimeError):
            return False

    def entries(self, directory, metadata=True):
        name = self.relative(directory)
        if not self.safe(directory):
            self.warn("unsafe_path", "Linked or escaping artifact path was ignored.", directory)
            self.fingerprints[name] = "unsafe"
            return []
        result, signature = [], []
        try:
            with os.scandir(directory) as stream:
                for index, entry in enumerate(stream):
                    if index >= MAX_DIRECTORY_ENTRIES:
                        raise _too_large()
                    path = directory / entry.name
                    try:
                        meta = entry.stat(follow_symlinks=False)
                    except OSError:
                        self.warn("unreadable_entry", "An artifact entry could not be inspected.", directory)
                        continue
                    signature.append((entry.name, meta.st_mode, meta.st_size, meta.st_mtime_ns) if metadata
                                     else (entry.name, meta.st_mode))
                    if _is_link(path) or not self.safe(path):
                        self.warn("unsafe_path", "Linked or escaping artifact path was ignored.", path)
                        continue
                    result.append((path, meta))
        except FileNotFoundError:
            self.fingerprints[name] = "missing"
            return []
        except OSError:
            self.warn("unreadable_directory", "Artifact directory is unavailable.", directory)
            self.fingerprints[name] = "unreadable"
            return []
        self.fingerprints[name] = sorted(signature)
        return result

    def read(self, path, limit=MAX_JSON_BYTES, optional=True):
        name = self.relative(path)
        if not self.safe(path):
            self.warn("unsafe_path", "Linked or escaping artifact path was ignored.", path)
            self.fingerprints[name] = "unsafe"
            return None
        descriptor = None
        try:
            preliminary = path.lstat()
            if not stat.S_ISREG(preliminary.st_mode):
                raise OSError("Not a regular file")
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0)
                                 | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
            with os.fdopen(descriptor, "rb") as stream:
                descriptor = None
                before = os.fstat(stream.fileno())
                if not stat.S_ISREG(before.st_mode):
                    raise OSError("Not a regular file")
                if before.st_size > limit or self.read_bytes + before.st_size > MAX_RESPONSE_READ_BYTES:
                    raise _too_large()
                data = stream.read(limit + 1)
                after = os.fstat(stream.fileno())
            self.read_bytes += len(data)
            if len(data) > limit or self.read_bytes > MAX_RESPONSE_READ_BYTES:
                raise _too_large()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or not self.safe(path):
                self.warn("changed_during_read", "Artifact changed while being read; this file was omitted.", path)
                self.fingerprints[name] = "changed"
                return None
            digest = hashlib.sha256(data).hexdigest()
            self.fingerprints[name] = (after.st_size, after.st_mtime_ns, digest)
            self.files[name] = {"path": name, "sizeBytes": after.st_size,
                                "modifiedAtMs": after.st_mtime_ns // 1_000_000, "recordedAtMs": None}
            return data
        except FileNotFoundError:
            self.fingerprints[name] = "missing"
            if not optional:
                self.warn("missing_artifact", "Expected artifact is unavailable.", path)
            return None
        except OSError:
            self.fingerprints[name] = "unreadable"
            self.warn("unreadable_artifact", "Artifact is temporarily unreadable.", path)
            return None
        finally:
            if descriptor is not None:
                os.close(descriptor)

    def json(self, path, optional=True):
        data = self.read(path, optional=optional)
        if data is None:
            return None
        try:
            value = _json(data)
        except (ValueError, UnicodeError, RecursionError):
            self.warn("invalid_json", "Artifact JSON is invalid or incomplete.", path)
            return None
        self.files[self.relative(path)]["recordedAtMs"] = _recorded_at(value)
        return value

    def response(self, value, revision=None, kind="artifact_files"):
        digest = hashlib.sha256(json.dumps([self.scope, self.fingerprints], sort_keys=True,
                                          ensure_ascii=True).encode()).hexdigest()
        actual = "sha256:" + digest
        if revision is not None and revision != actual:
            raise WorkbenchError("revision_conflict", 409, "Artifacts changed; restart pagination from the first page.")
        return {"schemaVersion": 1, "revision": actual, "readAtMs": self.read_at,
                "source": {"kind": kind, "consistency": "per_file", "files": list(self.files.values())},
                "warnings": self.warnings, **value}


class WorkbenchReader:
    """Read console/<mode>/<run> under a configured artifact root, without writes."""

    def __init__(self, artifacts):
        candidate = Path(artifacts).absolute()
        if _is_link(candidate):
            raise WorkbenchError("unsafe_path", 400, "Artifact root must not be a filesystem link.")
        try:
            self.root = candidate.resolve()
        except (OSError, RuntimeError):
            raise WorkbenchError("unsafe_path", 400, "Artifact root cannot be resolved safely.") from None
        self.console = self.root / "console"
        self.console_roots = WorkbenchStorage(self.root).registered_roots()
        self.roots = (self.root, *(root for root in self.console_roots if not root.is_relative_to(self.root)))

    @staticmethod
    def _mode(mode):
        if not isinstance(mode, str) or mode not in MODES:
            raise WorkbenchError("invalid_mode", 400, "Unknown control mode.")
        return mode

    @staticmethod
    def _page(offset, limit, revision, maximum):
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= maximum:
            raise WorkbenchError("invalid_pagination", 400, "Invalid offset or limit.")
        if revision is not None and (not isinstance(revision, str) or not revision or len(revision) > 128):
            raise WorkbenchError("invalid_revision", 400, "Invalid revision token.")

    def _runs(self, context, mode):
        runs = [(path, meta) for root in self.console_roots
                for path, meta in context.entries(root / mode, metadata=False)
                if stat.S_ISDIR(meta.st_mode) and _RUN_NAME.fullmatch(path.name)]
        if len(runs) > MAX_RUNS:
            raise _too_large()
        return sorted((path for path, _ in runs), key=lambda path: path.name, reverse=True)

    def _reference(self, context, raw, run):
        if not isinstance(raw, str) or not raw or len(raw) > 2048 or "\x00" in raw:
            return None
        path = Path(raw)
        if not path.is_absolute():
            path = self.root / path if path.parts and path.parts[0] == "console" else run / path
        try:
            if not context.safe(path):
                return None
            path = path.resolve()
            if not path.is_relative_to(run) or path == run:
                return None
            return path
        except (OSError, ValueError, RuntimeError):
            return None

    def _records(self, context, run, mode):
        records = []
        for name in ("report.json", "status.json"):
            record = context.json(run / name)
            if record is None:
                records.append(None)
                continue
            if not isinstance(record, dict):
                context.warn("invalid_record", "Run record is not a JSON object.", run / name)
                records.append(None)
                continue
            if record.get("mode") not in (None, mode) or record.get("run_id") not in (None, run.name):
                context.warn("identity_mismatch", "Run record identity does not match its directory.", run / name)
                return None
            if record.get("run_dir") is not None:
                raw = Path(record["run_dir"]) if isinstance(record["run_dir"], str) else None
                expected = raw if raw is not None and raw.is_absolute() else (run.parent / raw if raw is not None else None)
                try:
                    valid = expected is not None and context.safe(expected) and expected.resolve() == run
                except (OSError, ValueError, RuntimeError):
                    valid = False
                if not valid:
                    context.warn("identity_mismatch", "Run record path does not match its directory.", run / name)
                    return None
            records.append(record)
        report, status_record = records
        return report, status_record, {**(report or {}), **(status_record or {})}

    def _select(self, context, mode, run_id=None):
        if run_id is not None:
            if not isinstance(run_id, str) or not _RUN_NAME.fullmatch(run_id):
                raise WorkbenchError("invalid_run_id", 400, "Run identifier must be a single safe directory name.")
            candidates = [root / mode / run_id for root in self.console_roots if (root / mode / run_id).exists()]
            if len(candidates) > 1:
                raise WorkbenchError("ambiguous_run_id", 409, "Run identifier exists in more than one registered location.")
            run = candidates[0] if candidates else self.console / mode / run_id
            if not context.safe(run):
                raise WorkbenchError("unsafe_path", 400, "Linked or escaping run paths are not allowed.")
            if not run.is_dir():
                raise WorkbenchError("run_not_found", 404, "Requested run is unavailable.")
            records = self._records(context, run, mode)
            if records is None:
                raise WorkbenchError("run_not_found", 404, "Requested run identity is unavailable.")
            return run, records
        active = context.json(self.console / "active.json")
        if isinstance(active, dict) and active.get("mode") == mode:
            raw = active.get("run_dir")
            if isinstance(raw, str):
                path = Path(raw)
                if not path.is_absolute():
                    path = self.root / path if path.parts and path.parts[0] == "console" else self.console / mode / path
                try:
                    safe = context.safe(path)
                    run = path.resolve() if safe else None
                    valid = run is not None and run.parent in {root / mode for root in self.console_roots} and run.is_dir() and _RUN_NAME.fullmatch(run.name)
                    valid = valid and active.get("run_id") in (None, run.name)
                except (OSError, ValueError, RuntimeError):
                    valid = False
                if valid:
                    records = self._records(context, run, mode)
                    if records is not None:
                        return run, records
            context.warn("invalid_active_reference", "Active record reference was ignored.", self.console / "active.json")
        for run in self._runs(context, mode):
            records = self._records(context, run, mode)
            if records is not None:
                return run, records
        return None, None

    @staticmethod
    def _pagination(offset, limit, total):
        end = min(offset + limit, total)
        return {"offset": offset, "limit": limit, "total": total,
                "nextOffset": end if end < total else None}

    def _run_context(self, context, run, mode, records):
        path = run / "workbench-run.json"
        value = context.json(path)
        # A report must not supply a stale binding for newer unbound counters.
        report, status_record, _ = records
        source = status_record if status_record is not None else (report or {})
        anchor = source.get("workbench_context_id")
        if value is None:
            if anchor is not None:
                context.warn("missing_run_context", "Bound run context is unavailable; cumulative history is unknown.", path)
            return None
        if anchor is None:
            context.warn("unbound_run_context", "Run context is not bound to the reported counters.", path)
            return None
        try:
            result = validate_run_context(value, run_id=run.name, mode=mode)
            if result["id"] != anchor:
                raise MetadataError("context_id_mismatch", "Run context does not match its reported binding.")
            if report and status_record is not None and report.get("workbench_context_id", anchor) != anchor:
                raise MetadataError("context_id_mismatch", "Run report and current counters have different bindings.")
            if source.get("clock_policy_id", result["clockPolicyId"]) != result["clockPolicyId"]:
                raise MetadataError("clock_policy_mismatch", "Counter and context timing policies differ.")
            if source.get("control_hz", result["controlHz"]) != result["controlHz"]:
                raise MetadataError("control_hz_mismatch", "Counter and context control frequencies differ.")
            return result
        except MetadataError as exc:
            context.warn("invalid_run_context", "Run context was rejected (" + exc.code + "); cumulative history is unknown.", path)
            return None

    def _checkpoint_metadata(self, context, run, mode, checkpoint):
        path = checkpoint.with_name(checkpoint.name + CHECKPOINT_METADATA_SUFFIX)
        value = context.json(path)
        if value is None:
            status = "missing" if context.fingerprints.get(context.relative(path)) == "missing" else "invalid"
            return None, status
        try:
            result = validate_checkpoint(value, mode=mode)
            if result["createdByRunId"] != run.name:
                raise MetadataError("checkpoint_run_mismatch", "Checkpoint metadata belongs to another run.")
            return result, "valid"
        except MetadataError as exc:
            context.warn("invalid_checkpoint_metadata", "Checkpoint metadata was rejected (" + exc.code + ").", path)
            return None, "invalid"

    def state(self, mode="original"):
        mode = self._mode(mode)
        context = _Snapshot(self, ["state", mode])
        run, records = self._select(context, mode)
        if run is None:
            context.warn("no_run", "No readable run directory is available.")
            return context.response({"available": False, "mode": mode, "run": None,
                                     "progress": {key: value for key, value in calculate_progress(
                                         None, elapsed_seconds=None, steps=None, duration_seconds=None).items() if key != "warnings"}})
        report, status_record, record = records
        run_context = self._run_context(context, run, mode, records)
        if run_context is not None:
            # Only matching producers may contribute counters to a bound run.
            matching_report = report if report and report.get("workbench_context_id") == run_context["id"] else None
            report = matching_report
            record = {**(matching_report or {}), **(status_record or {})}
        raw_elapsed = _amount(record.get("elapsed_seconds"))
        stop_elapsed = next((value for source in (status_record, report) if source
                             and (value := _amount(source.get("training_stop_elapsed_seconds"))) is not None), None)
        finalization = _amount(record.get("finalization_seconds"))
        started = record.get("training_clock_started")
        started = started if isinstance(started, bool) else None
        preparation = _amount(record.get("preparation_seconds"))
        phase = _label(record.get("phase"))
        elapsed = stop_elapsed if stop_elapsed is not None else raw_elapsed
        if (mode_spec(mode)["hybrid"] or mode_spec(mode).get("teacher")) and started is False:
            preparation = max(preparation or 0, raw_elapsed or 0)
            elapsed = 0
        elif stop_elapsed is None and finalization is not None and raw_elapsed is not None:
            elapsed = max(0, raw_elapsed - finalization)
        elif stop_elapsed is None and phase in {"saving", "exporting", "verifying", "completed", "stopped"}:
            elapsed = None
            context.warn("budget_time_unavailable", "Finalization cannot be separated from reported elapsed time.")
        duration = _amount(record.get("duration_seconds"))
        steps = _count(record.get("steps"))
        calculation_steps = steps
        if steps is not None and steps > MAX_SAFE_STEPS:
            calculation_steps = None
            context.warn("step_count_out_of_range", "Reported steps exceed the cumulative metadata integer range.")
        progress = calculate_progress(run_context, elapsed_seconds=elapsed,
                                      steps=calculation_steps, duration_seconds=duration)
        for warning in progress.pop("warnings", []):
            context.warn(warning["code"], warning["message"])
        dto = {"id": run.name, "mode": mode, "reportedPhase": phase,
               "reportedStage": _label(record.get("stage")), "active": None, "liveness": "unverified",
               "steps": steps, "elapsedSeconds": elapsed,
               "durationSeconds": duration, "preparationSeconds": preparation, "finalizationSeconds": finalization,
               "trainingClockStarted": started, **progress,
               "recordedAtMs": _recorded_at(status_record or report)}
        available = bool(status_record or report)
        if not available:
            context.warn("no_state_record", "Run directory has no readable status or report.")
        return context.response({"available": available, "mode": mode, "run": dto,
                                 "progress": progress})

    def checkpoints(self, mode="original", offset=0, limit=100, revision=None):
        mode = self._mode(mode)
        self._page(offset, limit, revision, MAX_CHECKPOINT_LIMIT)
        context = _Snapshot(self, ["checkpoints", mode])
        candidates = {}
        for run in self._runs(context, mode):
            records = self._records(context, run, mode)
            if records is None:
                continue
            for path, meta in context.entries(run / "checkpoints"):
                if path.name.endswith(CHECKPOINT_METADATA_SUFFIX):
                    continue
                if stat.S_ISDIR(meta.st_mode) or stat.S_ISREG(meta.st_mode):
                    candidates[path] = (run, meta)
            for record in records[:2]:
                if not record or not record.get("checkpoint"):
                    continue
                path = self._reference(context, record["checkpoint"], run)
                if path is None or path.name.endswith(CHECKPOINT_METADATA_SUFFIX):
                    context.warn("invalid_checkpoint_reference", "Checkpoint reference was outside its run or linked.", run)
                    continue
                try:
                    meta = path.stat()
                except OSError:
                    context.warn("missing_checkpoint_reference", "Referenced checkpoint is unavailable.", run)
                    continue
                if stat.S_ISDIR(meta.st_mode) or stat.S_ISREG(meta.st_mode):
                    candidates[path] = (run, meta)
                    context.fingerprints[context.relative(path)] = (meta.st_mode, meta.st_size, meta.st_mtime_ns)
            if len(candidates) > MAX_CHECKPOINTS:
                raise _too_large()
        items = []
        for path, (run, meta) in sorted(candidates.items(), key=lambda item: item[0].as_posix()):
            relative = context.relative(path)
            metadata, metadata_status = self._checkpoint_metadata(context, run, mode, path)
            items.append({"referenceId": "location-sha256:" + hashlib.sha256(relative.encode()).hexdigest(),
                          "relativePath": relative, "runId": run.name, "mode": mode,
                          "kind": "directory" if stat.S_ISDIR(meta.st_mode) else "file",
                          "modifiedAtMs": meta.st_mtime_ns // 1_000_000,
                          "id": None, "alias": None, "trainedSeconds": None, "trainedSteps": None,
                          **({key: metadata[key] for key in (
                              "id", "createdByRunId", "parentCheckpointId", "clockPolicyId",
                              "baseTrainedSeconds", "baseTrainedSteps", "capturedSeconds", "capturedSteps",
                              "trainedSeconds", "trainedSteps")} if metadata else {}),
                          "metadataStatus": metadata_status,
                          "verification": "unverified", "canResume": False})
        return context.response({"available": bool(items), "mode": mode, "items": items[offset:offset + limit],
                                 "pagination": self._pagination(offset, limit, len(items))}, revision)

    @staticmethod
    def _definitions():
        return [{"id": key, "unit": unit, "stepsScope": "segment"} for key, unit in (
            ("episode_seconds", "seconds"), ("episode_reward", None), ("value_loss", None), ("policy_loss", None))]

    def descriptor(self, mode="original"):
        """Describe saved semantics without importing robot or policy implementations."""
        mode = self._mode(mode)
        context = _Snapshot(self, ["descriptor", mode])
        run, records = self._select(context, mode)
        spec = mode_spec(mode)
        policy, config, ppo = {}, {}, {}
        if run is not None:
            loaded = [context.json(run / name) for name in ("policy_metadata.json", "env_config.json", "ppo_config.json")]
            policy, config, ppo = [value if isinstance(value, dict) else {} for value in loaded]
            if policy.get("mode") not in (None, mode) or policy.get("control_hz", spec["control_hz"]) != spec["control_hz"]:
                context.warn("descriptor_identity_mismatch", "Saved policy semantics do not match the selected mode.", run / "policy_metadata.json")
                policy = {}

        def text(value):
            return value if isinstance(value, str) and 0 < len(value) <= 512 and "\x00" not in value else None

        def labels(value):
            return list(value) if isinstance(value, list) and len(value) <= 256 and all(text(item) is not None for item in value) else None

        dt = _amount(config.get("ctrl_dt"))
        episode = _count(config.get("episode_length"))
        try:
            episode_seconds = _number(episode * dt) if episode is not None and dt == 1 / spec["control_hz"] else None
        except OverflowError:
            episode_seconds = None
            context.warn("descriptor_value_out_of_range", "Saved episode length cannot be represented as finite seconds.")
        dto = {"available": run is not None, "mode": mode, "runId": run.name if run else None,
               "robot": {"id": text(policy.get("robot_id")), "modelId": text(policy.get("model_id")),
                         "sourceFingerprint": text(policy.get("source_fingerprint"))},
               "task": {"id": text(policy.get("task_id")), "profile": text(policy.get("training_profile"))},
               "observation": {"version": text(policy.get("observation_version")),
                               "actorDim": _count(policy.get("actor_dim")) if policy.get("actor_dim") is not None else spec["obs_dim"],
                               "criticDim": _count(policy.get("critic_dim"))},
               "action": {"mode": text(policy.get("action_mode")) or spec["action_mode"],
                          "order": labels(policy.get("action_order")), "units": labels(policy.get("action_units"))},
               "timing": {"controlHz": spec["control_hz"], "physicsHz": _count(policy.get("physics_hz")),
                          "episodeSeconds": episode_seconds},
               "runtime": {"numEnvs": _count(ppo.get("num_envs")), "batchSize": _count(ppo.get("batch_size")),
                           "numMinibatches": _count(ppo.get("num_minibatches")), "unrollLength": _count(ppo.get("unroll_length")),
                           "numUpdatesPerBatch": _count(ppo.get("num_updates_per_batch")),
                           "learningRate": _amount(ppo.get("learning_rate")), "numEvalEnvs": _count(ppo.get("num_eval_envs")),
                           "discounting": _amount(ppo.get("discounting")), "gaeLambda": _amount(ppo.get("gae_lambda")),
                           "clippingEpsilon": _amount(ppo.get("clipping_epsilon"))}}
        return context.response(dto)

    def _metric(self, event, sequence, dt, context, path, gap=None):
        values = {item["id"]: None for item in self._definitions()}
        result = {"sequence": sequence, "steps": None, "elapsedSeconds": None,
                  "recordedAtMs": None, "values": values, "gap": gap}
        if not isinstance(event, dict):
            result["gap"] = {"code": "invalid_record", "message": "Metric record is not a JSON object."}
            return result
        result.update(steps=_count(event.get("steps")), elapsedSeconds=_amount(event.get("elapsed_seconds")),
                      recordedAtMs=_recorded_at(event))
        metrics = event.get("metrics")
        if not isinstance(metrics, dict):
            result["gap"] = {"code": "invalid_metrics", "message": "Metric values are missing or invalid."}
            return result
        fields = {"episode_reward": "eval/episode_reward", "value_loss": "training/v_loss", "policy_loss": "training/policy_loss"}
        for key, source in fields.items():
            values[key] = _number(metrics.get(source))
        length = _number(metrics.get("eval/avg_episode_length"))
        seconds = length * dt if length is not None and dt is not None else None
        values["episode_seconds"] = _number(seconds)
        if result["steps"] is None:
            result["gap"] = {"code": "invalid_steps", "message": "Metric record has no valid segment step count."}
        return result

    def metrics(self, mode="original", run_id=None, offset=0, limit=500, revision=None):
        mode = self._mode(mode)
        self._page(offset, limit, revision, MAX_METRICS_LIMIT)
        context = _Snapshot(self, ["metrics", mode, run_id])
        run, records = self._select(context, mode, run_id)
        if run is not None:
            context.scope = ["metrics", mode, run.name]
            prefix = context.relative(run) + "/"
            context.fingerprints = {key: value for key, value in context.fingerprints.items() if key.startswith(prefix)}
            context.files = {key: value for key, value in context.files.items() if key.startswith(prefix)}
        base = {"available": False, "mode": mode, "runId": run.name if run else None, "items": [],
                "definitions": self._definitions(), "pagination": self._pagination(offset, limit, 0),
                "historyCompleteness": "unknown", "historyScope": "source_file", "producerHistoryComplete": None, "rawSource": None}
        if run is None:
            context.warn("no_run", "No readable run directory is available.")
            return context.response(base, revision)
        config = context.json(run / "env_config.json")
        dt = _number(config.get("ctrl_dt")) if isinstance(config, dict) else None
        if dt != 1 / mode_spec(mode)["control_hz"]:
            context.warn("control_timestep_unverified", "Episode length cannot be converted without a matching control timestep.", run / "env_config.json")
            dt = None
        path = run / "metrics.jsonl"
        data = context.read(path, MAX_JSONL_BYTES)
        items, raw_source, completeness = [], None, "unknown"
        if data is not None:
            raw_source, completeness = "metrics.jsonl", "full"
            stream = io.BytesIO(data)
            for index in range(1, MAX_RECORDS + 2):
                line = stream.readline(MAX_LINE_BYTES + 1)
                if not line:
                    break
                if index > MAX_RECORDS or len(line) > MAX_LINE_BYTES:
                    raise _too_large()
                terminated = line.endswith(b"\n")
                gap = None
                if not terminated:
                    context.warn("unterminated_record", "Last JSONL record has no final newline and may still be writing.", path)
                    gap = {"code": "unterminated_record", "message": "Last record has no final newline."}
                try:
                    event = _json(line)
                except (ValueError, UnicodeError, RecursionError):
                    code = "invalid_jsonl_record" if terminated else "partial_record"
                    context.warn(code, "A damaged metric record was retained as a gap.", path)
                    item = self._metric({}, index, dt, context, path)
                    item["gap"] = {"code": code, "message": "Metric record is invalid or incomplete."}
                else:
                    item = self._metric(event, index, dt, context, path, gap)
                items.append(item)
        elif context.fingerprints.get(context.relative(path)) == "missing":
            path = run / "progress.json"
            events = context.json(path)
            if events is not None:
                if not isinstance(events, list):
                    context.warn("invalid_progress", "Progress window is not an array.", path)
                else:
                    if len(events) > MAX_PROGRESS_RECORDS:
                        raise _too_large()
                    raw_source, completeness = "progress.json", "window"
                    context.warn("window_only", "Progress is a merged bounded window, not full callback history.", path)
                    items = [self._metric(event, index, dt, context, path) for index, event in enumerate(events, 1)]
        if raw_source is None:
            context.warn("metrics_unavailable", "No readable metric source is available.", run)
        base.update(available=bool(items), items=items[offset:offset + limit], rawSource=raw_source,
                    historyCompleteness=completeness, pagination=self._pagination(offset, limit, len(items)))
        return context.response(base, revision)

    def capabilities(self):
        context = _Snapshot(self, ["capabilities", 1, MODES])
        return context.response({"readOnly": True, "modes": list(MODES),
                                 "supported": ["state", "checkpoints", "metrics", "descriptor", "capabilities", "checkpointMetadata", "cumulativeProgress"],
                                 "checkpointMetadataSchemaVersion": 1, "runContextSchemaVersion": 1,
                                 "unsupported": ["startTraining", "pauseTraining", "resumeTraining", "saveSnapshot", "stopAndSave",
                                                 "resumeFromSelectedCheckpoint", "migrateCumulativeHistory", "openDirectory", "importCheckpoint", "connectPrototypeUI"],
                                 "limits": {"jsonBytes": MAX_JSON_BYTES, "jsonlBytes": MAX_JSONL_BYTES,
                                            "lineBytes": MAX_LINE_BYTES, "records": MAX_RECORDS,
                                            "responseReadBytes": MAX_RESPONSE_READ_BYTES, "directoryEntries": MAX_DIRECTORY_ENTRIES,
                                            "runs": MAX_RUNS, "checkpointCandidates": MAX_CHECKPOINTS,
                                            "metricsPageLimit": MAX_METRICS_LIMIT, "checkpointPageLimit": MAX_CHECKPOINT_LIMIT}}, kind="static_contract")
