"""Publish cumulative metadata beside saved weights, outside fingerprinted code."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import stat
from uuid import UUID, uuid4, uuid5

if __package__:
    from .control_modes import mode_spec
    from .workbench_api.checkpoint_metadata import (
        MetadataError, capture_checkpoint, make_run_context,
        validate_checkpoint, validate_run_context,
    )
else:
    from control_modes import mode_spec
    from workbench_api.checkpoint_metadata import (
        MetadataError, capture_checkpoint, make_run_context,
        validate_checkpoint, validate_run_context,
    )


MAX_METADATA_BYTES = 64 * 1024


class PersistenceError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code, self.message = code, message


def clock_policy_id(mode):
    spec = mode_spec(mode)
    return "console-ready-budget-v1" if spec["teacher"] or spec["hybrid"] else "console-entry-budget-v1"


def _linked(path):
    return path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())


def _inside(path, root):
    path, root = Path(path).absolute(), Path(root).absolute()
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        raise PersistenceError("unsafe_metadata_path", "Metadata path is outside its allowed directory.") from None
    if ".." in parts or _linked(root):
        raise PersistenceError("unsafe_metadata_path", "Metadata paths must not traverse links or parents.")
    current = root
    for part in parts:
        current = current / part
        if _linked(current):
            raise PersistenceError("unsafe_metadata_path", "Metadata paths must not traverse filesystem links.")
    try:
        if not path.resolve().is_relative_to(root.resolve()):
            raise PersistenceError("unsafe_metadata_path", "Metadata path escapes its allowed directory.")
    except (OSError, RuntimeError):
        raise PersistenceError("unsafe_metadata_path", "Metadata path cannot be resolved safely.") from None
    return path


def _reject_constant(_value):
    raise ValueError("Nonfinite metadata")


def _object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate metadata field")
        value[key] = item
    return value


def _read_json(path, root):
    path = _inside(path, root)
    preliminary = path.lstat()
    if not stat.S_ISREG(preliminary.st_mode) or preliminary.st_size > MAX_METADATA_BYTES:
        raise PersistenceError("invalid_metadata_file", "Metadata file is not a bounded regular file.")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_METADATA_BYTES:
            raise PersistenceError("invalid_metadata_file", "Metadata file is not a bounded regular file.")
        data = stream.read(MAX_METADATA_BYTES + 1)
        after = os.fstat(stream.fileno())
    _inside(path, root)
    if len(data) > MAX_METADATA_BYTES or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise PersistenceError("metadata_changed", "Metadata changed while being read.")
    try:
        return json.loads(data.decode("utf-8-sig"), parse_constant=_reject_constant, object_pairs_hook=_object)
    except (ValueError, UnicodeError, RecursionError):
        raise PersistenceError("invalid_metadata_json", "Metadata JSON is invalid or incomplete.") from None


def _atomic_json(path, value, root):
    path = _inside(path, root)
    if not path.parent.is_dir():
        raise PersistenceError("missing_metadata_directory", "Metadata directory does not exist.")
    if path.exists() and not path.is_file():
        raise PersistenceError("invalid_metadata_file", "Metadata destination is not a regular file.")
    payload = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf8")
    if len(payload) > MAX_METADATA_BYTES:
        raise PersistenceError("metadata_too_large", "Metadata exceeds its bounded size.")
    temporary = path.with_name("." + path.name + "." + uuid4().hex + ".tmp")
    descriptor = None
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(_inside(temporary, root), flags, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = None
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _inside(path, root)
        os.replace(temporary, path)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            _inside(temporary, root)
            if temporary.exists() and not _linked(temporary):
                temporary.unlink()
        except (PersistenceError, OSError):
            pass


class WorkbenchPersistence:
    """Single-run metadata publisher; metadata failures never claim weight verification."""

    def __init__(self, run_dir, mode, restore=None):
        self.run_dir = Path(run_dir).absolute()
        self.mode = mode
        self.policy_id = clock_policy_id(mode)
        self.context = None
        self.warnings = []
        self._start(restore)

    def _warn(self, code, message):
        warning = {"code": code, "message": message}
        if warning not in self.warnings and len(self.warnings) < 32:
            self.warnings.append(warning)

    def _start(self, restore):
        try:
            _inside(self.run_dir, self.run_dir)
            if not self.run_dir.is_dir():
                raise PersistenceError("missing_run_directory", "Run directory does not exist.")
            origin, parent = "new", None
            if restore is not None:
                origin = "legacy"
                checkpoint = Path(restore).absolute()
                sidecar = checkpoint.with_name(checkpoint.name + ".workbench.json")
                try:
                    _inside(checkpoint, checkpoint.parent)
                    if not checkpoint.is_dir():
                        raise PersistenceError("invalid_parent_checkpoint", "Parent checkpoint is unavailable.")
                    parent = validate_checkpoint(_read_json(sidecar, checkpoint.parent), mode=self.mode)
                    if parent["clockPolicyId"] != self.policy_id:
                        raise PersistenceError("parent_clock_policy_mismatch", "Parent checkpoint uses a different clock policy.")
                    origin = "checkpoint"
                except FileNotFoundError:
                    parent = None
                    self._warn("parent_metadata_missing", "Parent checkpoint has no cumulative metadata; historical counters remain unknown.")
                except (MetadataError, PersistenceError, OSError, ValueError):
                    parent = None
                    self._warn("parent_metadata_invalid", "Parent cumulative metadata is invalid or incompatible; historical counters remain unknown.")
            path = self.run_dir / "workbench-run.json"
            try:
                existing = validate_run_context(_read_json(path, self.run_dir), run_id=self.run_dir.name, mode=self.mode)
            except FileNotFoundError:
                existing = None
            if existing is not None:
                if (existing["clockPolicyId"], existing["origin"], existing["parentCheckpoint"]) != (self.policy_id, origin, parent):
                    raise PersistenceError("context_conflict", "Existing run context does not match the requested parent or clock policy.")
                self.context = existing
            else:
                candidate = make_run_context(run_id=self.run_dir.name, mode=self.mode, clock_policy_id=self.policy_id,
                                             origin=origin, parent=parent)
                _atomic_json(path, candidate, self.run_dir)
                self.context = candidate
        except (MetadataError, PersistenceError, OSError, ValueError):
            self.context = None
            self._warn("context_metadata_unavailable", "Cumulative run metadata could not be initialized; historical counters are unavailable.")

    def state_fields(self):
        return {"workbench_context_id": self.context["id"] if self.context else None,
                "clock_policy_id": self.policy_id,
                "workbench_metadata_status": "unavailable" if self.context is None else "warning" if self.warnings else "ready",
                "workbench_metadata_warnings": copy.deepcopy(self.warnings)}

    def bind_record(self, record):
        return dict(record, **self.state_fields())

    def publish_checkpoint(self, checkpoint, *, elapsed_seconds, steps):
        if self.context is None:
            self._warn("checkpoint_metadata_unavailable", "Checkpoint weights were kept without cumulative metadata.")
            return None
        try:
            checkpoint = _inside(Path(checkpoint).absolute(), self.run_dir)
            expected_parent = self.run_dir / "checkpoints"
            if checkpoint.parent != expected_parent or not checkpoint.name.isdigit() or not checkpoint.is_dir():
                raise PersistenceError("invalid_checkpoint_location", "Checkpoint must be an existing direct child of this run's checkpoint directory.")
            sidecar = checkpoint.with_name(checkpoint.name + ".workbench.json")
            try:
                existing = validate_checkpoint(_read_json(sidecar, self.run_dir), mode=self.mode)
            except FileNotFoundError:
                existing = None
            identity = existing["id"] if existing else str(uuid5(UUID(self.context["id"]), checkpoint.relative_to(self.run_dir).as_posix()))
            candidate = capture_checkpoint(self.context, elapsed_seconds=elapsed_seconds, steps=steps, checkpoint_id=identity)
            if existing is not None:
                fixed = ("createdByRunId", "mode", "controlHz", "clockPolicyId", "parentCheckpointId", "baseTrainedSeconds", "baseTrainedSteps")
                if any(existing[field] != candidate[field] for field in fixed):
                    raise PersistenceError("checkpoint_metadata_conflict", "Existing checkpoint metadata belongs to a different run baseline.")
                for field in ("capturedSeconds", "capturedSteps"):
                    if existing[field] is not None and (candidate[field] is None or candidate[field] < existing[field]):
                        raise PersistenceError("checkpoint_counter_regression", "Checkpoint captured counters must not move backwards.")
                if candidate == existing:
                    return copy.deepcopy(existing)
            _atomic_json(sidecar, candidate, self.run_dir)
            return copy.deepcopy(candidate)
        except (MetadataError, PersistenceError, OSError, ValueError):
            self._warn("checkpoint_metadata_failed", "Checkpoint weights were preserved, but cumulative metadata publication failed.")
            return None

    def publish_report(self, report):
        report = dict(report)
        report.pop("workbench_checkpoint_id", None)
        if report.get("checkpoint") is not None:
            if report.get("training_stop_elapsed_seconds") is None:
                self._warn("final_capture_time_unavailable", "Final cumulative metadata was not updated because the training cutoff is unavailable.")
            else:
                metadata = self.publish_checkpoint(report["checkpoint"], elapsed_seconds=report["training_stop_elapsed_seconds"],
                                                   steps=report.get("actual_timesteps", report.get("steps")))
                if metadata is not None:
                    report["workbench_checkpoint_id"] = metadata["id"]
        return self.bind_record(report)
