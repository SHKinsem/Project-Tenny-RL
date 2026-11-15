"""Pure versioned checkpoint metadata and continuation arithmetic; no I/O."""

from __future__ import annotations

import math
import re
from uuid import UUID, uuid4

from .mode_catalog import MODES, mode_spec


SCHEMA_VERSION = 1
MAX_SAFE_STEPS = 2**53 - 1
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}\Z")
_POLICY_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")
_CONTEXT_FIELDS = ("schemaVersion", "id", "runId", "mode", "controlHz", "clockPolicyId", "origin", "parentCheckpoint")
_CHECKPOINT_FIELDS = (
    "schemaVersion", "id", "createdByRunId", "mode", "controlHz", "clockPolicyId", "parentCheckpointId",
    "baseTrainedSeconds", "baseTrainedSteps", "capturedSeconds", "capturedSteps", "trainedSeconds", "trainedSteps",
)


class MetadataError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


def _error(code, message):
    raise MetadataError(code, message)


def _version_and_fields(value, fields):
    if not isinstance(value, dict):
        _error("invalid_metadata", "Metadata must be a JSON object.")
    if type(value.get("schemaVersion")) is not int or value["schemaVersion"] != SCHEMA_VERSION:
        _error("unsupported_schema_version", "Metadata schema version is unsupported.")
    if any(field not in value for field in fields):
        _error("missing_metadata_field", "Required metadata fields are missing; unknown counters must be explicit nulls.")


def _uuid(value):
    if not isinstance(value, str):
        _error("invalid_identity", "Persistent identity must be a canonical UUID string.")
    try:
        parsed = UUID(value)
        canonical = str(parsed)
    except (ValueError, AttributeError, TypeError):
        _error("invalid_identity", "Persistent identity must be a canonical UUID string.")
    if canonical != value:
        _error("invalid_identity", "Persistent identity must use canonical lowercase UUID form.")
    if parsed.int == 0:
        _error("invalid_identity", "Persistent identity must not use the nil UUID; unknown parents use null.")
    return canonical


def _run_id(value):
    if not isinstance(value, str) or not _RUN_ID.fullmatch(value):
        _error("invalid_run_id", "Run identity must be one safe directory name.")
    return value


def _mode(value):
    if not isinstance(value, str) or value not in MODES:
        _error("invalid_mode", "Unknown control mode.")
    return value


def _policy(value):
    if not isinstance(value, str) or not _POLICY_ID.fullmatch(value):
        _error("invalid_clock_policy", "Clock policy identity must be a safe short token.")
    return value


def _finite(value):
    try:
        return math.isfinite(value)
    except (OverflowError, TypeError, ValueError):
        return False


def _seconds(value):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not _finite(value) or value < 0:
        _error("invalid_seconds", "Seconds must be a nonnegative finite JSON number or null.")
    return value


def _steps(value):
    if value is None:
        return None
    if type(value) is not int or not 0 <= value <= MAX_SAFE_STEPS:
        _error("invalid_steps", "Steps must be a nonnegative JavaScript-safe integer or null.")
    return value


def _header(value, expected_mode=None):
    mode = _mode(value["mode"])
    if expected_mode is not None and mode != _mode(expected_mode):
        _error("mode_mismatch", "Metadata mode does not match the requested mode.")
    hz = value["controlHz"]
    if type(hz) is not int or hz != mode_spec(mode)["control_hz"]:
        _error("control_hz_mismatch", "Control frequency does not match the control mode.")
    return mode, hz, _policy(value["clockPolicyId"])


def _sum_strict(base, added, *, steps=False):
    if base is None or added is None:
        return None
    total = base + added
    if not _finite(total) or (steps and total > MAX_SAFE_STEPS):
        _error("derived_total_out_of_range", "Cumulative metadata exceeds its supported numeric range.")
    return total


def _same_seconds(actual, expected):
    if actual == expected:
        return True
    if actual is None or expected is None or expected == 0 or actual == 0:
        return False
    return abs(actual - expected) <= max(math.ulp(actual), math.ulp(expected)) * 2


def validate_checkpoint(value, *, mode=None):
    """Validate a flat metadata DTO; unknown time and step histories propagate separately."""
    _version_and_fields(value, _CHECKPOINT_FIELDS)
    actual_mode, hz, policy = _header(value, mode)
    identity = _uuid(value["id"])
    parent_id = _uuid(value["parentCheckpointId"]) if value["parentCheckpointId"] is not None else None
    if parent_id == identity:
        _error("self_parent", "A checkpoint cannot be its own parent.")
    result = {"schemaVersion": SCHEMA_VERSION, "id": identity,
              "createdByRunId": _run_id(value["createdByRunId"]), "mode": actual_mode,
              "controlHz": hz, "clockPolicyId": policy, "parentCheckpointId": parent_id}
    for field in ("baseTrainedSeconds", "capturedSeconds", "trainedSeconds"):
        result[field] = _seconds(value[field])
    for field in ("baseTrainedSteps", "capturedSteps", "trainedSteps"):
        result[field] = _steps(value[field])
    expected_seconds = _sum_strict(result["baseTrainedSeconds"], result["capturedSeconds"])
    expected_steps = _sum_strict(result["baseTrainedSteps"], result["capturedSteps"], steps=True)
    if not _same_seconds(result["trainedSeconds"], expected_seconds) or result["trainedSteps"] != expected_steps:
        _error("inconsistent_totals", "Cumulative counters must equal fixed base plus this capture, with unknowns preserved.")
    return result


def validate_run_context(value, *, run_id=None, mode=None):
    """Return an independent normalized context, without following external references."""
    _version_and_fields(value, _CONTEXT_FIELDS)
    actual_mode, hz, policy = _header(value, mode)
    actual_run = _run_id(value["runId"])
    if run_id is not None and actual_run != _run_id(run_id):
        _error("run_id_mismatch", "Context identity does not match the requested run.")
    origin = value["origin"]
    if origin not in ("new", "checkpoint", "legacy"):
        _error("invalid_origin", "Context origin must be new, checkpoint, or legacy.")
    parent = value["parentCheckpoint"]
    if origin == "checkpoint":
        if parent is None:
            _error("missing_parent", "Checkpoint continuation requires parent metadata.")
        parent = validate_checkpoint(parent, mode=actual_mode)
        if parent["clockPolicyId"] != policy:
            _error("clock_policy_mismatch", "Parent and run must use the same clock policy.")
    elif parent is not None:
        _error("unexpected_parent", "New and legacy contexts cannot attach parent metadata.")
    return {"schemaVersion": SCHEMA_VERSION, "id": _uuid(value["id"]), "runId": actual_run,
            "mode": actual_mode, "controlHz": hz, "clockPolicyId": policy,
            "origin": origin, "parentCheckpoint": parent}


def make_run_context(*, run_id, mode, clock_policy_id, origin="new", parent=None, context_id=None):
    mode = _mode(mode)
    return validate_run_context({"schemaVersion": SCHEMA_VERSION,
                                 "id": str(uuid4()) if context_id is None else context_id,
                                 "runId": run_id, "mode": mode, "controlHz": mode_spec(mode)["control_hz"],
                                 "clockPolicyId": clock_policy_id, "origin": origin, "parentCheckpoint": parent})


def _history(context):
    if context is None or context["origin"] == "legacy":
        return None, None
    if context["origin"] == "new":
        return 0, 0
    parent = context["parentCheckpoint"]
    return parent["trainedSeconds"], parent["trainedSteps"]


def capture_checkpoint(context, *, elapsed_seconds, steps, checkpoint_id=None):
    """Every capture derives from the same immutable run baseline, never a prior capture."""
    context = validate_run_context(context)
    elapsed, captured_steps = _seconds(elapsed_seconds), _steps(steps)
    base_seconds, base_steps = _history(context)
    parent = context["parentCheckpoint"]
    result = {"schemaVersion": SCHEMA_VERSION, "id": str(uuid4()) if checkpoint_id is None else checkpoint_id,
              "createdByRunId": context["runId"], "mode": context["mode"], "controlHz": context["controlHz"],
              "clockPolicyId": context["clockPolicyId"], "parentCheckpointId": parent["id"] if parent else None,
              "baseTrainedSeconds": base_seconds, "baseTrainedSteps": base_steps,
              "capturedSeconds": elapsed, "capturedSteps": captured_steps,
              "trainedSeconds": _sum_strict(base_seconds, elapsed),
              "trainedSteps": _sum_strict(base_steps, captured_steps, steps=True)}
    return validate_checkpoint(result, mode=context["mode"])


def _derived_warning(warnings, field):
    warnings.append({"code": "derived_value_out_of_range",
                     "message": f"Derived {field} exceeds its supported finite JSON range."})


def _sum_progress(base, added, warnings, field, *, steps=False):
    if base is None or added is None:
        return None
    total = base + added
    if not _finite(total) or (steps and total > MAX_SAFE_STEPS):
        _derived_warning(warnings, field)
        return None
    return total


def _percent(numerator, denominator, warnings, field):
    if numerator is None or denominator is None or denominator <= 0:
        return None
    try:
        result = numerator / denominator * 100
    except (OverflowError, ZeroDivisionError):
        result = None
    if result is None or not _finite(result):
        _derived_warning(warnings, field)
        return None
    return result


def calculate_progress(context, *, elapsed_seconds, steps, duration_seconds):
    """Calculate reported-counter progress; no wall-clock extrapolation or clamping."""
    context = validate_run_context(context) if context is not None else None
    elapsed, added_steps, duration = _seconds(elapsed_seconds), _steps(steps), _seconds(duration_seconds)
    base_seconds, base_steps = _history(context)
    warnings = []
    trained_seconds = _sum_progress(base_seconds, elapsed, warnings, "trainedSeconds")
    trained_steps = _sum_progress(base_steps, added_steps, warnings, "trainedSteps", steps=True)
    target = _sum_progress(base_seconds, duration, warnings, "targetTrainedSeconds") if duration else None
    remaining = max(0, duration - elapsed) if duration and elapsed is not None else None
    parent = context["parentCheckpoint"] if context else None
    return {"baseTrainedSeconds": base_seconds, "baseTrainedSteps": base_steps,
            "trainedSeconds": trained_seconds, "trainedSteps": trained_steps,
            "targetTrainedSeconds": target, "remainingSeconds": remaining,
            "segmentPercent": _percent(elapsed, duration, warnings, "segmentPercent") if duration else None,
            "cumulativePercent": _percent(trained_seconds, target, warnings, "cumulativePercent") if duration else None,
            "sourceCheckpointId": parent["id"] if parent else None,
            "contextId": context["id"] if context else None,
            "clockPolicyId": context["clockPolicyId"] if context else None, "warnings": warnings}
