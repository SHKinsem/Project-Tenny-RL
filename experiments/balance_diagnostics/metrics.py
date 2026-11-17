"""Pure SI-unit analysis of one frozen policy's first-episode trajectory.

Samples include the reset at t=0 and each post-control-step state. State RMS
uses trapezoidal time integration; reward and command occupancy belong to the
interval ending at their sample. Windows use recorded samples, never invented
states. Threshold events describe measurements, not success or a causal model.
"""
from bisect import bisect_right
from collections.abc import Mapping
import math


DEFAULT_THRESHOLDS = {
    "displacement_m": 0.05,
    "pitch_error_deg": 10.0,
    "horizontal_speed_m_s": 0.1,
    "persistence_s": 0.2,
}


def _number(value, label, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    try:
        value = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{label} must be finite" + (" and positive" if positive else ""))
    return value


def _sequence(value, label, count=None):
    if not isinstance(value, (list, tuple)) or (count is not None and len(value) != count):
        raise ValueError(f"{label} must be a list" + (f" of length {count}" if count is not None else ""))
    return value


def _validated(trace):
    if not isinstance(trace, Mapping) or type(trace.get("schema_version")) is not int or trace["schema_version"] != 1:
        raise ValueError("Expected trace schema_version=1")
    times = [_number(value, "t_s") for value in _sequence(trace.get("t_s"), "t_s")]
    count = len(times)
    if count < 2 or times[0] != 0 or any(right <= left for left, right in zip(times, times[1:])):
        raise ValueError("t_s must start at zero and strictly increase, with at least two samples")
    metadata = trace.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a mapping")
    reference = _number(metadata.get("reference_pitch_rad"), "reference_pitch_rad")
    limits = [_number(value, "torque_limits_nm", positive=True)
              for value in _sequence(metadata.get("torque_limits_nm"), "torque_limits_nm")]
    if not limits:
        raise ValueError("torque_limits_nm must not be empty")
    data = {"t_s": times}
    for name in ("pitch_rad", "roll_rad", "pitch_rate_rad_s", "reward"):
        data[name] = [_number(value, name) for value in _sequence(trace.get(name), name, count)]
    for name, width in (("base_position_m", 3), ("base_velocity_m_s", 3), ("actuator_command_nm", len(limits))):
        data[name] = [[_number(value, name) for value in _sequence(row, name, width)]
                      for row in _sequence(trace.get(name), name, count)]
    done = _sequence(trace.get("done"), "done", count)
    if any(type(value) is not bool for value in done):
        raise ValueError("done must contain booleans")
    if done[0] or data["reward"][0] != 0:
        raise ValueError("Reset sample must have done=False and reward=0")
    if any(done[:-1]):
        raise ValueError("Trace includes samples after its first done; split at the first episode")
    termination = trace.get("termination")
    if not isinstance(termination, Mapping):
        raise ValueError("termination must be a mapping")
    flags = [termination.get(name) for name in ("terminated", "truncated")]
    if any(type(value) is not bool for value in flags) or sum(flags) != 1:
        raise ValueError("Exactly one of terminated/truncated must be true")
    if not done[-1]:
        raise ValueError("Final sample must mark the episode done, including a time-limit truncation")
    end_time = _number(termination.get("time_s"), "termination.time_s")
    if not math.isclose(end_time, times[-1], rel_tol=0, abs_tol=1e-9):
        raise ValueError("termination.time_s must equal the last sample time")
    if not isinstance(termination.get("reason"), str) or not termination["reason"]:
        raise ValueError("termination.reason must be a nonempty string")
    data["displacement"] = [math.hypot(row[0] - data["base_position_m"][0][0],
                                      row[1] - data["base_position_m"][0][1])
                            for row in data["base_position_m"]]
    data["speed"] = [math.hypot(row[0], row[1]) for row in data["base_velocity_m_s"]]
    data["pitch_error"] = [math.degrees(value - reference) for value in data["pitch_rad"]]
    return data, limits, dict(termination)


def _rms(times, values):
    duration = times[-1] - times[0]
    if duration == 0:
        return abs(values[0])
    integral = math.fsum((right - left) * (a*a + b*b) / 2
                         for left, right, a, b in zip(times, times[1:], values, values[1:]))
    return math.sqrt(integral / duration)


def _measure(data, limits, start, end):
    """Inclusive recorded sample range; position reference remains the reset."""
    times = data["t_s"][start:end+1]
    duration = times[-1] - times[0]
    positions = data["base_position_m"][start:end+1]
    displacement = data["displacement"][start:end+1]
    speed = data["speed"][start:end+1]
    error = data["pitch_error"][start:end+1]
    rates = data["pitch_rate_rad_s"][start:end+1]
    reward = math.fsum(data["reward"][start+1:end+1])
    commands = data["actuator_command_nm"][start+1:end+1]
    at_limit_time = math.fsum(
        (right - left) * sum(abs(value) / limit >= 0.99 for value, limit in zip(command, limits)) / len(limits)
        for left, right, command in zip(times, times[1:], commands))
    origin = data["base_position_m"][0]
    return {
        "start_s": times[0], "end_s": times[-1], "observed_seconds": duration,
        "sample_count": len(times),
        "end_displacement_xy_m": [positions[-1][axis] - origin[axis] for axis in (0, 1)],
        "end_horizontal_displacement_m": displacement[-1],
        "max_horizontal_displacement_m": max(displacement),
        "position_rms_m": _rms(times, displacement),
        "path_length_m": math.fsum(math.hypot(right[0]-left[0], right[1]-left[1])
                                  for left, right in zip(positions, positions[1:])),
        "horizontal_speed_max_m_s": max(speed), "horizontal_speed_end_m_s": speed[-1],
        "horizontal_speed_rms_m_s": _rms(times, speed),
        "pitch_error_rms_deg": _rms(times, error), "pitch_error_max_abs_deg": max(map(abs, error)),
        "pitch_error_end_deg": error[-1],
        "pitch_max_abs_deg": max(abs(math.degrees(value)) for value in data["pitch_rad"][start:end+1]),
        "roll_max_abs_deg": max(abs(math.degrees(value)) for value in data["roll_rad"][start:end+1]),
        "pitch_rate_rms_rad_s": _rms(times, rates), "pitch_rate_end_rad_s": rates[-1],
        "reward_total": reward, "reward_per_second": reward / duration if duration else None,
        "command_at_limit_fraction": at_limit_time / duration if duration else None,
    }


def _window(data, limits, cutoff, requested):
    end = bisect_right(data["t_s"], cutoff + 1e-12) - 1
    result = _measure(data, limits, 0, end)
    result.update(requested_seconds=requested,
                  reached_window_end=data["t_s"][end] >= requested - 1e-9)
    # Include only samples on/after the final one-second boundary; never borrow
    # an earlier state and pretend its interval belongs to this tail.
    tail_start = bisect_right(data["t_s"], max(0, data["t_s"][end] - 1.0) - 1e-12)
    result["tail"] = _measure(data, limits, min(tail_start, end), end)
    return result


def _event(times, values, threshold, persistence):
    start = None
    for timestamp, value in zip(times, values):
        if value >= threshold:
            if start is None:
                start = timestamp
            if timestamp - start >= persistence - 1e-12:
                return {"first_exceeded_s": start, "confirmed_at_s": timestamp}
        else:
            start = None
    return {"first_exceeded_s": None, "confirmed_at_s": None}


def analyze_trace(trace, *, windows_s=(6.0,), thresholds=None):
    """Describe a complete first episode; never label it stable or infer cause.

    ``thresholds`` may override displacement_m, pitch_error_deg,
    horizontal_speed_m_s, and persistence_s. All are positive finite values.
    Window endpoints and tail starts fall on recorded samples (no interpolation).
    """
    data, limits, termination = _validated(trace)
    windows = [_number(value, "windows_s", positive=True)
               for value in _sequence(windows_s, "windows_s")]
    if len(set(windows)) != len(windows):
        raise ValueError("windows_s must not contain duplicate durations")
    settings = dict(DEFAULT_THRESHOLDS)
    if thresholds is not None:
        if not isinstance(thresholds, Mapping) or set(thresholds) - settings.keys():
            raise ValueError("Unknown thresholds or invalid thresholds mapping")
        settings.update(thresholds)
    settings = {name: _number(value, name, positive=True) for name, value in settings.items()}
    end = data["t_s"][-1]
    results = {f"first_{window:g}s": _window(data, limits, min(window, end), window) for window in windows}
    results["full_episode"] = _window(data, limits, end, end)
    events = {}
    for name, values, threshold in (
        ("displacement", data["displacement"], settings["displacement_m"]),
        ("pitch_error", [abs(value) for value in data["pitch_error"]], settings["pitch_error_deg"]),
        ("horizontal_speed", data["speed"], settings["horizontal_speed_m_s"]),
    ):
        events[name] = _event(data["t_s"], values, threshold, settings["persistence_s"])
    return {
        "schema_version": 1, "windows": results, "events": events,
        "thresholds": settings, "termination": termination,
        "scope": "Descriptive measurements of one complete first episode; thresholds are not validated success criteria or causal evidence.",
        "sampling": "Recorded endpoints only, no interpolation. State RMS is trapezoidal time-weighted; reward uses post-step intervals. Command-at-limit is the control-period-weighted fraction of last-substep command samples, not measured physics-substep saturation. Tail is at most the last one second.",
    }
