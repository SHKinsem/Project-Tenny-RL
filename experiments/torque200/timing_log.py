"""Exclusive lifecycle timing, independent of training frameworks and step clocks.

JSONL events are append-only.  ``segment`` events are the only closed business
intervals; ``checkpoint`` events observe an open interval without adding it a
second time.  Every event carries recorder overhead measured *before* that
event's append.  The next event therefore makes the preceding append auditable.
The final publication is outside the explicitly stated measurement cutoff.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Iterator
import uuid


SCHEMA_VERSION = 1
MEASUREMENT_NOTE = (
    "Seconds use one monotonic origin at recorder construction. The measurement "
    "cutoff is process_wall_seconds; phase fractions use that whole window, "
    "including recorder overhead. This is elapsed wall time, not device kernel time."
)
REPORTING_TAIL_NOTE = (
    "The final event and summary publication occur after the final measurement "
    "cutoff. Their small serialization/write/close tail is not included or assumed "
    "to be zero. For a running log, its last event's own append and all later work "
    "remain unobserved until a subsequent event."
)


@dataclass
class _PhaseStats:
    total_seconds: float = 0.0
    segments: int = 0
    min_seconds: float | None = None
    max_seconds: float | None = None
    outcomes: dict[str, int] = field(default_factory=dict)

    def add(self, seconds: float, outcome: str) -> None:
        self.total_seconds += seconds
        self.segments += 1
        self.min_seconds = seconds if self.min_seconds is None else min(self.min_seconds, seconds)
        self.max_seconds = seconds if self.max_seconds is None else max(self.max_seconds, seconds)
        self.outcomes[outcome] = self.outcomes.get(outcome, 0) + 1


@dataclass
class _OpenSegment:
    phase: str
    wall_start_seconds: float
    steps_start: int | None
    metadata: dict[str, Any]
    excluded_seconds: float = 0.0


def _phase_name(phase: str) -> str:
    if not isinstance(phase, str) or not phase.strip():
        raise ValueError("phase must be a non-empty string")
    return phase


def _paths(run_dir: Path) -> dict[str, str]:
    return {
        "timing_file": str(run_dir / "timing.jsonl"),
        "timing_summary_file": str(run_dir / "timing-summary.json"),
        "timing_text_file": str(run_dir / "timing-summary.txt"),
    }


def _summary(
    *,
    run_id: str,
    mode: str,
    session_id: str,
    process_wall_seconds: float,
    recorder_overhead_seconds: float,
    phases: dict[str, _PhaseStats],
    open_segment: dict[str, Any] | None,
    steps: int | None,
    outcome: str,
    finished: bool,
    paths: dict[str, str],
) -> dict[str, Any]:
    rows: dict[str, dict[str, Any]] = {}
    for name, stats in phases.items():
        rows[name] = {
            "total_seconds": stats.total_seconds,
            "segments": stats.segments,
            "closed_segments": stats.segments,
            "open_segments": 0,
            "min_seconds": stats.min_seconds,
            "max_seconds": stats.max_seconds,
            "outcomes": dict(stats.outcomes),
        }
    closed_exclusive = sum(row["total_seconds"] for row in rows.values())
    open_seconds = 0.0
    if open_segment is not None and open_segment.get("observed", True):
        open_seconds = float(open_segment["duration_seconds"])
        name = open_segment["phase"]
        row = rows.setdefault(
            name,
            {
                "total_seconds": 0.0,
                "segments": 0,
                "closed_segments": 0,
                "open_segments": 0,
                "min_seconds": None,
                "max_seconds": None,
                "outcomes": {},
            },
        )
        row["total_seconds"] += open_seconds
        row["segments"] += 1
        row["open_segments"] = 1
        row["min_seconds"] = open_seconds if row["min_seconds"] is None else min(row["min_seconds"], open_seconds)
        row["max_seconds"] = open_seconds if row["max_seconds"] is None else max(row["max_seconds"], open_seconds)
    for row in rows.values():
        row["mean_seconds"] = row["total_seconds"] / row["segments"] if row["segments"] else 0.0
        row["process_wall_fraction"] = row["total_seconds"] / process_wall_seconds if process_wall_seconds else 0.0
        row["process_wall_percent"] = 100.0 * row["process_wall_fraction"]
    exclusive = closed_exclusive + open_seconds
    residual = process_wall_seconds - exclusive - recorder_overhead_seconds
    if abs(residual) < 1e-12:
        residual = 0.0
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "mode": mode,
        "session_id": session_id,
        "outcome": outcome,
        "finished": finished,
        "process_wall_seconds": process_wall_seconds,
        "measurement_cutoff_seconds": process_wall_seconds,
        "exclusive_seconds": exclusive,
        "closed_exclusive_seconds": closed_exclusive,
        "current_open_seconds": open_seconds,
        "recorder_overhead_seconds": recorder_overhead_seconds,
        "unaccounted_seconds": max(0.0, residual),
        "accounting_residual_seconds": residual,
        "steps": steps,
        "phases": rows,
        "open_segment": open_segment,
        "paths": dict(paths),
        "measurement_note": MEASUREMENT_NOTE,
        "reporting_tail_note": REPORTING_TAIL_NOTE,
    }


def format_timing_summary(summary: dict[str, Any]) -> str:
    """Return a compact table using the entire measured window as denominator."""
    lines = [
        f"Run {summary['run_id']} | mode={summary['mode']} | outcome={summary['outcome']}",
        f"Measured wall: {summary['process_wall_seconds']:.6f} s",
        f"Exclusive phases: {summary['exclusive_seconds']:.6f} s",
        f"Recorder overhead: {summary['recorder_overhead_seconds']:.6f} s",
        f"Open phase (included above): {summary['current_open_seconds']:.6f} s",
        f"Unaccounted within cutoff: {summary['unaccounted_seconds']:.6f} s",
        "",
        f"{'phase':<28} {'seconds':>11} {'segments':>9} {'mean':>11} {'min':>11} {'max':>11} {'wall %':>8}",
    ]
    for name, row in sorted(summary["phases"].items(), key=lambda item: item[1]["total_seconds"], reverse=True):
        lines.append(
            f"{name:<28} {row['total_seconds']:>11.6f} {row['segments']:>9} "
            f"{row['mean_seconds']:>11.6f} {row['min_seconds']:>11.6f} "
            f"{row['max_seconds']:>11.6f} {row['process_wall_percent']:>8.2f}"
        )
    if summary["open_segment"] is not None:
        observed = summary["open_segment"].get("observed", True)
        lines.append(f"Open phase: {summary['open_segment']['phase']} ({'observed through cutoff' if observed else 'elapsed time after last event is not yet observed'})")
    lines.extend(["", summary["measurement_note"], summary["reporting_tail_note"]])
    for diagnostic in summary.get("diagnostics", []):
        lines.append(f"Note: {diagnostic}")
    return "\n".join(lines) + "\n"


class RunTiming:
    """Record flat, exclusive lifecycle segments with lightweight append logging.

    ``steps`` is an optional logical environment-step counter.  It never supplies
    the timer clock, and ``snapshot`` never resets either counter.  A span pauses
    its parent and resumes it as a new segment, so summing phases is meaningful.
    Instances are intended for one control thread, not concurrent callbacks.
    """

    def __init__(
        self,
        run_dir: str | os.PathLike[str],
        *,
        run_id: str,
        mode: str,
        clock: Callable[[], float] = time.monotonic,
        phase: str = "lifecycle",
        cancelled_exceptions: tuple[type[BaseException], ...] = (),
    ) -> None:
        _phase_name(phase)
        if not isinstance(cancelled_exceptions, tuple) or any(
            not isinstance(kind, type) or not issubclass(kind, BaseException)
            for kind in cancelled_exceptions
        ):
            raise TypeError("cancelled_exceptions must be a tuple of exception types")
        self._cancelled_exceptions = (KeyboardInterrupt, SystemExit, GeneratorExit) + cancelled_exceptions
        self._clock = clock
        self._origin = float(clock())
        self._last_clock = self._origin
        self.run_id = str(run_id)
        self.mode = str(mode)
        self.session_id = uuid.uuid4().hex
        self.paths = _paths(Path(run_dir))
        self._phases: dict[str, _PhaseStats] = {}
        self._steps: int | None = None
        self._overhead = 0.0
        self._active: _OpenSegment | None = None
        self._final_summary: dict[str, Any] | None = None
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        self._stream = Path(self.paths["timing_file"]).open("a", encoding="utf-8", newline="\n")
        self._append_event(
            self._event(
                "start", 0.0, phase=phase, outcome="running",
                next_phase={"phase": phase, "steps_start": None, "metadata": {}},
            )
        )
        end = self._now()
        self._overhead = end
        self._active = _OpenSegment(phase, end, None, {})

    def _now(self) -> float:
        value = float(self._clock())
        if not math.isfinite(value) or value < self._last_clock:
            raise RuntimeError("timing clock must be finite and monotonic")
        self._last_clock = value
        return value - self._origin

    def _event(self, event: str, now: float, **fields: Any) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "event": event,
            "run_id": self.run_id,
            "mode": self.mode,
            "session_id": self.session_id,
            "phase": None,
            "wall_start_seconds": now,
            "wall_end_seconds": now,
            "duration_seconds": 0.0,
            "steps_start": self._steps,
            "steps_end": self._steps,
            "outcome": "running",
            "metadata": {},
            "process_wall_seconds": now,
            "recorder_overhead_seconds": self._overhead,
            **fields,
        }

    def _append_event(self, event: dict[str, Any]) -> None:
        self._stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
        self._stream.flush()

    def _record_overhead(self, start: float, end: float) -> None:
        seconds = end - start
        self._overhead += seconds
        if self._active is not None:
            self._active.excluded_seconds += seconds

    def _open_observation(self, now: float) -> dict[str, Any] | None:
        if self._active is None:
            return None
        active = self._active
        return {
            "phase": active.phase,
            "wall_start_seconds": active.wall_start_seconds,
            "wall_end_seconds": now,
            "duration_seconds": max(0.0, now - active.wall_start_seconds - active.excluded_seconds),
            "recorder_excluded_seconds": active.excluded_seconds,
            "steps_start": active.steps_start,
            "steps_end": self._steps,
            "outcome": "running",
            "metadata": dict(active.metadata),
            "observed": True,
        }

    def _build_summary(self, now: float, *, outcome: str = "running", finished: bool = False) -> dict[str, Any]:
        return _summary(
            run_id=self.run_id,
            mode=self.mode,
            session_id=self.session_id,
            process_wall_seconds=now,
            recorder_overhead_seconds=self._overhead,
            phases=self._phases,
            open_segment=self._open_observation(now),
            steps=self._steps,
            outcome=outcome,
            finished=finished,
            paths=self.paths,
        )

    def _transition(
        self,
        phase: str | None,
        *,
        steps: int | None,
        metadata: dict[str, Any],
        outcome: str,
        exception_type: str | None = None,
    ) -> None:
        if self._final_summary is not None:
            raise RuntimeError("timing recorder is already finished")
        start = self._now()
        if steps is not None:
            self._steps = steps
        closed = self._open_observation(start)
        if closed is None:
            raise RuntimeError("timing recorder has no active phase")
        closed.pop("observed")
        closed["outcome"] = outcome
        if exception_type is not None:
            closed["metadata"]["exception_type"] = exception_type
        next_phase = None if phase is None else {"phase": phase, "steps_start": self._steps, "metadata": dict(metadata)}
        self._append_event(self._event("segment", start, **closed, next_phase=next_phase))
        self._phases.setdefault(closed["phase"], _PhaseStats()).add(closed["duration_seconds"], outcome)
        self._active = None
        end = self._now()
        self._record_overhead(start, end)
        if phase is not None:
            self._active = _OpenSegment(phase, end, self._steps, dict(metadata))

    def set_phase(self, phase: str, *, steps: int | None = None, **metadata: Any) -> None:
        """Close the current segment and open a new one, including same-name phases."""
        self._transition(_phase_name(phase), steps=steps, metadata=metadata, outcome="completed")

    @contextmanager
    def span(self, phase: str, *, steps: int | None = None, **metadata: Any) -> Iterator[RunTiming]:
        """Pause the parent, then resume it; exception outcomes belong to this span."""
        if self._active is None or self._final_summary is not None:
            raise RuntimeError("timing recorder is already finished")
        parent_phase = self._active.phase
        parent_metadata = dict(self._active.metadata)
        self.set_phase(phase, steps=steps, **metadata)
        try:
            yield self
        except BaseException as exc:
            if self._final_summary is None:
                outcome = "cancelled" if isinstance(exc, self._cancelled_exceptions) else "error"
                self._transition(parent_phase, steps=None, metadata=parent_metadata, outcome=outcome, exception_type=type(exc).__name__)
            raise
        else:
            if self._final_summary is None:
                self._transition(parent_phase, steps=None, metadata=parent_metadata, outcome="completed")

    def snapshot(self) -> dict[str, Any]:
        """Observe closed and currently open time without appending or resetting."""
        if self._final_summary is not None:
            return self._final_summary
        start = self._now()
        summary = self._build_summary(start)
        end = self._now()
        self._record_overhead(start, end)
        # Update the observation to include this snapshot's measured bookkeeping
        # overhead, without charging those seconds to the open business segment.
        summary["process_wall_seconds"] = summary["measurement_cutoff_seconds"] = end
        summary["recorder_overhead_seconds"] = self._overhead
        if summary["open_segment"] is not None:
            summary["open_segment"]["wall_end_seconds"] = end
            summary["open_segment"]["recorder_excluded_seconds"] += end - start
        for row in summary["phases"].values():
            row["process_wall_fraction"] = row["total_seconds"] / end if end else 0.0
            row["process_wall_percent"] = 100.0 * row["process_wall_fraction"]
        return summary

    @staticmethod
    def _atomic_text(path: Path, text: str) -> None:
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(text, encoding="utf-8", newline="\n")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def _write_summaries(self, summary: dict[str, Any]) -> None:
        self._atomic_text(Path(self.paths["timing_summary_file"]), json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
        self._atomic_text(Path(self.paths["timing_text_file"]), format_timing_summary(summary))

    def flush_summary(self) -> dict[str, Any]:
        """Publish an atomic observation; its I/O is excluded from business time."""
        if self._final_summary is not None:
            return self._final_summary
        start = self._now()
        summary = self._build_summary(start)
        self._append_event(self._event("checkpoint", start, phase=self._active.phase, open_segment=summary["open_segment"]))
        self._write_summaries(summary)
        end = self._now()
        self._record_overhead(start, end)
        return summary

    def finish(self, outcome: str = "completed", *, steps: int | None = None) -> dict[str, Any]:
        """Close once and publish final reports after an explicit timing cutoff."""
        if self._final_summary is not None:
            return self._final_summary
        self._transition(None, steps=steps, metadata={}, outcome=outcome)
        # The preceding append's end is the last fully measured boundary.  All
        # work below belongs to the explicitly excluded final publication tail.
        cutoff = self._last_clock - self._origin
        summary = self._build_summary(cutoff, outcome=outcome, finished=True)
        self._append_event(self._event("finish", cutoff, outcome=outcome, reporting_tail_note=REPORTING_TAIL_NOTE))
        self._write_summaries(summary)
        self._stream.close()
        self._final_summary = summary
        return summary


def _nonnegative_seconds(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"invalid {field_name} in timing log")
    return float(value)


def rebuild_summary(run_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """Rebuild the latest recorder session from JSONL, tolerating a torn last line.

    A running report stops at its last durable observation.  It does not invent
    elapsed seconds from file timestamps or a possibly rebooted monotonic clock.
    Multiple sessions remain append-only; the most recent ``start`` selects the
    session reported here, and a diagnostic identifies earlier sessions.
    """
    directory = Path(run_dir)
    path = directory / "timing.jsonl"
    if not path.is_file():
        raise ValueError(f"No timing.jsonl in {directory}; this run predates timing logs or has not started recording.")
    phases: dict[str, _PhaseStats] = {}
    latest: dict[str, Any] | None = None
    open_segment: dict[str, Any] | None = None
    sessions = 0
    ignored_tail = 0
    with path.open("rb") as stream:
        raw = stream.readline()
        line_number = 0
        while raw:
            line_number += 1
            following = stream.readline()
            if raw.strip():
                try:
                    event = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    if not following:
                        ignored_tail = 1
                        break
                    raise ValueError(f"Invalid timing JSON at line {line_number}") from exc
                if not isinstance(event, dict) or event.get("schema_version") != SCHEMA_VERSION:
                    raise ValueError(f"Unsupported timing event at line {line_number}")
                kind = event.get("event")
                if kind not in {"start", "segment", "checkpoint", "finish"}:
                    raise ValueError(f"Unknown timing event at line {line_number}: {kind!r}")
                if kind == "start":
                    phases = {}
                    sessions += 1
                elif latest is None or event.get("session_id") != latest.get("session_id"):
                    raise ValueError(f"Timing session is missing its start event at line {line_number}")
                _nonnegative_seconds(event.get("process_wall_seconds"), "process_wall_seconds")
                _nonnegative_seconds(event.get("recorder_overhead_seconds"), "recorder_overhead_seconds")
                if kind == "segment":
                    name = _phase_name(event.get("phase"))
                    seconds = _nonnegative_seconds(event.get("duration_seconds"), "duration_seconds")
                    phases.setdefault(name, _PhaseStats()).add(seconds, event.get("outcome", "completed"))
                if kind == "checkpoint":
                    open_segment = event.get("open_segment")
                    if open_segment is not None:
                        _phase_name(open_segment.get("phase"))
                        _nonnegative_seconds(open_segment.get("duration_seconds"), "open duration_seconds")
                elif kind == "finish":
                    open_segment = None
                else:
                    pending = event.get("next_phase")
                    open_segment = None if pending is None else {**pending, "observed": False, "duration_seconds": None, "wall_start_seconds": None, "wall_end_seconds": event["process_wall_seconds"]}
                latest = event
            raw = following
    if latest is None:
        raise ValueError(f"No complete timing events in {path}")
    summary = _summary(
        run_id=latest["run_id"], mode=latest["mode"], session_id=latest["session_id"],
        process_wall_seconds=float(latest["process_wall_seconds"]),
        recorder_overhead_seconds=float(latest["recorder_overhead_seconds"]),
        phases=phases, open_segment=open_segment, steps=latest.get("steps_end"),
        outcome=latest["outcome"] if latest["event"] == "finish" else "running",
        finished=latest["event"] == "finish", paths=_paths(directory),
    )
    summary["source"] = "timing.jsonl"
    summary["sessions_in_log"] = sessions
    summary["ignored_trailing_lines"] = ignored_tail
    summary["diagnostics"] = []
    if ignored_tail:
        summary["diagnostics"].append("Ignored an incomplete or invalid final JSONL line; all preceding complete events were retained.")
    if sessions > 1:
        summary["diagnostics"].append(f"Showing the latest of {sessions} append-only recorder sessions.")
    if not summary["finished"]:
        summary["diagnostics"].append("No final event was observed. Outcome and elapsed work after the last complete event are unknown.")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rebuild exclusive run timing directly from timing.jsonl.")
    parser.add_argument("run_dir", help="Run directory containing timing.jsonl")
    args = parser.parse_args(argv)
    try:
        summary = rebuild_summary(args.run_dir)
    except (OSError, ValueError) as exc:
        print(f"Timing analysis failed: {exc}", file=sys.stderr)
        return 2
    print(format_timing_summary(summary), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
