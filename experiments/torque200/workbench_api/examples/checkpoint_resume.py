"""Synthetic save/reload/continue example. Always uses a disposable directory."""
import json
from pathlib import Path
import tempfile

from .. import WorkbenchReader, capture_checkpoint, make_run_context


POLICY = "console-original-v1"


def _id(number):
    return f"30000000-0000-4000-8000-{number:012d}"


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False, indent=2), encoding="utf-8")


def run_example():
    """Return progress read back from real JSON files, with synthetic counts."""
    with tempfile.TemporaryDirectory(prefix="tenny-checkpoint-example-") as temporary:
        root = Path(temporary) / "artifacts"

        def save(metadata, name):
            path = root / "console" / "original" / metadata["createdByRunId"] / "checkpoints" / name
            path.mkdir(parents=True)
            (path / "weights.bin").write_bytes(b"synthetic, never loaded")
            _write(path.with_name(path.name + ".workbench.json"), metadata)

        def report(context, elapsed, steps, phase="training", **extra):
            path = root / "console" / "original" / context["runId"]
            _write(path / "workbench-run.json", context)
            _write(path / "status.json", dict(
                mode="original", run_id=context["runId"], phase=phase,
                workbench_context_id=context["id"], clock_policy_id=POLICY,
                training_clock_started=True, elapsed_seconds=elapsed, steps=steps,
                duration_seconds=600, **extra,
            ))

        def read(label):
            # Recreate the reader: history must survive serialization/reload.
            value = WorkbenchReader(root).state()["progress"]
            return {"stage": label, **{key: value[key] for key in (
                "baseTrainedSeconds", "trainedSeconds", "trainedSteps",
                "targetTrainedSeconds", "segmentPercent", "cumulativePercent")}}

        first = make_run_context(run_id="run-000", mode="original", clock_policy_id=POLICY,
                                 context_id=_id(1))
        parent = capture_checkpoint(first, elapsed_seconds=600, steps=100000, checkpoint_id=_id(2))
        save(parent, "000000100000")
        resumed = make_run_context(run_id="run-001", mode="original", clock_policy_id=POLICY,
                                   origin="checkpoint", parent=parent, context_id=_id(3))
        report(resumed, 0, 0)
        rows = [read("resume_600_plus_600")]
        report(resumed, 60, 10000)
        rows.append(read("after_60_seconds"))
        report(resumed, 60, 10000, "paused")
        rows.append(read("paused_same_counters"))
        saved = capture_checkpoint(resumed, elapsed_seconds=60, steps=10000, checkpoint_id=_id(4))
        save(saved, "000000010000")
        report(resumed, 75, 10000, "completed", training_stop_elapsed_seconds=60, finalization_seconds=15)
        rows.append(read("saved_without_15_seconds_finalization"))
        next_run = make_run_context(run_id="run-002", mode="original", clock_policy_id=POLICY,
                                    origin="checkpoint", parent=saved, context_id=_id(5))
        report(next_run, 60, 10000)
        rows.append(read("next_run_after_another_60_seconds"))
        return rows


if __name__ == "__main__":
    print(json.dumps(run_example(), indent=2, allow_nan=False))
