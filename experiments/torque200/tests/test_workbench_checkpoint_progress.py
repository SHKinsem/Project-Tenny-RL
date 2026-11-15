"""Cumulative progress through serialized, synthetic console artifacts only."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.torque200.workbench_api import WorkbenchReader
from experiments.torque200.workbench_api import reader as reader_module
from experiments.torque200.workbench_api.checkpoint_metadata import capture_checkpoint, make_run_context


POLICY = "console-original-v1"


def identity(number):
    return f"10000000-0000-4000-8000-{number:012d}"


class CheckpointProgressReadTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="workbench-cumulative-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "artifacts"
        self.reader = WorkbenchReader(self.root)
        self.parent_context = make_run_context(run_id="run-000", mode="original",
                                              clock_policy_id=POLICY, context_id=identity(1))
        self.parent = capture_checkpoint(self.parent_context, elapsed_seconds=600,
                                         steps=100000, checkpoint_id=identity(2))
        self.context = make_run_context(run_id="run-001", mode="original", clock_policy_id=POLICY,
                                        origin="checkpoint", parent=self.parent, context_id=identity(3))

    def write(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, allow_nan=False), encoding="utf-8")

    def run_path(self, name="run-001"):
        return self.root / "console" / "original" / name

    def status(self, context=None, **changes):
        context = self.context if context is None else context
        directory = self.run_path(context["runId"])
        record = dict(mode="original", run_id=context["runId"], phase="training", stage="optimization",
                      workbench_context_id=context["id"], clock_policy_id=POLICY,
                      training_clock_started=True, elapsed_seconds=0, steps=0, duration_seconds=600)
        record.update(changes)
        self.write(directory / "workbench-run.json", context)
        self.write(directory / "status.json", record)
        return record

    def checkpoint(self, metadata, name="000000100000"):
        directory = self.run_path(metadata["createdByRunId"]) / "checkpoints" / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "weights.bin").write_bytes(b"synthetic: never loaded")
        self.write(directory.with_name(directory.name + ".workbench.json"), metadata)
        return directory

    def test_resume_starts_at_half_then_reaches_55_percent(self):
        self.checkpoint(self.parent)
        self.status()
        initial = self.reader.state()
        self.assertEqual(initial["progress"]["cumulativePercent"], 50)
        self.assertEqual(initial["progress"]["segmentPercent"], 0)
        self.assertEqual(initial["progress"]["targetTrainedSeconds"], 1200)
        self.status(elapsed_seconds=60, steps=10000)
        current = WorkbenchReader(self.root).state()
        self.assertEqual(current["run"]["baseTrainedSeconds"], 600)
        self.assertEqual(current["run"]["trainedSeconds"], 660)
        self.assertEqual(current["run"]["trainedSteps"], 110000)
        self.assertAlmostEqual(current["progress"]["cumulativePercent"], 55, places=12)
        self.assertEqual(current["progress"]["segmentPercent"], 10)
        self.assertEqual(current["progress"]["remainingSeconds"], 540)
        self.assertIsNone(current["run"]["active"])
        self.assertEqual(current["run"]["liveness"], "unverified")

    def test_repeated_snapshots_and_next_resume_do_not_double_count(self):
        self.status(elapsed_seconds=120, steps=20000)
        first = capture_checkpoint(self.context, elapsed_seconds=60, steps=10000, checkpoint_id=identity(4))
        second = capture_checkpoint(self.context, elapsed_seconds=120, steps=20000, checkpoint_id=identity(5))
        self.checkpoint(first, "000000010000")
        self.checkpoint(second, "000000020000")
        items = self.reader.checkpoints()["items"]
        self.assertEqual(len(items), 2)  # Sidecars are metadata, never extra candidates.
        self.assertEqual([item["trainedSeconds"] for item in items], [660, 720])
        self.assertTrue(all(item["metadataStatus"] == "valid" for item in items))
        self.assertTrue(all(item["verification"] == "unverified" and not item["canResume"] for item in items))
        restarted = make_run_context(run_id="run-002", mode="original", clock_policy_id=POLICY,
                                     origin="checkpoint", parent=first, context_id=identity(6))
        self.status(restarted, elapsed_seconds=60, steps=10000)
        current = self.reader.state()["run"]
        self.assertEqual(current["trainedSeconds"], 720)
        self.assertEqual(current["trainedSteps"], 120000)
        self.assertEqual(current["sourceCheckpointId"], identity(4))

    def test_pause_reads_do_not_advance_and_finalization_is_not_added(self):
        self.status(phase="paused", elapsed_seconds=60, steps=10000)
        first = self.reader.state()
        with patch.object(reader_module.time, "time", return_value=4_000_000_000):
            later = self.reader.state()
        self.assertEqual(first["revision"], later["revision"])
        self.assertAlmostEqual(later["progress"]["cumulativePercent"], 55, places=12)
        self.status(phase="completed", elapsed_seconds=180, training_stop_elapsed_seconds=120,
                    finalization_seconds=60, steps=20000)
        self.assertEqual(self.reader.state()["run"]["trainedSeconds"], 720)

    def test_unknown_time_and_known_steps_are_independent(self):
        parent = deepcopy(self.parent)
        parent.update(baseTrainedSeconds=None, trainedSeconds=None)
        context = make_run_context(run_id="run-001", mode="original", clock_policy_id=POLICY,
                                   origin="checkpoint", parent=parent, context_id=identity(7))
        self.status(context, elapsed_seconds=60, steps=10000)
        progress = self.reader.state()["progress"]
        self.assertIsNone(progress["trainedSeconds"])
        self.assertIsNone(progress["targetTrainedSeconds"])
        self.assertIsNone(progress["cumulativePercent"])
        self.assertEqual(progress["trainedSteps"], 110000)
        self.assertEqual(progress["segmentPercent"], 10)

    def test_new_legacy_and_unlimited_origins(self):
        new = make_run_context(run_id="run-001", mode="original", clock_policy_id=POLICY,
                               context_id=identity(8))
        self.status(new, elapsed_seconds=60, steps=10000)
        self.assertEqual(self.reader.state()["progress"]["cumulativePercent"], 10)
        legacy = make_run_context(run_id="run-001", mode="original", clock_policy_id=POLICY,
                                  origin="legacy", context_id=identity(9))
        self.status(legacy, elapsed_seconds=60, steps=10000)
        self.assertIsNone(self.reader.state()["run"]["trainedSteps"])
        self.status(elapsed_seconds=60, steps=10000, duration_seconds=0)
        progress = self.reader.state()["progress"]
        self.assertEqual(progress["trainedSeconds"], 660)
        self.assertIsNone(progress["targetTrainedSeconds"])
        self.assertIsNone(progress["segmentPercent"])
        self.assertIsNone(progress["cumulativePercent"])

    def test_bad_unbound_missing_and_mismatched_contexts_degrade_to_unknown(self):
        record = self.status(elapsed_seconds=60, steps=10000)
        path = self.run_path() / "workbench-run.json"
        cases = [
            {**self.context, "schemaVersion": 999}, {**self.context, "id": identity(10)},
            {**self.context, "runId": "other-run"}, {**self.context, "mode": "torque200"},
            {**self.context, "clockPolicyId": "other-clock"},
        ]
        for value in cases:
            self.write(path, value)
            result = self.reader.state()
            self.assertIsNone(result["run"]["trainedSeconds"])
            self.assertEqual(result["progress"]["segmentPercent"], 10)
            self.assertIn("invalid_run_context", [item["code"] for item in result["warnings"]])
        self.write(path, self.context)
        record.pop("workbench_context_id")
        self.write(self.run_path() / "status.json", record)
        self.assertIsNone(self.reader.state()["run"]["trainedSeconds"])
        path.unlink()
        record["workbench_context_id"] = self.context["id"]
        self.write(self.run_path() / "status.json", record)
        self.assertIn("missing_run_context", [item["code"] for item in self.reader.state()["warnings"]])

    def test_legacy_report_cannot_supply_counters_to_bound_context(self):
        self.status(elapsed_seconds=60, steps=10000)
        self.write(self.run_path() / "report.json", {"mode": "original", "training_stop_elapsed_seconds": 999})
        self.assertEqual(self.reader.state()["run"]["trainedSeconds"], 660)
        self.write(self.run_path() / "report.json", {"mode": "original", "training_stop_elapsed_seconds": 999,
                                                    "workbench_context_id": identity(11)})
        result = self.reader.state()
        self.assertIsNone(result["run"]["trainedSeconds"])

    def test_metadata_identity_errors_do_not_promote_old_candidates(self):
        path = self.checkpoint(self.parent)
        metadata_path = path.with_name(path.name + ".workbench.json")
        for changed in ({"schemaVersion": 999}, {"createdByRunId": "other-run"},
                        {"trainedSeconds": 999}, {"controlHz": 200}):
            self.write(metadata_path, {**self.parent, **changed})
            result = self.reader.checkpoints()
            self.assertEqual(len(result["items"]), 1)
            self.assertEqual(result["items"][0]["metadataStatus"], "invalid")
            self.assertIsNone(result["items"][0]["id"])
            self.assertIsNone(result["items"][0]["trainedSeconds"])
            self.assertFalse(result["items"][0]["canResume"])

    def test_metadata_revision_changes_and_reads_do_not_write(self):
        self.status(elapsed_seconds=60, steps=10000)
        path = self.checkpoint(self.parent)
        first = self.reader.checkpoints()
        before = {p.relative_to(self.root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in self.root.rglob("*") if p.is_file()}
        self.reader.state()
        self.reader.checkpoints()
        self.reader.metrics()
        after = {p.relative_to(self.root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        changed = capture_checkpoint(self.parent_context, elapsed_seconds=660,
                                     steps=110000, checkpoint_id=self.parent["id"])
        self.write(path.with_name(path.name + ".workbench.json"), changed)
        with self.assertRaises(reader_module.WorkbenchError) as caught:
            self.reader.checkpoints(revision=first["revision"])
        self.assertEqual(caught.exception.status, 409)


if __name__ == "__main__":
    unittest.main()
