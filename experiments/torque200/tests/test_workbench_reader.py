"""Synthetic artifact tests; no training, process inspection, or production reads."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.torque200.workbench_api import WorkbenchError, WorkbenchReader
from experiments.torque200.workbench_api import reader as module
from experiments.torque200.control_modes import MODES, mode_spec


class WorkbenchReaderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="workbench-reader-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "artifacts"
        self.reader = WorkbenchReader(self.root)

    def write_json(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, allow_nan=False), encoding="utf8")

    def run_artifacts(self, name="20261115T120000Z_run", mode="original", **changes):
        run = self.root / "console" / mode / name
        state = dict(mode=mode, run_id=name, run_dir=str(run), phase="training", stage="optimization",
                     elapsed_seconds=60, steps=123456, duration_seconds=120,
                     training_clock_started=True, preparation_seconds=12)
        state.update(changes)
        self.write_json(run / "status.json", state)
        self.write_json(run / "env_config.json", {"ctrl_dt": 1 / mode_spec(mode)["control_hz"]})
        return run

    def jsonl(self, run, events):
        (run / "metrics.jsonl").write_text("".join(json.dumps(event, allow_nan=False) + "\n" for event in events), encoding="utf8")

    @staticmethod
    def metric(steps=10, **values):
        return {"steps": steps, "elapsed_seconds": 5,
                "metrics": {"eval/avg_episode_length": 24, "eval/episode_reward": -2,
                            "training/v_loss": 0, "training/policy_loss": -1e-12, **values}}

    def expect_error(self, code, status, action):
        with self.assertRaises(WorkbenchError) as caught:
            action()
        self.assertEqual(caught.exception.code, code)
        self.assertEqual(caught.exception.status, status)
        self.assertNotIn(str(self.root), caught.exception.message)

    def test_missing_root_is_read_only_and_has_no_invented_liveness(self):
        state = self.reader.state()
        self.assertFalse(state["available"])
        self.assertIsNone(state["run"])
        self.assertEqual(self.reader.checkpoints()["items"], [])
        self.assertEqual(self.reader.metrics()["items"], [])
        self.assertFalse(self.root.exists())
        capabilities = self.reader.capabilities()
        self.assertTrue(capabilities["readOnly"])
        self.assertEqual(capabilities["modes"], list(MODES))
        self.assertIn("saveSnapshot", capabilities["unsupported"])
        self.assertFalse(self.root.exists())

    def test_parameter_validation_precedes_missing_artifacts(self):
        for mode in ("unknown", True, None):
            self.expect_error("invalid_mode", 400, lambda mode=mode: self.reader.state(mode))
        for run_id in ("../escape", "a/b", "a\\b", ".", "..", "/absolute", "x:drive", ""):
            self.expect_error("invalid_run_id", 400, lambda run_id=run_id: self.reader.metrics(run_id=run_id))
        self.expect_error("run_not_found", 404, lambda: self.reader.metrics(run_id="unknown_run"))
        for offset, limit in ((-1, 10), (True, 10), (0, 0), (0, 2001), (0, True)):
            self.expect_error("invalid_pagination", 400, lambda offset=offset, limit=limit: self.reader.metrics(offset=offset, limit=limit))
        self.expect_error("invalid_pagination", 400, lambda: self.reader.checkpoints(limit=1001))

    def test_state_is_reported_only_and_never_extrapolates_elapsed(self):
        self.run_artifacts()
        first = self.reader.state()
        with patch.object(module.time, "time", return_value=3_000_000_000):
            later = self.reader.state()
        run = later["run"]
        self.assertEqual(run["reportedPhase"], "training")
        self.assertEqual(run["reportedStage"], "optimization")
        self.assertEqual(run["elapsedSeconds"], 60)
        self.assertEqual(run["steps"], 123456)
        self.assertEqual(run["segmentPercent"], 50)
        self.assertIsNone(run["active"])
        self.assertEqual(run["liveness"], "unverified")
        for field in ("trainedSeconds", "trainedSteps", "baseTrainedSeconds", "baseTrainedSteps", "cumulativePercent", "recordedAtMs"):
            self.assertIsNone(run[field])
        self.assertEqual(first["revision"], later["revision"])
        self.assertNotEqual(first["readAtMs"], later["readAtMs"])
        self.assertTrue(all(file["recordedAtMs"] is None for file in later["source"]["files"]))

    def test_stop_elapsed_precedes_total_elapsed_and_finalization_is_separate(self):
        self.run_artifacts(phase="completed", elapsed_seconds=180, training_stop_elapsed_seconds=120,
                           finalization_seconds=60, duration_seconds=120)
        run = self.reader.state()["run"]
        self.assertEqual(run["elapsedSeconds"], 120)
        self.assertEqual(run["finalizationSeconds"], 60)
        self.assertEqual(run["segmentPercent"], 100)
        self.assertIsNone(run["cumulativePercent"])

    def test_extreme_finite_counters_do_not_emit_nonfinite_derived_percent(self):
        self.run_artifacts(elapsed_seconds=1e308, duration_seconds=1e-308)
        response = self.reader.state()
        self.assertEqual(response["run"]["elapsedSeconds"], 1e308)
        self.assertEqual(response["run"]["durationSeconds"], 1e-308)
        self.assertIsNone(response["run"]["segmentPercent"])
        self.assertIn("derived_value_out_of_range", [item["code"] for item in response["warnings"]])
        json.dumps(response, allow_nan=False)

    def test_old_final_elapsed_can_only_exclude_known_finalization(self):
        run = self.run_artifacts(phase="completed", elapsed_seconds=180, finalization_seconds=60)
        self.assertEqual(self.reader.state()["run"]["elapsedSeconds"], 120)
        state = json.loads((run / "status.json").read_text())
        state.pop("finalization_seconds")
        self.write_json(run / "status.json", state)
        response = self.reader.state()
        self.assertIsNone(response["run"]["elapsedSeconds"])
        self.assertIn("budget_time_unavailable", [item["code"] for item in response["warnings"]])

    def test_hybrid_preparation_does_not_consume_training_budget(self):
        self.run_artifacts(mode="hybrid_virtual_leg_force_100", phase="preparing", elapsed_seconds=37,
                           training_clock_started=False, preparation_seconds=0, duration_seconds=900)
        run = self.reader.state("hybrid_virtual_leg_force_100")["run"]
        self.assertEqual(run["elapsedSeconds"], 0)
        self.assertEqual(run["preparationSeconds"], 37)
        self.assertEqual(run["segmentPercent"], 0)

    def test_safe_active_reference_wins_over_latest_directory_and_bad_identity_falls_back(self):
        older = self.run_artifacts("20261114_old", steps=1)
        newer = self.run_artifacts("20261115_new", steps=2)
        active = self.root / "console" / "active.json"
        self.write_json(active, {"mode": "original", "run_dir": str(older), "run_id": older.name})
        self.assertEqual(self.reader.state()["run"]["id"], older.name)
        self.write_json(active, {"mode": "original", "run_dir": str(older), "run_id": "wrong_identity"})
        response = self.reader.state()
        self.assertEqual(response["run"]["id"], newer.name)
        self.assertIn("invalid_active_reference", [item["code"] for item in response["warnings"]])

    def test_outside_references_are_ignored_without_path_disclosure(self):
        run = self.run_artifacts()
        outside = Path(self.temporary.name) / "outside"
        outside.mkdir()
        self.write_json(self.root / "console" / "active.json", {"mode": "original", "run_dir": str(outside)})
        self.write_json(run / "report.json", {"mode": "original", "checkpoint": str(outside)})
        state = self.reader.state()
        checkpoints = self.reader.checkpoints()
        self.assertEqual(state["run"]["id"], run.name)
        self.assertEqual(checkpoints["items"], [])
        self.assertNotIn(str(outside), json.dumps(state))
        self.assertNotIn(str(outside), json.dumps(checkpoints))
        self.assertNotIn(str(self.root), json.dumps(checkpoints))

    def test_checkpoint_candidates_do_not_claim_verification_or_persistent_identity(self):
        run = self.run_artifacts()
        checkpoint = run / "checkpoints" / "000000123456"
        checkpoint.mkdir(parents=True)
        (checkpoint / "weights.bin").write_bytes(b"not loaded")
        self.write_json(run / "report.json", {"mode": "original", "checkpoint": str(checkpoint),
                                               "completed": True, "checkpoint_validated": True,
                                               "trainedSeconds": 600, "trainedSteps": 123456})
        items = self.reader.checkpoints()["items"]
        self.assertEqual(len(items), 1)
        item = items[0]
        self.assertEqual(item["relativePath"], f"console/original/{run.name}/checkpoints/000000123456")
        self.assertEqual(item["verification"], "unverified")
        self.assertFalse(item["canResume"])
        for field in ("id", "alias", "trainedSeconds", "trainedSteps"):
            self.assertIsNone(item[field])
        self.assertEqual(self.reader.checkpoints()["items"][0]["referenceId"], item["referenceId"])

    def test_checkpoint_pagination_revision_detects_new_candidate(self):
        run = self.run_artifacts()
        directory = run / "checkpoints"
        directory.mkdir()
        (directory / "001").mkdir()
        first = self.reader.checkpoints(limit=1)
        self.assertIsNone(first["pagination"]["nextOffset"])
        self.assertEqual(self.reader.checkpoints(limit=1, revision=first["revision"])["revision"], first["revision"])
        (directory / "002").mkdir()
        self.expect_error("revision_conflict", 409, lambda: self.reader.checkpoints(offset=1, limit=1, revision=first["revision"]))

    def test_10000_callbacks_preserve_duplicates_order_and_raw_numbers(self):
        run = self.run_artifacts()
        events = [self.metric(index // 2 if index != 9999 else 1) for index in range(10000)]
        self.jsonl(run, events)
        self.write_json(run / "progress.json", [self.metric(999999)])
        response = self.reader.metrics(run_id=run.name, offset=9990, limit=20)
        self.assertEqual(response["pagination"]["total"], 10000)
        self.assertEqual(response["rawSource"], "metrics.jsonl")
        self.assertEqual(response["historyCompleteness"], "full")
        self.assertIsNone(response["producerHistoryComplete"])
        self.assertEqual([item["steps"] for item in response["items"]], [event["steps"] for event in events[9990:]])
        self.assertEqual([item["sequence"] for item in response["items"]], list(range(9991, 10001)))
        values = response["items"][0]["values"]
        self.assertEqual(values["episode_seconds"], 0.24)
        self.assertEqual(values["episode_reward"], -2)
        self.assertEqual(values["value_loss"], 0)
        self.assertEqual(values["policy_loss"], -1e-12)
        self.assertTrue(all(item["recordedAtMs"] is None for item in response["items"]))

    def test_bad_and_partial_lines_keep_gaps_without_dropping_the_page(self):
        run = self.run_artifacts()
        data = json.dumps(self.metric(10)) + "\nnot-json\n" + json.dumps(self.metric(10)) + '\n{"steps":'
        (run / "metrics.jsonl").write_text(data, encoding="utf8")
        response = self.reader.metrics(run_id=run.name)
        self.assertEqual(response["pagination"]["total"], 4)
        self.assertEqual(response["items"][0]["steps"], 10)
        self.assertEqual(response["items"][2]["steps"], 10)
        self.assertEqual(response["items"][1]["gap"]["code"], "invalid_jsonl_record")
        self.assertEqual(response["items"][3]["gap"]["code"], "partial_record")
        self.assertTrue(all(value is None for value in response["items"][3]["values"].values()))

    def test_valid_unterminated_line_is_retained_and_marked_as_potentially_writing(self):
        run = self.run_artifacts()
        (run / "metrics.jsonl").write_text(json.dumps(self.metric(25)), encoding="utf8")
        response = self.reader.metrics(run_id=run.name)
        self.assertEqual(response["items"][0]["steps"], 25)
        self.assertEqual(response["items"][0]["gap"]["code"], "unterminated_record")
        self.assertIn("unterminated_record", [warning["code"] for warning in response["warnings"]])

    def test_progress_fallback_is_a_window_and_bad_jsonl_is_not_silently_replaced(self):
        run = self.run_artifacts()
        self.write_json(run / "progress.json", [self.metric(10), self.metric(20)])
        response = self.reader.metrics(run_id=run.name)
        self.assertEqual(response["rawSource"], "progress.json")
        self.assertEqual(response["historyCompleteness"], "window")
        (run / "metrics.jsonl").mkdir()
        response = self.reader.metrics(run_id=run.name)
        self.assertFalse(response["available"])
        self.assertIsNone(response["rawSource"])

    def test_metric_pagination_revision_changes_when_file_content_changes(self):
        run = self.run_artifacts()
        self.jsonl(run, [self.metric(10), self.metric(10), self.metric(5)])
        first = self.reader.metrics(run_id=run.name, limit=2)
        second = self.reader.metrics(run_id=run.name, offset=2, limit=2, revision=first["revision"])
        self.assertEqual(first["pagination"]["nextOffset"], 2)
        self.assertEqual(second["items"][0]["steps"], 5)
        self.jsonl(run, [self.metric(10), self.metric(10), self.metric(5), self.metric(30)])
        self.expect_error("revision_conflict", 409,
                          lambda: self.reader.metrics(run_id=run.name, offset=2, revision=first["revision"]))

    def test_default_run_page_can_continue_with_the_returned_run_identity(self):
        run = self.run_artifacts()
        self.jsonl(run, [self.metric(10), self.metric(20), self.metric(30)])
        first = self.reader.metrics(limit=2)
        second = self.reader.metrics(run_id=first["runId"], offset=2, limit=2, revision=first["revision"])
        self.assertEqual(second["revision"], first["revision"])
        self.assertEqual(second["items"][0]["steps"], 30)

    def test_jsonl_uses_lf_boundaries_and_accepts_crlf_whitespace(self):
        run = self.run_artifacts()
        good = json.dumps(self.metric(10)).encode()
        (run / "metrics.jsonl").write_bytes(good + b"\r\n" + good + b"\r" + good + b"\n")
        response = self.reader.metrics(run_id=run.name)
        self.assertEqual(response["pagination"]["total"], 2)
        self.assertEqual(response["items"][0]["steps"], 10)
        self.assertEqual(response["items"][1]["gap"]["code"], "invalid_jsonl_record")

    def test_nonregular_metric_source_is_refused_before_open(self):
        run = self.run_artifacts()
        path = run / "metrics.jsonl"
        if hasattr(os, "mkfifo"):
            os.mkfifo(path)
        else:
            path.mkdir()
        actual_open = module.os.open

        def guarded_open(candidate, *args, **kwargs):
            if Path(candidate) == path:
                self.fail("A nonregular metric source must be rejected before open can block.")
            return actual_open(candidate, *args, **kwargs)

        with patch.object(module.os, "open", side_effect=guarded_open):
            response = self.reader.metrics(run_id=run.name)
        self.assertFalse(response["available"])
        self.assertIn("unreadable_artifact", [item["code"] for item in response["warnings"]])

    def test_dt_mismatch_does_not_convert_episode_length_and_other_values_survive(self):
        run = self.run_artifacts(mode="torque200")
        self.jsonl(run, [self.metric()])
        self.write_json(run / "env_config.json", {"ctrl_dt": 0.01})
        response = self.reader.metrics("torque200", run_id=run.name)
        self.assertIsNone(response["items"][0]["values"]["episode_seconds"])
        self.assertEqual(response["items"][0]["values"]["episode_reward"], -2)
        units = {item["id"]: item["unit"] for item in response["definitions"]}
        self.assertEqual(units["episode_seconds"], "seconds")
        self.assertIsNone(units["episode_reward"])
        self.assertIsNone(units["value_loss"])
        self.assertTrue(all(item["stepsScope"] == "segment" for item in response["definitions"]))

    def test_explicit_record_time_stays_distinct_from_read_and_disk_mtime(self):
        run = self.run_artifacts(recorded_at_ms=1000)
        event = self.metric()
        event["recorded_at"] = "2026-11-15T04:45:00+00:00"
        self.jsonl(run, [event])
        state = self.reader.state()
        self.assertEqual(state["run"]["recordedAtMs"], 1000)
        self.assertNotEqual(state["readAtMs"], 1000)
        status_source = next(file for file in state["source"]["files"] if file["path"].endswith("status.json"))
        self.assertEqual(status_source["recordedAtMs"], 1000)
        self.assertNotEqual(status_source["modifiedAtMs"], 1000)
        metrics = self.reader.metrics(run_id=run.name)
        self.assertEqual(metrics["items"][0]["recordedAtMs"], 1794717900000)

    def test_null_invalid_and_nonfinite_metrics_are_not_zero(self):
        run = self.run_artifacts()
        events = [self.metric(10, **{"training/v_loss": None, "training/policy_loss": "bad"}),
                  {"steps": 11, "metrics": {"eval/episode_reward": -0.5}}]
        self.jsonl(run, events)
        with (run / "metrics.jsonl").open("a", encoding="utf8") as stream:
            stream.write('{"steps":12,"metrics":{"training/v_loss":NaN}}\n')
        items = self.reader.metrics(run_id=run.name)["items"]
        self.assertIsNone(items[0]["values"]["value_loss"])
        self.assertIsNone(items[0]["values"]["policy_loss"])
        self.assertIsNone(items[1]["values"]["episode_seconds"])
        self.assertEqual(items[1]["values"]["episode_reward"], -0.5)
        self.assertIsNotNone(items[2]["gap"])
        self.assertIsNone(items[2]["values"]["value_loss"])

    def test_unreadable_or_half_written_state_does_not_invent_running(self):
        run = self.run_artifacts()
        (run / "status.json").write_text('{"phase":', encoding="utf8")
        state = self.reader.state()
        self.assertFalse(state["available"])
        self.assertIsNone(state["run"]["reportedPhase"])
        self.assertIsNone(state["run"]["active"])

    def test_oversize_bytes_lines_records_and_directory_scans_are_bounded(self):
        run = self.run_artifacts()
        with (run / "metrics.jsonl").open("wb") as stream:
            stream.truncate(module.MAX_JSONL_BYTES + 1)
        self.expect_error("resource_too_large", 413, lambda: self.reader.metrics(run_id=run.name))
        (run / "metrics.jsonl").write_bytes(b" " * (module.MAX_LINE_BYTES + 1))
        self.expect_error("resource_too_large", 413, lambda: self.reader.metrics(run_id=run.name))
        (run / "metrics.jsonl").write_bytes(b"{}\n" * (module.MAX_RECORDS + 1))
        self.expect_error("resource_too_large", 413, lambda: self.reader.metrics(run_id=run.name))
        with (run / "status.json").open("wb") as stream:
            stream.truncate(module.MAX_JSON_BYTES + 1)
        self.expect_error("resource_too_large", 413, lambda: self.reader.state())
        with patch.object(module, "MAX_DIRECTORY_ENTRIES", 0):
            self.expect_error("resource_too_large", 413, lambda: self.reader.checkpoints())

    def test_linked_run_and_checkpoint_do_not_escape(self):
        run = self.run_artifacts()
        outside = Path(self.temporary.name) / "outside"
        outside.mkdir()
        link = run.parent / "linked_run"
        try:
            link.symlink_to(outside, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("Filesystem symlink creation is unavailable")
        self.expect_error("unsafe_path", 400, lambda: self.reader.metrics(run_id=link.name))
        directory = run / "checkpoints"
        directory.mkdir()
        (directory / "linked_checkpoint").symlink_to(outside, target_is_directory=True)
        result = self.reader.checkpoints()
        self.assertEqual(result["items"], [])
        self.assertIn("unsafe_path", [item["code"] for item in result["warnings"]])

    def test_every_query_leaves_existing_artifacts_unchanged(self):
        run = self.run_artifacts()
        (run / "checkpoints" / "0001").mkdir(parents=True)
        self.jsonl(run, [self.metric(2), self.metric(2)])

        def snapshot():
            return [(path.relative_to(self.root).as_posix(), path.stat().st_mtime_ns,
                     hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None)
                    for path in sorted(self.root.rglob("*"))]

        before = snapshot()
        self.reader.state()
        self.reader.checkpoints()
        self.reader.metrics(run_id=run.name)
        self.reader.capabilities()
        self.assertEqual(snapshot(), before)
        self.assertFalse((self.root / "console" / "manager.lock").exists())


if __name__ == "__main__":
    unittest.main()
