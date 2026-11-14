"""Runner lifecycle tests: no ML imports, GPU, WSL, or training required."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "console_train.py"
spec = importlib.util.spec_from_file_location("console_train_under_test", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name).resolve()
        self.now = [0.]
        self.args = argparse.Namespace(run_dir=self.directory, mode="original",
                                      duration_seconds=0., restore=None, seed=0)

    def monitor(self):
        return runner.RunMonitor(self.args, clock=lambda: self.now[0])

    def read(self, name):
        return json.loads((self.directory / name).read_text(encoding="utf8"))

    def test_initial_status_before_ml_imports(self):
        self.monitor()
        state = self.read("status.json")
        self.assertEqual(state["phase"], "preparing")
        self.assertIsNone(state["deadline_at"])
        self.assertIsNone(state["remaining_seconds"])
        self.assertIsNone(state["checkpoint"])
        self.assertEqual(self.read("progress.json"), [])
        self.assertEqual((self.directory / "metrics.jsonl").read_text(), "")

    def test_unlimited_remains_running_after_large_elapsed(self):
        monitor = self.monitor()
        self.now[0] = 10**9
        monitor.check_stop()
        self.assertEqual(monitor.remaining, float("inf"))

    def test_requested_stop_is_cooperative(self):
        monitor = self.monitor()
        (self.directory / "stop.request").write_text("{}")
        with self.assertRaisesRegex(runner.StopRequested, "user_stop"):
            monitor.check_stop()
        self.assertEqual(self.read("status.json")["phase"], "stopping")
        self.assertFalse(self.read("status.json")["completed"])

    def test_deadline_includes_setup(self):
        self.args.duration_seconds = 30.
        monitor = self.monitor()
        self.now[0] = 30.
        with self.assertRaisesRegex(runner.StopRequested, "wall_budget"):
            monitor.check_stop()
        self.assertEqual(self.read("status.json")["remaining_seconds"], 0)

    def test_full_stream_and_bounded_window(self):
        monitor = self.monitor()
        # Atomic writes are separately exercised in the other tests. Avoid 2005
        # filesystem replacements here while retaining the actual JSONL stream.
        with patch.object(runner, "atomic_json"):
            for i in range(2005):
                monitor.event(i, {"training/v_loss": 0.5})
            monitor.event(2004, {"training/policy_loss": 0.25})
        self.assertEqual(len(monitor.events), 2000)
        self.assertEqual(monitor.events[0]["steps"], 5)
        self.assertTrue(monitor.state["progress_window_truncated"])
        self.assertEqual(monitor.total_events, 2006)
        stream = (self.directory / "metrics.jsonl").read_text().splitlines()
        self.assertEqual(len(stream), 2006)
        self.assertEqual(json.loads(stream[0])["steps"], 0)
        self.assertEqual(monitor.events[-1]["metrics"]["training/v_loss"], .5)
        self.assertEqual(monitor.events[-1]["metrics"]["training/policy_loss"], .25)
        monitor.check_stop()  # A full UI window must never stop training.

    def test_metrics_reject_regression_and_nonfinite(self):
        monitor = self.monitor()
        monitor.event(8, {"training/v_loss": .1})
        with self.assertRaises(ValueError):
            monitor.event(7, {})
        with self.assertRaises(FloatingPointError):
            monitor.event(9, {"training/v_loss": float("nan")})
        self.assertEqual(monitor.total_events, 1)

    def test_initial_stop_cancels_without_import_or_checkpoint(self):
        (self.directory / "stop.request").write_text("{}")
        runner.main(["--mode", "original", "--run-dir", str(self.directory)])
        state = self.read("status.json")
        self.assertEqual(state["phase"], "cancelled")
        self.assertIsNone(state["checkpoint"])
        self.assertFalse(state["completed"])
        self.assertTrue((self.directory / "report.json").is_file())

    def test_setup_failure_is_published(self):
        with patch.object(runner, "run_training", side_effect=RuntimeError("setup failed")):
            with self.assertRaisesRegex(RuntimeError, "setup failed"):
                runner.main(["--mode", "torque200", "--run-dir", str(self.directory)])
        self.assertEqual(self.read("status.json")["phase"], "failed")
        self.assertIn("setup failed", self.read("report.json")["error"])

    def test_restore_metadata_rejects_other_mode_and_dimension(self):
        checkpoint = self.directory / "checkpoints" / "000000000100"
        checkpoint.mkdir(parents=True)
        for report in ({"mode": "torque200"}, {"obs_dim": 24}):
            runner.atomic_json(self.directory / "report.json", report)
            with self.assertRaises(ValueError):
                runner.validate_restore_metadata(checkpoint, "original", 26)
        runner.atomic_json(self.directory / "report.json", {})
        runner.validate_restore_metadata(checkpoint, "original", 26)

    def test_atomic_json_replaces_and_disallows_nan(self):
        path = self.directory / "sample.json"
        runner.atomic_json(path, {"a": 1})
        runner.atomic_json(path, {"a": 2})
        self.assertEqual(self.read("sample.json"), {"a": 2})
        self.assertFalse(path.with_name("sample.json.tmp").exists())
        with self.assertRaises(ValueError):
            runner.atomic_json(path, {"a": float("nan")})
        self.assertEqual(self.read("sample.json"), {"a": 2})


if __name__ == "__main__":
    unittest.main()
