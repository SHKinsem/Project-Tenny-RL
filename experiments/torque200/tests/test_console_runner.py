"""Runner lifecycle tests: no ML imports, GPU, WSL, or training required."""
import argparse
from contextlib import nullcontext, redirect_stderr
import io
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock

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

    def test_teacher_preparation_does_not_consume_ppo_budget(self):
        self.args.mode = "wheel_torque_bc"
        self.args.duration_seconds = 30.
        monitor = self.monitor()
        self.now[0] = 70.
        monitor.check_stop()
        self.assertEqual(monitor.remaining, 30.)
        monitor.begin_training_clock()
        self.assertEqual(monitor.state["preparation_seconds"], 70.)
        self.now[0] = 100.
        with self.assertRaisesRegex(runner.StopRequested, "wall_budget"):
            monitor.check_stop()

    def test_teacher_preparation_remains_cancellable(self):
        self.args.mode = "wheel_torque_bc"
        monitor = self.monitor()
        (self.directory / "stop.request").write_text("{}")
        with self.assertRaisesRegex(runner.StopRequested, "user_stop"):
            monitor.check_stop()
        self.assertFalse(monitor.training_clock_started)

    def test_leg_force_preparation_preserves_full_twenty_minute_budget(self):
        self.args.mode = "leg_force_bc"
        self.args.duration_seconds = 1200.
        monitor = self.monitor()
        self.now[0] = 180.
        monitor.check_stop()
        self.assertFalse(monitor.training_clock_started)
        self.assertEqual(monitor.remaining, 1200.)
        monitor.begin_training_clock()
        self.now[0] = 1379.
        monitor.check_stop()
        self.assertEqual(monitor.remaining, 1.)
        self.now[0] = 1380.
        with self.assertRaisesRegex(runner.StopRequested, "wall_budget"):
            monitor.check_stop()

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

    def force_cli(self, *extra):
        return runner.parse_args(["--mode", "leg_force_bc", "--run-dir", str(self.directory), *extra])

    def test_cli_option_file_is_normalized_and_seed_is_not_defaulted_over_it(self):
        from force_teacher.options import normalize_options
        path = self.directory / "options.json"
        runner.atomic_json(path, {"seed": 47, "wheel_head": "residual", "fixed_normalization": False})
        args = self.force_cli("--experiment-config", str(path))
        self.assertEqual(args.seed, 47)
        self.assertEqual(args.experiment_config, normalize_options(
            dict(seed=47, wheel_head="residual", fixed_normalization=False)))
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.force_cli("--experiment-config", str(path), "--seed", "0")

    def test_seed_only_cli_is_folded_into_saved_experiment(self):
        args = self.force_cli("--seed", "52")
        self.assertEqual(args.seed, 52)
        self.assertEqual(args.experiment_config["seed"], 52)

    def test_option_file_and_run_request_cannot_disagree(self):
        runner.atomic_json(self.directory / "request.json", {"experiment_config": {"seed": 5}})
        path = self.directory / "options.json"
        runner.atomic_json(path, {"seed": 6})
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.force_cli("--experiment-config", str(path))

    def test_option_file_wrong_mode_invalid_type_and_unknown_key_fail_before_monitor(self):
        path = self.directory / "options.json"
        for value in (None, [], {"bogus": True}, {"seed": True}):
            path.write_text(json.dumps(value))
            with self.subTest(value=value), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.force_cli("--experiment-config", str(path))
        runner.atomic_json(path, {})
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            runner.parse_args(["--mode", "original", "--run-dir", str(self.directory),
                               "--experiment-config", str(path)])
        self.assertFalse((self.directory / "status.json").exists())

    def test_direct_resume_reads_v3_metadata_and_rejects_new_head(self):
        from force_teacher.options import legacy_options
        checkpoint = self.directory / "saved" / "checkpoints" / "000000000100"
        checkpoint.mkdir(parents=True)
        runner.atomic_json(checkpoint.parent.parent / "policy_metadata.json", dict(mode="leg_force_bc",
            teacher_schema="leg-force-reachable-teacher-bc-v3", policy_head="wheel_logit_residual_v1",
            wheel_residual_scale=.05))
        args = self.force_cli("--restore", str(checkpoint))
        self.assertEqual(args.experiment_config, legacy_options())
        path = self.directory / "options.json"
        runner.atomic_json(path, {"wheel_head": "absolute"})
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.force_cli("--restore", str(checkpoint), "--experiment-config", str(path))

    def test_main_saves_options_and_status_before_training_and_preserves_request_fields(self):
        runner.atomic_json(self.directory / "request.json", dict(experiment_config={"seed": 61}, created_at="preserved"))
        def entered(args, monitor):
            self.assertEqual(self.read("request.json")["experiment_config"], args.experiment_config)
            self.assertEqual(self.read("status.json")["experiment_config"], args.experiment_config)
            self.assertEqual(self.read("request.json")["created_at"], "preserved")
            self.assertEqual(args.seed, 61)
        with patch("hybrid.managed_training.runtime_locks", return_value=nullcontext()), \
             patch.object(runner, "run_training", side_effect=entered):
            runner.main(["--mode", "leg_force_bc", "--run-dir", str(self.directory)])

    def test_adaptive_normalizer_reports_real_change_and_fixed_mode_rejects_it(self):
        monitor = Mock()
        runner.record_normalizer_change(monitor, .25, {"fixed_normalization": False})
        monitor.publish.assert_called_with(ppo_normalizer_max_abs_change=.25)
        with self.assertRaisesRegex(RuntimeError, "frozen"):
            runner.record_normalizer_change(monitor, .25, {"fixed_normalization": True})
        runner.record_normalizer_change(monitor, 0., {"fixed_normalization": True})


if __name__ == "__main__":
    unittest.main()
