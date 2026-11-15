"""Real CPU-only fake children exercise the supervisor's lifecycle and lock."""
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import threading
import time
import unittest
from contextlib import nullcontext
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training_control import ControlError, TrainingController, _write
from force_teacher.options import normalize_options, legacy_options


class ExperimentLifecycleTests(unittest.TestCase):
    """No children or ML; exercise the actual request and CLI boundary on all hosts."""
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name).resolve()
        self.script = self.base / "console_train.py"
        self.script.write_text("# unused fake child\n")
        self.controller = TrainingController(self.base / "artifacts", self.base, self.base, self.script)
        for name, value in (("_locked", nullcontext()), ("_current", ({}, {}, False)),
                            ("_latest", None), ("_process", None)):
            mocked = patch.object(self.controller, name, return_value=value)
            setattr(self, name + "_mock", mocked.start())
            self.addCleanup(mocked.stop)
        state = patch.object(self.controller, "_status", side_effect=lambda mode: json.loads(
            (self.controller.root / "active.json").read_text()))
        state.start(); self.addCleanup(state.stop)
        child = patch("training_control.subprocess.Popen", return_value=MagicMock(pid=43210))
        self.child = child.start(); self.addCleanup(child.stop)
        environment = patch("hybrid.readiness.prepare_environment", return_value={})
        environment.start(); self.addCleanup(environment.stop)

    def saved_checkpoint(self, legacy=False, options=None):
        run = self.controller.root / "leg_force_bc" / "saved"
        checkpoint = run / "checkpoints" / "000000000128"
        checkpoint.mkdir(parents=True)
        meta = dict(mode="leg_force_bc")
        if legacy:
            meta.update(teacher_schema="leg-force-reachable-teacher-bc-v3",
                        policy_head="wheel_logit_residual_v1", wheel_residual_scale=.05)
        else:
            meta.update(teacher_schema="leg-force-configurable-bc-v4", experiment_config=options)
        _write(run / "policy_metadata.json", meta)
        self._latest_mock.return_value = {"checkpoint": str(checkpoint)}
        return checkpoint

    def test_invalid_options_rejected_before_run_directory_or_process(self):
        bad = [[], {"unknown": True}, {"seed": True}, {"wheel_head": "servo"},
               {"teacher_source": "reference", "dagger_rounds": 2}]
        for value in bad:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.controller.start("leg_force_bc", experiment_config=value)
        with self.assertRaises(ValueError):
            self.controller.start("original", experiment_config={})
        self.assertFalse((self.controller.root / "leg_force_bc").exists())
        self.child.assert_not_called()

    def test_request_status_child_seed_and_runner_load_preserve_full_options(self):
        from console_train import parse_args
        supplied = dict(wheel_head="residual", wheel_residual_scale=.2,
                        fixed_normalization=False, deterministic_eval=False, seed=37)
        state = self.controller.start("leg_force_bc", 30, experiment_config=supplied)
        expected = normalize_options(supplied)
        run = Path(state["run_dir"])
        for path in (run / "request.json", run / "status.json"):
            self.assertEqual(json.loads(path.read_text())["experiment_config"], expected)
        self.assertEqual(state["experiment_config"], expected)
        argv = self.child.call_args.args[0]
        parsed = parse_args(argv[2:])
        self.assertEqual(parsed.experiment_config, expected)
        self.assertEqual(parsed.seed, 37)

    def test_resume_inherits_saved_options_and_seed_instead_of_defaults(self):
        from console_train import parse_args
        expected = normalize_options(dict(wheel_head="residual", seed=91, fixed_normalization=False))
        self.saved_checkpoint(options=expected)
        state = self.controller.start("leg_force_bc", resume=True)
        self.assertEqual(state["experiment_config"], expected)
        parsed = parse_args(self.child.call_args.args[0][2:])
        self.assertEqual(parsed.experiment_config, expected)
        self.assertEqual(parsed.seed, 91)

    def test_legacy_v3_resume_retains_residual_head_through_runner(self):
        from console_train import parse_args
        self.saved_checkpoint(legacy=True)
        state = self.controller.start("leg_force_bc", resume=True)
        self.assertEqual(state["experiment_config"], legacy_options())
        parsed = parse_args(self.child.call_args.args[0][2:])
        self.assertEqual(parsed.experiment_config["wheel_head"], "residual")
        self.assertEqual(parsed.experiment_config, legacy_options())

    def test_resume_rejects_reinterpretation_before_creating_run(self):
        self.saved_checkpoint(legacy=True)
        before = list((self.controller.root / "leg_force_bc").iterdir())
        with self.assertRaisesRegex(ValueError, "differs"):
            self.controller.start("leg_force_bc", resume=True, experiment_config={"wheel_head": "absolute"})
        self.assertEqual(list((self.controller.root / "leg_force_bc").iterdir()), before)
        self.child.assert_not_called()


FAKE_RUNNER = r'''
import argparse, json, os, pathlib, time
p = argparse.ArgumentParser()
for flag in ("mode", "run-dir", "original-root", "torque-root", "duration-seconds", "seed", "restore"):
    p.add_argument("--" + flag)
a = p.parse_args()
run = pathlib.Path(a.run_dir)
def write(name, value):
    t = run / (name + ".tmp")
    t.write_text(json.dumps(value))
    t.replace(run / name)
write("env.json", {k: os.environ.get(k) for k in ("JAX_PLATFORMS", "CUDA_VISIBLE_DEVICES", "XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_MEM_FRACTION", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "JAX_COMPILATION_CACHE_DIR")})
write("status.json", {"phase": "training", "mode": a.mode})
while not (run / "stop.request").exists():
    time.sleep(.01)
write("status.json", {"phase": "saving", "mode": a.mode})
while not (run / "allow_save").exists():
    time.sleep(.01)
(run / "checkpoint").mkdir()
(run / "export.json").write_text("{}")
write("report.json", {"mode": a.mode, "checkpoint": str(run / "checkpoint"), "export_path": str(run / "export.json"), "all_parameters_finite": True, "parameter_tree_max_abs_error": 0, "export_max_abs_error": 0, "export_atol": .001})
write("status.json", {"phase": "completed", "mode": a.mode, "stop_reason": "user_requested"})
while not (run / "allow_exit").exists():
    time.sleep(.01)
'''


@unittest.skipUnless(sys.platform.startswith("linux"), "fcntl and /proc are Linux runtime requirements")
class TrainingControlTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="console-control-test-")
        self.base = Path(self.temporary.name)
        self.script = self.base / "console_train.py"
        self.script.write_text(FAKE_RUNNER)
        self.controllers = []
        self.controller = self.new_controller()
        self.children = []

    def new_controller(self):
        controller = TrainingController(self.base / "artifacts", self.base, self.base, self.script)
        self.controllers.append(controller)
        return controller

    def tearDown(self):
        for child in self.children:
            try:
                os.kill(child, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for controller in self.controllers:
            for child in controller._children.values():
                try:
                    child.wait(timeout=2)
                except Exception:
                    pass
        self.temporary.cleanup()

    def wait(self, predicate):
        limit = time.monotonic() + 4
        while time.monotonic() < limit:
            value = predicate()
            if value:
                return value
            time.sleep(.015)
        self.fail("Fake child did not reach expected state")

    def start(self, mode="original"):
        state = self.controller.start(mode)
        self.children.append(state["pid"])
        self.wait(lambda: self.controller.status(mode)["phase"] == "training")
        return Path(state["run_dir"])

    def test_preview_request_cancel_and_legacy(self):
        run = self.start()
        with self.assertRaises(ControlError):
            self.controller.request_preview("original")
        _write(run / "status.json", {"phase": "training", "mode": "original", "preview_protocol": 1})
        first = self.controller.request_preview("original")
        self.assertEqual(first["kind"], "live")
        second = self.controller.request_preview("original")
        self.controller.cancel_preview(first)
        self.assertEqual(json.loads((run / "preview.request.json").read_text())["request_id"], second["request_id"])
        self.controller.cancel_preview(second)
        self.assertFalse((run / "preview.request.json").exists())
        with self.assertRaises(ControlError):
            self.controller.request_preview("torque200")

    def finish(self, run):
        self.controller.stop()
        (run / "allow_save").touch()
        (run / "allow_exit").touch()
        self.wait(lambda: not self.controller.status()["active"])

    def test_two_instances_and_concurrent_start_share_global_lock(self):
        controllers = [self.controller, self.new_controller()]
        barrier = threading.Barrier(2)
        outcomes = []
        def start(controller):
            barrier.wait()
            try:
                outcomes.append(controller.start("original"))
            except ControlError as error:
                outcomes.append(error)
        threads = [threading.Thread(target=start, args=(c,)) for c in controllers]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        started = [x for x in outcomes if isinstance(x, dict)]
        self.assertEqual(len(started), 1)
        self.assertEqual(sum(isinstance(x, ControlError) for x in outcomes), 1)
        self.children.append(started[0]["pid"])
        with self.assertRaises(ControlError):
            self.new_controller().start("torque200")

    def test_restart_reconnect_and_explicit_environment(self):
        run = self.start()
        state = self.new_controller().status("original")
        self.assertTrue(state["active"])
        self.assertEqual(state["run_dir"], str(run))
        env = json.loads((run / "env.json").read_text())
        self.assertEqual(env["JAX_PLATFORMS"], "cuda")
        self.assertEqual(env["XLA_PYTHON_CLIENT_MEM_FRACTION"], "0.35")
        self.assertEqual(env["XLA_PYTHON_CLIENT_PREALLOCATE"], "false")
        self.assertEqual(json.loads((run / "request.json").read_text())["restore"], None)

    def test_stop_is_cooperative_and_completed_waits_for_exit(self):
        run = self.start()
        self.controller.stop()
        self.assertTrue((run / "stop.request").exists())
        self.wait(lambda: self.controller.status()["phase"] == "saving")
        self.assertEqual(self.controller.stop()["phase"], "saving")
        self.assertTrue(self.controller.status()["active"])
        (run / "allow_save").touch()
        self.wait(lambda: json.loads((run / "status.json").read_text())["phase"] == "completed")
        self.assertTrue(self.controller.status()["active"])
        self.assertIsNone(self.controller.latest("original"))
        with self.assertRaises(ControlError):
            self.controller.start("original")
        (run / "allow_exit").touch()
        self.wait(lambda: not self.controller.status()["active"])
        result = self.controller.latest("original")
        self.assertTrue(result["verified"])
        self.assertEqual(result["run_id"], run.name)
        resumed = self.controller.start("original", resume=True)
        self.children.append(resumed["pid"])
        request = json.loads((Path(resumed["run_dir"]) / "request.json").read_text())
        self.assertEqual(request["restore"], str(run / "checkpoint"))

    def test_dead_child_is_interrupted_and_keeps_artifacts(self):
        run = self.start()
        (run / "partial-checkpoint").write_text("keep")
        os.kill(self.children[-1], signal.SIGTERM)
        self.wait(lambda: not self.new_controller().status()["active"])
        state = self.controller.status()
        self.assertEqual(state["phase"], "interrupted")
        self.assertEqual((run / "partial-checkpoint").read_text(), "keep")

    def test_mode_isolation_and_previous_run_status(self):
        run = self.start("original")
        self.finish(run)
        self.assertIsNone(self.controller.latest("torque200"))
        with self.assertRaises(ControlError):
            self.controller.start("torque200", resume=True)
        self.start("torque200")
        original = self.controller.status("original")
        self.assertEqual(original["run_dir"], str(run))
        self.assertEqual(original["phase"], "completed")
        self.assertTrue(original["active"])
        self.assertEqual(original["active_mode"], "torque200")

    def test_invalid_inputs_and_untrusted_artifact_paths(self):
        for mode in ("../original", "", None, 3):
            with self.assertRaises(ValueError):
                self.controller.start(mode)
        for duration in (-1, float("nan"), float("inf"), "1", True, 604801):
            with self.assertRaises(ValueError):
                self.controller.start("original", duration)
        run = self.start()
        self.finish(run)
        report = json.loads((run / "report.json").read_text())
        report["checkpoint"] = str(self.base)
        _write(run / "report.json", report)
        self.assertIsNone(self.new_controller().latest("original"))
        report["checkpoint"] = str(run / "checkpoint")
        report["mode"] = "torque200"
        _write(run / "report.json", report)
        self.assertIsNone(self.new_controller().latest("original"))

    def test_reused_pid_requires_exact_run_argument_and_start_time(self):
        run = self.start()
        record = json.loads((self.controller.root / "active.json").read_text())
        self.assertTrue(self.controller._alive(record))
        record["process_start"] = "not-the-process-start-time"
        self.assertFalse(self.controller._alive(record))
        record["process_start"] = None
        record["run_dir"] = str(run) + "suffix"
        self.assertFalse(self.controller._alive(record))


if __name__ == "__main__":
    unittest.main()
