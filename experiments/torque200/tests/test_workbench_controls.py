"""Controller contracts without GPU, live services, or a Linux child process."""
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training_control import ControlError, TrainingController, _read, _write
from workbench_api import WorkbenchReader


class ControlTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.artifacts = self.base / "artifacts"
        self.controller = TrainingController(self.artifacts, self.base / "reference", self.base)
        self.controller._locked = nullcontext
        self.controller._alive = lambda _: True
        self.controller._latest = lambda *_: None
        self.run = self.controller.root / "original" / "current"
        self.run.mkdir(parents=True)
        _write(self.controller.root / "active.json", dict(mode="original", run_dir=str(self.run), pid=123, started_at="2000-01-01T00:00:00Z"))
        self.status()

    def status(self, **changes):
        value = dict(mode="original", run_id=self.run.name, run_dir=str(self.run), phase="training",
                     workbench_control_protocol=1, elapsed_seconds=17., started_at="2000-01-01T00:00:00Z")
        value.update(changes)
        _write(self.run / "status.json", value)

    def test_old_runner_rejects_new_controls_without_writing_requests(self):
        self.status(workbench_control_protocol=None)
        for method in (self.controller.pause, self.controller.resume, self.controller.save):
            with self.assertRaisesRegex(ControlError, "下一轮"):
                method()
        self.assertFalse((self.run / "workbench-control.json").exists())
        self.assertFalse((self.run / "workbench-save-requests").exists())
        self.assertFalse(self.controller.capabilities("original")["pauseTraining"])

    def test_pause_resume_keep_process_and_save_queue_independent(self):
        saved = self.controller.save()
        pause = self.controller.pause()
        resumed = self.controller.resume()
        self.assertNotEqual(pause["requestId"], resumed["requestId"])
        self.assertEqual(_read(self.run / "workbench-control.json")["action"], "resume")
        request = _read(self.run / "workbench-save-requests" / (saved["requestId"] + ".json"))
        self.assertEqual(request["runId"], self.run.name)
        self.assertEqual(request["action"], "save")
        self.assertEqual(_read(self.controller.root / "active.json")["pid"], 123)
        self.assertFalse((self.run / "stop.request").exists())

    def test_retried_pause_does_not_override_a_later_resume(self):
        identifier = str(uuid.uuid4())
        self.controller.pause(identifier)
        self.controller.resume()
        result = self.controller.pause(identifier)
        self.assertEqual(result["requestId"], identifier)
        self.assertEqual(_read(self.run / "workbench-control.json")["action"], "resume")
        with self.assertRaises(ValueError):
            self.controller.resume(identifier)
        with self.assertRaises(ValueError):
            self.controller.save(identifier)
        saved = self.controller.save()
        with self.assertRaises(ValueError):
            self.controller.pause(saved["requestId"])

    def test_saved_receipt_is_idempotent_and_queue_is_bounded(self):
        identifier = str(uuid.uuid4())
        result = self.controller.save(identifier)
        target = self.run / "workbench-save-requests" / (identifier + ".json")
        before = target.read_bytes()
        self.controller.save(identifier)
        self.assertEqual(before, target.read_bytes())
        receipts = self.run / "workbench-save-results"
        receipts.mkdir()
        _write(receipts / (identifier + ".json"), dict(requestId=identifier, status="completed", checkpoint="saved"))
        target.unlink()
        result = self.controller.save(identifier)
        self.assertEqual(result["result"]["status"], "completed")
        self.assertFalse(target.exists())
        for _ in range(32):
            self.controller.save()
        with self.assertRaisesRegex(ControlError, "Too many"):
            self.controller.save()

    def test_protocol_elapsed_is_not_replaced_by_wall_clock_during_pause(self):
        self.status(phase="paused", paused=True, elapsed_seconds=17.)
        self.assertEqual(self.controller.status()["elapsed_seconds"], 17.)
        self.assertTrue(self.controller.capabilities("original")["saveSnapshot"])
        for phase in ("saving", "exporting", "verifying", "completed", "stopping"):
            self.status(phase=phase)
            with self.subTest(phase=phase), self.assertRaises(ControlError):
                self.controller.pause()

    def test_exact_historical_checkpoint_is_selected_and_cross_mode_rejected(self):
        older = self.verified_run("older", 11)
        newer = self.verified_run("newer", 22)
        selected = self.controller._select_checkpoint("original", str(older))
        self.assertEqual(selected["checkpoint"], str(older))
        self.assertNotEqual(selected["checkpoint"], str(newer))
        option = self.controller._checkpoint_option(selected)
        self.assertEqual(option["path"], str(older))
        self.assertIsNone(option["trainedSeconds"])
        self.assertIsNone(option["trainedSteps"])
        self.assertFalse(option["optimizerRestored"])
        with self.assertRaises(ControlError):
            self.controller._select_checkpoint("torque200", str(older))
        with self.assertRaises((ControlError, ValueError)):
            self.controller._select_checkpoint("original", str(self.base))

    def verified_run(self, name, steps):
        run = self.controller.root / "original" / name
        checkpoint = run / "checkpoints" / f"{steps:012d}"
        checkpoint.mkdir(parents=True, exist_ok=True)
        (run / "policy.cpp.txt").write_text("fixture")
        _write(run / "status.json", dict(mode="original", run_id=name, run_dir=str(run), phase="completed", steps=steps))
        _write(run / "report.json", dict(mode="original", all_parameters_finite=True,
                   parameter_tree_max_abs_error=0, export_max_abs_error=.00001, export_atol=.001,
                   checkpoint=str(checkpoint), export_path=str(run / "policy.cpp.txt"), steps=steps))
        return checkpoint

    def test_snapshot_completion_invalidates_latest_and_final_save_remains_newest(self):
        old = self.verified_run("older", 11)
        snapshot = self.controller.root / "original" / "later-snapshot"
        snapshot.mkdir()
        receipts = self.run / "workbench-save-results"
        receipts.mkdir()
        record = _read(self.controller.root / "active.json")
        first = TrainingController._latest(self.controller, "original", record, True)
        self.assertEqual(first["checkpoint"], str(old))
        saved = self.verified_run("later-snapshot", 22)
        _write(receipts / (str(uuid.uuid4()) + ".json"), {"status":"completed"})
        second = TrainingController._latest(self.controller, "original", record, True)
        self.assertEqual(second["checkpoint"], str(saved))
        final = self.verified_run("current", 33)
        record["ended_at"] = "2026-01-01T00:00:00Z"
        latest = TrainingController._latest(self.controller, "original", record, False)
        self.assertEqual(latest["checkpoint"], str(final))

    def test_launch_passes_exact_selected_path_and_explicit_parameters_with_cuda(self):
        self.controller._alive = lambda _: False
        selected = self.verified_run("older", 11)
        self.verified_run("newer", 22)
        child = SimpleNamespace(pid=9999999, poll=lambda: None)
        with patch("training_control.subprocess.Popen", return_value=child) as spawn:
            self.controller.start("original", checkpoint=str(selected), parameters={"num_envs":64,"learning_rate":2e-5})
        argv = spawn.call_args.args[0]
        environment = spawn.call_args.kwargs["env"]
        self.assertEqual(argv[argv.index("--restore")+1], str(selected))
        self.assertEqual(environment["JAX_PLATFORMS"], "cuda")
        self.assertEqual(environment["XLA_PYTHON_CLIENT_PREALLOCATE"], "false")
        self.assertEqual(json.loads(environment["TENNY_WORKBENCH_PARAMETERS"]), {"num_envs":64,"learning_rate":2e-5})
        request = _read(Path(argv[argv.index("--run-dir")+1]) / "request.json")
        self.assertEqual(request["requested_effective_parameters"]["num_envs"], 64)
        self.assertEqual(request["restore"], str(selected))

    def test_empty_edits_do_not_add_environment_override_or_disable_checkpoint_profile(self):
        self.controller._alive = lambda _: False
        selected = self.verified_run("legacy-profile", 11)
        prior = dict(num_envs=64, batch_size=128, num_minibatches=1,
                     unroll_length=32, num_updates_per_batch=2)
        _write(selected.parent.parent / "ppo_config.json", prior)
        environment = {key: value for key, value in os.environ.items() if key != "TENNY_WORKBENCH_PARAMETERS"}
        child = SimpleNamespace(pid=9999999, poll=lambda: None)
        for checkpoint in (None, str(selected)):
            for edits in (None, {}):
                with self.subTest(checkpoint=checkpoint, edits=edits), \
                        patch.dict(os.environ, environment, clear=True), \
                        patch("training_control.subprocess.Popen", return_value=child) as spawn:
                    self.controller.start("original", checkpoint=checkpoint, parameters=edits)
                    self.assertNotIn("TENNY_WORKBENCH_PARAMETERS", spawn.call_args.kwargs["env"])
                    argv = spawn.call_args.args[0]
                    request = _read(Path(argv[argv.index("--run-dir") + 1]) / "request.json")
                    self.assertEqual(request["parameters"], {})
                    if checkpoint:
                        self.assertEqual({key: request["requested_effective_parameters"][key] for key in prior}, prior)

    def test_empty_current_edits_still_pass_inherited_explicit_checkpoint_parameters(self):
        self.controller._alive = lambda _: False
        selected = self.verified_run("edited-profile", 11)
        _write(selected.parent.parent / "request.json", {"parameters": {"learning_rate": 1e-5}})
        child = SimpleNamespace(pid=9999999, poll=lambda: None)
        with patch("training_control.subprocess.Popen", return_value=child) as spawn:
            self.controller.start("original", checkpoint=str(selected), parameters={})
        self.assertEqual(json.loads(spawn.call_args.kwargs["env"]["TENNY_WORKBENCH_PARAMETERS"]),
                         {"learning_rate": 1e-5})
        argv = spawn.call_args.args[0]
        request = _read(Path(argv[argv.index("--run-dir") + 1]) / "request.json")
        self.assertEqual(request["parameters"], {"learning_rate": 1e-5})
        self.assertEqual(request["requested_effective_parameters"]["learning_rate"], 1e-5)

    def test_external_save_location_is_visible_to_read_only_reader(self):
        self.controller._alive = lambda _: False
        external = self.base / "external"
        external.mkdir()
        self.controller.configure_storage(str(external))
        run = external / "original" / "external-run"
        run.mkdir(parents=True)
        _write(run / "status.json", dict(mode="original", run_id=run.name, run_dir=str(run), phase="training", steps=20))
        _write(self.controller.root / "active.json", dict(mode="original", run_dir=str(run), pid=123))
        self.assertEqual(self.controller._run_path({"mode":"original", "run_dir":str(run)}), run)
        result = WorkbenchReader(self.artifacts).state("original")
        self.assertEqual(result["run"]["id"], "external-run")
        self.assertEqual(result["run"]["steps"], 20)
        self.assertTrue(self.controller.resolve_directory(str(run))["ok"])
        with self.assertRaises(ValueError):
            self.controller.resolve_directory(str(self.base))

    def test_configure_is_locked_during_training_and_invalid_ids_are_rejected(self):
        with self.assertRaises(ControlError):
            self.controller.configure_storage(str(self.base))
        for identifier in ("../outside", "0"*32, str(uuid.UUID(int=0)), 23, True):
            with self.subTest(identifier=identifier), self.assertRaises(ValueError):
                self.controller.save(identifier)


if __name__ == "__main__":
    unittest.main()
