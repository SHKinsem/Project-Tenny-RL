"""Safe-boundary runner tests using temporary files, fake clocks and fake JAX."""

import copy
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4


_previous = list(sys.path)
try:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    timed = importlib.import_module("timed_console_train")
    runtime_module = importlib.import_module("workbench_runtime")
finally:
    sys.path[:] = _previous


class Clock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value


class WorkbenchRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="workbench-runtime-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        environment = patch.dict(os.environ, {"TENNY_WORKBENCH_PARAMETERS": "{}", "TENNY_EVAL_BACKEND": "auto"})
        environment.start()
        self.addCleanup(environment.stop)

    def monitor(self, mode="original", parameters=None):
        run = self.root / "console" / mode / ("run_" + uuid4().hex)
        run.mkdir(parents=True)
        args = SimpleNamespace(run_dir=run, mode=mode, restore=None, duration_seconds=600, seed=0,
                               original_root=self.root / "reference", torque_root=self.root / "torque",
                               runtime_profile_requested="auto")
        clock = Clock()
        with patch.dict(os.environ, {"TENNY_WORKBENCH_PARAMETERS": json.dumps(parameters or {})}):
            monitor = timed.TimedRunMonitor(args, clock=clock)
        self.addCleanup(lambda: monitor.timing.finish("fixture_complete", steps=timed.observed_steps(monitor)))
        if not monitor.training_clock_started:
            monitor.begin_training_clock()
        clock.value = 10
        monitor.publish(phase="training", stage="optimization", steps=100)
        return monitor, clock

    def request(self, monitor, action, identity=None):
        value = dict(schemaVersion=1, runId=monitor.args.run_dir.name, requestId=identity or str(uuid4()),
                     action=action, createdAt="2026-11-15T04:45:00+00:00")
        self.write_request(monitor, value)
        return value

    @staticmethod
    def write_request(monitor, value):
        run = monitor.args.run_dir
        if value["action"] == "save":
            directory = run / "workbench-save-requests"
            directory.mkdir(exist_ok=True)
            path = directory / (value["requestId"] + ".json")
        else:
            path = run / "workbench-control.json"
        path.write_text(json.dumps(value), encoding="utf8")

    @staticmethod
    def configurations(monitor):
        for name, value in (("env_config.json", {"ctrl_dt": 0.01}),
                            ("ppo_config.json", {"num_envs": 128, "batch_size": 128, "num_minibatches": 1,
                                                 "unroll_length": 75, "num_updates_per_batch": 4})):
            (monitor.args.run_dir / name).write_text(json.dumps(value), encoding="utf8")

    @staticmethod
    def fake_builder(run, *, steps, factory, params, elapsed_seconds, stage):
        checkpoint = run / "checkpoints" / f"{steps:012d}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "weights.bin").write_bytes(b"synthetic complete parameters")
        export = run / "policy.cpp.txt"
        export.write_text("synthetic exported actor", encoding="utf8")
        stage("verifying")
        return dict(checkpoint=str(checkpoint), export_path=str(export), actual_timesteps=steps,
                    training_stop_elapsed_seconds=elapsed_seconds, all_parameters_finite=True,
                    parameter_tree_max_abs_error=0, restore_max_abs_error=0.0,
                    export_max_abs_error=0.0, export_atol=1e-3)

    def test_pause_waits_for_boundary_and_freezes_budget_until_same_run_resumes(self):
        monitor, clock = self.monitor()
        control = monitor._workbench_runtime
        pause = self.request(monitor, "pause")
        monitor.check_stop()
        self.assertTrue(monitor.state["pause_requested"])
        self.assertFalse(monitor.state["paused"])
        self.assertEqual(monitor.state["workbench_control_result"]["status"], "pending")
        observed = []

        def sleep(_duration):
            clock.value += 1
            observed.append(monitor.elapsed)
            if len(observed) == 5:
                self.request(monitor, "resume")

        control.sleep = sleep
        control.boundary(100, "factory", "params")
        self.assertEqual(observed, [10] * 5)
        self.assertEqual(monitor.elapsed, 10)
        self.assertEqual(control.excluded_seconds(), 5)
        self.assertFalse(monitor.state["paused"])
        self.assertFalse(monitor.state["pause_requested"])
        self.assertEqual(monitor.state["phase"], "training")
        self.assertEqual(monitor.state["workbench_control_result"]["action"], "resume")
        self.assertEqual(monitor.state["workbench_control_result"]["status"], "applied")
        self.assertEqual(monitor.state["workbench_control_protocol"], 1)
        clock.value += 2
        self.assertEqual(monitor.elapsed, 12)

    def test_stop_while_paused_raises_original_cooperative_stop_and_excludes_wait(self):
        monitor, clock = self.monitor()
        control = monitor._workbench_runtime
        self.request(monitor, "pause")
        waits = []

        def sleep(_duration):
            clock.value += 1
            waits.append(True)
            if len(waits) == 2:
                (monitor.args.run_dir / "stop.request").write_text("stop", encoding="utf8")

        control.sleep = sleep
        with self.assertRaises(timed.console.StopRequested):
            control.boundary(100, "factory", "params")
        self.assertEqual(monitor.elapsed, 10)
        self.assertEqual(monitor.state["phase"], "stopping")
        self.assertFalse(monitor.state["paused"])
        self.assertFalse(monitor.state["pause_requested"])

    def test_save_while_paused_has_its_own_receipt_and_does_not_resume_training(self):
        monitor, clock = self.monitor()
        control = monitor._workbench_runtime
        self.request(monitor, "pause")
        waits, saves, requests = [], [], []

        def writer(request, *, steps, factory, params, elapsed_seconds, stage):
            saves.append((control.paused, monitor.elapsed, steps, factory, params))
            stage("verifying")
            clock.value += 3
            return dict(checkpoint="checkpoint-path", checkpointId=str(uuid4()), capturedSteps=steps,
                        capturedElapsedSeconds=elapsed_seconds)

        def sleep(_duration):
            clock.value += 1
            waits.append(True)
            if len(waits) == 1:
                requests.append(self.request(monitor, "save"))
            else:
                self.request(monitor, "resume")

        control.snapshot_writer, control.sleep = writer, sleep
        factory, params = object(), object()
        control.boundary(100, factory, params)
        self.assertEqual(len(saves), 1)
        self.assertEqual(saves[0][:3], (True, 10, 100))
        self.assertIs(saves[0][3], factory)
        self.assertIs(saves[0][4], params)
        self.assertEqual(monitor.elapsed, 10)
        result = monitor.state["workbench_save_results"][0]
        self.assertEqual(result["requestId"], requests[0]["requestId"])
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["capturedElapsedSeconds"], 10)
        self.assertTrue((monitor.args.run_dir / "workbench-save-results" / (result["requestId"] + ".json")).is_file())
        self.assertEqual(list((monitor.args.run_dir / "workbench-save-requests").iterdir()), [])
        self.write_request(monitor, requests[0])
        control.boundary(100, factory, params)
        self.assertEqual(len(saves), 1)
        self.assertEqual(list((monitor.args.run_dir / "workbench-save-requests").iterdir()), [])

    def test_queue_does_not_lose_save_when_pause_intent_is_replaced_by_resume(self):
        monitor, clock = self.monitor()
        control = monitor._workbench_runtime
        self.request(monitor, "pause")
        save = self.request(monitor, "save")
        self.request(monitor, "resume")
        writer = Mock(return_value=dict(checkpoint="checkpoint-path", checkpointId=str(uuid4())))
        control.snapshot_writer = writer
        control.boundary(100, "factory", "params")
        writer.assert_called_once()
        self.assertEqual(monitor.state["workbench_save_results"][0]["requestId"], save["requestId"])
        self.assertFalse(monitor.state["paused"])

    def test_saved_receipt_is_idempotent_after_runtime_object_recreation(self):
        monitor, clock = self.monitor()
        request = self.request(monitor, "save")
        first = monitor._workbench_runtime
        first.snapshot_writer = Mock(return_value=dict(checkpoint="checkpoint-path", checkpointId=str(uuid4())))
        first.boundary(100, "factory", "params")
        receipt = copy.deepcopy(first.results[request["requestId"]])
        self.write_request(monitor, request)
        second = runtime_module.WorkbenchRuntime(monitor, clock=clock)
        monitor._workbench_runtime = second
        second.snapshot_writer = Mock(side_effect=AssertionError("must not save again"))
        second.boundary(100, "factory", "params")
        second.snapshot_writer.assert_not_called()
        self.assertEqual(second.results[request["requestId"]], receipt)

    def test_failed_save_does_not_report_completion_and_finish_terminates_unhandled_requests(self):
        monitor, clock = self.monitor()
        control = monitor._workbench_runtime
        failed = self.request(monitor, "save")
        control.snapshot_writer = Mock(side_effect=RuntimeError("synthetic verification failure"))
        with patch.object(runtime_module.traceback, "print_exc") as diagnostic:
            control.boundary(100, "factory", "params")
        diagnostic.assert_called_once()
        self.assertEqual(control.results[failed["requestId"]]["status"], "failed")
        self.assertNotIn("checkpointId", control.results[failed["requestId"]])
        pending = self.request(monitor, "save")
        monitor.state["phase"] = "completed"
        control.finish()
        self.assertEqual(control.results[pending["requestId"]]["error"]["code"], "run_finished")
        self.assertEqual(list((monitor.args.run_dir / "workbench-save-requests").iterdir()), [])

    def test_wrong_identity_and_invalid_time_do_not_pause_or_crash_runner(self):
        monitor, clock = self.monitor()
        for changed in ({"runId": "other_run"}, {"createdAt": 1e308}, {"schemaVersion": True}):
            request = dict(schemaVersion=1, runId=monitor.args.run_dir.name, requestId=str(uuid4()),
                           action="pause", createdAt="2026-11-15T04:45:00Z")
            request.update(changed)
            self.write_request(monitor, request)
            monitor._workbench_runtime.poll_intent(force=True)
        self.assertFalse(monitor._workbench_runtime.desired_paused)
        self.assertTrue(monitor._workbench_runtime.warnings)

    def test_full_snapshot_run_keeps_source_report_and_uses_fixed_cumulative_context(self):
        monitor, clock = self.monitor(parameters={"learning_rate": 0.0002})
        self.configurations(monitor)
        source_report = monitor.args.run_dir / "report.json"
        source_report.write_bytes(b"source report remains unchanged")
        request = self.request(monitor, "save")
        builder = Mock(side_effect=self.fake_builder)
        publisher = runtime_module.ManualSnapshotPublisher(monitor, builder)
        result = publisher(request, steps=100, factory="factory", params="params", elapsed_seconds=10, stage=lambda _: None)
        snapshot = Path(result["checkpoint"]).parent.parent
        report = json.loads((snapshot / "report.json").read_text())
        status = json.loads((snapshot / "status.json").read_text())
        manifest = json.loads(Path(result["checkpoint"]).with_name("000000000100.workbench.json").read_text())
        self.assertEqual(snapshot.parent, monitor.args.run_dir.parent)
        self.assertNotEqual(snapshot, monitor.args.run_dir)
        self.assertEqual(status["phase"], "completed")
        self.assertEqual(report["source_run_id"], monitor.args.run_dir.name)
        self.assertEqual(result["checkpointId"], manifest["id"])
        self.assertEqual(manifest["trainedSeconds"], 10)
        self.assertEqual(manifest["trainedSteps"], 100)
        self.assertEqual(json.loads((snapshot / "request.json").read_text())["parameters"], {"learning_rate": 0.0002})
        self.assertEqual(source_report.read_bytes(), b"source report remains unchanged")
        repeated = publisher(request, steps=200, factory="changed", params="changed", elapsed_seconds=20, stage=lambda _: None)
        self.assertEqual(repeated, result)
        builder.assert_called_once()

    def test_failed_snapshot_validation_never_publishes_a_completed_run(self):
        monitor, clock = self.monitor()
        self.configurations(monitor)
        request = self.request(monitor, "save")

        def bad(run, **options):
            result = self.fake_builder(run, **options)
            result["export_max_abs_error"] = 0.5
            return result

        publisher = runtime_module.ManualSnapshotPublisher(monitor, bad)
        with self.assertRaises(ValueError):
            publisher(request, steps=100, factory="factory", params="params", elapsed_seconds=10, stage=lambda _: None)
        other = next(path for path in monitor.args.run_dir.parent.iterdir() if path != monitor.args.run_dir)
        self.assertEqual(json.loads((other / "status.json").read_text())["phase"], "failed")
        self.assertFalse((other / "report.json").exists())

    def test_parameter_overrides_reach_saved_config_and_real_brax_boundary_identically(self):
        edits = {"num_envs": 256, "batch_size": 256, "learning_rate": 0.0002, "num_eval_envs": 4}
        monitor, clock = self.monitor(mode="leg_force_bc", parameters=edits)
        config = dict(num_envs=128, batch_size=128, num_minibatches=1, unroll_length=16,
                      num_updates_per_batch=1, learning_rate=3e-5, num_eval_envs=8,
                      network_factory={"opaque": "preserved"})
        writer = timed.timed_atomic_json(timed.console.atomic_json, monitor)
        writer(monitor.args.run_dir / "ppo_config.json", config)
        saved = json.loads((monitor.args.run_dir / "ppo_config.json").read_text())
        seen = []
        jax = SimpleNamespace(block_until_ready=Mock(side_effect=lambda value: value))

        def train(**kwargs):
            seen.append(kwargs)
            kwargs["policy_params_fn"](100, "factory", "params")
            return "original result"

        actual = timed.timed_ppo_train(train, jax, monitor)(**config, policy_params_fn=lambda *args: None)
        self.assertEqual(actual, "original result")
        for key, value in edits.items():
            self.assertEqual(saved[key], value)
            self.assertEqual(seen[0][key], value)
        self.assertEqual(seen[0]["network_factory"], config["network_factory"])
        self.assertEqual(monitor.state["workbench_parameters"], edits)
        self.assertTrue(monitor.state["workbench_parameters_applied"])
        self.assertEqual(monitor.state["workbench_parameters_actual"]["learning_rate"], 0.0002)
        self.assertEqual(monitor.state["eval_envs"], 4)
        writer(monitor.args.run_dir / "report.json", {"num_eval_envs": 8})
        self.assertEqual(json.loads((monitor.args.run_dir / "report.json").read_text())["num_eval_envs"], 4)
        self.assertEqual(dict(monitor.runtime_profile.settings)["num_envs"], 512)
        self.assertEqual(jax.block_until_ready.call_count, 1)

    def test_invalid_parameters_are_rejected_before_actual_brax_call(self):
        monitor, clock = self.monitor(parameters={"learning_rate": -1})
        function = Mock()
        with self.assertRaises(ValueError):
            timed.timed_ppo_train(function, SimpleNamespace(), monitor)()
        function.assert_not_called()

    def test_network_configuration_observer_preserves_identity_and_return_value(self):
        monitor, clock = self.monitor()
        expected = object()
        factory = Mock(return_value=expected)
        wrapper = timed.remember_workbench_network_config(factory, monitor)
        result = wrapper("sizes", 4, True, "network factory")
        factory.assert_called_once_with("sizes", 4, True, "network factory")
        self.assertIs(result, expected)
        self.assertIs(monitor._workbench_network_config, expected)

    def test_brax_rejects_dropped_or_mutated_explicit_parameters_before_execution(self):
        monitor, clock = self.monitor(parameters={"learning_rate": 2e-5})
        config = dict(num_envs=128, batch_size=8, num_minibatches=16, unroll_length=75,
                      num_updates_per_batch=4, learning_rate=1e-4)
        timed.timed_atomic_json(timed.console.atomic_json, monitor)(monitor.args.run_dir / "ppo_config.json", config)
        function = Mock()
        wrapper = timed.timed_ppo_train(function, SimpleNamespace(), monitor)
        dropped = {key: value for key, value in config.items() if key != "learning_rate"}
        with self.assertRaises(ValueError):
            wrapper(**dropped)
        with self.assertRaises(ValueError):
            wrapper(**{**config, "learning_rate": 0.1})
        function.assert_not_called()
        self.assertFalse(monitor.state["workbench_parameters_applied"])


if __name__ == "__main__":
    unittest.main()
