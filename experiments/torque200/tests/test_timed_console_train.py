"""Timing-entry integration with fake device completion and no ML dependencies."""

import argparse
from contextlib import contextmanager
import importlib
import inspect
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import timed_console_train as timed
import training_control
from runtime_defaults import RUNTIME_FIELDS
from timing_log import rebuild_summary


class FakeClock:
    def __init__(self):
        self.value = 1000.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


def fake_training_modules(args, train_function, clock):
    """Supply every import used by training_hooks without importing numpy/JAX."""
    modules = {}

    def module(name, **attributes):
        item = ModuleType(name)
        item.__dict__.update(attributes)
        modules[name] = item
        if "." in name:
            parent, leaf = name.rsplit(".", 1)
            if parent not in modules:
                module(parent)
            setattr(modules[parent], leaf, item)
        return item

    def jit(function=None, *, static_argnums=()):
        return function if function is not None else lambda function: function

    def block_until_ready(value):
        return value

    class Evaluator:
        def run_evaluation(self, params, training_metrics, aggregate_episodes=True):
            return dict(training_metrics)

    def save(path, steps, params, config):
        return path

    def network_config(observation_size, action_size, normalize_observations, network_factory):
        return {
            "observation_size": observation_size,
            "action_size": action_size,
            "normalize_observations": normalize_observations,
            "network_factory": network_factory,
        }

    def save_cpu_checkpoint(params, path):
        return path

    def load(path):
        return {"loaded": path}

    def export_params(path, output):
        return output

    def make_ppo_networks(observation_size, action_size, *, preprocess_observations_fn=None):
        return (observation_size, action_size)

    def check_parameter_trees(before, after):
        return {"all_parameters_finite": True}

    class SnapshotPublisher:
        def service(self, steps, params):
            return None

    source = args.original_root / "mujoco_legwheel_rl_public" if args.mode == "original" else args.torque_root / "src"
    module("jax", jit=jit, block_until_ready=block_until_ready)
    module("brax.training.acting", Evaluator=Evaluator)
    module("brax.training.agents.ppo.checkpoint", save=save, network_config=network_config)
    module("brax.training.agents.ppo.networks", make_ppo_networks=make_ppo_networks)
    module("brax.training.agents.ppo.train", train=train_function)
    module("legwheel", __file__=str(source / "legwheel" / "__init__.py"))
    module("legwheel.utils", save_cpu_checkpoint=save_cpu_checkpoint)
    module("legwheel.deployment.extract", load=load, export_params=export_params)
    module("preview_sync", SnapshotPublisher=SnapshotPublisher)
    module("verify_artifacts", check_parameter_trees=check_parameter_trees)
    return modules


class TimedConsoleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name).resolve()
        self.clock = FakeClock()
        self.args = argparse.Namespace(
            run_dir=self.directory, mode="original", duration_seconds=0.0,
            restore=None, seed=0, original_root=self.directory / "original",
            torque_root=self.directory / "torque", force_revalidate=False,
        )

    def monitor(self):
        monitor = timed.TimedRunMonitor(self.args, clock=self.clock)
        self.addCleanup(lambda: monitor.timing.finish("test_cleanup"))
        return monitor

    def read(self, name):
        return json.loads((self.directory / name).read_text(encoding="utf-8"))

    def segments(self):
        events = [json.loads(line) for line in (self.directory / "timing.jsonl").read_text(encoding="utf-8").splitlines()]
        return [event for event in events if event["event"] == "segment"]

    def assertBalanced(self, summary):
        self.assertAlmostEqual(summary["accounting_residual_seconds"], 0.0)
        self.assertAlmostEqual(summary["process_wall_seconds"], summary["exclusive_seconds"] + summary["recorder_overhead_seconds"])

    @contextmanager
    def fake_main_clock(self):
        real_monitor = timed.TimedRunMonitor
        created = []

        def create(args, **kwargs):
            monitor = real_monitor(args, clock=self.clock)
            created.append(monitor)
            return monitor

        with patch.object(timed, "TimedRunMonitor", side_effect=create):
            yield created

    def assertSidecarsAndPaths(self, expected_outcome):
        for name in ("timing.jsonl", "timing-summary.json", "timing-summary.txt"):
            self.assertTrue((self.directory / name).is_file(), name)
        status = self.read("status.json")
        report = self.read("report.json")
        for key, name in (("timing_file", "timing.jsonl"), ("timing_summary_file", "timing-summary.json"), ("timing_text_file", "timing-summary.txt")):
            self.assertEqual(status[key], str(self.directory / name))
            self.assertEqual(report[key], str(self.directory / name))
        summary = self.read("timing-summary.json")
        self.assertEqual(summary["outcome"], expected_outcome)
        self.assertTrue(summary["finished"])
        self.assertEqual(rebuild_summary(self.directory)["outcome"], expected_outcome)
        self.assertBalanced(summary)
        return summary

    def test_device_wait_belongs_to_ppo_and_cpu_capture_is_exclusive(self):
        monitor = self.monitor()
        waits = []

        def wait(params):
            waits.append(params)
            self.clock.advance(5)
            return params

        def capture(steps, factory, params):
            monitor.publish(steps=int(steps))
            self.clock.advance(2)

        def train(*, policy_params_fn, progress_fn=None):
            self.clock.advance(3)
            policy_params_fn(0, "factory", "initial")
            self.clock.advance(7)
            policy_params_fn(100, "factory", "updated")
            self.clock.advance(11)
            return "trained"

        wrapped = timed.timed_ppo_train(train, SimpleNamespace(block_until_ready=wait), monitor)
        self.assertEqual(wrapped(policy_params_fn=capture), "trained")
        summary = monitor.timing.finish(steps=100)
        self.assertEqual(waits, ["initial", "updated"])
        self.assertEqual(summary["phases"]["ppo.initialize"]["total_seconds"], 8)
        self.assertEqual(summary["phases"]["ppo.first_epoch"]["total_seconds"], 12)
        self.assertEqual(summary["phases"]["ppo.train"]["total_seconds"], 11)
        self.assertEqual(summary["phases"]["callback.snapshot_and_checks"]["total_seconds"], 4)
        self.assertEqual(summary["process_wall_seconds"], 35)
        self.assertFalse(monitor._timing_in_ppo)
        snapshots = [event for event in self.segments() if event["phase"] == "callback.snapshot_and_checks"]
        self.assertEqual([event["duration_seconds"] for event in snapshots], [2, 2])
        self.assertEqual([event["steps_start"] for event in snapshots], [0, 100])
        self.assertBalanced(summary)

    def test_first_evaluation_includes_compile_and_later_evaluation_is_separate(self):
        monitor = self.monitor()
        calls = []

        def evaluation(evaluator, params, metrics, aggregate_episodes=True):
            if not calls:
                self.clock.advance(5)  # First-use compilation.
            self.clock.advance(3)  # Completed evaluator work, including wait.
            calls.append(params)
            return dict(metrics, evaluated=True)

        monitor.timing.set_phase("ppo.train", steps=10)
        wrapped = timed.timed_evaluation(evaluation, monitor)
        self.clock.advance(2)
        self.assertTrue(wrapped(None, "first", {})["evaluated"])
        self.clock.advance(2)
        self.assertTrue(wrapped(None, "later", {})["evaluated"])
        summary = monitor.timing.finish()
        self.assertEqual(summary["phases"]["ppo.evaluate_first"]["total_seconds"], 8)
        self.assertEqual(summary["phases"]["ppo.evaluate"]["total_seconds"], 3)
        self.assertEqual(summary["phases"]["ppo.train"]["total_seconds"], 4)
        evaluations = [event for event in self.segments() if event["phase"].startswith("ppo.evaluate")]
        self.assertEqual([event["metadata"]["compilation_included"] for event in evaluations], [True, False])
        self.assertBalanced(summary)

    def test_nested_snapshot_preview_and_atomic_io_do_not_regress_observed_steps(self):
        monitor = self.monitor()

        def preview(steps, params):
            # Original capture services preview before publishing the new steps.
            self.assertEqual(monitor.state["steps"], 0)
            self.clock.advance(1)

        measured_preview = timed.timed_call(preview, "callback.preview", monitor)

        def capture(steps, factory, params):
            measured_preview(steps, params)
            self.clock.advance(2)
            timed.console.atomic_json(self.directory / "snapshot-metadata.json", {"steps": steps})
            monitor.event(steps, {})
            monitor.publish(phase="training", stage="optimization")

        def train(*, policy_params_fn):
            self.clock.advance(2)
            policy_params_fn(8192, "factory", "params")

        with patch.object(timed.console, "atomic_json", timed.timed_atomic_json(timed.console.atomic_json, monitor)):
            timed.timed_ppo_train(train, SimpleNamespace(block_until_ready=lambda value: value), monitor)(policy_params_fn=capture)
        summary = monitor.timing.finish(steps=8192)
        endpoints = [event[key] for event in self.segments() for key in ("steps_start", "steps_end") if event[key] is not None]
        self.assertEqual(endpoints, sorted(endpoints))
        preview_segments = [event for event in self.segments() if event["phase"] == "callback.preview"]
        self.assertEqual([(event["steps_start"], event["steps_end"]) for event in preview_segments], [(8192, 8192)])
        config_segments = [event for event in self.segments() if event["phase"] == "io.config"]
        self.assertEqual([(event["steps_start"], event["steps_end"]) for event in config_segments], [(8192, 8192)])
        self.assertEqual(summary["phases"]["callback.preview"]["total_seconds"], 1)
        self.assertEqual(summary["phases"]["callback.snapshot_and_checks"]["total_seconds"], 2)
        self.assertEqual(summary["process_wall_seconds"], 5)
        self.assertBalanced(summary)

    def test_callback_progress_is_exclusive_and_preserves_return_value(self):
        monitor = self.monitor()

        def progress(steps, metrics):
            self.clock.advance(4)
            monitor.event(steps, metrics)
            return "progress recorded"

        def train(*, progress_fn):
            self.clock.advance(3)
            result = progress_fn(12, {"training/loss": 0.25})
            self.clock.advance(2)
            return result

        result = timed.timed_ppo_train(train, SimpleNamespace(), monitor)(progress_fn=progress)
        self.assertEqual(result, "progress recorded")
        summary = monitor.timing.finish(steps=12)
        self.assertEqual(summary["phases"]["callback.progress"]["total_seconds"], 4)
        self.assertEqual(summary["phases"]["ppo.initialize"]["total_seconds"], 5)
        self.assertEqual(self.read("progress.json")[-1]["steps"], 12)
        self.assertBalanced(summary)

    def test_hooks_record_errors_and_cancellation_and_restore_every_patch(self):
        for exception, expected in ((RuntimeError("training failed"), "error"), (timed.console.StopRequested("user_stop"), "cancelled"), (KeyboardInterrupt(), "cancelled")):
            with self.subTest(exception=type(exception).__name__):
                monitor = self.monitor()

                def train(*, policy_params_fn=None, progress_fn=None):
                    self.clock.advance(4)
                    raise exception

                modules = fake_training_modules(self.args, train, self.clock)
                bindings = [
                    (modules["brax.training.agents.ppo.train"], "train"),
                    (modules["jax"], "jit"),
                    (modules["brax.training.acting"].Evaluator, "run_evaluation"),
                    (modules["brax.training.agents.ppo.checkpoint"], "save"),
                    (modules["brax.training.agents.ppo.checkpoint"], "network_config"),
                    (modules["legwheel.utils"], "save_cpu_checkpoint"),
                    (modules["legwheel.deployment.extract"], "load"),
                    (modules["legwheel.deployment.extract"], "export_params"),
                    (modules["brax.training.agents.ppo.networks"], "make_ppo_networks"),
                    (modules["verify_artifacts"], "check_parameter_trees"),
                    (modules["preview_sync"].SnapshotPublisher, "service"),
                ]
                originals = [getattr(module, name) for module, name in bindings]
                previous_path = list(sys.path)
                with patch.dict(sys.modules, modules):
                    with self.assertRaises(type(exception)):
                        with timed.training_hooks(self.args, monitor, train.__code__):
                            for (module, name), original in zip(bindings, originals):
                                self.assertIsNot(getattr(module, name), original)
                                self.assertEqual(inspect.signature(getattr(module, name)), inspect.signature(original))
                            modules["brax.training.agents.ppo.train"].train()
                    for (module, name), original in zip(bindings, originals):
                        self.assertIs(getattr(module, name), original)
                self.assertEqual(sys.path, previous_path)
                self.assertFalse(monitor._timing_in_ppo)
                self.assertEqual(monitor.timing.snapshot()["open_segment"]["phase"], "lifecycle.training_return")
                summary = monitor.timing.finish(expected)
                self.assertEqual(summary["phases"]["ppo.initialize"]["outcomes"].get(expected), 1)
                self.assertEqual(summary["phases"]["ppo.initialize"]["total_seconds"], 4)
                self.assertBalanced(summary)

    def test_runner_budget_reset_does_not_reset_process_wall(self):
        self.args.mode = "leg_force_bc"
        self.args.duration_seconds = 30.0
        monitor = self.monitor()
        self.clock.advance(70)
        monitor.check_stop()
        self.assertEqual(monitor.remaining, 30)
        self.assertEqual(monitor.timing.snapshot()["process_wall_seconds"], 70)
        monitor.begin_training_clock()
        self.assertEqual(monitor.state["preparation_seconds"], 70)
        self.assertEqual(monitor.elapsed, 0)
        self.clock.advance(29)
        monitor.check_stop()
        self.assertEqual(monitor.remaining, 1)
        self.assertEqual(monitor.timing.snapshot()["process_wall_seconds"], 99)
        self.clock.advance(1)
        with self.assertRaisesRegex(timed.console.StopRequested, "wall_budget"):
            monitor.check_stop()
        summary = monitor.timing.finish("cancelled")
        self.assertEqual(summary["process_wall_seconds"], 100)
        self.assertEqual(summary["phases"]["preparation.runtime_lock"]["total_seconds"], 70)
        self.assertEqual(summary["phases"]["ppo.setup"]["total_seconds"], 30)
        self.assertBalanced(summary)

    def test_reset_jit_targets_only_the_runner_setup_callsite(self):
        monitor = self.monitor()
        waits = []

        def jit(function=None, *, static_argnums=()):
            return function

        def wait(value):
            self.clock.advance(3)
            waits.append(value)
            return value

        def reset(key):
            self.clock.advance(2)
            return {"key": key}

        def other_function(key):
            self.clock.advance(1)
            return key

        def runner_setup(function):
            return wrapped(function, static_argnums=())

        def internal_training_reset():
            return wrapped(reset)

        wrapped = timed.timed_reset_jit(jit, SimpleNamespace(block_until_ready=wait), monitor, runner_setup.__code__)
        setup_reset = runner_setup(reset)
        self.assertEqual(setup_reset(7), {"key": 7})
        internal = internal_training_reset()
        self.assertIs(internal, reset)
        self.assertEqual(internal(8), {"key": 8})
        other = runner_setup(other_function)
        self.assertIs(other, other_function)
        self.assertEqual(other(9), 9)
        summary = monitor.timing.finish()
        self.assertEqual(waits, [{"key": 7}])
        self.assertEqual(summary["phases"]["setup.reset_first"]["total_seconds"], 5)
        self.assertEqual(summary["process_wall_seconds"], 8)
        self.assertEqual(inspect.signature(wrapped), inspect.signature(jit))
        self.assertEqual(inspect.signature(setup_reset), inspect.signature(reset))
        self.assertBalanced(summary)

    def test_main_early_stop_publishes_timing_without_entering_training_hooks(self):
        source_path = Path(timed.console.__file__)
        source_before = source_path.read_bytes()
        hooks_entered = []

        @contextmanager
        def hooks(args, monitor, code):
            hooks_entered.append(args.mode)
            yield

        (self.directory / "stop.request").write_text("{}", encoding="utf-8")
        original_monitor = timed.console.RunMonitor
        original_atomic = timed.console.atomic_json
        with self.fake_main_clock(), patch.object(timed, "training_hooks", hooks), patch.object(timed.console, "run_training") as training:
            timed.main(["--mode", "original", "--run-dir", str(self.directory)])
            training.assert_not_called()
        self.assertEqual(hooks_entered, [])
        self.assertIs(timed.console.RunMonitor, original_monitor)
        self.assertIs(timed.console.atomic_json, original_atomic)
        self.assertEqual(self.read("status.json")["phase"], "cancelled")
        self.assertSidecarsAndPaths("cancelled")
        self.assertEqual(source_path.read_bytes(), source_before)

    def test_main_mock_failure_runs_real_console_lifecycle_and_restores_patches(self):
        source_path = Path(timed.console.__file__)
        source_before = source_path.read_bytes()
        hooks_entered = []
        hooks_exited = []

        @contextmanager
        def hooks(args, monitor, code):
            hooks_entered.append(code)
            try:
                yield
            finally:
                hooks_exited.append(args.mode)

        def failing_training(args, monitor):
            self.clock.advance(2)
            monitor.publish(stage="environment_setup")
            self.clock.advance(3)
            raise RuntimeError("mock device setup failed")

        original_monitor = timed.console.RunMonitor
        original_atomic = timed.console.atomic_json
        with self.fake_main_clock(), patch.object(timed, "training_hooks", hooks), patch.object(timed.console, "run_training", failing_training):
            with self.assertRaisesRegex(RuntimeError, "mock device setup failed"):
                timed.main(["--mode", "torque200", "--run-dir", str(self.directory)])
            self.assertIs(timed.console.run_training, failing_training)
        self.assertEqual(hooks_entered, [failing_training.__code__])
        self.assertEqual(hooks_exited, ["torque200"])
        self.assertIs(timed.console.RunMonitor, original_monitor)
        self.assertIs(timed.console.atomic_json, original_atomic)
        self.assertEqual(self.read("status.json")["phase"], "failed")
        self.assertIn("mock device setup failed", self.read("report.json")["error"])
        summary = self.assertSidecarsAndPaths("failed")
        self.assertEqual(summary["process_wall_seconds"], 5)
        self.assertEqual(source_path.read_bytes(), source_before)

    def test_main_success_still_uses_original_lifecycle_and_atomic_report(self):
        @contextmanager
        def hooks(args, monitor, code):
            yield

        def successful_training(args, monitor):
            self.clock.advance(2)
            monitor.publish(phase="completed", stage="verified", steps=128, completed=True)
            timed.console.atomic_json(args.run_dir / "report.json", dict(monitor.state))

        with self.fake_main_clock(), patch.object(timed, "training_hooks", hooks), patch.object(timed.console, "run_training", successful_training):
            timed.main(["--mode", "original", "--run-dir", str(self.directory)])
        summary = self.assertSidecarsAndPaths("completed")
        self.assertEqual(summary["steps"], 128)
        phases = {event["phase"] for event in self.segments()}
        self.assertIn("io.status", phases)
        self.assertIn("io.report", phases)

    def test_main_selects_model_roots_before_package_dispatch_and_restores_path(self):
        release_source = self.directory / "release" / "src"
        torque_root = self.directory / "models"
        original_root = self.directory / "reference"
        torque_source = torque_root / "src"
        reference_source = original_root / "mujoco_legwheel_rl_public"
        deploy_source = original_root / "mujoco_esp32_robot_deploy_public"
        deploy_source.mkdir(parents=True)
        for source in (release_source, torque_source, reference_source):
            legwheel = source / "legwheel"
            legwheel.mkdir(parents=True)
            (legwheel / "__init__.py").write_text("", encoding="utf8")
        for source in (release_source, torque_source):
            hybrid = source / "hybrid"
            hybrid.mkdir()
            (hybrid / "__init__.py").write_text("", encoding="utf8")
            (hybrid / "managed_training.py").write_text(
                "from contextlib import contextmanager\n"
                "@contextmanager\n"
                "def runtime_locks(root):\n"
                "    yield\n"
                "def run_managed(args, monitor, run_training):\n"
                "    return run_training(args, monitor)\n", encoding="utf8")

        original_sys_path = list(sys.path)
        for mode in ("original", "wheel_torque_bc", "torque200", "leg_force_bc",
                     "hybrid_virtual_leg_force_100"):
            for failure in (False, True):
                with self.subTest(mode=mode, failure=failure), patch.dict(sys.modules):
                    # Each run represents a fresh runner, with a versioned wrapper
                    # visible first and distinct model/assets roots in its args.
                    for name in ("hybrid.managed_training", "hybrid", "legwheel"):
                        sys.modules.pop(name, None)
                    importlib.invalidate_caches()
                    sys.path[:] = [str(release_source)] + original_sys_path
                    before_main = list(sys.path)
                    run = self.directory / (mode + ("-failure" if failure else "-success"))
                    run.mkdir()
                    expected_source = (reference_source if mode in ("original", "wheel_torque_bc")
                                       else torque_source)
                    entered = []

                    @contextmanager
                    def hooks(args, monitor, code):
                        self.assertEqual(sys.path[:2], [str(expected_source), str(deploy_source)])
                        package = importlib.import_module("legwheel")
                        self.assertEqual(Path(package.__file__).resolve().parent.parent, expected_source)
                        if mode in ("leg_force_bc", "hybrid_virtual_leg_force_100"):
                            # This package was bound by console.main, before hooks.
                            managed = sys.modules["hybrid.managed_training"]
                            self.assertEqual(Path(managed.__file__).resolve().parent.parent, torque_source)
                        entered.append(True)
                        yield

                    def train(args, monitor):
                        if failure:
                            raise RuntimeError("fixture training failure")
                        monitor.publish(phase="completed", stage="verified", completed=True)
                        timed.console.atomic_json(run / "report.json", dict(monitor.state))

                    try:
                        with self.fake_main_clock(), patch.object(timed, "training_hooks", hooks), \
                                patch.object(timed.console, "run_training", train):
                            argv = ["--mode", mode, "--run-dir", str(run),
                                    "--original-root", str(original_root), "--torque-root", str(torque_root)]
                            if failure:
                                with self.assertRaisesRegex(RuntimeError, "fixture training failure"):
                                    timed.main(argv)
                            else:
                                timed.main(argv)
                        self.assertEqual(entered, [True])
                        self.assertEqual(sys.path, before_main)
                    finally:
                        sys.path[:] = original_sys_path

    def test_manual_force_snapshot_restores_experiment_normalizer(self):
        monitor = self.monitor()
        monitor._workbench_network_config = {"fixture": True}
        options = {"fixed_normalization": False}
        args = SimpleNamespace(mode="leg_force_bc", experiment_config=options)
        restored = ["saved_normalizer", "actor", "critic"]
        normalizer = ModuleType("force_teacher.ppo")
        normalizer.restore_normalizer = lambda state, config: (
            "welford" if state == "saved_normalizer" and config is options else "wrong")

        class VerificationReached(Exception):
            pass

        def verify(params, recovered):
            self.assertEqual(recovered[0], "welford")
            raise VerificationReached

        builder = timed.manual_snapshot_builder(
            args, monitor, SimpleNamespace(),
            SimpleNamespace(save=lambda *args: None),
            SimpleNamespace(load=lambda path: restored, export_params=lambda *args: None),
            SimpleNamespace(check_parameter_trees=verify))
        with patch.dict(sys.modules, {"force_teacher.ppo": normalizer}):
            with self.assertRaises(VerificationReached):
                builder(self.directory, steps=123, factory=None, params=None,
                        elapsed_seconds=10, stage=lambda value: None)

    def test_wrappers_preserve_the_public_signatures_brax_inspects(self):
        monitor = self.monitor()

        def call(a, /, b=1, *, option=False):
            return a

        def train(environment, *, policy_params_fn=None, progress_fn=None, num_evals=1):
            return environment

        def evaluate(self, params, metrics, aggregate_episodes=True):
            return metrics

        functions = [
            (call, timed.timed_call(call, "test.call", monitor)),
            (train, timed.timed_ppo_train(train, SimpleNamespace(), monitor)),
            (evaluate, timed.timed_evaluation(evaluate, monitor)),
            (timed.console.atomic_json, timed.timed_atomic_json(timed.console.atomic_json, monitor)),
        ]
        for original, wrapped in functions:
            with self.subTest(function=original.__name__):
                self.assertEqual(inspect.signature(wrapped), inspect.signature(original))
                self.assertIs(wrapped.__wrapped__, original)

    def test_controller_defaults_to_timed_entry_and_accepts_both_exact_runner_names(self):
        controller = training_control.TrainingController(self.directory / "artifacts", self.directory, self.directory)
        self.assertEqual(controller.script.name, "timed_console_train.py")
        run = controller.root / "original" / "run-1"
        run.mkdir(parents=True)
        for name in ("console_train.py", "timed_console_train.py", "timed_console_train.py.extra"):
            with self.subTest(name=name):
                script = self.directory / name
                record = dict(pid=12345, mode="original", run_dir=str(run), script=str(script), process_start="42")
                argv = [sys.executable, str(script), "--mode", "original", "--run-dir", str(run)]
                with patch.object(controller, "_process", return_value=(argv, "42")):
                    self.assertEqual(controller._alive(record), name != "timed_console_train.py.extra")
                with patch.object(controller, "_process", return_value=(argv, "43")):
                    self.assertFalse(controller._alive(record))
                with patch.object(controller, "_process", return_value=(argv[:-1] + [str(run) + "suffix"], "42")):
                    self.assertFalse(controller._alive(record))
        invalid = self.directory / "untrusted.py"
        invalid.write_text("raise AssertionError('must not launch')", encoding="utf-8")
        controller.script = invalid
        with self.assertRaisesRegex(ValueError, "console_train.py or timed_console_train.py"):
            controller.start("original")

    def test_runtime_default_config_report_and_actual_brax_call_agree(self):
        self.args.mode = "leg_force_bc"
        monitor = self.monitor()
        options = dict(num_envs=128, batch_size=128, num_minibatches=1,
                       unroll_length=16, num_updates_per_batch=1, learning_rate=3e-5)
        received = []

        def train(**kwargs):
            received.append(kwargs)
            return "trained"

        with patch.object(timed.console, "atomic_json", timed.timed_atomic_json(timed.console.atomic_json, monitor)):
            timed.console.atomic_json(self.directory / "ppo_config.json", options)
            result = timed.timed_ppo_train(train, SimpleNamespace(), monitor)(**options)
            timed.console.atomic_json(self.directory / "report.json", dict(monitor.state))
        self.assertEqual(result, "trained")
        saved = self.read("ppo_config.json")
        self.assertEqual([saved[key] for key in RUNTIME_FIELDS], [512, 512, 1, 16, 1])
        self.assertEqual({key: received[0][key] for key in RUNTIME_FIELDS},
                         {key: saved[key] for key in RUNTIME_FIELDS})
        self.assertEqual(received[0]["learning_rate"], options["learning_rate"])
        self.assertEqual(options["num_envs"], 128)
        self.assertEqual(self.read("report.json")["num_envs"], 512)
        self.assertTrue(self.read("report.json")["runtime_profile"]["applied_to_brax"])
        self.assertTrue(self.read("runtime_profile.json")["applied_to_brax"])
        self.assertEqual(len(self.read("runtime_profile.json")["runtime_module_sha256"]), 64)

    def test_auto_resume_uses_saved_batch_instead_of_new_default(self):
        self.args.mode = "leg_force_bc"
        prior = self.directory / "prior"
        checkpoint = prior / "checkpoints" / "000001000000"
        checkpoint.mkdir(parents=True)
        options = dict(num_envs=128, batch_size=128, num_minibatches=1,
                       unroll_length=16, num_updates_per_batch=1)
        saved_path = prior / "ppo_config.json"
        saved_path.write_text(json.dumps(options), encoding="utf8")
        before = saved_path.read_bytes()
        self.args.restore = checkpoint
        monitor = self.monitor()
        with patch.object(timed.console, "atomic_json", timed.timed_atomic_json(timed.console.atomic_json, monitor)):
            timed.console.atomic_json(self.directory / "ppo_config.json", options)
            actual = timed.timed_ppo_train(lambda **kwargs: kwargs, SimpleNamespace(), monitor)(**options)
        self.assertEqual(actual, options)
        self.assertEqual(self.read("ppo_config.json"), options)
        self.assertEqual(saved_path.read_bytes(), before)
        self.assertEqual(monitor.state["num_envs"], 128)

    def test_recorded_settings_mismatch_is_rejected_before_training(self):
        monitor = self.monitor()  # original mode: preserves entrypoint options
        options = dict(num_envs=128, batch_size=8, num_minibatches=16,
                       unroll_length=75, num_updates_per_batch=4)
        train = unittest.mock.Mock()
        with patch.object(timed.console, "atomic_json", timed.timed_atomic_json(timed.console.atomic_json, monitor)):
            timed.console.atomic_json(self.directory / "ppo_config.json", options)
            with self.assertRaisesRegex(ValueError, "differ from the saved"):
                timed.timed_ppo_train(train, SimpleNamespace(), monitor)(**dict(options, num_envs=64))
        train.assert_not_called()

    def test_runtime_cli_option_is_consumed_before_original_parser(self):
        with patch.object(timed.console, "main", return_value="parsed") as original:
            self.assertEqual(timed.main(["--runtime-profile", "legacy", "--mode", "original"]), "parsed")
        original.assert_called_once_with(["--mode", "original"])

    def test_invalid_resume_runtime_metadata_still_publishes_failed_artifacts(self):
        checkpoint = self.directory / "prior" / "checkpoints" / "000000000001"
        checkpoint.mkdir(parents=True)
        (checkpoint.parent.parent / "ppo_config.json").write_text('{"num_envs":128}', encoding="utf8")
        entered = []

        @contextmanager
        def hooks(args, monitor, code):
            entered.append(True)
            yield

        with self.fake_main_clock(), patch.object(timed, "training_hooks", hooks):
            with self.assertRaisesRegex(ValueError, "Invalid runtime configuration"):
                timed.main(["--mode", "original", "--run-dir", str(self.directory),
                            "--restore", str(checkpoint)])
        self.assertEqual(entered, [])
        self.assertEqual(self.read("status.json")["phase"], "failed")
        self.assertEqual(self.read("report.json")["runtime_profile"]["effective"], "invalid")
        self.assertFalse(self.read("report.json")["runtime_profile"]["applied_to_brax"])
        self.assertSidecarsAndPaths("failed")


if __name__ == "__main__":
    unittest.main()
