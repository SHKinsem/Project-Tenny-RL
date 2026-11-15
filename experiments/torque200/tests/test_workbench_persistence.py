"""Temporary synthetic checkpoints and the real standard-library timing seams."""

import copy
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


_previous_path = list(sys.path)
try:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    persistence = importlib.import_module("workbench_persistence")
    timed = importlib.import_module("timed_console_train")
finally:
    sys.path[:] = _previous_path


class Clock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value


class WorkbenchPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="workbench-persistence-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        environment = patch.dict(os.environ, {"TENNY_EVAL_BACKEND": "auto", "TENNY_TRAINING_PROFILE": "auto"})
        environment.start()
        self.addCleanup(environment.stop)

    def run_dir(self, name="run_001"):
        path = self.root / name
        path.mkdir(exist_ok=True)
        return path

    def checkpoint(self, run, steps):
        path = run / "checkpoints" / f"{steps:012d}"
        path.mkdir(parents=True, exist_ok=True)
        (path / "weights.bin").write_bytes(b"synthetic weights")
        return path

    @staticmethod
    def sidecar(checkpoint):
        return checkpoint.with_name(checkpoint.name + ".workbench.json")

    @staticmethod
    def read(path):
        return json.loads(path.read_text(encoding="utf8"))

    def publisher(self, name="run_001", mode="original", restore=None):
        return persistence.WorkbenchPersistence(self.run_dir(name), mode, restore=restore)

    def parent(self, mode="original"):
        publisher = self.publisher("parent_run", mode)
        checkpoint = self.checkpoint(publisher.run_dir, 524288)
        publisher.publish_checkpoint(checkpoint, elapsed_seconds=600, steps=524288)
        return checkpoint

    def monitor(self, name="timed_run", mode="original", restore=None):
        clock = Clock()
        args = SimpleNamespace(run_dir=self.run_dir(name), mode=mode, restore=restore,
                               runtime_profile_requested="auto", duration_seconds=600,
                               original_root=self.root / "reference", torque_root=self.root / "torque")
        monitor = timed.TimedRunMonitor(args, clock=clock)
        self.addCleanup(lambda: monitor.timing.finish("fixture_complete", steps=timed.observed_steps(monitor)))
        return monitor, clock

    def test_new_context_is_created_once_and_bound_to_records(self):
        publisher = self.publisher()
        context = self.read(publisher.run_dir / "workbench-run.json")
        self.assertEqual(context["origin"], "new")
        self.assertIsNone(context["parentCheckpoint"])
        fields = publisher.state_fields()
        self.assertEqual(fields["workbench_context_id"], context["id"])
        self.assertEqual(fields["clock_policy_id"], "console-entry-budget-v1")
        self.assertEqual(fields["workbench_metadata_status"], "ready")
        self.assertEqual(publisher.bind_record({"steps": 0})["workbench_context_id"], context["id"])
        retried = persistence.WorkbenchPersistence(publisher.run_dir, "original")
        self.assertEqual(retried.context["id"], context["id"])

    def test_legacy_restore_keeps_history_unknown_without_rewriting_source(self):
        old = self.checkpoint(self.run_dir("old_run"), 9000)
        before = (old / "weights.bin").read_bytes()
        publisher = self.publisher(restore=old)
        self.assertEqual(publisher.context["origin"], "legacy")
        self.assertTrue(publisher.warnings)
        checkpoint = self.checkpoint(publisher.run_dir, 10)
        metadata = publisher.publish_checkpoint(checkpoint, elapsed_seconds=60, steps=10)
        self.assertIsNone(metadata["trainedSeconds"])
        self.assertIsNone(metadata["trainedSteps"])
        self.assertEqual(metadata["capturedSeconds"], 60)
        self.assertEqual((old / "weights.bin").read_bytes(), before)
        self.assertFalse(self.sidecar(old).exists())

    def test_valid_parent_and_multiple_snapshots_use_one_fixed_baseline(self):
        old = self.parent()
        source = self.sidecar(old).read_bytes()
        publisher = self.publisher(restore=old)
        self.assertEqual(publisher.context["origin"], "checkpoint")
        first = self.checkpoint(publisher.run_dir, 81920)
        second = self.checkpoint(publisher.run_dir, 163840)
        a = publisher.publish_checkpoint(first, elapsed_seconds=60, steps=81920)
        b = publisher.publish_checkpoint(second, elapsed_seconds=120, steps=163840)
        self.assertEqual(a["trainedSeconds"], 660)
        self.assertEqual(b["trainedSeconds"], 720)
        self.assertEqual(b["trainedSteps"], 688128)
        self.assertEqual(a["parentCheckpointId"], b["parentCheckpointId"])
        self.assertEqual(self.sidecar(old).read_bytes(), source)

    def test_invalid_and_incompatible_parent_metadata_becomes_legacy(self):
        old = self.parent()
        path = self.sidecar(old)
        valid = self.read(path)
        for value in (b'{"schemaVersion":', json.dumps({**valid, "clockPolicyId": "different-policy"}).encode()):
            path.write_bytes(value)
            publisher = self.publisher("run_" + str(len(value)), restore=old)
            self.assertEqual(publisher.context["origin"], "legacy")
            self.assertIsNone(publisher.context["parentCheckpoint"])
            self.assertEqual(path.read_bytes(), value)
            self.assertTrue(publisher.warnings)

    def test_same_checkpoint_identity_is_reused_and_counters_cannot_regress(self):
        publisher = self.publisher()
        checkpoint = self.checkpoint(publisher.run_dir, 10)
        first = publisher.publish_checkpoint(checkpoint, elapsed_seconds=60, steps=10)
        retried = persistence.WorkbenchPersistence(publisher.run_dir, "original")
        second = retried.publish_checkpoint(checkpoint, elapsed_seconds=61, steps=10)
        self.assertEqual(first["id"], second["id"])
        before = self.sidecar(checkpoint).read_bytes()
        self.assertIsNone(retried.publish_checkpoint(checkpoint, elapsed_seconds=59, steps=10))
        self.assertEqual(self.sidecar(checkpoint).read_bytes(), before)
        self.assertEqual(retried.state_fields()["workbench_metadata_status"], "warning")

    def test_existing_bad_or_conflicting_context_is_not_overwritten(self):
        run = self.run_dir()
        path = run / "workbench-run.json"
        path.write_bytes(b'{"invalid":')
        publisher = persistence.WorkbenchPersistence(run, "original")
        self.assertIsNone(publisher.context)
        self.assertIsNone(publisher.state_fields()["workbench_context_id"])
        self.assertEqual(path.read_bytes(), b'{"invalid":')
        path.unlink()
        original = persistence.WorkbenchPersistence(run, "original")
        before = path.read_bytes()
        conflict = persistence.WorkbenchPersistence(run, "original", restore=self.parent())
        self.assertIsNone(conflict.context)
        self.assertEqual(path.read_bytes(), before)
        self.assertIsNotNone(original.context)

    def test_invalid_existing_manifest_does_not_damage_weights_or_manifest(self):
        publisher = self.publisher()
        checkpoint = self.checkpoint(publisher.run_dir, 10)
        weight_hash = hashlib.sha256((checkpoint / "weights.bin").read_bytes()).hexdigest()
        path = self.sidecar(checkpoint)
        path.write_bytes(b"broken manifest")
        self.assertIsNone(publisher.publish_checkpoint(checkpoint, elapsed_seconds=60, steps=10))
        self.assertEqual(path.read_bytes(), b"broken manifest")
        self.assertEqual(hashlib.sha256((checkpoint / "weights.bin").read_bytes()).hexdigest(), weight_hash)

    def test_manifest_from_another_run_is_not_rebound(self):
        publisher = self.publisher()
        checkpoint = self.checkpoint(publisher.run_dir, 10)
        metadata = publisher.publish_checkpoint(checkpoint, elapsed_seconds=60, steps=10)
        metadata["createdByRunId"] = "another_run"
        path = self.sidecar(checkpoint)
        path.write_text(json.dumps(metadata), encoding="utf8")
        before = path.read_bytes()
        self.assertIsNone(publisher.publish_checkpoint(checkpoint, elapsed_seconds=61, steps=10))
        self.assertEqual(path.read_bytes(), before)

    def test_final_report_uses_training_cutoff_and_preserves_finalization_fields(self):
        publisher = self.publisher(restore=self.parent())
        checkpoint = self.checkpoint(publisher.run_dir, 81920)
        report = {"checkpoint": str(checkpoint), "actual_timesteps": 81920, "steps": 81920,
                  "training_stop_elapsed_seconds": 60, "elapsed_seconds": 180, "finalization_seconds": 120}
        bound = publisher.publish_report(report)
        metadata = self.read(self.sidecar(checkpoint))
        self.assertEqual(metadata["capturedSeconds"], 60)
        self.assertEqual(metadata["trainedSeconds"], 660)
        self.assertEqual(bound["elapsed_seconds"], 180)
        self.assertEqual(bound["finalization_seconds"], 120)
        self.assertEqual(bound["workbench_context_id"], publisher.context["id"])
        self.assertNotIn("verification", metadata)

    def test_missing_final_cutoff_does_not_guess_or_update_previous_manifest(self):
        publisher = self.publisher()
        checkpoint = self.checkpoint(publisher.run_dir, 10)
        publisher.publish_checkpoint(checkpoint, elapsed_seconds=60, steps=10)
        before = self.sidecar(checkpoint).read_bytes()
        report = publisher.publish_report({"checkpoint": str(checkpoint), "steps": 10, "elapsed_seconds": 999})
        self.assertEqual(self.sidecar(checkpoint).read_bytes(), before)
        self.assertIn("final_capture_time_unavailable", [warning["code"] for warning in report["workbench_metadata_warnings"]])

    def test_successful_final_report_exposes_the_published_checkpoint_identity(self):
        publisher = self.publisher()
        checkpoint = self.checkpoint(publisher.run_dir, 10)
        source = {"checkpoint": str(checkpoint), "steps": 10, "training_stop_elapsed_seconds": 60,
                  "workbench_checkpoint_id": "stale-id"}
        report = publisher.publish_report(source)
        metadata = self.read(self.sidecar(checkpoint))
        self.assertEqual(report["workbench_checkpoint_id"], metadata["id"])
        self.assertNotEqual(report["workbench_checkpoint_id"], "stale-id")
        self.assertEqual(source["workbench_checkpoint_id"], "stale-id")

    def test_failed_final_metadata_publication_does_not_forward_an_old_identity(self):
        publisher = self.publisher()
        checkpoint = self.checkpoint(publisher.run_dir, 10)
        source = {"checkpoint": str(checkpoint), "steps": 10, "training_stop_elapsed_seconds": 60,
                  "workbench_checkpoint_id": "stale-id"}
        with patch.object(persistence.os, "replace", side_effect=OSError("synthetic failure")):
            report = publisher.publish_report(source)
        self.assertNotIn("workbench_checkpoint_id", report)
        self.assertFalse(self.sidecar(checkpoint).exists())
        self.assertTrue(report["workbench_metadata_warnings"])

    def test_atomic_failure_leaves_existing_manifest_and_no_false_success(self):
        publisher = self.publisher()
        checkpoint = self.checkpoint(publisher.run_dir, 10)
        publisher.publish_checkpoint(checkpoint, elapsed_seconds=60, steps=10)
        before = self.sidecar(checkpoint).read_bytes()
        with patch.object(persistence.os, "replace", side_effect=OSError("synthetic failure")):
            result = publisher.publish_checkpoint(checkpoint, elapsed_seconds=61, steps=10)
        self.assertIsNone(result)
        self.assertEqual(self.sidecar(checkpoint).read_bytes(), before)
        self.assertFalse(list(checkpoint.parent.glob(".*.tmp")))
        self.assertTrue(publisher.warnings)
        self.assertNotIn(str(self.root), json.dumps(publisher.warnings))

    def test_context_write_failure_does_not_publish_a_binding(self):
        run = self.run_dir()
        with patch.object(persistence.os, "replace", side_effect=OSError("synthetic failure")):
            publisher = persistence.WorkbenchPersistence(run, "original")
        self.assertIsNone(publisher.context)
        self.assertIsNone(publisher.state_fields()["workbench_context_id"])
        self.assertFalse((run / "workbench-run.json").exists())

    def test_outside_or_linked_checkpoint_paths_are_rejected(self):
        publisher = self.publisher()
        outside = self.checkpoint(self.run_dir("outside"), 10)
        self.assertIsNone(publisher.publish_checkpoint(outside, elapsed_seconds=60, steps=10))
        self.assertFalse(self.sidecar(outside).exists())
        directory = publisher.run_dir / "checkpoints"
        try:
            directory.symlink_to(outside.parent, target_is_directory=True)
        except (OSError, NotImplementedError):
            return
        self.assertIsNone(publisher.publish_checkpoint(directory / outside.name, elapsed_seconds=60, steps=10))
        self.assertFalse(self.sidecar(outside).exists())

    def test_actual_timed_monitor_keeps_512_native_backend_and_brax_kwargs(self):
        monitor, clock = self.monitor(mode="leg_force_bc")
        expected = dict(num_envs=512, batch_size=512, num_minibatches=1, unroll_length=16, num_updates_per_batch=1)
        self.assertEqual(dict(monitor.runtime_profile.settings), expected)
        self.assertEqual(monitor.state["evaluation_backend"], "native_mujoco_cpp")
        self.assertEqual(monitor.state["clock_policy_id"], "console-ready-budget-v1")
        monitor.begin_training_clock()
        seen, captured = {}, []
        params = object()
        jax = SimpleNamespace(block_until_ready=Mock(side_effect=lambda value: value))

        def train(*, num_envs, batch_size, num_minibatches, unroll_length, num_updates_per_batch,
                  policy_params_fn, progress_fn, discounting):
            seen.update(num_envs=num_envs, batch_size=batch_size, num_minibatches=num_minibatches,
                        unroll_length=unroll_length, num_updates_per_batch=num_updates_per_batch, discounting=discounting)
            policy_params_fn(81920, "factory", params)
            progress_fn(81920, {"loss": 0.5})
            return "original-result"

        wrapper = timed.timed_ppo_train(train, jax, monitor)
        self.assertEqual(inspect.signature(wrapper), inspect.signature(train))
        result = wrapper(num_envs=128, batch_size=128, num_minibatches=1, unroll_length=75, num_updates_per_batch=4,
                         policy_params_fn=lambda *values: captured.append(values), progress_fn=lambda *values: None,
                         discounting=0.987)
        self.assertEqual(result, "original-result")
        self.assertEqual({field: seen[field] for field in expected}, expected)
        self.assertEqual(seen["discounting"], 0.987)
        self.assertIs(captured[0][2], params)
        self.assertEqual(jax.block_until_ready.call_count, 1)
        self.assertTrue(monitor._runtime_applied)
        self.assertIn("ppo.initialize", json.dumps(monitor.timing.snapshot()))
        self.assertEqual(monitor._timing_steps, 81920)

    def test_actual_timed_config_hook_keeps_checkpoint_performance_settings(self):
        old = self.checkpoint(self.run_dir("old"), 10)
        expected = dict(num_envs=128, batch_size=128, num_minibatches=1, unroll_length=75, num_updates_per_batch=4)
        (old.parent.parent / "ppo_config.json").write_text(json.dumps(expected), encoding="utf8")
        monitor, clock = self.monitor(mode="leg_force_bc", restore=old)
        self.assertEqual(dict(monitor.runtime_profile.settings), expected)
        caller = dict(num_envs=512, batch_size=512, num_minibatches=1, unroll_length=16, num_updates_per_batch=1, seed=7)
        original = copy.deepcopy(caller)
        writer = timed.timed_atomic_json(timed.console.atomic_json, monitor)
        writer(monitor.args.run_dir / "ppo_config.json", caller)
        actual = self.read(monitor.args.run_dir / "ppo_config.json")
        self.assertEqual({field: actual[field] for field in expected}, expected)
        self.assertEqual(actual["seed"], 7)
        self.assertEqual(caller, original)
        self.assertEqual(monitor.runtime_profile.source, "checkpoint")
        self.assertEqual(monitor._workbench.context["origin"], "legacy")

    def test_actual_timed_save_hook_captures_before_io_without_extra_sync(self):
        monitor, clock = self.monitor()
        monitor.state["phase"] = "training"
        clock.value = 60
        params, config, marker = object(), object(), object()
        seen = []

        def save(path, step, passed_params, passed_config):
            seen.append((path, step, passed_params, passed_config))
            clock.value += 3
            self.checkpoint(monitor.args.run_dir, step)
            return marker

        wrapped = timed.workbench_checkpoint_save(timed.timed_call(save, "save.checkpoint", monitor), monitor)
        self.assertEqual(inspect.signature(wrapped), inspect.signature(save))
        directory = monitor.args.run_dir / "checkpoints"
        self.assertIs(wrapped(str(directory), 10, params, config), marker)
        metadata = self.read(self.sidecar(directory / f"{10:012d}"))
        self.assertEqual(metadata["capturedSeconds"], 60)
        self.assertEqual(metadata["capturedSteps"], 10)
        self.assertIs(seen[0][2], params)
        self.assertIs(seen[0][3], config)
        self.assertIn("save.checkpoint", json.dumps(monitor.timing.snapshot()))

    def test_actual_final_hook_waits_for_report_cutoff_and_keeps_timing_fields(self):
        monitor, clock = self.monitor()
        monitor.state["phase"] = "saving"
        clock.value = 90

        def save(path, step, params, config):
            self.checkpoint(monitor.args.run_dir, step)
            clock.value += 5

        wrapped = timed.workbench_checkpoint_save(timed.timed_call(save, "save.checkpoint", monitor), monitor)
        checkpoint = monitor.args.run_dir / "checkpoints" / f"{10:012d}"
        wrapped(str(checkpoint.parent), 10, {}, {})
        self.assertFalse(self.sidecar(checkpoint).exists())
        clock.value = 130
        writer = timed.timed_atomic_json(timed.console.atomic_json, monitor)
        writer(monitor.args.run_dir / "report.json", {"checkpoint": str(checkpoint), "actual_timesteps": 10,
               "training_stop_elapsed_seconds": 90, "elapsed_seconds": 130, "finalization_seconds": 40})
        report = self.read(monitor.args.run_dir / "report.json")
        metadata = self.read(self.sidecar(checkpoint))
        self.assertEqual(metadata["capturedSeconds"], 90)
        self.assertEqual(report["finalization_seconds"], 40)
        self.assertEqual(report["workbench_context_id"], monitor.state["workbench_context_id"])
        self.assertIn("timing_summary", report)
        self.assertIn("runtime_profile", report)

    def test_preparation_and_failed_weight_saves_never_publish_capture_success(self):
        monitor, clock = self.monitor(mode="leg_force_bc")
        clock.value = 60

        def save(path, step, params, config):
            self.checkpoint(monitor.args.run_dir, step)

        wrapped = timed.workbench_checkpoint_save(save, monitor)
        checkpoint = monitor.args.run_dir / "checkpoints" / f"{10:012d}"
        wrapped(str(checkpoint.parent), 10, {}, {})
        self.assertFalse(self.sidecar(checkpoint).exists())
        failing = timed.workbench_checkpoint_save(Mock(side_effect=RuntimeError("weight failure")), monitor)
        with self.assertRaises(RuntimeError):
            failing(str(checkpoint.parent), 11, {}, {})
        self.assertFalse(self.sidecar(checkpoint.parent / f"{11:012d}").exists())


if __name__ == "__main__":
    unittest.main()
