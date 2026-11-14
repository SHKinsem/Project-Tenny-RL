"""No ML imports: cache evidence, invalidation, failure and cancellation tests."""
from contextlib import nullcontext
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/torque200"))
from hybrid import readiness as ready
from hybrid import stage_evidence as evidence
from hybrid import train
from hybrid.core import digest_file, digest_tree, write_json
from hybrid.timing import Timing


class RuntimeIdentityTests(unittest.TestCase):
    def test_console_and_cli_defaults_have_identical_actual_identity(self):
        root = Path("/home/test/runtime")
        cli = ready.prepare_environment(root, {})
        # Web preview is CPU; its training supervisor explicitly requests CUDA.
        web_parent = {"JAX_PLATFORMS": "cpu", "LD_LIBRARY_PATH": "/preview/lib"}
        web_parent["JAX_PLATFORMS"] = "cuda"
        console = ready.prepare_environment(root, web_parent)
        driver = SimpleNamespace(stdout="gpu-id, GPU name, 580.0, 7.5\n")
        with patch.object(ready.importlib.metadata, "distributions", return_value=[]), \
                patch.object(ready.subprocess, "run", return_value=driver):
            with patch.dict(os.environ, cli, clear=True):
                a = ready.runtime_identity()
            with patch.dict(os.environ, console, clear=True):
                b = ready.runtime_identity()
        self.assertEqual(a, b)
        self.assertEqual(a["settings"]["CUDA_VISIBLE_DEVICES"], "0")
        self.assertEqual(a["settings"]["XLA_PYTHON_CLIENT_MEM_FRACTION"], "0.35")
        self.assertEqual(a["settings"]["LD_LIBRARY_PATH"], "")

    def test_explicit_device_and_memory_changes_are_not_normalized_away(self):
        root = Path("/home/test/runtime")
        first = ready.prepare_environment(root, {})
        other = ready.prepare_environment(root, {"CUDA_VISIBLE_DEVICES": "1", "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.5"})
        driver = SimpleNamespace(stdout="gpu-id, GPU name, 580.0, 7.5\n")
        with patch.object(ready.importlib.metadata, "distributions", return_value=[]), \
                patch.object(ready.subprocess, "run", return_value=driver):
            with patch.dict(os.environ, first, clear=True):
                a = ready.runtime_identity()
            with patch.dict(os.environ, other, clear=True):
                b = ready.runtime_identity()
        self.assertNotEqual(a, b)


class ReadinessTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.environment = dict(packages=[["jax", "test"]], devices=["gpu-a, driver-a"],
                                settings={"JAX_PLATFORMS": "cuda"})
        self.sources = {"assets/model.xml": "model", "hybrid/core.py": "core", "hybrid/env.py": "env",
                        "hybrid/train.py": "train", "hybrid/readiness.py": "orchestration",
                        "legwheel/env.py": "physics"}
        handle = patch.object(evidence, "inventory", lambda: dict(self.sources))
        handle.start(); self.addCleanup(handle.stop)
        for target, value in (("runtime_identity", lambda: copy.deepcopy(self.environment)),
                              ("source_fingerprint", lambda: "source-v1"),
                              ("_cache_lock", lambda *a: nullcontext())):
            handle = patch.object(ready, target, value)
            handle.start(); self.addCleanup(handle.stop)
        handle = patch.object(ready, "_run_stage", side_effect=self.stage)
        self.stages = handle.start(); self.addCleanup(handle.stop)

    def stage(self, label, module, argv, log, report, cancelled):
        Path(log).write_bytes(b"real test fixture, no ML\n")
        hz = int(argv[argv.index("--control-hz") + 1])
        if module == "hybrid.validate":
            write_json(report, dict(schema="hybrid-local-validation-v1", runtime_validated=True,
                       source_fingerprint=ready.source_fingerprint(), runtime_packages={"jax": "test"},
                       **Timing(hz).contract()))
            return
        mode = argv[argv.index("--mode") + 1]
        run = Path(report).parent
        checkpoint = run / "checkpoints/000000000016"
        checkpoint.mkdir(parents=True)
        (checkpoint / "weights").write_bytes(b"checkpoint bytes")
        (run / "policy.cpp.txt").write_text("export bytes")
        export_hash = digest_file(run / "policy.cpp.txt")
        write_json(run / "policy_metadata.json", dict(source_fingerprint=ready.source_fingerprint(),
                   action_mode=mode, export_sha256=export_hash, **Timing(hz).contract()))
        value = dict(action_mode=mode, control_hz=hz, checkpoint_validated=True,
                     all_parameters_finite=True, parameter_tree_max_abs_error=0,
                     restore_max_abs_error=0, export_max_abs_error=1e-5, adapter_max_abs_error=1e-8,
                     finite_training_losses_checked=["training/total_loss"], actual_progress_steps=16,
                     checkpoint=str(checkpoint), checkpoint_name=checkpoint.name,
                     checkpoint_sha256=digest_tree(checkpoint), export_sha256=export_hash)
        if run.name == "restart":
            value.update(warm_restart_changed_network_leaves=16, warm_restart_network_max_abs_change=1e-4)
        write_json(report, value)

    def ensure(self, **kw):
        mode = kw.pop("mode", "virtual_leg_force")
        hz = kw.pop("hz", 100)
        return ready.ensure_ready(self.root, mode, hz, **kw)

    def test_first_full_then_hit_does_not_launch_children(self):
        states = []
        first = self.ensure(on_status=states.append)
        self.assertFalse(first["cache_hit"])
        self.assertEqual(self.stages.call_count, 3)
        second = self.ensure()
        self.assertTrue(second["cache_hit"])
        self.assertEqual(self.stages.call_count, 3)
        self.assertEqual(second["validation_path"], first["validation_path"])
        self.assertEqual(states[-1]["stage"], "readiness_ready")
        restart_argv = self.stages.call_args_list[2].args[2]
        self.assertIn("--restore", restart_argv)

    def test_source_runtime_mode_frequency_are_separate(self):
        original = self.ensure()
        with patch.object(ready, "source_fingerprint", return_value="source-v2"):
            self.sources["hybrid/readiness.py"] = "display-only-change"
            self.assertTrue(self.ensure()["cache_hit"])
        self.sources["hybrid/core.py"] = "changed-control"
        self.assertFalse(self.ensure()["cache_hit"])
        self.environment["devices"] = ["gpu-b, driver-b"]
        self.assertNotEqual(self.ensure()["cache_key"], original["cache_key"])
        self.assertFalse(self.ensure(mode="matched_envelope")["cache_hit"])
        self.assertFalse(self.ensure(hz=200)["cache_hit"])
        self.environment["packages"].append(["flax", "different"])
        self.assertFalse(self.ensure()["cache_hit"])
        self.environment["settings"]["XLA_FLAGS"] = "different"
        self.assertFalse(self.ensure()["cache_hit"])

    def test_modified_export_checkpoint_report_are_not_reused(self):
        for relative, expected_runs in (("first/policy.cpp.txt", 2), ("first/checkpoints/000000000016/weights", 2),
                                       ("restart/report.json", 1), ("validation.json", 1)):
            with self.subTest(relative=relative):
                result = self.ensure()
                stage = relative.split('/')[0] if '/' in relative else 'validation'
                ref = ready._read(result["stage_references"][stage])
                base = Path(ref['report_path']).parent
                path = base / (relative.split('/', 1)[1] if '/' in relative else 'validation.json')
                path.write_bytes(b"damaged")
                before = self.stages.call_count
                fresh = self.ensure()
                self.assertFalse(fresh["cache_hit"])
                self.assertEqual(self.stages.call_count, before + expected_runs)
                self.assertTrue(path.exists())  # Keep the failure evidence.

    def test_failed_force_revokes_old_cache(self):
        first = self.ensure()
        with patch.object(ready, "_run_stage", side_effect=RuntimeError("preflight failed")):
            with self.assertRaisesRegex(RuntimeError, "preflight failed"):
                self.ensure(force=True)
        self.assertEqual(ready.discover_ready(self.root, 'virtual_leg_force', 100)['reused_stages'], [])
        self.assertFalse(self.ensure()["cache_hit"])

    def test_cancellation_keeps_completed_validation_and_resumes_first(self):
        flag = [False]
        def cancel_after_stage(label, module, argv, log, report, cancelled):
            self.stage(label, module, argv, log, report, cancelled)
            flag[0] = True
        self.stages.side_effect = cancel_after_stage
        with self.assertRaises(ready.ReadinessCancelled):
            self.ensure(cancelled=lambda: flag[0])
        self.assertEqual(list((self.root / "readiness-cache").glob("*.json")), [])
        self.assertEqual(self.stages.call_count, 1)
        self.assertEqual(ready.discover_ready(self.root, 'virtual_leg_force', 100)['reused_stages'], ['validation'])
        self.stages.side_effect = self.stage
        result = self.ensure()
        self.assertEqual(result['reused_stages'], ['validation'])
        self.assertEqual(self.stages.call_count, 3)

    def test_changed_environment_during_gate_cannot_publish(self):
        def change(label, module, argv, log, report, cancelled):
            self.stage(label, module, argv, log, report, cancelled)
            self.environment["devices"] = ["changed mid-run"]
        self.stages.side_effect = change
        with self.assertRaisesRegex(RuntimeError, "changed during preparation"):
            self.ensure()
        self.assertEqual(list((self.root / "readiness-cache").glob("*.json")), [])

    def test_missing_network_update_rejected(self):
        def no_update(label, module, argv, log, report, cancelled):
            self.stage(label, module, argv, log, report, cancelled)
            if Path(report).parent.name == "restart":
                value = ready._read(report)
                value["warm_restart_changed_network_leaves"] = 0
                write_json(report, value)
        self.stages.side_effect = no_update
        with self.assertRaisesRegex(ValueError, "network update"):
            self.ensure()

    def test_subprocess_failure_preserves_exit_code(self):
        self.stages.side_effect = subprocess.CalledProcessError(7, ["fake-ML"])
        with self.assertRaises(subprocess.CalledProcessError) as caught:
            self.ensure()
        self.assertEqual(caught.exception.returncode, 7)
        self.assertEqual(list((self.root / "readiness-cache").glob("*.json")), [])

    def test_cache_manifest_tamper_forces_new_evidence(self):
        result = self.ensure()
        manifest = Path(result['stage_references']['validation'])
        content = ready._read(manifest)
        content['report_sha256'] = 'invalid'
        write_json(manifest, content)
        self.assertFalse(self.ensure()["cache_hit"])

    def test_train_dependency_change_reuses_validation_only(self):
        self.ensure()
        self.sources['hybrid/train.py'] = 'changed-training'
        result = self.ensure()
        self.assertEqual(result['reused_stages'], ['validation'])

    def test_unknown_new_module_invalidates_all_stages(self):
        self.ensure()
        self.sources['hybrid/new_physics.py'] = 'new'
        self.assertEqual(self.ensure()['reused_stages'], [])

    def test_failed_restart_reuses_first_even_after_orchestration_change(self):
        original = self.stage
        def fail_restart(label, module, argv, log, report, cancelled):
            if Path(report).parent.name == 'restart':
                raise subprocess.CalledProcessError(7, ['restart'])
            return original(label, module, argv, log, report, cancelled)
        self.stages.side_effect = fail_restart
        with self.assertRaises(subprocess.CalledProcessError):
            self.ensure()
        self.sources['hybrid/readiness.py'] = 'fixed-orchestration'
        self.stages.side_effect = original
        with patch.object(ready, 'source_fingerprint', return_value='source-v2'):
            result = self.ensure()
        self.assertEqual(result['reused_stages'], ['validation', 'first'])
        argv = self.stages.call_args.args[2]
        self.assertIn('--restore-stage-reference', argv)
        reference = argv[argv.index('--restore-stage-reference')+1]
        checkpoint = argv[argv.index('--restore')+1]
        # The explicit reference accepts only the audited unchanged training
        # dependencies, never arbitrary metadata/source mismatches.
        expected = dict(action_mode='virtual_leg_force', source_fingerprint='source-v2', **Timing(100).contract())
        train.restore_guard(checkpoint, expected, reference)
        with self.assertRaises(ValueError):
            train.restore_guard(checkpoint, expected)

    def test_audited_partial_validation_import_and_original_report_preserved(self):
        directory = self.root / 'artifacts/readiness-old'
        directory.mkdir(parents=True)
        expected = ready.identity('virtual_leg_force', 100)
        write_json(directory/'identity.json', expected)
        self.stage('预检', 'hybrid.validate', ['--control-hz', '100'], directory/'validation.log', directory/'validation.json', None)
        log = directory/'managed.log'
        log.write_text('[通过] 预检\n[准备] 新训练\n', encoding='utf8')
        att = dict(schema=evidence.ATTESTATION, source_fingerprint='source-v1',
                   files=dict(self.sources), validation_sha256=digest_file(directory/'validation.json'),
                   identity_sha256=digest_file(directory/'identity.json'), validation_exit_code=0,
                   exit_evidence=dict(log=str(log), sha256=digest_file(log)))
        write_json(directory/'source-manifest.json', att)
        before = (directory/'validation.json').read_bytes()
        self.sources['hybrid/readiness.py'] = 'new-cache-logic'
        with patch.object(ready, 'source_fingerprint', return_value='source-v2'):
            plan = ready.discover_ready(self.root, 'virtual_leg_force', 100)
            self.assertEqual(plan['reused_stages'], ['validation'])
            result = self.ensure()
            self.assertEqual(result['reused_stages'], ['validation'])
            accepted = train.check_validation(result['validation_path'], 100)
            self.assertEqual(accepted['source_fingerprint'], 'source-v1')
        self.assertEqual((directory/'validation.json').read_bytes(), before)
        self.assertEqual(self.stages.call_count, 2)

    def test_missing_first_report_never_counts_as_passed_stage(self):
        def incomplete(label, module, argv, log, report, cancelled):
            self.stage(label, module, argv, log, report, cancelled)
            if Path(report).parent.name == 'first':
                Path(report).unlink()
                raise ready.ReadinessCancelled('cancelled after export before acceptance')
        self.stages.side_effect = incomplete
        with self.assertRaises(ready.ReadinessCancelled):
            self.ensure()
        self.assertEqual(ready.discover_ready(self.root, 'virtual_leg_force', 100)['reused_stages'], ['validation'])


if __name__ == "__main__":
    unittest.main()
