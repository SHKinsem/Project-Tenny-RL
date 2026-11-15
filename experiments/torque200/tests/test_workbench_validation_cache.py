"""Real contract/hash validation cached only for reads; no simulator or process."""
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training_control import ControlError, TrainingController, _read, _write
from workbench_validation import Dependency, filesystem_signature, source_signature
from hybrid import core, managed_training
from force_teacher import contract as force_contract
from wheel_teacher import contract as wheel_contract


class ValidationCacheTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="workbench-validation-")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name).resolve()
        # Deliberately distinct roots: verifier module ROOT, configured runtime,
        # and the author's reference source need not be the same directory.
        self.source = self.base / "release-model-source"
        self.reference = self.base / "reference"
        self.controller = TrainingController(self.base / "artifacts", self.reference, self.base / "runtime")
        self.controller._locked = nullcontext
        for module in (core, force_contract, wheel_contract):
            self.enterContext(patch.object(module, "ROOT", self.source))
        # This is a protocol/hash test, not a geometry or dynamics test. Keep
        # actual metadata, source fingerprints and verified_report functions.
        for module in (core, force_contract):
            self.enterContext(patch.object(module, "read_geometry", return_value=[]))
            self.enterContext(patch.object(module, "scan_geometry", return_value=[]))
        for name in ("assets/robot.xml", "legwheel/env.py", "hybrid/core.py",
                     "force_teacher/networks.py", "wheel_teacher/policy.py", "console_train.py",
                     "control_modes.py", "preview_sync.py", "verify_artifacts.py", "hybrid-wsl.sh"):
            self.file(self.source / name, b"source-v1\n")
        for name in ("mujoco_legwheel_rl_public/legwheel/config.py", "mujoco_legwheel_rl_public/assets/robot.xml",
                     "mujoco_esp32_robot_deploy_public/host/policy/rl_policy.py"):
            self.file(self.reference / name, b"reference-v1\n")
        for suffix in ("cpp", "h"):
            self.file(self.reference / ("mujoco_esp32_robot_deploy_public/src/policy/rl_policy." + suffix),
                      (wheel_contract.AUTHOR_CHECKPOINT + "\nweights-v1\n").encode())

    @staticmethod
    def file(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def verifier(self, mode):
        return (force_contract if mode == "leg_force_bc" else wheel_contract
                if mode == "wheel_torque_bc" else managed_training)

    def saved_run(self, mode="leg_force_bc", name="saved", steps=128, report_time=None):
        run = self.controller.root / mode / name
        checkpoint = run / "checkpoints" / f"{steps:012d}"
        self.file(checkpoint / "weights.bin", b"checkpoint-weights-v1")
        export = run / "policy.cpp.txt"
        self.file(export, b"exported-weights-v1")
        spec = __import__("control_modes").mode_spec(mode)
        metadata = (force_contract.metadata(self.reference) if mode == "leg_force_bc"
                    else wheel_contract.metadata(self.reference) if mode == "wheel_torque_bc"
                    else core.metadata(spec["action_mode"], [], spec["control_hz"]))
        metadata["export_sha256"] = core.digest_file(export)
        _write(run / "policy_metadata.json", metadata)
        _write(run / "env_config.json", {})
        _write(run / "ppo_config.json", dict(num_envs=128, batch_size=128, num_minibatches=1,
                                             unroll_length=16, num_updates_per_batch=1))
        _write(run / "status.json", dict(mode=mode, phase="completed", steps=steps))
        _write(run / "report.json", dict(mode=mode, checkpoint_validated=True,
               all_parameters_finite=True, parameter_tree_max_abs_error=0,
               export_max_abs_error=0., export_atol=.001, checkpoint=str(checkpoint), export_path=str(export),
               checkpoint_name=checkpoint.name, checkpoint_sha256=core.digest_tree(checkpoint),
               export_sha256=metadata["export_sha256"], source_fingerprint=metadata["source_fingerprint"],
               action_mode=spec["action_mode"], control_hz=spec["control_hz"], steps=steps,
               details={"labels": ["verified"]}))
        if report_time is not None:
            os.utime(run / "report.json", ns=(report_time, report_time))
        return run, checkpoint

    def test_warm_status_capabilities_lists_and_parameter_schema_do_not_revalidate(self):
        for mode in ("leg_force_bc", "wheel_torque_bc", "hybrid_virtual_leg_force_100"):
            with self.subTest(mode=mode):
                _, checkpoint = self.saved_run(mode)
                module = self.verifier(mode)
                with patch.object(module, "verified_report", wraps=module.verified_report) as verify:
                    first = self.controller.status(mode)
                    self.assertTrue(first["can_resume"])
                    self.assertIn(verify.call_count, (1, 2))
                    cold_calls = verify.call_count
                    first["latest"]["details"]["labels"].append("caller mutation")
                    first["latest"]["checkpoint"] = "caller mutation"
                    for _ in range(3):
                        current = self.controller.status(mode)
                        self.assertEqual(current["latest"]["details"]["labels"], ["verified"])
                        self.assertEqual(current["latest_checkpoint"], str(checkpoint))
                        self.assertTrue(self.controller.capabilities(mode)["resumeLatest"])
                        options = self.controller.checkpoint_options(mode)
                        self.assertEqual(options["checkpoints"][0]["path"], str(checkpoint))
                        self.assertEqual(len(self.controller.runs(mode)), 1)
                        self.controller.parameters(mode, checkpoint=str(checkpoint))
                    self.assertEqual(verify.call_count, cold_calls)

    def test_same_size_weight_export_and_required_metadata_edits_invalidate(self):
        run, checkpoint = self.saved_run()
        module = force_contract
        with patch.object(module, "verified_report", wraps=module.verified_report) as verify:
            self.assertIsNotNone(self.controller.latest("leg_force_bc"))
            for path in (checkpoint / "weights.bin", run / "policy.cpp.txt", run / "policy_metadata.json"):
                with self.subTest(path=path.name):
                    before = path.read_bytes()
                    changed = (before.replace(b'"control_hz": 100', b'"control_hz": 200')
                               if path.name == "policy_metadata.json" else b"!" + before[1:])
                    self.assertEqual(len(before), len(changed))
                    self.assertNotEqual(before, changed)
                    path.write_bytes(changed)
                    calls = verify.call_count
                    self.assertIsNone(self.controller.latest("leg_force_bc"))
                    self.assertIn(verify.call_count - calls, (1, 2))
                    changed_calls = verify.call_count
                    # Invalid unchanged artifacts are cheap during polling too.
                    self.assertIsNone(self.controller.latest("leg_force_bc"))
                    self.assertEqual(verify.call_count, changed_calls)
                    path.write_bytes(before)
                    self.assertIsNotNone(self.controller.latest("leg_force_bc"))

    def test_actual_force_reference_source_assets_host_and_teacher_weights_invalidate(self):
        self.saved_run()
        paths = [self.source / "hybrid" / "core.py", self.source / "legwheel" / "env.py",
                 self.source / "assets" / "robot.xml", self.source / "force_teacher" / "networks.py",
                 self.source / "wheel_teacher" / "policy.py", self.source / "console_train.py",
                 self.reference / "mujoco_legwheel_rl_public" / "legwheel" / "config.py",
                 self.reference / "mujoco_legwheel_rl_public" / "assets" / "robot.xml",
                 self.reference / "mujoco_esp32_robot_deploy_public" / "host" / "policy" / "rl_policy.py",
                 self.reference / "mujoco_esp32_robot_deploy_public" / "src" / "policy" / "rl_policy.cpp",
                 self.reference / "mujoco_esp32_robot_deploy_public" / "src" / "policy" / "rl_policy.h"]
        with patch.object(force_contract, "verified_report", wraps=force_contract.verified_report) as verify:
            self.assertIsNotNone(self.controller.latest("leg_force_bc"))
            for path in paths:
                with self.subTest(path=path.relative_to(self.base)):
                    before = path.read_bytes()
                    path.write_bytes(before.replace(b"-v1", b"-v2"))
                    calls = verify.call_count
                    self.assertIsNone(self.controller.latest("leg_force_bc"))
                    self.assertIn(verify.call_count - calls, (1, 2))
                    path.write_bytes(before)
                    self.assertIsNotNone(self.controller.latest("leg_force_bc"))

    def test_source_and_checkpoint_entries_added_or_removed_invalidate(self):
        _, checkpoint = self.saved_run()
        with patch.object(force_contract, "verified_report", wraps=force_contract.verified_report) as verify:
            self.assertIsNotNone(self.controller.latest("leg_force_bc"))
            for path in (self.source / "force_teacher" / "new.py", checkpoint / "new-weights.bin"):
                with self.subTest(path=path.name):
                    path.write_bytes(b"new")
                    calls = verify.call_count
                    self.assertIsNone(self.controller.latest("leg_force_bc"))
                    self.assertIn(verify.call_count - calls, (1, 2))
                    path.unlink()
                    self.assertIsNotNone(self.controller.latest("leg_force_bc"))
            directory = checkpoint / "new-empty-directory"
            directory.mkdir()
            calls = verify.call_count
            self.assertIsNotNone(self.controller.latest("leg_force_bc"))
            self.assertIn(verify.call_count - calls, (1, 2))
            calls = verify.call_count
            directory.rmdir()
            self.assertIsNotNone(self.controller.latest("leg_force_bc"))
            self.assertIn(verify.call_count - calls, (1, 2))

    def test_report_replacement_and_checkpoint_metadata_changes_invalidate(self):
        run, checkpoint = self.saved_run()
        with patch.object(force_contract, "verified_report", wraps=force_contract.verified_report) as verify:
            self.assertIsNotNone(self.controller.latest("leg_force_bc"))
            report = run / "report.json"
            before = report.stat()
            replacement = report.with_name("replacement.json")
            replacement.write_bytes(report.read_bytes())
            os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
            os.replace(replacement, report)
            calls = verify.call_count
            self.assertIsNotNone(self.controller.latest("leg_force_bc"))
            self.assertIn(verify.call_count - calls, (1, 2))
            calls = verify.call_count
            self.assertIsNotNone(self.controller.latest("leg_force_bc"))
            self.assertEqual(verify.call_count, calls)
            _write(checkpoint.with_name(checkpoint.name + ".workbench.json"), {"trainedSeconds": None})
            self.assertIsNotNone(self.controller.latest("leg_force_bc"))
            self.assertIn(verify.call_count - calls, (1, 2))
            calls = verify.call_count
            self.assertIsNotNone(self.controller.latest("leg_force_bc"))
            self.assertEqual(verify.call_count, calls)

    def test_hardlink_count_changes_invalidate_without_changing_weight_bytes(self):
        _, checkpoint = self.saved_run()
        weights = checkpoint / "weights.bin"
        outside_link = self.base / "weight-alias.bin"
        def scanned_identity():
            # Windows DirEntry metadata can omit identity/link counts even when
            # Path.stat reports them. Probe the scanner's actual stat API.
            with os.scandir(checkpoint) as entries:
                info = next(entry for entry in entries if entry.name == weights.name).stat(follow_symlinks=False)
            return info.st_nlink, info.st_dev, info.st_ino
        with patch.object(force_contract, "verified_report", wraps=force_contract.verified_report) as verify:
            self.assertIsNotNone(self.controller.latest("leg_force_bc"))
            before = weights.stat()
            before_identity = scanned_identity()
            try:
                os.link(weights, outside_link)
            except OSError:
                self.skipTest("Host cannot create fixture hardlinks")
            after = weights.stat()
            if scanned_identity() == before_identity:
                self.skipTest("Host DirEntry stat omits hardlink identity/link-count changes; required on Linux")
            self.assertEqual(after.st_mtime_ns, before.st_mtime_ns)
            self.assertEqual(after.st_size, before.st_size)
            calls = verify.call_count
            self.assertIsNotNone(self.controller.latest("leg_force_bc"))
            self.assertIn(verify.call_count - calls, (1, 2))
            calls = verify.call_count
            self.assertIsNotNone(self.controller.latest("leg_force_bc"))
            self.assertEqual(verify.call_count, calls)
            outside_link.unlink()
            self.assertIsNotNone(self.controller.latest("leg_force_bc"))
            self.assertIn(verify.call_count - calls, (1, 2))
            calls = verify.call_count
            self.assertIsNotNone(self.controller.latest("leg_force_bc"))
            self.assertEqual(verify.call_count, calls)

    def test_manual_snapshot_and_late_final_report_switch_latest_without_ttl(self):
        mode = "leg_force_bc"
        _, old = self.saved_run(mode, "old", 10, 1_000_000_000)
        active = self.controller.root / mode / "early-started-long-running"
        active.mkdir()
        record = dict(mode=mode, run_dir=str(active))
        self.assertEqual(self.controller._latest(mode, record, True)["checkpoint"], str(old))
        snapshot_run = self.controller.root / mode / "manual-snapshot"
        snapshot_run.mkdir()
        # The folder already exists: it is the later report/artifact publication,
        # not a mode-folder mtime heuristic, that must make this model visible.
        _, snapshot = self.saved_run(mode, "manual-snapshot", 20, 2_000_000_000)
        self.assertEqual(self.controller._latest(mode, record, True)["checkpoint"], str(snapshot))
        _, final = self.saved_run(mode, active.name, 30, 3_000_000_000)
        self.assertEqual(self.controller._latest(mode, record, False)["checkpoint"], str(final))
        self.assertEqual(self.controller._latest(mode, record, False)["checkpoint"], str(final))

    def test_fresh_selected_latest_and_import_reject_tampered_cached_weights(self):
        for action in ("select", "resume", "import"):
            with self.subTest(action=action):
                self.controller._validation_cache.clear()
                mode = "hybrid_virtual_leg_force_100"
                _, checkpoint = self.saved_run(mode, action)
                module = managed_training
                # A fresh action must also bypass the historical internal hybrid
                # shortcut, even if that weaker signature does not see an edit.
                with patch.object(module, "verification_signature", return_value=("unchanged",)), \
                        patch.object(module, "verified_report", wraps=module.verified_report) as verify:
                    self.assertIsNotNone(self.controller._validated(checkpoint.parent.parent, mode, fresh=False))
                    self.assertIn(verify.call_count, (1, 2))
                    cold_calls = verify.call_count
                    weights = checkpoint / "weights.bin"
                    before = weights.read_bytes()
                    weights.write_bytes(b"!" + before[1:])
                    if action == "select":
                        with self.assertRaises(ControlError):
                            self.controller._select_checkpoint(mode, str(checkpoint))
                    elif action == "resume":
                        # No other valid run may silently replace the exact
                        # candidate under this test's resumeLatest action.
                        with patch.object(self.controller, "_current", return_value=({}, {}, False)), \
                                patch("training_control.subprocess.Popen") as spawn:
                            with self.assertRaises(ControlError):
                                self.controller.start(mode, resume=True)
                            spawn.assert_not_called()
                    else:
                        with self.assertRaises(ValueError):
                            self.controller.import_checkpoint(str(checkpoint), mode)
                    self.assertGreater(verify.call_count, cold_calls)
                    weights.write_bytes(before)
                    # Remove this fixture so later subtests cannot choose it.
                    _write(checkpoint.parent.parent / "status.json", {"phase": "failed"})

    def test_unchanged_fresh_selection_still_performs_full_validation(self):
        mode = "hybrid_virtual_leg_force_100"
        _, checkpoint = self.saved_run(mode)
        with patch.object(managed_training, "verified_report", wraps=managed_training.verified_report) as verify:
            self.assertIsNotNone(self.controller.latest(mode))
            self.assertIn(verify.call_count, (1, 2))
            calls = verify.call_count
            self.assertIsNotNone(self.controller.latest(mode))
            self.assertEqual(verify.call_count, calls)
            self.controller._select_checkpoint(mode, str(checkpoint))
            self.assertGreater(verify.call_count, calls)
            calls = verify.call_count
            self.controller._select_checkpoint(mode, str(checkpoint))
            self.assertGreater(verify.call_count, calls)

    def test_partial_validation_does_not_poison_following_cache_hits(self):
        run, _ = self.saved_run()
        original = force_contract.verified_report
        calls = []
        def changing_verifier(*args):
            original(*args)
            calls.append(True)
            if len(calls) == 1:
                _write(run / "status.json", {"phase": "failed"})
        with patch.object(force_contract, "verified_report", changing_verifier), \
                patch.object(self.controller, "_validate_artifacts", wraps=self.controller._validate_artifacts) as validate:
            self.assertIsNone(self.controller.latest("leg_force_bc"))
            self.assertIsNone(self.controller.latest("leg_force_bc"))
            self.assertEqual(validate.call_count, 2)
            self.assertIsNone(self.controller.latest("leg_force_bc"))
            self.assertEqual(validate.call_count, 2)

    def test_signature_scan_limit_and_links_never_produce_cache_hits(self):
        _, checkpoint = self.saved_run()
        dependency = (Dependency(checkpoint, recursive=True),)
        self.assertIsNone(filesystem_signature(dependency, max_entries=1))
        outside = self.base / "outside.bin"
        outside.write_bytes(b"checkpoint-weights-v1")
        link = checkpoint / "linked.bin"
        try:
            link.symlink_to(outside)
        except OSError:
            self.skipTest("Host cannot create fixture symlinks")
        self.assertIsNone(filesystem_signature(dependency))
        with patch.object(force_contract, "verified_report", wraps=force_contract.verified_report) as verify:
            self.assertIsNone(self.controller.latest("leg_force_bc"))
            self.assertIsNone(self.controller.latest("leg_force_bc"))
            self.assertEqual(verify.call_count, 2)

    @unittest.skipUnless(os.name == "posix", "Linux ctime protects restored mtimes")
    def test_same_size_rewrite_with_restored_mtime_is_detected(self):
        _, checkpoint = self.saved_run()
        mode = "leg_force_bc"
        self.assertIsNotNone(self.controller.latest(mode))
        weights = checkpoint / "weights.bin"
        before = weights.stat()
        data = weights.read_bytes()
        weights.write_bytes(b"!" + data[1:])
        os.utime(weights, ns=(before.st_atime_ns, before.st_mtime_ns))
        self.assertEqual(weights.stat().st_mtime_ns, before.st_mtime_ns)
        self.assertIsNone(self.controller.latest(mode))


if __name__ == "__main__":
    unittest.main()
