"""CPU-only contracts for new defaults and unchanged checkpoint inheritance."""
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime_defaults import (PROFILE_CHOICES, RUNTIME_FIELDS, RuntimeProfile, apply_profile,
                              resolve_profile)


class RuntimeDefaultTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.run = Path(self.directory.name) / "run"
        self.checkpoint = self.run / "checkpoints" / "000012345678"
        self.checkpoint.mkdir(parents=True)
        self.config = self.run / "ppo_config.json"
        self.old = dict(num_envs=128, batch_size=128, num_minibatches=1,
                        unroll_length=16, num_updates_per_batch=1)
        self.new = dict(num_envs=512, batch_size=512, num_minibatches=1,
                        unroll_length=16, num_updates_per_batch=1)

    def write_config(self, values=None):
        self.config.write_text(json.dumps(self.old if values is None else values), encoding="utf8")

    def test_exported_choices_and_runtime_fields_are_the_shared_contract(self):
        self.assertEqual(PROFILE_CHOICES, ("auto", "throughput512", "legacy", "checkpoint"))
        self.assertEqual(RUNTIME_FIELDS, tuple(self.old))

    def test_auto_new_force_training_uses_verified_throughput_profile(self):
        profile = resolve_profile("leg_force_bc")
        self.assertEqual(profile.requested, "auto")
        self.assertEqual(profile.effective, "throughput512")
        self.assertEqual(profile.source, "builtin")
        self.assertEqual(dict(profile.settings), self.new)
        self.assertFalse(profile.overrides_checkpoint)

    def test_auto_other_new_modes_leave_entry_options_alone(self):
        for mode in ("original", "torque200", "wheel_torque_bc", "future_mode"):
            with self.subTest(mode=mode):
                profile = resolve_profile(mode)
                self.assertEqual(profile.effective, "legacy")
                self.assertEqual(dict(profile.settings), {})
                self.assertEqual(apply_profile(profile, self.old), self.old)

    def test_auto_restore_inherits_all_five_existing_settings(self):
        old = dict(self.old, num_envs=64, batch_size=16, num_minibatches=4,
                   unroll_length=32, num_updates_per_batch=3,
                   learning_rate=1e-4, network_factory={"shape": [8, 16]})
        self.write_config(old)
        expected = {key: old[key] for key in RUNTIME_FIELDS}
        for mode in ("leg_force_bc", "original", "future_mode"):
            with self.subTest(mode=mode):
                profile = resolve_profile(mode, self.checkpoint)
                self.assertEqual(profile.effective, "checkpoint")
                self.assertEqual(profile.source, "checkpoint")
                self.assertEqual(dict(profile.settings), expected)
                self.assertEqual(profile.checkpoint_config_path, str(self.config.resolve()))

    def test_explicit_throughput_override_does_not_read_corrupt_old_settings(self):
        self.config.write_text("corrupt prior configuration", encoding="utf8")
        profile = resolve_profile("leg_force_bc", self.checkpoint, "throughput512")
        self.assertEqual(profile.settings, self.new)
        self.assertTrue(profile.overrides_checkpoint)
        self.assertEqual(profile.as_dict()["requested"], "throughput512")
        self.assertEqual(apply_profile(profile, self.old), self.new)

    def test_explicit_throughput_is_rejected_for_other_modes(self):
        for mode in ("original", "torque200", "wheel_torque_bc"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                resolve_profile(mode, requested="throughput512")

    def test_explicit_legacy_never_reads_or_overrides_checkpoint_settings(self):
        self.config.write_text("damaged JSON", encoding="utf8")
        for restore in (None, self.checkpoint):
            with self.subTest(restore=restore):
                profile = resolve_profile("leg_force_bc", restore, "legacy")
                self.assertEqual(profile.effective, "legacy")
                self.assertEqual(profile.settings, {})
                self.assertEqual(apply_profile(profile, self.old), self.old)

    def test_explicit_checkpoint_requires_restore_and_existing_config(self):
        with self.assertRaises(ValueError):
            resolve_profile("leg_force_bc", requested="checkpoint")
        with self.assertRaises(ValueError):
            resolve_profile("leg_force_bc", self.checkpoint, "checkpoint")
        self.write_config()
        profile = resolve_profile("leg_force_bc", self.checkpoint, "checkpoint")
        self.assertEqual(profile.effective, "checkpoint")
        self.assertEqual(profile.settings, self.old)

    def test_missing_whole_config_is_explicit_legacy_without_fabricated_settings(self):
        profile = resolve_profile("leg_force_bc", self.checkpoint)
        self.assertEqual(profile.effective, "legacy_missing_config")
        self.assertEqual(profile.settings, {})
        self.assertEqual(profile.as_dict()["source"], "entrypoint")
        self.assertEqual(profile.as_dict()["checkpoint_config_path"], str(self.config.resolve()))
        self.assertEqual(apply_profile(profile, self.old), self.old)
        self.assertFalse(self.config.exists())

    def test_present_but_corrupt_incomplete_or_ambiguous_config_fails(self):
        sources = ("broken", "[]", "null", "true", "{}",
                   json.dumps({k: v for k, v in self.old.items() if k != "unroll_length"}),
                   json.dumps(dict(self.old, num_envs=256)),
                   '{"num_envs":128,"num_envs":512}',
                   json.dumps(dict(self.old, unused=float("nan"))))
        for source in sources:
            self.config.write_text(source, encoding="utf8")
            for request in ("auto", "checkpoint"):
                with self.subTest(source=source, request=request), self.assertRaises(ValueError):
                    resolve_profile("leg_force_bc", self.checkpoint, request)

    def test_nonpositive_boolean_float_or_string_settings_are_rejected(self):
        for field in RUNTIME_FIELDS:
            for invalid in (0, -1, True, 1.0, "1", None):
                self.write_config(dict(self.old, **{field: invalid}))
                with self.subTest(field=field, invalid=invalid), self.assertRaises(ValueError):
                    resolve_profile("leg_force_bc", self.checkpoint)

    def test_settings_and_profile_are_immutable_and_export_is_independent(self):
        self.write_config()
        profile = resolve_profile("leg_force_bc", self.checkpoint)
        with self.assertRaises(TypeError):
            profile.settings["num_envs"] = 512
        with self.assertRaises(FrozenInstanceError):
            profile.effective = "throughput512"
        exported = profile.as_dict()
        exported["settings"]["num_envs"] = 9
        self.assertEqual(profile.settings["num_envs"], 128)
        self.assertEqual(json.loads(json.dumps(profile.as_dict())), profile.as_dict())

    def test_direct_profile_construction_defensively_copies_settings(self):
        supplied = dict(self.old)
        profile = RuntimeProfile("auto", "checkpoint", "checkpoint", supplied)
        supplied["num_envs"] = 512
        self.assertEqual(profile.settings["num_envs"], 128)
        with self.assertRaises(ValueError):
            RuntimeProfile("auto", "throughput512", "builtin", {})
        with self.assertRaises(ValueError):
            RuntimeProfile("auto", "checkpoint", "checkpoint", dict(self.old, unknown=3))

    def test_apply_is_copy_preserves_other_options_and_validates_actual_result(self):
        callback = object()
        original = dict(self.old, learning_rate=3e-5, entropy_cost=.001,
                        network_factory={"width": [64, 64]}, policy_params_fn=callback)
        unchanged = dict(original)
        result = apply_profile(resolve_profile("leg_force_bc"), original)
        self.assertIsNot(result, original)
        self.assertEqual(original, unchanged)
        for key in ("learning_rate", "entropy_cost", "network_factory", "policy_params_fn"):
            self.assertIs(result[key], original[key])
        self.assertEqual({key: result[key] for key in RUNTIME_FIELDS}, self.new)
        result["num_envs"] = 1024
        self.assertEqual(original["num_envs"], 128)

    def test_inactive_profile_still_rejects_invalid_or_incomplete_actual_kwargs(self):
        profile = resolve_profile("original")
        for invalid in ({}, {"num_envs": 128}, dict(self.old, num_envs=512),
                        dict(self.old, num_updates_per_batch=False)):
            with self.subTest(options=invalid), self.assertRaises(ValueError):
                apply_profile(profile, invalid)

    def test_batch_constraint_uses_product_not_batch_size_alone(self):
        options = dict(self.old, num_envs=96, batch_size=128, num_minibatches=3)
        result = apply_profile(resolve_profile("original"), options)
        self.assertEqual(result, options)

    def test_same_merged_dict_can_drive_runtime_and_saved_json(self):
        self.write_config(dict(self.old, num_envs=256, batch_size=512, unroll_length=32))
        profile = resolve_profile("leg_force_bc", self.checkpoint)
        options = dict(self.old, learning_rate=3e-5, normalize_until_count=0,
                       network_factory={"policy_hidden_layer_sizes": [64, 64]})
        actual_kwargs = apply_profile(profile, options)
        persisted = json.loads(json.dumps(actual_kwargs))
        self.assertEqual(actual_kwargs, persisted)
        self.assertEqual({key: persisted[key] for key in RUNTIME_FIELDS}, dict(profile.settings))
        self.assertEqual(persisted["learning_rate"], options["learning_rate"])

    def test_resolving_and_applying_do_not_write_existing_run_files(self):
        self.write_config(dict(self.old, learning_rate=.001))
        model = self.checkpoint / "model.bin"
        model.write_bytes(b"original checkpoint")
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.run.rglob("*") if p.is_file()}
        profile = resolve_profile("leg_force_bc", self.checkpoint)
        apply_profile(profile, self.old)
        after = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.run.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_invalid_requests_and_empty_modes_are_rejected(self):
        for request in ("unknown", "AUTO", "", None, True):
            with self.subTest(request=request), self.assertRaises(ValueError):
                resolve_profile("leg_force_bc", requested=request)
        for mode in (None, "", 1):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                resolve_profile(mode)
        with self.assertRaises(ValueError):
            resolve_profile("leg_force_bc", "")
        with self.assertRaises(TypeError):
            apply_profile({}, self.old)
        with self.assertRaises(TypeError):
            apply_profile(resolve_profile("original"), [])


if __name__ == "__main__":
    unittest.main()
