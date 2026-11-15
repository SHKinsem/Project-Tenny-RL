"""Force preview applies the shared head to raw observations exactly once."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace, ModuleType
import importlib.util
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/torque200"))
from preview_sync import SnapshotPolicy, SnapshotPublisher, read_snapshot
from force_teacher import contract
from force_teacher.options import normalize_options, legacy_options


_make_metadata = contract.metadata


def metadata_fixture(original_root=None, experiment_config=None):
    """Use real head/config metadata without requiring a teacher installation."""
    with patch.object(contract, "hybrid_metadata", return_value=dict(source_fingerprint="hybrid", control_hz=100)), \
         patch.object(contract, "read_geometry", return_value=()), \
         patch.object(contract, "scan_geometry", return_value=()), \
         patch.object(contract, "source_fingerprint", return_value="force"), \
         patch.object(contract, "teacher_policy_path", return_value=Path("teacher.cpp")), \
         patch.object(contract, "file_digest", return_value="teacher-bytes"):
        return _make_metadata(original_root, experiment_config)


def arrays(size=24):
    values = {"mean": np.linspace(-.7, .7, size, dtype=np.float32),
              "std": np.linspace(.6, 1.8, size, dtype=np.float32)}
    for i, (a, b) in enumerate(zip((size, 64, 64), (64, 64, 8))):
        values[f"w{i}"] = np.zeros((a, b), dtype=np.float32)
        values[f"b{i}"] = np.zeros(b, dtype=np.float32)
    values["b2"][:] = [.25, -.4, 2., -3., .2, .3, .4, .5]
    return values


class ForcePreviewHeadTests(unittest.TestCase):
    def test_saved_export_and_live_snapshot_agree_with_nontrivial_parameters(self):
        from force_teacher.policy import ForcePolicy, _write_export, file_digest
        values = arrays()
        rng = np.random.default_rng(42)
        for i in range(3):
            values[f"w{i}"] = rng.normal(0, .04, values[f"w{i}"].shape).astype(np.float32)
        values["std"][0] = np.float32(1.2345678e-7)
        params = [SimpleNamespace(mean={"state": values["mean"]}, std={"state": values["std"]}),
                  {"params": {f"hidden_{i}": {"kernel": values[f"w{i}"], "bias": values[f"b{i}"]}
                              for i in range(3)}}, {}]
        reader_spec = importlib.util.spec_from_file_location("_force_test_reader",
            ROOT / "experiments/torque200/legwheel/deployment/numpy_policy.py")
        reader = importlib.util.module_from_spec(reader_spec)
        reader_spec.loader.exec_module(reader)
        modules = {"legwheel.deployment.numpy_policy": reader}
        for name in ("legwheel", "legwheel.deployment"):
            module = ModuleType(name)
            module.__path__ = []
            modules[name] = module
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, modules), patch.object(contract, "metadata", side_effect=metadata_fixture):
            run = Path(tmp)
            export = _write_export(params, run / "policy.cpp.txt", "test-only")
            for options in (normalize_options(), legacy_options()):
                with self.subTest(head=options["wheel_head"]):
                    expected = metadata_fixture(experiment_config=options)
                    (run / "policy_metadata.json").write_text(json.dumps(dict(expected, export_sha256=file_digest(export))))
                    saved = ForcePolicy(export, run / "policy_metadata.json")
                    live = SnapshotPolicy.from_params(params, "leg_force_bc", options)
                    np.testing.assert_array_equal(saved.arrays["std"], values["std"])
                    for wheel in ((0., 0.), (.8, -.6), (1., -1.), (-1.2, 1.2)):
                        obs = values["mean"] + values["std"] * rng.uniform(-1, 1, 24).astype(np.float32)
                        obs[-2:] = wheel
                        np.testing.assert_allclose(saved.action(obs), live.run(obs), atol=1e-6, rtol=1e-6)

    def test_wheel_skip_uses_raw_previous_torque_not_normalized_observation(self):
        from force_teacher.networks import transform_logits
        values = arrays()
        actor = SnapshotPolicy(values, "leg_force_bc")
        for left, right in ((.6, -.4), (1., -1.), (1.4, -2.), (0., 0.)):
            obs = np.zeros(24, dtype=np.float32)
            obs[-2:] = left, right
            expected = np.tanh(transform_logits(values["b2"], obs, xp=np)[:4])
            np.testing.assert_allclose(actor.run(obs), expected, atol=1e-7, rtol=1e-6)
            np.testing.assert_allclose(actor.run(obs)[:2], np.tanh(values["b2"][:2]), atol=1e-7)
        # A zero residual preserves the measured wheel torque, independent of
        # the actor normalizer; this catches missing, doubled or normalized skips.
        values["b2"][2:4] = 0
        actor = SnapshotPolicy(values, "leg_force_bc")
        obs[-2:] = .71, -.37
        np.testing.assert_allclose(actor.run(obs)[2:], obs[-2:], atol=1e-7)

    def test_legacy_modes_keep_the_plain_tanh_head(self):
        for mode, size in (("original", 26), ("wheel_torque_bc", 26),
                           ("torque200", 24), ("hybrid_virtual_leg_force_100", 24)):
            with self.subTest(mode=mode):
                values = arrays(size)
                obs = np.zeros(size, dtype=np.float32)
                obs[-2:] = .8, -.8
                np.testing.assert_allclose(SnapshotPolicy(values, mode).run(obs),
                                           np.tanh(values["b2"][:4]), atol=1e-7)

    def test_published_npz_preserves_actor_arrays_and_head_semantics(self):
        values = arrays()
        params = [SimpleNamespace(mean={"state": values["mean"]}, std={"state": values["std"]}),
                  {"params": {f"hidden_{i}": {"kernel": values[f"w{i}"], "bias": values[f"b{i}"]}
                              for i in range(3)}}, {}]
        variants = [dict(mode="leg_force_bc", teacher_schema=contract.SCHEMA,
                         experiment_config=normalize_options()),
                    dict(mode="leg_force_bc", teacher_schema=contract.LEGACY_SCHEMA,
                         policy_head="wheel_logit_residual_v1", wheel_residual_scale=.05)]
        for saved in variants:
            with self.subTest(schema=saved["teacher_schema"]), tempfile.TemporaryDirectory() as tmp, \
                 patch("preview_sync.snapshot_fingerprint", return_value="force-source"):
                run = Path(tmp)
                (run / "policy_metadata.json").write_text(json.dumps(saved))
                options = saved.get("experiment_config", legacy_options())
                (run / "preview.request.json").write_text(json.dumps({"mode": "leg_force_bc", "request_id": "open"}))
                report = SnapshotPublisher(run, "leg_force_bc").service(32, params)
                self.assertNotIn("error", report)
                loaded, metadata = read_snapshot(run / "preview-policy.npz", "leg_force_bc", "open")
                self.assertEqual(metadata["steps"], 32)
                self.assertEqual(loaded.experiment_config, options)
                self.assertEqual(metadata["experiment_config"], options)
                for key in values:
                    np.testing.assert_array_equal(loaded.arrays[key], values[key])
                direct = SnapshotPolicy.from_params(params, "leg_force_bc", options)
                for obs in np.random.default_rng(11).uniform(-1, 1, (12, 24)).astype(np.float32):
                    np.testing.assert_array_equal(loaded.run(obs), direct.run(obs))

    def test_publisher_rejects_missing_or_conflicting_saved_configuration(self):
        with tempfile.TemporaryDirectory() as tmp, patch("preview_sync.snapshot_fingerprint", return_value="force-source"):
            run = Path(tmp)
            (run / "preview.request.json").write_text(json.dumps({"mode": "leg_force_bc", "request_id": "open"}))
            missing = SnapshotPublisher(run, "leg_force_bc").service(32, [])
            self.assertIn("saved experiment configuration", missing["error"])
            self.assertFalse((run / "preview-policy.npz").exists())
            (run / "policy_metadata.json").write_text(json.dumps(dict(mode="leg_force_bc",
                teacher_schema=contract.SCHEMA, experiment_config=normalize_options())))
            mismatch = SnapshotPublisher(run, "leg_force_bc", experiment_config=legacy_options()).service(32, [])
            self.assertIn("differs from saved model", mismatch["error"])
            self.assertFalse((run / "preview-policy.npz").exists())


if __name__ == "__main__":
    unittest.main()
