"""Wheel-torque contract and label tests; optional CPU simulator checks are explicit."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/torque200"))
from wheel_teacher.teacher import student_observation, torque_label
from wheel_teacher import contract


class WheelLabelTests(unittest.TestCase):
    def test_label_is_actual_mean_torque_not_policy_velocity(self):
        action = np.array([.1, -.2, .8, -.7])
        actual = np.array([[.001, -.001], [.002, -.003], [.003, -.005], [.004, -.007]])
        result = torque_label(action, actual)
        np.testing.assert_allclose(result[:2], action[:2])
        np.testing.assert_allclose(result[2:], [.0025 / .0295, -.004 / .0295])
        self.assertFalse(np.allclose(result[2:], action[2:]))

    def test_invalid_labels_fail_instead_of_clipping(self):
        for actual in (np.full((4, 2), .03), np.full((4, 2), np.nan), np.zeros((1, 2))):
            with self.subTest(actual=actual.shape), self.assertRaises((ValueError, FloatingPointError)):
                torque_label(np.zeros(4), actual)
        with self.assertRaises(ValueError):
            torque_label(np.array([0, 0, 2., 0]), np.zeros((4, 2)))

    def test_student_tail_has_no_teacher_controller_memory(self):
        original = np.arange(26, dtype=np.float32)
        history = np.array([[.2, -.1], [.3, -.4]], dtype=np.float32)
        wheels = np.array([5., -6.])
        first = student_observation(original, history, wheels)
        original[20:] = 12345
        second = student_observation(original, history, wheels)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first[:20], original[:20])
        np.testing.assert_array_equal(first[20:24], history.flatten())
        np.testing.assert_array_equal(first[24:], wheels)

    def test_observation_rejects_bad_dimensions_and_nonfinite(self):
        with self.assertRaises(ValueError):
            student_observation(np.zeros(24), np.zeros((2, 2)), np.zeros(2))
        with self.assertRaises(FloatingPointError):
            student_observation(np.zeros(26), np.full((2, 2), np.nan), np.zeros(2))


class ContractTests(unittest.TestCase):
    def test_metadata_refuses_old_same_dimension_original_policy(self):
        expected = dict(mode="wheel_torque_bc", actor_dim=26, control_hz=100,
                        observation_version=contract.OBSERVATION_VERSION)
        with self.assertRaises(ValueError):
            contract.require_metadata({**expected, "mode": "original"}, expected)
        with self.assertRaises(ValueError):
            contract.require_metadata({**expected, "observation_version": "old"}, expected)

    def test_validated_report_binds_model_and_export_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ckpt = root / "checkpoints" / "000016"
            ckpt.mkdir(parents=True)
            (ckpt / "params").write_bytes(b"policy parameters")
            export = root / "policy.cpp.txt"
            export.write_bytes(b"policy export")
            expected = dict(schema=contract.SCHEMA, mode=contract.MODE, source_fingerprint="current")
            report = dict(mode=contract.MODE, checkpoint_validated=True, source_fingerprint="current",
                          checkpoint=str(ckpt), export_path=str(export),
                          checkpoint_sha256=contract.tree_digest(ckpt), export_sha256=contract.file_digest(export))
            (root / "policy_metadata.json").write_text(json.dumps({**expected, "export_sha256": report["export_sha256"]}))
            (root / "report.json").write_text(json.dumps(report))
            with patch.object(contract, "metadata", return_value=expected):
                contract.verified_report(root, report)
                contract.restore_guard(ckpt)
                export.write_bytes(b"tampered")
                with self.assertRaises(ValueError):
                    contract.verified_report(root, report)


@unittest.skipUnless(os.environ.get("TENNY_TEST_WHEEL_MJX") == "1", "Explicit CPU MJX validation only")
class StudentRuntimeTests(unittest.TestCase):
    def test_real_student_observation_and_torque_mapping(self):
        source = os.environ.get("TENNY_ORIGINAL_ROOT")
        self.assertTrue(source, "TENNY_ORIGINAL_ROOT must select the existing original runtime")
        code = r'''
import sys
from pathlib import Path
sys.path[:0] = [str(Path(sys.argv[1]) / 'mujoco_legwheel_rl_public'), sys.argv[2]]
import jax, jax.numpy as jp, numpy as np
from wheel_teacher.env import StudentEnv, student_config
cfg = student_config()
cfg.noise_config.level = 0.
cfg.push_config.enable = False
cfg.cmd_config.cmd_prob = 0.
env = StudentEnv(cfg)
state = jax.jit(env.reset)(jax.random.PRNGKey(0))
assert state.obs['state'].shape == (26,) and state.obs['privileged_state'].shape == (55,)
np.testing.assert_array_equal(state.obs['state'][20:24], np.zeros(4))
state.info['last_target_wheel_vel'] = jp.array([1e5,-1e5])
state.info['wheels_commands_history'] = jp.ones((2,2))*1e5
clean = env._get_obs(state.data,state.info)
np.testing.assert_array_equal(clean['state'][20:24], np.zeros(4))
step = jax.jit(env.step)
out = step(state,jp.array([0.,0.,2.,-2.]))
jax.block_until_ready(out)
np.testing.assert_allclose(out.data.ctrl[2:], [.0295,-.0295], rtol=1e-5)
np.testing.assert_allclose(out.obs['state'][20:24], [0.,0.,1.,-1.], atol=1e-6)
np.testing.assert_allclose(out.obs['privileged_state'][:26],out.obs['state'])
np.testing.assert_allclose(out.obs['privileged_state'][-6:-2],out.obs['state'][20:24])
bad = step(state,jp.array([0.,0.,float('nan'),0.]))
jax.block_until_ready(bad)
assert bool(bad.done)
np.testing.assert_allclose(bad.data.ctrl[2:], [0.,0.], atol=1e-9)
print('STUDENT_CPU_RUNTIME_OK')
'''
        env = {**os.environ, "JAX_PLATFORMS": "cpu", "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
        result = subprocess.run([sys.executable, "-c", code, source, str(ROOT / "experiments/torque200")],
                                env=env, text=True, capture_output=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("STUDENT_CPU_RUNTIME_OK", result.stdout)


if __name__ == "__main__":
    unittest.main()
