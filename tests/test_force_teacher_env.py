"""Force labels must preserve measured signs/limits and explicit approximation error."""
import json
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/torque200"))
from force_teacher.teacher import force_label, projected_action, _observations, collect_dataset
from force_teacher import contract
from force_teacher.options import normalize_options, legacy_options
from hybrid.core import XML, map_action


_make_metadata = contract.metadata


def metadata_fixture(original_root=None, experiment_config=None):
    """Exercise the real metadata builder without external teacher artifacts."""
    with patch.object(contract, "hybrid_metadata", return_value=dict(source_fingerprint="hybrid", control_hz=100)), \
         patch.object(contract, "read_geometry", return_value=()), \
         patch.object(contract, "scan_geometry", return_value=()), \
         patch.object(contract, "source_fingerprint", return_value="force"), \
         patch.object(contract, "teacher_policy_path", return_value=Path("teacher.cpp")), \
         patch.object(contract, "file_digest", return_value="teacher-bytes"):
        return _make_metadata(original_root, experiment_config)


class StandingConfigTests(unittest.TestCase):
    def test_standing_defaults_preserve_rewards_commands_and_timing(self):
        # Host check of the real factory against a small config object; this
        # deliberately does not pretend to execute the simulator or JAX sampler.
        rewards = dict(pitch_scale=0., roll_scale=4., action_rate_scale=-.5)
        enables = dict(lin_vel_x=True, ang_vel_yaw=True, height_value=True, roll_value=True)
        cfg = SimpleNamespace(control_hz=100, ctrl_dt=.01, sim_dt=.0025,
            cmd_config=SimpleNamespace(cmd_prob=.9, enable=enables.copy()),
            noise_config=SimpleNamespace(level=1.), push_config=SimpleNamespace(enable=True),
            reward_config=rewards.copy())
        fake = ModuleType("hybrid.env")
        fake.HybridLegWheelRobot = object
        with patch.dict(sys.modules, {"hybrid.env": fake}):
            with patch.object(fake, "hybrid_config", create=True, return_value=cfg) as factory:
                path = Path(__file__).resolve().parents[1] / "experiments/torque200/force_teacher/env.py"
                spec = importlib.util.spec_from_file_location("_force_env_config_test", path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                result = module.student_config()
        factory.assert_called_once_with(100)
        self.assertEqual(result.cmd_config.cmd_prob, 0.)
        self.assertEqual(result.noise_config.level, 0.)
        self.assertFalse(result.push_config.enable)
        self.assertEqual(result.cmd_config.enable, enables)
        self.assertEqual(result.reward_config, rewards)
        self.assertEqual((result.control_hz, result.ctrl_dt, result.sim_dt), (100, .01, .0025))

    @unittest.skipUnless(os.environ.get("TENNY_TEST_FORCE_CONFIG") == "1",
                         "Explicit runtime dependency check; no dynamics")
    def test_real_command_sampler_is_stationary_for_all_reset_seeds(self):
        import inspect
        import jax
        from force_teacher.env import StudentEnv, student_config
        from legwheel.env import LegWheelRobot
        cfg = student_config()
        env = SimpleNamespace(_config=cfg)
        keys = jax.random.split(jax.random.PRNGKey(71), 128)
        commands = jax.vmap(lambda key: LegWheelRobot.sample_command(env, key))(keys)
        np.testing.assert_array_equal(np.asarray(commands), np.zeros((128, 4)))
        # This is the reset API used by the training wrapper. NativeStudent.reset
        # likewise uses the nominal qpos/qvel unless an explicit state is passed.
        self.assertIs(inspect.signature(StudentEnv.reset).parameters["random"].default, False)
        self.assertEqual((cfg.control_hz, cfg.ctrl_dt, cfg.sim_dt), (100, .01, .0025))
        self.assertEqual(tuple(cfg.torque_control.scales), (1., 1., .0295, .0295))


class ForceLabelTests(unittest.TestCase):
    def test_projected_oracle_is_explicitly_lossy_and_reachable(self):
        label = projected_action([-16., 14., .04, -.04], [-.1, -.09], [8., 8.])
        np.testing.assert_allclose(label["action"], [1., -1., 1., -1.])
        np.testing.assert_array_equal(label["projected"], [True] * 4)
        np.testing.assert_allclose(label["projected_torque_nm"], [-.8, .72, .0295, -.0295])
        self.assertGreater(np.max(np.abs(label["projection_error_nm"])), 13.)

    def test_held_force_updates_only_torque_mapping_with_substep_geometry(self):
        label = projected_action([-.4, .2, .01, -.02], [-.1, -.1], [8., 8.])
        sent = np.asarray([map_action(label["action"], j, [8., 8.], "virtual_leg_force")
                           for j in ([-.1, -.1], [-.09, -.08], [-.08, -.09], [-.1, -.1])])
        np.testing.assert_allclose(sent[:, :2], [[-.4,.2],[-.36,.16],[-.32,.18],[-.4,.2]])
        np.testing.assert_allclose(sent[:, 2:], np.tile([.01,-.02], (4,1)), atol=1e-8)

    def test_dagger_rejects_unavailable_or_invalid_student(self):
        for probability, policy in ((.5, None), (-.1, lambda o:o), (1.1, lambda o:o)):
            with self.assertRaises(ValueError):
                collect_dataset("unused", student_probability=probability, policy_callable=policy)

    def test_actual_negative_hip_torque_maps_to_positive_extending_force(self):
        jac = np.full((4, 2), -.1)
        actual = np.tile([-.4, .2, .01, -.02], (4, 1))
        label = force_label(actual, jac, [8., 8.])
        self.assertTrue(label["reachable"])
        np.testing.assert_allclose(label["action"], [.5, -.25, .01/.0295, -.02/.0295])
        np.testing.assert_allclose(label["instantaneous_force_n"], np.tile([4., -2.], (4, 1)))
        np.testing.assert_allclose(label["reconstructed_torque_nm"], actual)

    def test_constant_force_is_weighted_fit_with_real_substep_error(self):
        jac = np.array([[-.1, -.09], [-.08, -.1], [-.09, -.08], [-.1, -.1]])
        torques = np.column_stack((jac * [8., 8.] * [.4, -.3], np.zeros((4, 2))))
        torques[:, 0] += [.02, -.02, .01, -.01]
        result = force_label(torques, jac, [8., 8.])
        gain = jac * [8., 8.]
        expected = np.sum(gain * torques[:, :2], axis=0) / np.sum(gain**2, axis=0)
        np.testing.assert_allclose(result["action"][:2], expected)
        self.assertGreater(np.max(np.abs(result["torque_error_nm"])), 0)
        np.testing.assert_allclose(result["reconstructed_torque_nm"][:, :2], gain * expected)

    def test_original_servo_overload_is_not_silently_clipped(self):
        result = force_label(np.tile([-16., 14., 0, 0], (4, 1)), np.full((4, 2), -.1), [8., 8.])
        self.assertFalse(result["reachable"])
        self.assertGreater(np.max(np.abs(result["action"])), 1.)
        np.testing.assert_allclose(result["action"][:2], [20., -17.5])

    def test_substep_infeasibility_is_not_hidden_by_feasible_mean(self):
        torques = np.zeros((4, 4))
        torques[:, 0] = [.9, -.9, .9, -.9]
        result = force_label(torques, np.full((4, 2), -.1), [8., 8.])
        self.assertAlmostEqual(float(result["action"][0]), 0.)
        self.assertFalse(result["reachable"])

    def test_invalid_jacobian_or_shape_fails(self):
        for jac in (np.zeros((4, 2)), np.full((4, 2), np.nan), np.ones((2, 2))):
            with self.assertRaises(ValueError):
                force_label(np.zeros((4, 4)), jac, [8., 8.])

    def test_observation_uses_real_hip_velocity_and_four_torque_history(self):
        class Teacher:
            cfg = SimpleNamespace(wheel_radius=.0195)
            q, v = [0, 1, 2, 3], [0, 1, 2, 3]
            data = SimpleNamespace(qpos=np.arange(4.), qvel=np.array([1.5, 2., -3., -4.]))
            def observation(self):
                obs = np.arange(26, dtype=np.float32)
                obs[14], obs[16] = 0, 0
                obs[20:] = 10000  # Teacher wheel target state must not leak.
                return obs
            def sensor(self, name):
                return np.zeros(3)
        obs, privileged = _observations(Teacher(), [-.4, .2, .01, -.01])
        self.assertEqual(obs.shape, (24,))
        self.assertEqual(privileged.shape, (51,))
        np.testing.assert_allclose(obs[[14, 16]], [1.5, -3.])
        np.testing.assert_allclose(obs[-4:], [-.4, .2, .01/.0295, -.01/.0295])
        np.testing.assert_array_equal(privileged[:24], obs)
        np.testing.assert_allclose(privileged[-4:], obs[-4:])


class ForceContractTests(unittest.TestCase):
    def test_v1_teacher_data_cannot_match_new_semantics(self):
        self.assertEqual(contract.SCHEMA, "leg-force-configurable-bc-v4")
        with self.assertRaises(ValueError):
            contract.validate_metadata(dict(mode=contract.MODE, teacher_schema="leg-force-teacher-bc-v1"))

    def test_v4_checks_complete_options_hash_head_and_resume_configuration(self):
        expected = metadata_fixture()
        with patch.object(contract, "metadata", side_effect=metadata_fixture):
            self.assertEqual(contract.validate_metadata(expected), normalize_options())
            invalid = [dict(expected, policy_head="wheel_logit_residual_v1"),
                       dict(expected, experiment_config_sha256="changed"),
                       dict(expected, unexpected=True),
                       dict(expected, experiment_config={"wheel_head": "absolute"})]
            for actual in invalid:
                with self.subTest(actual=actual), self.assertRaises(ValueError):
                    contract.validate_metadata(actual)
            with self.assertRaisesRegex(ValueError, "Resume configuration conflicts"):
                contract.validate_metadata(expected, experiment_config=dict(wheel_head="residual"))

    def test_v3_migration_requires_exact_source_pair_and_residual_semantics(self):
        legacy = metadata_fixture(experiment_config=legacy_options())
        legacy.pop("experiment_config")
        legacy.pop("experiment_config_sha256")
        source, force_source = next(iter(contract.LEGACY_SOURCE_PAIRS))
        legacy.update(teacher_schema=contract.LEGACY_SCHEMA, source_fingerprint=source,
                      force_source_fingerprint=force_source)
        with patch.object(contract, "metadata", side_effect=metadata_fixture):
            self.assertEqual(contract.validate_metadata(legacy), legacy_options())
            invalid = [dict(legacy, source_fingerprint="unknown"),
                       dict(legacy, force_source_fingerprint="unknown"),
                       dict(legacy, policy_head="absolute_tanh_v1"),
                       dict(legacy, wheel_residual_scale=.1),
                       dict(legacy, experiment_config=legacy_options())]
            for actual in invalid:
                with self.subTest(actual=actual), self.assertRaises(ValueError):
                    contract.validate_metadata(actual)
            with self.assertRaisesRegex(ValueError, "Resume configuration conflicts"):
                contract.validate_metadata(legacy, experiment_config=normalize_options())

    def test_actual_student_xml_has_four_motors_not_leg_servos(self):
        actuators = list(ET.parse(XML).getroot().find("actuator"))
        self.assertEqual(len(actuators), 4)
        self.assertTrue(all(a.tag == "motor" for a in actuators))

    def test_extra_force_identity_is_checked_even_with_same_hybrid_fingerprint(self):
        expected = dict(mode=contract.MODE, source_fingerprint="hybrid", force_source_fingerprint="teacher-new")
        with self.assertRaises(ValueError):
            contract.require_metadata({**expected, "force_source_fingerprint": "teacher-old"}, expected)
        with self.assertRaises(ValueError):
            contract.require_metadata({**expected, "mode": "hybrid_virtual_leg_force_100"}, expected)

    def test_saved_report_binds_contract_and_both_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoints" / "000016"
            checkpoint.mkdir(parents=True)
            (checkpoint / "weights").write_bytes(b"verified")
            export = root / "policy.cpp.txt"
            export.write_bytes(b"export")
            expected = metadata_fixture()
            report = dict(mode=contract.MODE, checkpoint_validated=True, source_fingerprint="hybrid",
                          checkpoint=str(checkpoint), export_path=str(export),
                          checkpoint_sha256=contract.tree_digest(checkpoint), export_sha256=contract.file_digest(export))
            (root / "policy_metadata.json").write_text(json.dumps({**expected, "export_sha256": report["export_sha256"]}))
            (root / "report.json").write_text(json.dumps(report))
            with patch.object(contract, "metadata", side_effect=metadata_fixture):
                self.assertEqual(contract.verified_report(root, report), normalize_options())
                self.assertEqual(contract.restore_guard(checkpoint), normalize_options())
                export.write_bytes(b"changed export")
                with self.assertRaises(ValueError):
                    contract.verified_report(root, report)
                export.write_bytes(b"export")
                (checkpoint / "weights").write_bytes(b"changed")
                with self.assertRaises(ValueError):
                    contract.verified_report(root, report)


if __name__ == "__main__":
    unittest.main()
