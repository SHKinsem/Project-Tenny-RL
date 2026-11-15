"""Bounded CPU-native oracle contract check, available in the existing WSL runtime."""
import importlib.util
import math
import os
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/torque200"))
from force_teacher.teacher import NativeStudent, ReachableTeacher, _observations, collect_dataset, evaluate_student
from hybrid.core import map_action, TORQUE_LIMITS

ROOT = Path(os.environ.get("TENNY_ORIGINAL_ROOT", str(Path.home() / "projects/tenny-reference")))
AVAILABLE = (ROOT / "mujoco_legwheel_rl_public").is_dir() and importlib.util.find_spec("mujoco") is not None


@unittest.skipUnless(AVAILABLE, "requires existing native MuJoCo/reference runtime")
class ReachableOraclePhysicsTests(unittest.TestCase):
    def test_three_nominal_six_second_episodes_and_held_action_mapping(self):
        student, oracle = NativeStudent(ROOT), ReachableTeacher(ROOT)
        for episode in range(3):
            command = np.zeros(4)
            student.reset(command)
            pitch = (0., .01, -.01)[episode]
            qpos = student.state.data.qpos.copy()
            qpos[3:7] = [math.cos(pitch/2), 0., math.sin(pitch/2), 0.]
            student.reset(command, qpos, student.state.data.qvel.copy())
            oracle.reset(command, student)
            for _ in range(600):
                label = oracle.action(student)
                self.assertLessEqual(np.max(np.abs(label["action"])), 1.)
                result = student.step(label["action"])
                oracle.executed(label["action"])
                self.assertFalse(result["done"], result["termination"])
                expected = np.asarray([map_action(label["action"], j, student.scales, "virtual_leg_force")
                                       for j in result["jacobians"]])
                np.testing.assert_allclose(result["sent_torques"], expected, atol=1e-8)
                np.testing.assert_allclose(result["actual_torques"], expected, atol=1e-8)
                obs, priv = _observations(student.state, student.previous)
                np.testing.assert_allclose(obs[-4:], expected.mean(axis=0) / TORQUE_LIMITS, atol=1e-7)
                self.assertEqual((obs.shape, priv.shape), ((24,), (51,)))

    def test_isolated_dagger_labels_differ_from_executed_student_actions(self):
        data = collect_dataset(ROOT, episodes=2, steps_per_episode=15, seed=42,
                               policy_callable=lambda obs: np.zeros(4), student_probability=1.)
        self.assertEqual(data["report"]["mixed_student_cycles"], len(data["actions"]))
        self.assertGreater(np.max(np.abs(data["actions"])), 0.)
        np.testing.assert_array_equal(data["observations"][:, -4:], 0.)
        self.assertEqual(data["report"]["retention_fraction"], 1.)

    def test_standing_evaluation_records_distinct_starts_and_termination(self):
        report = evaluate_student(ROOT, lambda obs: np.zeros(4), episodes=3, steps_per_episode=60)
        self.assertEqual(report["evaluation_profile"], "standing_gate")
        starts = []
        for episode in report["episodes"]:
            np.testing.assert_array_equal(episode["command"], np.zeros(4))
            self.assertIn("reasons", episode["termination"])
            starts.append(episode["initial_qpos"][5])
        self.assertEqual(len(set(starts)), 3)
        # A short diagnostic must never satisfy the required three 6-second gate.
        self.assertFalse(report["gate_passed"])


if __name__ == "__main__":
    unittest.main()
