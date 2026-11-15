"""Verify BC weights and student evaluation reach PPO without running ML."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/torque200"))
from wheel_teacher.ppo import train_student


class StudentPpoTests(unittest.TestCase):
    def setUp(self):
        self.seen = {}

        def train(*, environment, eval_env, wrap_env_fn, network_factory,
                  randomization_fn, progress_fn, policy_params_fn=None,
                  restore_params=None, restore_checkpoint_path=None, num_envs=128):
            self.seen.update(locals())
            return "trained"

        self.modules = {
            "brax.training.agents.ppo": SimpleNamespace(
                networks=SimpleNamespace(make_ppo_networks=lambda **kw: kw),
                train=SimpleNamespace(train=train)),
            "mujoco_playground": SimpleNamespace(wrapper=SimpleNamespace(wrap_for_brax_training=object())),
            "legwheel.randomization": SimpleNamespace(domain_randomize=object()),
        }
        self.options = dict(network_factory={}, num_envs=128)

    def call(self, **kw):
        with patch.dict(sys.modules, self.modules):
            return train_student(env="student", eval_env="student_eval", env_cfg=None,
                                 ppo_params=self.options, progress_fn=None, **kw)

    def test_bc_bundle_is_restored_and_evaluation_uses_student(self):
        bundle = (object(), object(), object())
        self.assertEqual(self.call(params=bundle), "trained")
        self.assertIs(self.seen["restore_params"], bundle)
        self.assertEqual(self.seen["environment"], "student")
        self.assertEqual(self.seen["eval_env"], "student_eval")

    def test_resume_does_not_reinitialize_from_teacher(self):
        self.options["restore_checkpoint_path"] = "verified-student-checkpoint"
        self.call()
        self.assertIsNone(self.seen["restore_params"])
        self.assertEqual(self.seen["restore_checkpoint_path"], "verified-student-checkpoint")
        with self.assertRaisesRegex(ValueError, "not both"):
            self.call(params=(1, 2, 3))

    def test_unknown_training_option_is_not_silently_dropped(self):
        self.options["wrong_option"] = True
        with self.assertRaisesRegex(ValueError, "wrong_option"):
            self.call()


if __name__ == "__main__":
    unittest.main()
