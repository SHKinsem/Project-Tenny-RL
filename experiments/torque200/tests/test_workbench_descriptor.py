"""Describe reported semantic contracts; do not infer model compatibility from shape."""
import json
from pathlib import Path
import tempfile
import unittest

from experiments.torque200.workbench_api import WorkbenchReader
from experiments.torque200.workbench_web import mode_catalog


class DescriptorTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run = self.root / "console/leg_force_bc/run-1"
        self.run.mkdir(parents=True)
        self.write("status.json", {"mode": "leg_force_bc", "phase": "completed"})
        self.reader = WorkbenchReader(self.root)

    def write(self, name, value):
        (self.run / name).write_text(json.dumps(value), encoding="utf-8")

    def test_missing_identity_units_and_task_are_not_invented(self):
        value = self.reader.descriptor("leg_force_bc")
        self.assertIsNone(value["robot"]["id"])
        self.assertIsNone(value["task"]["id"])
        self.assertIsNone(value["action"]["order"])
        self.assertIsNone(value["action"]["units"])
        self.assertEqual(value["timing"]["controlHz"], 100)
        self.assertEqual(value["observation"]["actorDim"], 24)
        self.assertIsNone(value["runtime"]["numEnvs"])

    def test_saved_semantics_and_ppo_are_read_without_fixed_action_count(self):
        self.write("policy_metadata.json", {"mode": "leg_force_bc", "robot_id": "fixture-six-actuator",
                   "task_id": "fixture-stand", "observation_version": "fixture-v1", "actor_dim": 32,
                   "action_order": [f"motor_{i}" for i in range(6)], "action_units": ["N.m"] * 6,
                   "action_mode": "fixture_torque", "control_hz": 100})
        self.write("env_config.json", {"ctrl_dt": 0.01, "episode_length": 600})
        self.write("ppo_config.json", {"num_envs": 512, "batch_size": 512, "num_minibatches": 1,
                   "unroll_length": 16, "num_updates_per_batch": 1, "learning_rate": 0.0003})
        value = self.reader.descriptor("leg_force_bc")
        self.assertEqual(value["robot"]["id"], "fixture-six-actuator")
        self.assertEqual(len(value["action"]["order"]), 6)
        self.assertEqual(value["runtime"]["numEnvs"], 512)
        self.assertEqual(value["timing"]["episodeSeconds"], 6)
        self.assertEqual(value["observation"]["actorDim"], 32)
        self.assertNotIn("canResume", value)  # Describing bytes never verifies weights.
        self.assertTrue(any(item["path"].endswith("policy_metadata.json") for item in value["source"]["files"]))

    def test_wrong_mode_semantics_are_ignored_and_catalog_comes_from_server(self):
        self.write("policy_metadata.json", {"mode": "original", "robot_id": "wrong", "control_hz": 100})
        value = self.reader.descriptor("leg_force_bc")
        self.assertIsNone(value["robot"]["id"])
        self.assertIn("descriptor_identity_mismatch", [warning["code"] for warning in value["warnings"]])
        catalog = mode_catalog("leg_force_bc")
        self.assertEqual(catalog["currentMode"], "leg_force_bc")
        force = next(mode for mode in catalog["modes"] if mode["id"] == "leg_force_bc")
        self.assertEqual(force["controlHz"], 100)
        self.assertTrue(force["teacher"])


if __name__ == "__main__":
    unittest.main()
