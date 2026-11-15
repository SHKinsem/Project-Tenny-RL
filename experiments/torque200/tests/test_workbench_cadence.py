"""Sparse evaluation is distinct from explicitly invalid measurements."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from learning_curves import load_console_learning_data


class MetricCadenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.run = Path(temporary.name)
        self.write("env_config.json", {"ctrl_dt": 0.01, "episode_length": 600})
        self.write("ppo_config.json", {"num_eval_envs": 8})

    def write(self, name, value):
        (self.run / name).write_text(json.dumps(value), encoding="utf8")

    def test_absent_metric_is_not_a_sample_or_a_zero(self):
        self.write("progress.json", [
            {"steps": 0, "metrics": {"eval/avg_episode_length": 100, "eval/episode_reward": 5}},
            {"steps": 10, "metrics": {"training/v_loss": 0.2, "training/policy_loss": -1e-9}},
            {"steps": 20, "metrics": {"eval/avg_episode_length": 200, "eval/episode_reward": 7}},
            {"steps": 30, "metrics": {}},
        ])
        value = load_console_learning_data(self.run, "original")
        self.assertTrue(value["available"])
        points = value["points"]
        self.assertEqual(points[0]["measured"], ["episode_seconds", "episode_reward"])
        self.assertEqual(points[1]["measured"], ["value_loss", "policy_loss"])
        self.assertEqual(points[-1]["measured"], [])
        self.assertIsNone(points[1]["episode_seconds"])
        self.assertEqual(points[1]["policy_loss"], -1e-9)
        self.assertEqual(value["metadata"]["eval_point_count"], 2)
        self.assertIn("每次8个评估环境", " ".join(value["metadata"]["limitations"]))

    def test_explicit_null_or_invalid_is_still_a_measurement_gap(self):
        self.write("progress.json", [
            {"steps": 0, "metrics": {"eval/avg_episode_length": None, "training/v_loss": True}},
            {"steps": 10, "metrics": {"training/v_loss": 0}},
        ])
        value = load_console_learning_data(self.run, "original")
        point = value["points"][0]
        self.assertEqual(point["measured"], ["episode_seconds", "value_loss"])
        self.assertIsNone(point["episode_seconds"])
        self.assertIsNone(point["value_loss"])
        self.assertEqual(value["points"][1]["value_loss"], 0)


if __name__ == "__main__":
    unittest.main()
