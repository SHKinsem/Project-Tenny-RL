"""Check live telemetry without importing the simulator or starting training."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from learning_curves import load_learning_data


class LiveCurves(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run = self.root / "training-test"
        self.run.mkdir()
        self.write(self.run / "env_config.json", {"ctrl_dt": .005, "episode_length": 1200})
        self.write(self.run / "ppo_config.json", {"num_eval_envs": 2})
        self.write(self.root / "training-live.json", {
            "run_dir": str(self.run), "progress_file": str(self.run / "progress.json"), "phase": "training"})

    def write(self, path, value):
        path.write_text(json.dumps(value))

    def test_sparse_evaluation_is_not_fabricated(self):
        self.write(self.run / "progress.json", [
            {"steps": 0, "metrics": {"eval/avg_episode_length": 40, "eval/episode_reward": -10}},
            {"steps": 8192, "metrics": {"training/v_loss": .5}},
            {"steps": 16384, "metrics": {"eval/avg_episode_length": 80, "eval/episode_reward": 5}}])
        data = load_learning_data(self.root)
        self.assertTrue(data["available"])
        self.assertEqual(data["metadata"]["eval_point_count"], 2)
        self.assertIsNone(data["points"][1]["episode_seconds"])
        self.assertIsNone(data["points"][0]["value_loss"])
        self.assertEqual(data["points"][-1]["episode_seconds"], .4)

    def test_empty_run_does_not_show_older_training(self):
        self.write(self.run / "progress.json", [])
        self.assertFalse(load_learning_data(self.root)["available"])

    def test_duplicate_steps_rejected(self):
        self.write(self.run / "progress.json", [{"steps": 1, "metrics": {}}] * 2)
        self.assertFalse(load_learning_data(self.root)["available"])


if __name__ == "__main__":
    unittest.main()
