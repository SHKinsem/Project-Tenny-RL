"""Lightweight mode/record contracts; no JAX, renderer, server or training."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest

from learning_curves import load_learning_data
from web_preview import persist_mode, restart_arguments


class ModeTests(unittest.TestCase):
    def test_mode_persistence_and_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = Path(directory)
            persist_mode(artifacts, "original")
            self.assertEqual(json.loads((artifacts / "selected-mode.json").read_text()), {"mode": "original"})
            args = argparse.Namespace(artifacts=artifacts, original_root=Path("original tree"), host="127.0.0.1", port=8770)
            command = restart_arguments(args, "torque200")
            self.assertEqual(command[-2:], ["--mode", "torque200"])
            self.assertIn("original tree", command)
            with self.assertRaises(ValueError):
                persist_mode(artifacts, "arbitrary-command")
            with self.assertRaises(ValueError):
                restart_arguments(args, "arbitrary-command")

    def test_original_reads_only_verified_original_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "train-gpu-20261114-145047"
            checkpoint = run / "checkpoints" / "000000000016"
            checkpoint.mkdir(parents=True)
            def write(name, value):
                (run / name).write_text(json.dumps(value))
            write("report.json", {"checkpoint": str(checkpoint), "restore_max_abs_error": 0,
                                  "export_max_abs_error": 1e-5, "export_atol": 1e-3})
            write("env_config.json", {"ctrl_dt": .01, "episode_length": 600})
            write("ppo_config.json", {"num_eval_envs": 2})
            write("progress.json", [{"steps": 0, "metrics": {"eval/avg_episode_length": 3}},
                                     {"steps": 16, "metrics": {"eval/avg_episode_length": 5}}])
            (root / "training-live.json").write_text("invalid torque record")
            result = load_learning_data(root, mode="original")
            self.assertTrue(result["available"])
            self.assertEqual(result["points"][-1]["episode_seconds"], .05)
            self.assertIn("与作者预训练展示模型不同", result["metadata"]["source"])
            self.assertFalse(load_learning_data(root, mode="torque200")["available"])
            write("env_config.json", {"ctrl_dt": .005, "episode_length": 1200})
            self.assertFalse(load_learning_data(root, mode="original")["available"])
            write("env_config.json", {"ctrl_dt": .01, "episode_length": 600})
            write("report.json", {"checkpoint": str(checkpoint), "restore_max_abs_error": .1,
                                  "export_max_abs_error": 1e-5, "export_atol": 1e-3})
            self.assertFalse(load_learning_data(root, mode="original")["available"])

    def test_empty_records_and_invalid_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertFalse(load_learning_data(root, mode="original")["available"])
            with self.assertRaises(ValueError):
                load_learning_data(root, mode="unexpected")


if __name__ == "__main__":
    unittest.main()
