"""Check fixed PPO work and untraced performance accounting without a GPU."""
import copy
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import launch_scaling
from launch_scaling import matrix_counts
from run_scaling import case_config, timing_summary


class ScalingContractTests(unittest.TestCase):
    def setUp(self):
        self.original = {"num_envs": 128, "batch_size": 128, "num_minibatches": 1,
            "num_updates_per_batch": 1, "unroll_length": 16, "action_repeat": 1,
            "num_timesteps": 10**9, "num_evals": 100,
            "learning_rate": 3e-5, "discounting": .98, "normalize_until_count": 0,
            "network_factory": {"policy_hidden_layer_sizes": [64, 64]},
            "restore_checkpoint_path": "verified-input", "save_checkpoint_path": "unused"}

    def test_every_concurrency_has_same_samples_and_update_work(self):
        cases = []
        for count in (128, 256, 512):
            config, steps, epochs = case_config(self.original, num_envs=count, batch_size=512)
            self.assertEqual(steps, 8192)
            self.assertEqual(epochs, 12)
            self.assertEqual(config["batch_size"] // count * count * config["unroll_length"], steps)
            self.assertEqual(math.ceil(config["num_timesteps"] / ((config["num_evals"]-1)*steps)), 1)
            self.assertEqual(config["num_minibatches"] * config["num_updates_per_batch"], 1)
            cases.append({key: value for key, value in config.items() if key != "num_envs"})
        self.assertEqual(cases[0], cases[1])
        self.assertEqual(cases[1], cases[2])

    def test_original_configuration_and_learning_options_are_preserved(self):
        before = copy.deepcopy(self.original)
        config, _, _ = case_config(self.original, num_envs=256, batch_size=512)
        self.assertEqual(self.original, before)
        allowed = {"num_envs", "batch_size", "num_timesteps", "num_evals",
                   "run_evals", "save_checkpoint_path", "restore_checkpoint_path"}
        self.assertTrue(all(config[key] == value for key, value in before.items() if key not in allowed))
        self.assertIsNone(config["save_checkpoint_path"])
        self.assertIsNone(config["restore_checkpoint_path"])
        self.assertFalse(config["run_evals"])

    def test_invalid_batch_or_changed_update_semantics_are_rejected(self):
        with self.assertRaises(ValueError):
            case_config(self.original, num_envs=256, batch_size=128)
        for key in ("num_updates_per_batch", "num_minibatches", "num_resets_per_eval"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                case_config(dict(self.original, **{key: 2}), num_envs=128, batch_size=512)

    def test_performance_excludes_warmup_trace_and_recovery(self):
        rows = [{"kind": kind, "complete_seconds": seconds, "env_steps": 8192}
                for kind, seconds in (("warmup", 20), ("measure", 1), ("measure", 2),
                                      ("trace", 30), ("recovery", 4))]
        result = timing_summary(rows)
        self.assertEqual(result["samples"], 2)
        self.assertEqual(result["seconds"], 3)
        self.assertAlmostEqual(result["steps_per_second"], 16384/3)

    def test_invalid_times_are_not_published(self):
        for value in (0, -1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                timing_summary([{"kind": "measure", "complete_seconds": value, "env_steps": 8192}])

    def test_matrix_validation_preserves_order_and_rejects_duplicate_work(self):
        self.assertEqual(matrix_counts([512, 128, 256], 512), [512, 128, 256])
        for counts in ([], [128, 128], [128, 300], [0, 128]):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                matrix_counts(counts, 512)


@unittest.skipUnless(os.name == "posix", "The launcher uses Linux process groups")
class SpawnCleanupTests(unittest.TestCase):
    def test_receipt_failure_immediately_after_spawn_cleans_up_child(self):
        """No GPU/runtime imports: exercise the actual launcher exception path."""
        children = []
        real_popen = subprocess.Popen
        real_write = launch_scaling.write_json
        modules = {name: types.ModuleType(name) for name in
                   ("hybrid", "hybrid.managed_training", "hybrid.readiness", "hybrid.core",
                    "force_teacher", "force_teacher.contract")}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            run.mkdir()
            export = root / "export.npz"
            export.write_bytes(b"verified-model")
            (run / "report.json").write_text(json.dumps({"run_id": "input",
                "checkpoint": str(run / "checkpoint"), "checkpoint_sha256": "checkpoint",
                "source_fingerprint": "source", "export_path": str(export)}), encoding="utf8")
            modules["hybrid.managed_training"].runtime_root = lambda: root
            modules["hybrid.readiness"].prepare_environment = lambda base, env: env
            modules["hybrid.core"].source_fingerprint = lambda: "source"
            modules["force_teacher.contract"].tree_digest = lambda value: "checkpoint"
            state = json.dumps({"training": {"active": False, "run_id": "input"}}).encode()

            def spawn(*args, **kwargs):
                child = real_popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, start_new_session=True)
                children.append(child)
                self.addCleanup(launch_scaling.stop_process_group, child, .1, .1)
                return child

            def fail_receipt(path, value):
                if children:
                    raise OSError("simulated write failure immediately after spawn")
                return real_write(path, value)

            with patch.dict(sys.modules, modules), patch.object(sys, "path", list(sys.path)), \
                 patch.object(launch_scaling, "urlopen", side_effect=lambda *a, **kw: io.BytesIO(state)), \
                 patch.object(launch_scaling.subprocess, "Popen", side_effect=spawn), \
                 patch.object(launch_scaling, "write_json", side_effect=fail_receipt):
                with self.assertRaisesRegex(OSError, "simulated write failure"):
                    launch_scaling.main(["--torque-root", str(root), "--run-dir", str(run),
                                         "--output", str(root / "output"), "--envs", "128"])
            self.assertEqual(len(children), 1)
            self.assertIsNotNone(children[0].poll())


if __name__ == "__main__":
    unittest.main()
