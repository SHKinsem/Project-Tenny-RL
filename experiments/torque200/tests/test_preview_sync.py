"""Protocol and numerical safety checks for on-open preview synchronization."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preview_sync import SnapshotPolicy, SnapshotPublisher, read_snapshot


def parameters(mode):
    n = 26 if mode == "original" else 24
    rng = np.random.default_rng(17)
    layers = {f"hidden_{i}": {"kernel": rng.normal(0, .1, (a,b)).astype(np.float32),
                              "bias": rng.normal(0,.1,b).astype(np.float32)}
              for i, (a,b) in enumerate(zip((n,64,64),(64,64,8)))}
    return [SimpleNamespace(mean={"state": np.zeros(n)}, std={"state": np.ones(n)}),
            {"params": layers}, {"critic": "must not be published"}]


class PreviewSyncTests(unittest.TestCase):
    def test_on_demand_once_and_reopen(self):
        for mode in ("original", "torque200"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                run = Path(directory)
                params = parameters(mode)
                publisher = SnapshotPublisher(run, mode)
                self.assertIsNone(publisher.service(10, params))
                self.assertFalse((run / "preview-policy.npz").exists())
                request = run / "preview.request.json"
                request.write_text(json.dumps({"request_id":"first", "mode":mode}))
                first = publisher.service(20, params)
                self.assertNotIn("error", first)
                self.assertLess(first["bytes"], 40000)
                policy, metadata = read_snapshot(run / "preview-policy.npz", mode, "first")
                self.assertEqual(metadata["steps"], 20)
                obs = np.linspace(-1,1,len(params[0].mean["state"])).astype(np.float32)
                np.testing.assert_array_equal(policy.run(obs), SnapshotPolicy.from_params(params, mode).run(obs))
                original_bytes = (run / "preview-policy.npz").read_bytes()
                self.assertIsNone(publisher.service(30, params))
                self.assertEqual(original_bytes, (run / "preview-policy.npz").read_bytes())
                request.unlink()
                self.assertIsNone(publisher.service(40, params))
                request.write_text(json.dumps({"request_id":"second", "mode":mode}))
                self.assertIsNone(read_snapshot(run / "preview-policy.npz", mode, "second"))
                publisher.service(50, params)
                self.assertEqual(read_snapshot(run / "preview-policy.npz", mode, "second")[1]["steps"], 50)
                self.assertEqual(publisher.count, 2)

    def test_wrong_mode_and_bad_parameters_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            params = parameters("original")
            (run / "preview.request.json").write_text(json.dumps({"request_id":"one", "mode":"original"}))
            publisher = SnapshotPublisher(run, "original")
            publisher.service(10, params)
            with self.assertRaises(ValueError):
                read_snapshot(run / "preview-policy.npz", "torque200", "one")
            params[0].std["state"][0] = 0
            (run / "preview.request.json").write_text(json.dumps({"request_id":"two", "mode":"original"}))
            self.assertIn("error", publisher.service(20, params))
            self.assertIsNone(read_snapshot(run / "preview-policy.npz", "original", "two"))


if __name__ == "__main__":
    unittest.main()
