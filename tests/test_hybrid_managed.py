"""Host-only checks for hybrid console identity, readiness and cooperative lifecycle."""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/torque200"))
from console_train import RunMonitor, StopRequested, parse_args
from control_modes import mode_spec
from hybrid.managed_training import prepare, verified_report, environment_pair
from hybrid.core import metadata, read_geometry, scan_geometry, digest_file, digest_tree
from hybrid.readiness import prepare_environment, runtime_identity
from preview_sync import SnapshotPublisher, read_snapshot
from training_control import TrainingController
import numpy as np


MODE = "hybrid_virtual_leg_force_100"


class ManagedTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.now = [0.0]
        self.args = argparse.Namespace(mode=MODE, run_dir=self.root, duration_seconds=30,
                                       restore=None, seed=0, force_revalidate=False)
        self.monitor = RunMonitor(self.args, clock=lambda: self.now[0])

    def ready(self, *args, **kw):
        self.now[0] = 600
        kw["on_status"]({"stage": "readiness_ready", "cache_hit": True, "message": "ready"})
        return dict(validation_path="validation.json", cache_hit=True,
                    evidence_dir="evidence", cache_key="verified-key")

    def test_preflight_does_not_consume_training_duration(self):
        self.now[0] = 300
        self.monitor.check_stop()
        self.assertEqual(self.monitor.remaining, 30)
        result = prepare(self.args, self.monitor, ensure=self.ready)
        self.assertTrue(result["cache_hit"])
        self.assertEqual(self.monitor.state["preparation_seconds"], 600)
        self.assertEqual(self.monitor.elapsed, 0)
        self.assertTrue(self.monitor.state["training_clock_started"])
        self.now[0] = 630
        with self.assertRaisesRegex(StopRequested, "wall_budget"):
            self.monitor.check_stop()

    def test_stop_before_or_during_preflight_never_starts_clock(self):
        ensure = Mock(side_effect=self.ready)
        (self.root / "stop.request").touch()
        with self.assertRaises(StopRequested):
            prepare(self.args, self.monitor, ensure=ensure)
        ensure.assert_not_called()
        (self.root / "stop.request").unlink()

        def stop(*args, **kw):
            (self.root / "stop.request").touch()
            raise RuntimeError("cancelled child")

        with self.assertRaises(StopRequested):
            prepare(self.args, self.monitor, ensure=stop)
        self.assertFalse(self.monitor.training_clock_started)

    def test_failed_gate_propagates_without_training(self):
        with self.assertRaisesRegex(ValueError, "invalid gate"):
            prepare(self.args, self.monitor, ensure=Mock(side_effect=ValueError("invalid gate")))
        self.assertFalse(self.monitor.training_clock_started)

    def test_parser_and_readiness_keep_mode_frequency_and_force(self):
        parsed = parse_args(["--mode", MODE, "--run-dir", str(self.root), "--force-revalidate"])
        self.assertTrue(parsed.force_revalidate)
        self.args.force_revalidate = True
        ensure = Mock(side_effect=self.ready)
        prepare(self.args, self.monitor, ensure=ensure)
        self.assertEqual(ensure.call_args.args[1:], ("virtual_leg_force", 100))
        self.assertTrue(ensure.call_args.kwargs["force"])

    def test_bad_mode_is_not_a_dimension_based_fallback(self):
        with self.assertRaises(ValueError):
            mode_spec("hybrid_virtual_leg_force_150")
        with self.assertRaises(ValueError):
            mode_spec("hybrid_unknown_100")


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name).resolve()
        self.controller = TrainingController(self.base / "artifacts", self.base, self.base)
        self.run = self.controller.root / MODE / "run"
        self.run.mkdir(parents=True)
        self.checkpoint = self.run / "checkpoints" / "000000000016"
        self.checkpoint.mkdir(parents=True)
        (self.checkpoint / "params").write_bytes(b"verified training parameters")
        self.export = self.run / "policy.cpp.txt"
        self.export.write_bytes(b"verified export")
        self.contract = metadata("virtual_leg_force", scan_geometry(read_geometry()), 100)
        self.contract["export_sha256"] = digest_file(self.export)
        self.report = dict(mode=MODE, phase="completed", checkpoint_validated=True,
            action_mode="virtual_leg_force", control_hz=100,
            source_fingerprint=self.contract["source_fingerprint"],
            checkpoint=str(self.checkpoint), checkpoint_name=self.checkpoint.name,
            checkpoint_sha256=digest_tree(self.checkpoint), export_path=str(self.export),
            export_sha256=self.contract["export_sha256"],
            all_parameters_finite=True, parameter_tree_max_abs_error=0,
            export_max_abs_error=1e-5, export_atol=1e-3, actual_timesteps=16)
        (self.run / "policy_metadata.json").write_text(json.dumps(self.contract))
        (self.run / "report.json").write_text(json.dumps(self.report))
        (self.run / "status.json").write_text(json.dumps({"phase": "completed", "mode": MODE}))

    def test_validated_model_can_be_exposed_with_identity(self):
        actual = self.controller._validated(self.run, MODE)
        self.assertTrue(actual["verified"])
        self.assertEqual(actual["actual_timesteps"], 16)
        self.assertEqual(actual["control_hz"], 100)

    def test_tampered_checkpoint_export_and_source_rejected(self):
        with patch("hybrid.core.source_fingerprint", return_value="changed"):
            self.assertIsNone(self.controller._validated(self.run, MODE))
        original = self.export.read_bytes()
        self.export.write_bytes(b"tampered")
        self.assertIsNone(self.controller._validated(self.run, MODE))
        self.export.write_bytes(original)
        (self.checkpoint / "params").write_bytes(b"tampered")
        self.assertIsNone(self.controller._validated(self.run, MODE))

    def test_wrong_mode_frequency_and_false_report_rejected(self):
        for mode in ("hybrid_virtual_leg_force_200", "hybrid_direct_torque_100"):
            with self.assertRaises(ValueError):
                verified_report(self.run, mode, self.report)
        with self.assertRaises(ValueError):
            verified_report(self.run, MODE, {**self.report, "checkpoint_validated": False})


class PreviewTests(unittest.TestCase):
    def test_preview_binds_compound_mode_and_source(self):
        rng = np.random.default_rng(7)
        layers = {f"hidden_{i}": {"kernel": rng.normal(0, .1, (a, b)), "bias": np.zeros(b)}
                  for i, (a, b) in enumerate(zip((24, 64, 64), (64, 64, 8)))}
        params = [SimpleNamespace(mean={"state": np.zeros(24)}, std={"state": np.ones(24)}),
                  {"params": layers}]
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "preview.request.json").write_text(json.dumps({"mode": MODE, "request_id": "open"}))
            publisher = SnapshotPublisher(run, MODE)
            result = publisher.service(8192, params)
            self.assertNotIn("error", result)
            policy, meta = read_snapshot(run / "preview-policy.npz", MODE, "open")
            self.assertEqual(meta["control_hz"], 100)
            self.assertEqual(policy.run(np.zeros(24)).shape, (4,))
            with self.assertRaises(ValueError):
                read_snapshot(run / "preview-policy.npz", "hybrid_virtual_leg_force_200", "open")
            with patch("hybrid.core.source_fingerprint", return_value="changed"):
                with self.assertRaises(ValueError):
                    read_snapshot(run / "preview-policy.npz", MODE, "open")


class EnvironmentTests(unittest.TestCase):
    def test_training_and_evaluation_use_same_hybrid_action_and_frequency(self):
        class Legacy:
            def __init__(self, config):
                self.cfg = config

        class Hybrid:
            def __init__(self, action, config):
                self.action, self.cfg = action, config

        for action in ("virtual_leg_force", "matched_envelope", "direct_torque"):
            for hz in (100, 200):
                with self.subTest(action=action, hz=hz):
                    mode = f"hybrid_{action}_{hz}"
                    cfg = SimpleNamespace(control_hz=hz)
                    environments = environment_pair(mode, cfg, Legacy, Hybrid,
                                                    lambda rate: SimpleNamespace(control_hz=rate))
                    train = Mock(return_value="trained")
                    train(**environments, env_cfg=cfg)
                    supplied = train.call_args.kwargs
                    for key in ("env", "eval_env"):
                        self.assertIsInstance(supplied[key], Hybrid)
                        self.assertEqual(supplied[key].action, action)
                        self.assertEqual(supplied[key].cfg.control_hz, hz)
                    self.assertIsNot(supplied["env"], supplied["eval_env"])
                    self.assertIsNot(supplied["env"].cfg, supplied["eval_env"].cfg)
        original = environment_pair("original", SimpleNamespace(control_hz=100), Legacy)
        self.assertIsInstance(original["env"], Legacy)
        self.assertIsNone(original["eval_env"])
        with self.assertRaises(ValueError):
            environment_pair(MODE, SimpleNamespace(control_hz=100), Legacy)

    def test_controller_preserves_explicit_gpu_settings_and_matches_cli_identity(self):
        for overrides in ({}, {"CUDA_VISIBLE_DEVICES": "2", "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.22"}):
            with self.subTest(overrides=overrides), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                controller = TrainingController(root / "artifacts", root, root)
                source = {k: v for k, v in os.environ.items()
                          if not k.startswith(("JAX_", "XLA_", "CUDA_"))}
                source.update(overrides)
                source["LD_LIBRARY_PATH"] = "/preview-only"
                # Simulate the CPU preview parent; CLI's matching training backend is CUDA.
                source["JAX_PLATFORMS"] = "cpu"
                with patch.dict(os.environ, source, clear=True), \
                     patch.object(controller, "_locked", return_value=nullcontext()), \
                     patch.object(controller, "_current", return_value=({}, {}, False)), \
                     patch.object(controller, "_process", return_value=None), \
                     patch.object(controller, "_status", return_value={}), \
                     patch("training_control.subprocess.Popen") as spawn:
                    spawn.return_value.pid = 42
                    controller.start(MODE)
                    managed_env = spawn.call_args.kwargs["env"]
                    from hybrid.managed_training import runtime_root
                    cli_env = prepare_environment(runtime_root(), {**source, "JAX_PLATFORMS": "cuda"})
                self.assertEqual(managed_env, cli_env)
                self.assertEqual(managed_env["CUDA_VISIBLE_DEVICES"], overrides.get("CUDA_VISIBLE_DEVICES", "0"))
                self.assertEqual(managed_env["XLA_PYTHON_CLIENT_MEM_FRACTION"], overrides.get("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.35"))
                # Identity itself, including real installed-package metadata, must match.
                gpu = SimpleNamespace(stdout="gpu-uuid, model, driver, capability\n")
                driver_check = (nullcontext() if os.environ.get("TENNY_VERIFY_GPU_IDENTITY") == "1"
                                else patch("hybrid.readiness.subprocess.run", return_value=gpu))
                with driver_check:
                    with patch.dict(os.environ, cli_env, clear=True):
                        cli_identity = runtime_identity()
                    with patch.dict(os.environ, managed_env, clear=True):
                        managed_identity = runtime_identity()
                self.assertEqual(cli_identity, managed_identity)


if __name__ == "__main__":
    unittest.main()
