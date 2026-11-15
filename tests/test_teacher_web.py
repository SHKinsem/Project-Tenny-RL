"""Teacher-student web identity and source isolation without ML imports."""
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/torque200"))
from control_modes import MODES, mode_spec
from hybrid_web import Console, Preview, load_saved, select_teacher_sources
from learning_curves import load_console_learning_data
from preview_sync import SnapshotPublisher, read_snapshot
import web_preview

MODE = "wheel_torque_bc"


def fake_contract():
    package, contract = ModuleType("wheel_teacher"), ModuleType("wheel_teacher.contract")
    package.__path__ = []
    contract.source_fingerprint = Mock(return_value="teacher-source")
    contract.verified_report = Mock(return_value=None)
    return contract, {"wheel_teacher": package, "wheel_teacher.contract": contract}


class TeacherWebTests(unittest.TestCase):
    def test_mode_identity_keeps_other_modes_separate(self):
        spec = mode_spec(MODE)
        self.assertTrue(spec["teacher"])
        self.assertFalse(spec["hybrid"])
        self.assertEqual((spec["obs_dim"], spec["control_hz"]), (26, 100))
        self.assertTrue(all(not mode_spec(m)["teacher"] for m in MODES if m not in (MODE, "leg_force_bc")))

    def test_empty_teacher_console_is_ready_without_model_or_simulator(self):
        control = Mock()
        control.status.return_value = {"active": False, "phase": "idle", "latest": None}
        factory = Mock(side_effect=AssertionError("Must remain lazy"))
        preview = Preview(MODE, control, factory)
        console = Console(SimpleNamespace(mode=MODE), control, preview)
        preview.tick()
        state = console.state()
        self.assertEqual(state["obs_dim"], 26)
        self.assertEqual(state["input_units"], ["rad", "rad", "N·m", "N·m"])
        self.assertFalse(state["preview_available"])
        factory.assert_not_called()
        control.start.assert_not_called()
        console.training({"action": "start", "duration_seconds": 60, "resume": True})
        control.start.assert_called_once_with(MODE, duration_seconds=60, resume=True, force_revalidate=False)

    def test_teacher_dispatch_uses_lazy_server_before_any_environment_import(self):
        with tempfile.TemporaryDirectory() as tmp, patch("hybrid_web.main") as start:
            argv = ["web_preview.py", "--artifacts", tmp, "--mode", MODE]
            with patch.object(sys, "argv", argv):
                web_preview.main()
            self.assertEqual(start.call_args.args[0].mode, MODE)
            self.assertEqual(json.loads((Path(tmp) / "selected-mode.json").read_text())["mode"], MODE)

    def test_teacher_preview_rejects_already_loaded_torque_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "mujoco_legwheel_rl_public/legwheel"
            source.mkdir(parents=True)
            (source / "__init__.py").write_text("")
            deploy = root / "mujoco_esp32_robot_deploy_public/host/policy"
            deploy.mkdir(parents=True)
            (deploy / "rl_policy.py").write_text("")
            wrong = SimpleNamespace(__file__=str(root / "torque/legwheel/__init__.py"))
            with patch.dict(sys.modules, {"legwheel": wrong}):
                with self.assertRaisesRegex(RuntimeError, "Wrong legwheel source"):
                    select_teacher_sources(root)
            with patch.dict(sys.modules, {"legwheel": SimpleNamespace(__file__=str(source / "__init__.py"))}), patch.object(sys, "path", list(sys.path)):
                self.assertEqual(select_teacher_sources(root), root.resolve())
                self.assertEqual(Path(sys.path[0]), source.parent.resolve())

    def test_saved_teacher_model_must_pass_contract_before_reader(self):
        contract, modules = fake_contract()
        contract.verified_report.side_effect = ValueError("student metadata mismatch")
        reader = ModuleType("host.policy.rl_policy")
        reader.RL_Policy = Mock()
        modules["host.policy.rl_policy"] = reader
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, modules):
            root = Path(tmp)
            (root / "report.json").write_text(json.dumps({"export_path": str(root / "policy.cpp.txt")}))
            with patch("hybrid_web.select_teacher_sources", return_value=root):
                with self.assertRaisesRegex(ValueError, "student metadata"):
                    load_saved({"run_dir": str(root)}, MODE, root)
            reader.RL_Policy.assert_not_called()

    def test_live_teacher_snapshot_binds_reference_source_and_26d_actor(self):
        contract, modules = fake_contract()
        params = [SimpleNamespace(mean={"state": np.zeros(26)}, std={"state": np.ones(26)}),
            {"params": {f"hidden_{i}": {"kernel": np.zeros((a, b)), "bias": np.zeros(b)}
                        for i, (a, b) in enumerate(zip((26, 64, 64), (64, 64, 8)))}}, {}]
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, modules):
            root = Path(tmp)
            (root / "preview.request.json").write_text(json.dumps({"request_id": "one", "mode": MODE}))
            publisher = SnapshotPublisher(root, MODE, original_root=root / "reference")
            response = publisher.service(10, params)
            self.assertNotIn("error", response)
            actor, metadata = read_snapshot(root / "preview-policy.npz", MODE, "one", original_root=root / "reference")
            self.assertEqual(actor.run(np.zeros(26)).shape, (4,))
            self.assertEqual(metadata["source_fingerprint"], "teacher-source")
            contract.source_fingerprint.assert_called_with(root / "reference")
            with self.assertRaisesRegex(ValueError, "mode mismatch"):
                read_snapshot(root / "preview-policy.npz", "original", "one")
            contract.source_fingerprint.return_value = "changed"
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                read_snapshot(root / "preview-policy.npz", MODE, "one", original_root=root / "reference")

    def test_teacher_curves_require_student_metadata_not_just_same_frequency(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = {"env_config.json": {"ctrl_dt": .01, "episode_length": 600},
                    "ppo_config.json": {"num_eval_envs": 2},
                    "policy_metadata.json": {"mode": MODE, "control_hz": 100, "actor_dim": 26},
                    "progress.json": [{"steps": 64, "metrics": {"eval/avg_episode_length": 20}}]}
            for name, value in data.items():
                (root / name).write_text(json.dumps(value))
            self.assertTrue(load_console_learning_data(root, MODE)["available"])
            data["policy_metadata.json"]["mode"] = "original"
            (root / "policy_metadata.json").write_text(json.dumps(data["policy_metadata.json"]))
            self.assertFalse(load_console_learning_data(root, MODE)["available"])


if __name__ == "__main__":
    unittest.main()
