"""Force-teacher preview and web routing remain distinct from position control."""
import json
from pathlib import Path
import sys
import tempfile
import tomllib
import shutil
import subprocess
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/torque200"))
from control_modes import mode_spec
from hybrid_web import Console, Preview, load_saved, select_torque_sources
from learning_curves import load_console_learning_data
from preview_sync import SnapshotPublisher, read_snapshot
import web_preview
from force_teacher.options import normalize_options

MODE = "leg_force_bc"


def fake_contract():
    package, contract = ModuleType("force_teacher"), ModuleType("force_teacher.contract")
    package.__path__ = [str(ROOT / "experiments/torque200/force_teacher")]
    contract.source_fingerprint = Mock(return_value="force-teacher-source")
    contract.verified_report = Mock(return_value=None)
    return contract, {"force_teacher": package, "force_teacher.contract": contract}


class ForceTeacherWebTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node required for dashboard behavior checks")
    def test_dashboard_gate_failure_and_standing_labels_use_explicit_fields(self):
        html = (ROOT / "experiments/torque200/dashboard.html").read_text(encoding="utf8")
        script = html.split("let latest=", 1)[1].split("async function refresh()", 1)[0]
        cases = [
            {"active": False, "phase": "failed", "stage": "teacher_gate_failed", "bc_gate_passed": False,
             "training_clock_started": False, "error": "minimum episode failed"},
            {"active": True, "phase": "preparing", "stage": "teacher_gate", "bc_gate_passed": False,
             "training_clock_started": False},
            {"active": True, "phase": "training", "training_clock_started": True,
             "training_profile": "standing_v2", "evaluation_policy": "deterministic"},
            {"active": False, "phase": "completed"},
        ]
        harness = """const nodes={};global.document={getElementById:id=>nodes[id]??=( {value:'timed',textContent:'',removeAttribute(){},setAttribute(){}} )};\n"""
        harness += "let latest=" + script
        harness += "\nconst cases=" + json.dumps(cases) + ";const results=[];"
        harness += "for(const t of cases){latest={mode:'leg_force_bc',training:t};renderTraining();results.push({phase:el('training-phase').textContent,detail:el('training-detail').textContent,config:el('training-config').textContent});}process.stdout.write(JSON.stringify(results));"
        result = subprocess.run([shutil.which("node")], input=harness, capture_output=True, encoding="utf8")
        self.assertEqual(result.returncode, 0, result.stderr)
        rendered = json.loads(result.stdout)
        self.assertIn("未进入PPO", rendered[0]["phase"])
        self.assertIn("minimum episode failed", rendered[0]["detail"])
        self.assertNotIn("失败", rendered[1]["phase"])
        self.assertIn("确定性评估", rendered[2]["config"])
        self.assertNotIn("确定性评估", rendered[3]["config"])

    def test_gate_failure_curve_message_does_not_require_nonexistent_ppo_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "status.json").write_text(json.dumps({"phase": "failed", "stage": "teacher_gate_failed",
                "bc_gate_passed": False, "training_clock_started": False}))
            result = load_console_learning_data(run, MODE)
            self.assertFalse(result["available"])
            self.assertIn("未进入PPO", result["metadata"]["message"])
            self.assertEqual(result["points"], [])

    def test_force_mode_contract_and_defaults(self):
        spec = mode_spec(MODE)
        self.assertTrue(spec["teacher"])
        self.assertTrue(spec["hybrid"])
        self.assertEqual((spec["action_mode"], spec["obs_dim"], spec["control_hz"]), ("virtual_leg_force", 24, 100))
        self.assertEqual(mode_spec("wheel_torque_bc")["obs_dim"], 26)
        tasks = tomllib.loads((ROOT / "tenny/tasks.toml").read_text(encoding="utf8"))["task"]
        web = next(t for t in tasks if t["id"] == "web")
        selected = next(p for p in web["params"] if p["name"] == "mode")
        self.assertEqual(selected["default"], MODE)
        self.assertIn("wheel_torque_bc", selected["choices"])
        with tempfile.TemporaryDirectory() as tmp, patch("hybrid_web.main") as launch:
            with patch.object(sys, "argv", ["web_preview.py", "--artifacts", tmp]):
                web_preview.main()
            self.assertEqual(launch.call_args.args[0].mode, MODE)

    def test_idle_force_page_displays_actuator_torque_units_without_loading_ml(self):
        control = Mock()
        control.status.return_value = {"active": False, "phase": "idle", "latest": None}
        factory = Mock(side_effect=AssertionError("Must remain lazy"))
        preview = Preview(MODE, control, factory)
        preview.tick()
        state = Console(SimpleNamespace(mode=MODE), control, preview).state()
        self.assertEqual(state["obs_dim"], 24)
        self.assertEqual(state["input_units"], ["N·m"] * 4)
        self.assertIn("策略腿命令", state["actuator_description"])
        self.assertIn("下表", state["actuator_description"])
        self.assertIn("N·m", state["actuator_description"])
        self.assertFalse(state["preview_available"])
        factory.assert_not_called()

    def test_force_source_rejects_original_legwheel(self):
        wrong = SimpleNamespace(__file__=str(ROOT / "reference/mujoco_legwheel_rl_public/legwheel/__init__.py"))
        with patch.dict(sys.modules, {"legwheel": wrong}):
            with self.assertRaisesRegex(RuntimeError, "Wrong legwheel source"):
                select_torque_sources()
        expected = ROOT / "experiments/torque200"
        current = SimpleNamespace(__file__=str(expected / "legwheel/__init__.py"))
        with patch.dict(sys.modules, {"legwheel": current}), patch.object(sys, "path", list(sys.path)):
            self.assertEqual(select_torque_sources(), expected.resolve())
            self.assertEqual(Path(sys.path[0]), expected.resolve())

    def test_force_live_snapshot_uses_only_force_contract_and_rejects_same_shape_legacy(self):
        contract, modules = fake_contract()
        params = [SimpleNamespace(mean={"state": np.zeros(24)}, std={"state": np.ones(24)}),
            {"params": {f"hidden_{i}": {"kernel": np.zeros((a, b)), "bias": np.zeros(b)}
                        for i, (a, b) in enumerate(zip((24, 64, 64), (64, 64, 8)))}}, {}]
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, modules):
            run = Path(tmp)
            reference = run / "teacher"
            options = normalize_options()
            (run / "policy_metadata.json").write_text(json.dumps(dict(mode=MODE,
                teacher_schema="leg-force-configurable-bc-v4", experiment_config=options)))
            (run / "preview.request.json").write_text(json.dumps({"request_id": "one", "mode": MODE}))
            result = SnapshotPublisher(run, MODE, reference).service(128, params)
            self.assertNotIn("error", result)
            actor, metadata = read_snapshot(run / "preview-policy.npz", MODE, "one", original_root=reference)
            self.assertEqual(actor.run(np.zeros(24)).shape, (4,))
            self.assertEqual(metadata["source_fingerprint"], "force-teacher-source")
            self.assertEqual(actor.experiment_config, options)
            self.assertEqual(metadata["experiment_config"], options)
            contract.source_fingerprint.assert_called_with(reference)
            with self.assertRaisesRegex(ValueError, "mode mismatch"):
                read_snapshot(run / "preview-policy.npz", "hybrid_virtual_leg_force_100", "one")
            contract.source_fingerprint.return_value = "changed-force-source"
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                read_snapshot(run / "preview-policy.npz", MODE, "one", original_root=reference)

    def test_saved_force_policy_uses_contract_aware_force_reader_and_never_reference_loader(self):
        contract, modules = fake_contract()
        for name in ("legwheel", "legwheel.deployment"):
            package = ModuleType(name)
            package.__path__ = []
            modules[name] = package
        reader = ModuleType("force_teacher.policy")
        reader.ForcePolicy = Mock()
        modules[reader.__name__] = reader
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, modules), patch("hybrid_web.select_torque_sources"), patch("hybrid_web.select_teacher_sources", side_effect=AssertionError("Wrong source")):
            run = Path(tmp)
            export = run / "policy.cpp.txt"
            report = {"export_path": str(export)}
            (run / "report.json").write_text(json.dumps(report))
            options = normalize_options(dict(wheel_head="residual"))
            (run / "policy_metadata.json").write_text(json.dumps(dict(mode=MODE,
                teacher_schema="leg-force-configurable-bc-v4", experiment_config=options)))
            model = {"run_dir": str(run), "checkpoint": str(run / "checkpoints/128"), "actual_timesteps": 128}
            _, displayed = load_saved(model, MODE, run / "reference")
            self.assertIsNone(displayed["preview_training_profile"])
            self.assertEqual(displayed["preview_experiment_config"], options)
            contract.verified_report.assert_called_once_with(run, report, run / "reference")
            reader.ForcePolicy.assert_called_once_with(export, run / "policy_metadata.json", run / "reference")
            reader.ForcePolicy.reset_mock()
            report.update(training_profile="standing_v2", evaluation_policy="deterministic")
            (run / "report.json").write_text(json.dumps(report))
            _, displayed = load_saved(model, MODE, run / "reference")
            self.assertEqual(displayed["preview_training_profile"], "standing_v2")
            self.assertEqual(displayed["preview_evaluation_policy"], "deterministic")
            reader.ForcePolicy.reset_mock()
            contract.verified_report.side_effect = ValueError("Invalid force metadata")
            with self.assertRaisesRegex(ValueError, "force metadata"):
                load_saved(model, MODE, run / "reference")
            reader.ForcePolicy.assert_not_called()

    def test_curves_reject_position_teacher_and_legacy_hybrid(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            files = {"env_config.json": {"ctrl_dt": .01, "episode_length": 600},
                "ppo_config.json": {"num_eval_envs": 2},
                "policy_metadata.json": {"mode": MODE, "control_hz": 100, "actor_dim": 24},
                "progress.json": [{"steps": 128, "metrics": {"eval/avg_episode_length": 40}}]}
            for name, value in files.items():
                (run / name).write_text(json.dumps(value))
            legacy = load_console_learning_data(run, MODE)
            self.assertTrue(legacy["available"])
            self.assertNotIn("站立阶段", legacy["metadata"]["source"])
            (run / "status.json").write_text(json.dumps({"training_profile": "standing_v2", "evaluation_policy": "deterministic"}))
            standing = load_console_learning_data(run, MODE)
            self.assertIn("确定性站立评估", standing["metadata"]["source"])
            for wrong in ("wheel_torque_bc", "hybrid_virtual_leg_force_100"):
                files["policy_metadata.json"]["mode"] = wrong
                (run / "policy_metadata.json").write_text(json.dumps(files["policy_metadata.json"]))
                self.assertFalse(load_console_learning_data(run, MODE)["available"])


if __name__ == "__main__":
    unittest.main()
