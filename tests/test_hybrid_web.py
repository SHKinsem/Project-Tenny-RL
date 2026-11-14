import io
import json
from pathlib import Path
import sys
import tempfile
import shutil
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments/torque200"
sys.path.insert(0, str(SOURCE))
from hybrid_web import Console, Preview, make_handler
from learning_curves import load_console_learning_data
from training_control import ControlError
from web_preview import restart_arguments

MODE = "hybrid_virtual_leg_force_100"


def controller():
    value = Mock()
    value.status.return_value = {"active": False, "phase": "idle", "latest": None}
    value.request_preview.return_value = {"kind": "saved", "model": None, "request_id": "request"}
    return value


class HybridConsoleTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node is required for browser script syntax check")
    def test_dashboard_script_syntax(self):
        html = (SOURCE / "dashboard.html").read_text(encoding="utf8")
        script = html.split("<script>", 1)[1].split("</script>", 1)[0]
        result = subprocess.run([shutil.which("node"), "--check"], input=script,
                                encoding="utf8", capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_first_load_without_model_does_not_import_ml_or_start_training(self):
        control = controller()
        factory = Mock(side_effect=AssertionError("No simulator during first load"))
        preview = Preview(MODE, control, factory)
        console = Console(SimpleNamespace(mode=MODE), control, preview)
        state = console.state()
        preview.tick()
        self.assertFalse(state["preview_available"])
        self.assertTrue(state["suspended"])
        self.assertEqual(state["control_hz"], 100)
        self.assertEqual(state["training"]["phase"], "idle")
        factory.assert_not_called()
        control.start.assert_not_called()
        with self.assertRaises(ControlError):
            console.control({"action": "resume"})

    def test_start_routes_mode_time_and_force_without_mixing_legacy(self):
        control = controller()
        console = Console(SimpleNamespace(mode=MODE), control)
        console.training({"action": "start", "duration_seconds": 60, "resume": True, "force_revalidate": True})
        control.start.assert_called_once_with(MODE, duration_seconds=60, resume=True, force_revalidate=True)
        console.training({"action": "stop"})
        control.stop.assert_called_once_with()

    def test_switch_rejected_while_any_training_active(self):
        control = controller()
        control.status.return_value = {"active": True, "active_mode": "original"}
        console = Console(SimpleNamespace(mode=MODE), control)
        with self.assertRaises(ControlError):
            console.control({"action": "switch-mode", "mode": "hybrid_direct_torque_200"})
        self.assertIsNone(console.switch_mode)

    def test_switch_and_new_training_cannot_overlap(self):
        control = controller()
        console = Console(SimpleNamespace(mode=MODE), control)
        console.control({"action": "switch-mode", "mode": "original"})
        with self.assertRaises(ControlError):
            console.training({"action": "start"})
        self.assertTrue(console.state()["switching"])
        control.start.assert_not_called()

    def test_post_rejects_non_object_json_without_starting(self):
        control = controller()
        console = Console(SimpleNamespace(mode=MODE), control)
        handler = object.__new__(make_handler(console))
        handler.path = "/training"
        handler.headers = {"Content-Type": "application/json", "Content-Length": "2"}
        handler.rfile = io.BytesIO(b"[]")
        handler.send = Mock()
        handler.do_POST()
        self.assertEqual(handler.send.call_args.args[1], 400)
        control.start.assert_not_called()

    def test_hybrid_restart_keeps_original_root_and_port(self):
        args = SimpleNamespace(artifacts=Path("/tmp/artifacts"), original_root=Path("/tmp/reference"),
                               host="127.0.0.1", port=8770)
        argv = restart_arguments(args, MODE)
        self.assertEqual(argv[-2:], ["--mode", MODE])
        self.assertIn("8770", argv)


class PreviewTests(unittest.TestCase):
    def setUp(self):
        self.control = controller()
        self.control.request_preview.return_value = {"kind": "saved", "model": {"run_id": "saved"}, "request_id": "request"}
        self.simulator = Mock()
        self.simulator.frame.return_value = (b"jpeg", {"episode_seconds": .05, "reset_delta": 0})
        self.factory = Mock(return_value=self.simulator)
        self.preview = Preview(MODE, self.control, self.factory)

    def test_repeated_open_only_requests_once_and_closed_view_does_no_steps(self):
        self.preview.open()
        self.preview.open()
        self.control.request_preview.assert_called_once_with(MODE)
        with patch("hybrid_web.load_saved", return_value=(object(), {"preview_training_steps": 16})):
            self.preview.tick()
        self.factory.assert_called_once_with(MODE)
        self.simulator.frame.assert_not_called()  # No browser stream clients.
        self.preview.state["clients"] = 1
        self.preview.tick()
        self.simulator.frame.assert_called_once()
        self.preview.suspend()
        self.preview.tick()
        self.simulator.frame.assert_called_once()
        self.assertEqual(self.preview.jpg, b"")

    def test_cancel_during_slow_load_never_reopens(self):
        self.preview.open()
        def late_load(*_):
            self.preview.suspend()
            return object(), {"preview_training_steps": 16}
        with patch("hybrid_web.load_saved", side_effect=late_load):
            self.preview.tick()
        self.assertTrue(self.preview.snapshot()["suspended"])
        self.factory.assert_not_called()

    def test_cancel_during_first_simulator_setup_never_publishes(self):
        self.preview.open()
        def factory(_):
            self.preview.suspend()
            return self.simulator
        self.preview.factory = factory
        with patch("hybrid_web.load_saved", return_value=(object(), {})):
            self.preview.tick()
        self.assertTrue(self.preview.snapshot()["suspended"])
        self.assertEqual(self.preview.snapshot()["frames"], 0)
        self.simulator.frame.assert_not_called()

    def test_wrong_saved_contract_surfaces_error_without_fallback(self):
        self.preview.open()
        with patch("hybrid_web.load_saved", side_effect=ValueError("Incompatible source fingerprint")):
            self.preview.tick()
        state = self.preview.snapshot()
        self.assertTrue(state["suspended"])
        self.assertEqual(state["sync_status"], "error")
        self.assertIn("fingerprint", state["sync_error"])
        self.factory.assert_not_called()


class HybridCurvesTests(unittest.TestCase):
    def test_frequency_and_action_identity_are_checked_before_showing_curves(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            files = {"env_config.json": {"ctrl_dt": .01, "episode_length": 600},
                "ppo_config.json": {"num_eval_envs": 2},
                "policy_metadata.json": {"action_mode": "virtual_leg_force", "control_hz": 100},
                "progress.json": [{"steps": 16, "metrics": {"eval/avg_episode_length": 50, "eval/episode_reward": 2}}]}
            for name, value in files.items():
                (directory / name).write_text(json.dumps(value))
            result = load_console_learning_data(directory, MODE)
            self.assertTrue(result["available"])
            self.assertEqual(result["points"][0]["episode_seconds"], .5)
            self.assertFalse(load_console_learning_data(directory, "hybrid_virtual_leg_force_200")["available"])
            self.assertFalse(load_console_learning_data(directory, "hybrid_direct_torque_100")["available"])


if __name__ == "__main__":
    unittest.main()
