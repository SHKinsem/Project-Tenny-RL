"""Backend selection stays explicit and requires no ML imports."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from native_console_evaluation import selected_backend


class BackendSelectionTests(unittest.TestCase):
    def test_auto_selects_verified_force_mode(self):
        self.assertEqual(selected_backend("leg_force_bc", "auto"), "native_mujoco_cpp")

    def test_auto_preserves_other_modes(self):
        for mode in ("original", "torque200", "wheel_torque_bc", "hybrid_virtual_leg_force_100"):
            with self.subTest(mode=mode):
                self.assertEqual(selected_backend(mode, "auto"), "mjx_gpu")

    def test_explicit_gpu_for_force_comparison(self):
        self.assertEqual(selected_backend("leg_force_bc", "gpu"), "mjx_gpu")

    def test_explicit_native_rejects_unsupported_mode(self):
        with self.assertRaises(ValueError):
            selected_backend("original", "native")

    def test_invalid_backend_rejected(self):
        for value in ("cuda", "", True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                selected_backend("leg_force_bc", value)

    def test_environment_override(self):
        with patch.dict("os.environ", {"TENNY_EVAL_BACKEND": "gpu"}):
            self.assertEqual(selected_backend("leg_force_bc"), "mjx_gpu")


if __name__ == "__main__":
    unittest.main()
