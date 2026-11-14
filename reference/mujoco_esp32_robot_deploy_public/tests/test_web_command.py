"""Web 操作员映射单测。不经过 HTTP，也不经过 sim/real 本体。

锁的是摇杆 / 滑条 → command[4]，与 src/web/command.cpp 对着写。
sim 和 real 都接这一套，所以映射错了两边一起错。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "host"))
from web.command import apply_control_page, COMMAND_RANGE
from web.state import DEFAULT_K, Operator


class WebCommandTests(unittest.TestCase):
    def test_forward_stick_sets_vx_not_yaw(self) -> None:
        cmd = [0.0, 0.0, 0.0, 0.0]
        apply_control_page(0.0, 1.0, 0.0, 0.0, cmd)
        self.assertAlmostEqual(cmd[0], 0.2, places=5)
        self.assertAlmostEqual(cmd[1], 0.0, places=5)

    def test_back_stick_negative_vx(self) -> None:
        cmd = [0.0, 0.0, 0.0, 0.0]
        apply_control_page(0.0, -1.0, 0.0, 0.0, cmd)
        self.assertAlmostEqual(cmd[0], -0.2, places=5)

    def test_half_stick_is_symmetric(self) -> None:
        fwd = [0.0, 0.0, 0.0, 0.0]
        back = [0.0, 0.0, 0.0, 0.0]
        apply_control_page(0.0, 0.25, 0.0, 0.0, fwd)
        apply_control_page(0.0, -0.25, 0.0, 0.0, back)
        self.assertAlmostEqual(fwd[0], 0.2, places=5)
        self.assertAlmostEqual(back[0], -0.2, places=5)

    def test_small_down_stick_already_full_reverse(self) -> None:
        cmd = [0.0, 0.0, 0.0, 0.0]
        apply_control_page(0.0, -0.2, 0.0, 0.0, cmd)
        self.assertAlmostEqual(cmd[0], -0.2, places=5)

    def test_center_stick_is_stop(self) -> None:
        cmd = [0.1, 0.1, 0.0, 0.0]
        apply_control_page(0.0, 0.0, 0.0, 0.0, cmd)
        self.assertAlmostEqual(cmd[0], 0.0, places=6)
        self.assertAlmostEqual(cmd[1], 0.0, places=6)

    def test_height_bar_100_is_crouch(self) -> None:
        cmd = [0.0, 0.0, 0.0, 0.0]
        apply_control_page(0.0, 0.0, 100.0, 0.0, cmd)
        self.assertAlmostEqual(cmd[2], -0.5, places=5)

    def test_height_bar_0_is_stand(self) -> None:
        cmd = [0.0, 0.0, 0.0, 0.0]
        apply_control_page(0.0, 0.0, 0.0, 0.0, cmd)
        self.assertAlmostEqual(cmd[2], 0.0, places=5)

    def test_roll_bar_extremes(self) -> None:
        cmd = [0.0, 0.0, 0.0, 0.0]
        apply_control_page(0.0, 0.0, 0.0, 50.0, cmd)
        self.assertAlmostEqual(cmd[3], 0.2, places=5)
        apply_control_page(0.0, 0.0, 0.0, -50.0, cmd)
        self.assertAlmostEqual(cmd[3], -0.2, places=5)

    def test_main_page_keeps_k_commands(self) -> None:
        op = Operator()
        op.page = "main"
        op.k["cmd_lin_vel_x(-0.2,0.2)"] = 0.15
        op.set_joystick(0.0, 1.0, now_ms=0)
        cmd = op.command(now_ms=0)
        self.assertAlmostEqual(cmd[0], 0.15, places=5)

    def test_control_page_overrides_k_with_stick(self) -> None:
        op = Operator()
        op.page = "control"
        op.k["cmd_lin_vel_x(-0.2,0.2)"] = 0.15
        op.set_joystick(0.0, 1.0, now_ms=10)
        cmd = op.command(now_ms=10)
        self.assertAlmostEqual(cmd[0], 0.2, places=5)


    def test_k_tables_match_holdobs_and_firmware(self) -> None:
        import re

        from controller.hold_obs import HoldObs

        self.assertAlmostEqual(DEFAULT_K["pitch_max"], HoldObs().pitch_max, places=6)
        self.assertAlmostEqual(DEFAULT_K["sinroll_scale"], 1.0, places=6)
        cpp = Path(__file__).resolve().parents[1] / "src" / "web" / "lqrwebserver.cpp"
        block = re.search(r"KDict Kweb = \{([\s\S]*?)\};", cpp.read_text())
        self.assertIsNotNone(block)
        parsed = {
            k: float(v)
            for k, v in re.findall(r'\{\s*"([^"]+)"\s*,\s*([-+0-9.eE]+)f?\s*\}', block.group(1))
        }
        self.assertEqual(set(parsed), set(DEFAULT_K))
        for key, value in DEFAULT_K.items():
            self.assertAlmostEqual(parsed[key], value, places=5, msg=key)

    def test_command_range_matches_firmware(self) -> None:
        import re

        cpp = Path(__file__).resolve().parents[1] / "src" / "web" / "command.cpp"
        block = re.search(
            r"kCommandRange\[4\]\[2\] = \{([\s\S]*?)\};", cpp.read_text()
        )
        self.assertIsNotNone(block)
        nums = [float(x) for x in re.findall(r"[-+]?\d+\.\d+|[-+]?\d+", block.group(1))]
        self.assertEqual(len(nums), 8)
        flat = [v for pair in COMMAND_RANGE for v in pair]
        for a, b in zip(nums, flat):
            self.assertAlmostEqual(a, b, places=6)


if __name__ == "__main__":
    unittest.main()
