"""共享控制核 HoldObs 的离线单测（不经过 real / sim 支路）。

覆盖与训练仓 tests/test_env.py 相同的不变量。
输入已是仿真系 SimFrame。不覆盖 IMU 安装符号、舵机映射、环频率。

运行：python tests/test_hold_obs.py
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "host"))
from controller.hold_obs import HoldObs


class HoldObsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.obs = HoldObs()
        self.obs.step([0, 0, 0, 0], q_l=0.0, q_r=0.0, yaw_rad=0.0)

    def test_common_mode_wheel_only_grows_ds(self) -> None:
        out = self.obs.step([0, 0, 0, 0], q_l=1.0, q_r=1.0, yaw_rad=0.0)
        self.assertAlmostEqual(out["obs_ds_l"], 1.0, places=6)
        self.assertAlmostEqual(out["obs_ds_r"], 1.0, places=6)
        self.assertAlmostEqual(out["obs_dpsi"], 0.0, places=6)

    def test_differential_wheel_does_not_grow_ds(self) -> None:
        out = self.obs.step([0, 0, 0, 0], q_l=1.0, q_r=-1.0, yaw_rad=0.0)
        self.assertAlmostEqual(out["obs_ds_l"], 0.0, places=6)
        self.assertAlmostEqual(out["obs_ds_r"], 0.0, places=6)

    def test_imu_heading_only_grows_dpsi(self) -> None:
        out0 = self.obs.step([0, 0, 0, 0], q_l=0.0, q_r=0.0, yaw_rad=0.0)
        delta = 0.3
        out1 = self.obs.step([0, 0, 0, 0], q_l=0.0, q_r=0.0, yaw_rad=delta)
        expected = math.sin(delta * 0.5) / 0.5
        self.assertGreater(out1["obs_dpsi"] - out0["obs_dpsi"], 0.1)
        self.assertAlmostEqual(out1["obs_dpsi"] - out0["obs_dpsi"], expected, places=6)
        self.assertAlmostEqual(out1["obs_ds_l"], out0["obs_ds_l"], places=6)
        self.assertAlmostEqual(out1["obs_ds_l"], out1["obs_ds_r"], places=6)

    def test_vx_does_not_refresh_heading_anchor(self) -> None:
        far_yaw0 = self.obs.sinyaw(1.0)
        self.obs.yaw_sin_zero = far_yaw0
        out = self.obs.step([0.2, 0.0, 0.0, 0.0], q_l=0.4, q_r=0.4, yaw_rad=0.0)
        self.assertAlmostEqual(out["yaw0"], far_yaw0, places=6)
        self.assertAlmostEqual(out["s0"], 0.4, places=6)

    def test_yaw_cmd_does_not_refresh_position_anchor(self) -> None:
        self.obs.position_start = 3.0
        out = self.obs.step([0.0, 0.5, 0.0, 0.0], q_l=0.2, q_r=0.2, yaw_rad=0.4)
        self.assertAlmostEqual(out["s0"], 3.0, places=6)
        self.assertAlmostEqual(out["yaw0"], self.obs.sinyaw(0.4), places=6)

    def test_both_wheel_slots_are_the_same_ds(self) -> None:
        out = self.obs.step([0, 0, 0, 0], q_l=-1.5, q_r=-1.5, yaw_rad=0.0)
        self.assertAlmostEqual(out["obs_ds_l"], out["obs_ds_r"], places=6)
        self.assertAlmostEqual(out["ds_true"], -1.5, places=6)
        self.assertAlmostEqual(out["obs_ds_l"], out["ds_true"], places=6)

    def test_clip_obs_not_anchor(self) -> None:
        self.obs.position_start = self.obs.hold_position(0.0, 0.0) - 8.0
        out = self.obs.step([0, 0, 0, 0], q_l=0.0, q_r=0.0, yaw_rad=0.0)
        self.assertAlmostEqual(out["s0"], -8.0, places=6)
        self.assertGreater(abs(out["ds_true"]), 5.0)
        self.assertLessEqual(abs(out["obs_ds_l"]), 3.0 + 1e-6)
        self.assertAlmostEqual(out["obs_ds_l"], out["obs_ds_r"], places=6)

    def test_deadzone_command_is_zero_in_obs(self) -> None:
        out = self.obs.step([1e-5, 0.005, 0, 0], q_l=0.0, q_r=0.0, yaw_rad=0.0)
        self.assertEqual(out["cmd"][0], 0.0)
        self.assertEqual(out["cmd"][1], 0.0)
        self.assertAlmostEqual(out["s0"], 0.0, places=6)
        self.assertAlmostEqual(out["yaw0"], 0.0, places=6)

    def test_fallen_resets_both_anchors(self) -> None:
        self.obs.position_start = 2.0
        self.obs.yaw_sin_zero = 0.5
        out = self.obs.step(
            [0, 0, 0, 0], q_l=0.8, q_r=0.8, yaw_rad=0.2, pitch_rad=1.2
        )
        self.assertAlmostEqual(out["s0"], 0.8, places=6)
        self.assertAlmostEqual(out["yaw0"], self.obs.sinyaw(0.2), places=6)
        self.assertAlmostEqual(out["obs_ds_l"], 0.0, places=6)
        self.assertAlmostEqual(out["obs_dpsi"], 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
