"""Real 支路：传感器 → SimFrame。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "host"))
from real.io import DEG2RAD, to_sim_frame


class RealIoTests(unittest.TestCase):
    def test_wheel_sign_flip_and_deg_to_rad(self) -> None:
        sensors = [0.0] * 18
        sensors[1] = 10.0
        sensors[2] = -20.0
        sensors[3] = 30.0
        sensors[10] = 1.5
        sensors[11] = -0.4
        sensors[12] = 2.0
        sensors[13] = 3.0
        sensors[14] = 2136.0
        sensors[15] = 2019.0
        frame = to_sim_frame(sensors)
        self.assertAlmostEqual(frame.q_l, -1.5, places=6)
        self.assertAlmostEqual(frame.q_r, 0.4, places=6)
        self.assertAlmostEqual(frame.wheel_vel_l, -2.0, places=6)
        self.assertAlmostEqual(frame.wheel_vel_r, -3.0, places=6)
        self.assertAlmostEqual(frame.yaw_rad, 30.0 * DEG2RAD, places=6)
        self.assertAlmostEqual(frame.pitch_rad, -20.0 * DEG2RAD, places=6)
        self.assertAlmostEqual(frame.roll_rad, 10.0 * DEG2RAD, places=6)

    def test_squat_servos_map_to_sim_zero_thigh(self) -> None:
        sensors = [0.0] * 18
        sensors[14] = 2136.0
        sensors[15] = 2019.0
        frame = to_sim_frame(sensors)
        self.assertAlmostEqual(frame.thigh_l, 0.0, places=5)
        self.assertAlmostEqual(frame.thigh_r, 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
