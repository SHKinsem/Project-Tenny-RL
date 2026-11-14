"""共享 26 维观测拼装单测。不经过 real / sim 本体。

运行：python tests/test_obs.py
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "host"))
from controller.hold_obs import HoldObs
from controller.obs import OBS_DIM, ObsPack
from controller.sim_frame import SimFrame


class ObsPackTests(unittest.TestCase):
    def test_slots_and_pitch_curve(self) -> None:
        packer = ObsPack()
        hold = HoldObs()
        frame = SimFrame(
            q_l=0.4,
            q_r=0.4,
            yaw_rad=0.0,
            pitch_rad=0.1,
            roll_rad=0.05,
            gyro=[0.1, 0.2, 0.3],
            thigh_l=-0.2,
            thigh_r=-0.2,
            thigh_vel_l=0.01,
            thigh_vel_r=0.02,
            wheel_vel_l=1.0,
            wheel_vel_r=2.0,
        )
        out = hold.step([0.2, 0.0, -0.25, 0.1], frame, turn_on=True)
        obs = packer.pack(out["cmd"], frame, out)
        self.assertEqual(len(obs), OBS_DIM)
        self.assertAlmostEqual(obs[0], 0.2, places=6)
        self.assertAlmostEqual(obs[3], 0.1, places=6)
        self.assertAlmostEqual(obs[4], 0.1, places=6)
        self.assertAlmostEqual(obs[5], 0.2, places=6)
        self.assertAlmostEqual(obs[6], 0.3, places=6)
        self.assertAlmostEqual(obs[7], out["obs_dpsi"], places=6)
        self.assertAlmostEqual(obs[11], out["obs_ds_l"], places=6)
        self.assertAlmostEqual(obs[13], out["obs_ds_r"], places=6)
        self.assertAlmostEqual(obs[10], -0.2, places=6)
        self.assertAlmostEqual(obs[12], -0.2, places=6)
        self.assertAlmostEqual(obs[14], 0.0, places=6)
        self.assertAlmostEqual(obs[16], 0.0, places=6)
        self.assertAlmostEqual(obs[15], 1.0, places=6)
        self.assertAlmostEqual(obs[17], 2.0, places=6)
        self.assertAlmostEqual(obs[18], 1.0 * 0.0195, places=6)
        self.assertAlmostEqual(obs[19], 2.0 * 0.0195, places=6)
        self.assertAlmostEqual(obs[24], -1.0, places=6)
        self.assertAlmostEqual(obs[25], -2.0, places=6)

        cfg = packer.cfg
        avg = (frame.thigh_l + frame.thigh_r) * 0.5 - cfg.thigh_hi
        ratio = avg / (cfg.thigh_lo - cfg.thigh_hi)
        pitch_zero = cfg.pitch_zero_low + (cfg.pitch_zero_high - cfg.pitch_zero_low) * ratio
        expected_pitch = (math.sin(0.1) - math.sin(pitch_zero)) * cfg.sinpitch_scale
        expected_roll = (math.sin(0.05) - math.sin(0.1)) * cfg.sinroll_scale
        self.assertAlmostEqual(obs[8], expected_pitch, places=6)
        self.assertAlmostEqual(obs[9], expected_roll, places=6)

    def test_note_action_shifts_history_and_smooths_target(self) -> None:
        packer = ObsPack()
        frame = SimFrame(wheel_vel_l=1.0, wheel_vel_r=2.0)
        packer.note_policy_action([0.0, 0.0, 1.0, 0.0], frame)
        self.assertAlmostEqual(packer.wheels_commands_history[2], 30.0, places=6)
        self.assertAlmostEqual(packer.wheels_commands_history[3], 0.0, places=6)
        self.assertAlmostEqual(packer.last_target_wheel_vel[0], 15.0, places=6)
        packer.note_policy_action([0.0, 0.0, 0.0, 1.0], frame)
        self.assertAlmostEqual(packer.wheels_commands_history[0], 30.0, places=6)
        self.assertAlmostEqual(packer.wheels_commands_history[1], 0.0, places=6)
        self.assertAlmostEqual(packer.wheels_commands_history[2], 0.0, places=6)
        self.assertAlmostEqual(packer.wheels_commands_history[3], 30.0, places=6)

        hold = {
            "obs_dpsi": 0.0,
            "obs_ds_l": 0.0,
            "obs_ds_r": 0.0,
        }
        obs = packer.pack([0, 0, 0, 0], frame, hold)
        self.assertAlmostEqual(obs[20], 30.0, places=6)
        self.assertAlmostEqual(obs[23], 30.0, places=6)

    def test_reset_clears_history(self) -> None:
        packer = ObsPack()
        packer.note_policy_action([0, 0, 1, 1], SimFrame(wheel_vel_l=0, wheel_vel_r=0))
        packer.reset()
        self.assertEqual(packer.wheels_commands_history, [0.0, 0.0, 0.0, 0.0])
        self.assertEqual(packer.last_target_wheel_vel, [0.0, 0.0])

    def test_geom_pitch_zero_matches_training_formula(self) -> None:
        packer = ObsPack()
        hold = HoldObs()
        frame = SimFrame(pitch_rad=0.1, pitch_zero_rad=0.07)
        out = hold.step([0.0, 0.0, -0.25, 0.0], frame, turn_on=True)
        obs = packer.pack(out["cmd"], frame, out)
        self.assertAlmostEqual(obs[8], math.sin(0.1) - math.sin(0.07), places=6)

    def test_sinroll_scale_default_is_identity(self) -> None:
        self.assertAlmostEqual(ObsPack().cfg.sinroll_scale, 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
