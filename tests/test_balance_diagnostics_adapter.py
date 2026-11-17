import math
import unittest

from experiments.balance_diagnostics.legwheel import quaternion_euler


class QuaternionMeasurementTests(unittest.TestCase):
    def test_pitch_has_mujoco_wxyz_sign_and_normalizes(self):
        angle = -.4
        roll, pitch, yaw = quaternion_euler([2*math.cos(angle/2), 0, 2*math.sin(angle/2), 0])
        self.assertAlmostEqual(pitch, angle)
        self.assertAlmostEqual(roll, 0)
        self.assertAlmostEqual(yaw, 0)

    def test_invalid_quaternions_fail(self):
        for quaternion in ([0, 0, 0, 0], [1, 0, 0], [float('nan'), 0, 0, 0]):
            with self.subTest(quaternion=quaternion), self.assertRaises(ValueError):
                quaternion_euler(quaternion)
