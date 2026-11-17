"""Physical quantities, first-episode bounds, and time-weighted diagnostics."""
from copy import deepcopy
import math
import unittest

from experiments.balance_diagnostics.metrics import analyze_trace


def trajectory(times=(0., 1., 2., 3., 4., 5., 6., 7., 8.)):
    count = len(times)
    return {
        "schema_version": 1, "t_s": list(times),
        "base_position_m": [[1 + .01*t, 2., .1] for t in times],
        "base_velocity_m_s": [[.01, 0., 0.] for _ in times],
        "pitch_rad": [.02 + .01*t for t in times], "roll_rad": [0.] * count,
        "pitch_rate_rad_s": [.01] * count, "reward": [0.] + [2.] * (count - 1),
        "actuator_command_nm": [[0., 0.]] + [[1., 0.]] * (count - 1),
        "done": [False] * (count - 1) + [True],
        "metadata": {"reference_pitch_rad": .02, "torque_limits_nm": [1., 2.]},
        "termination": {"terminated": False, "truncated": True, "reason": "time_limit", "time_s": times[-1]},
    }


class BalanceMetricsTests(unittest.TestCase):
    def test_first_six_and_full_episode_are_distinct(self):
        result = analyze_trace(trajectory())
        first, full = result["windows"]["first_6s"], result["windows"]["full_episode"]
        self.assertEqual(first["observed_seconds"], 6)
        self.assertEqual(full["observed_seconds"], 8)
        self.assertTrue(first["reached_window_end"])
        self.assertAlmostEqual(first["end_horizontal_displacement_m"], .06)
        self.assertAlmostEqual(full["end_horizontal_displacement_m"], .08)
        self.assertEqual(first["reward_total"], 12)
        self.assertEqual(full["reward_total"], 16)
        self.assertEqual(first["reward_per_second"], full["reward_per_second"])
        self.assertAlmostEqual(full["command_at_limit_fraction"], .5)
        self.assertEqual(full["tail"]["observed_seconds"], 1)

    def test_early_failure_does_not_claim_full_window(self):
        trace = trajectory((0., .5, 1.))
        trace["termination"].update(terminated=True, truncated=False, reason="pitch")
        first = analyze_trace(trace)["windows"]["first_6s"]
        self.assertFalse(first["reached_window_end"])
        self.assertEqual(first["observed_seconds"], 1)

    def test_measured_displacement_does_not_use_wheel_rotation(self):
        trace = trajectory()
        trace["base_position_m"] = [[7., -2., .1] for _ in trace["t_s"]]
        trace["wheel_angle_distance_m"] = [100*t for t in trace["t_s"]]
        result = analyze_trace(trace)["windows"]["full_episode"]
        self.assertEqual(result["max_horizontal_displacement_m"], 0)
        self.assertEqual(result["path_length_m"], 0)

    def test_path_length_is_not_net_displacement(self):
        trace = trajectory((0., 1., 2.))
        trace["base_position_m"] = [[0., 0., .1], [3., 4., .1], [0., 0., .1]]
        result = analyze_trace(trace)["windows"]["full_episode"]
        self.assertEqual(result["end_horizontal_displacement_m"], 0)
        self.assertEqual(result["max_horizontal_displacement_m"], 5)
        self.assertEqual(result["path_length_m"], 10)

    def test_state_rms_and_command_occupancy_use_irregular_dt(self):
        trace = trajectory((0., 1., 3.))
        trace["base_velocity_m_s"] = [[0., 0., 0.], [2., 0., 0.], [2., 0., 0.]]
        trace["actuator_command_nm"] = [[0., 0.], [0., 0.], [1., 2.]]
        result = analyze_trace(trace)["windows"]["full_episode"]
        self.assertAlmostEqual(result["horizontal_speed_rms_m_s"], math.sqrt(10 / 3))
        self.assertAlmostEqual(result["command_at_limit_fraction"], 2 / 3)
        self.assertAlmostEqual(result["reward_per_second"], 4 / 3)

    def test_window_does_not_invent_endpoint_or_borrow_later_reward(self):
        trace = trajectory((0., .4, .9, 1.5))
        result = analyze_trace(trace, windows_s=(1.,))["windows"]["first_1s"]
        self.assertEqual(result["observed_seconds"], .9)
        self.assertFalse(result["reached_window_end"])
        self.assertEqual(result["reward_total"], 4)

    def test_persistent_events_ignore_brief_crossing_and_restart_after_recovery(self):
        trace = trajectory((0., .1, .2, .3, .4, .5, .6))
        trace["base_position_m"] = [[x, 0., 0.] for x in (0., .06, 0., .06, .06, .06, .06)]
        result = analyze_trace(trace)
        self.assertEqual(result["events"]["displacement"]["first_exceeded_s"], .3)
        self.assertEqual(result["events"]["displacement"]["confirmed_at_s"], .5)
        self.assertIsNone(result["events"]["horizontal_speed"]["first_exceeded_s"])

    def test_reference_pitch_and_threshold_override(self):
        trace = trajectory((0., .1, .2, .3))
        trace["pitch_rad"] = [.02 + math.radians(2)] * 4
        result = analyze_trace(trace, thresholds={"pitch_error_deg": 1., "persistence_s": .1})
        self.assertAlmostEqual(result["windows"]["full_episode"]["pitch_error_rms_deg"], 2)
        self.assertEqual(result["events"]["pitch_error"]["first_exceeded_s"], 0)
        self.assertEqual(result["events"]["pitch_error"]["confirmed_at_s"], .1)

    def test_input_not_modified(self):
        trace = trajectory()
        before = deepcopy(trace)
        analyze_trace(trace)
        self.assertEqual(trace, before)

    def test_rejects_samples_after_terminal_step(self):
        trace = trajectory()
        trace["done"][3] = True
        with self.assertRaisesRegex(ValueError, "after its first done"):
            analyze_trace(trace)

    def test_rejects_inconsistent_termination(self):
        for edit in (
            lambda t: t["termination"].update(terminated=True),
            lambda t: t["termination"].update(time_s=7.),
            lambda t: t["termination"].update(reason=""),
            lambda t: t["done"].__setitem__(-1, False),
        ):
            trace = trajectory()
            edit(trace)
            with self.assertRaises(ValueError):
                analyze_trace(trace)

    def test_rejects_nonfinite_wrong_dimensions_and_bad_time(self):
        for edit in (
            lambda t: t["pitch_rad"].__setitem__(1, math.nan),
            lambda t: t["base_position_m"].__setitem__(2, [0., 0.]),
            lambda t: t["t_s"].__setitem__(1, 0.),
            lambda t: t["reward"].pop(),
            lambda t: t["actuator_command_nm"].__setitem__(2, [1.]),
            lambda t: t["metadata"].update(torque_limits_nm=[0., 1.]),
            lambda t: t["done"].__setitem__(1, 0),
        ):
            trace = trajectory()
            edit(trace)
            with self.assertRaises(ValueError):
                analyze_trace(trace)

    def test_rejects_invalid_analysis_options(self):
        for options in (
            {"thresholds": {"displacement_m": 0}},
            {"thresholds": {"persistence_s": math.inf}},
            {"thresholds": {"unknown": 1}},
            {"windows_s": (6., 6.)}, {"windows_s": (-1.,)},
        ):
            with self.assertRaises(ValueError):
                analyze_trace(trajectory(), **options)


if __name__ == "__main__":
    unittest.main()
