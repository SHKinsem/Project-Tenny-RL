from copy import deepcopy
import math
import unittest

from experiments.reward_ablation.evaluation import standing_metrics


def trace_at(times, *, terminated=True, origin=(0., 0.), reference=.03):
    count = len(times)
    return {
        'schema_version': 1,
        't_s': list(times),
        'base_position_m': [[origin[0], origin[1], .1] for _ in times],
        'base_velocity_m_s': [[0., 0., 0.] for _ in times],
        'pitch_rad': [reference] * count,
        'roll_rad': [0.] * count,
        'pitch_rate_rad_s': [0.] * count,
        'actuator_command_nm': [[0.] for _ in times],
        'reward': [0.] * count,
        'done': [False] * (count - 1) + [True],
        'metadata': {'reference_pitch_rad': reference, 'torque_limits_nm': [1.]},
        'termination': {'terminated': terminated, 'truncated': not terminated,
                        'reason': 'pitch' if terminated else 'time_limit',
                        'time_s': times[-1]},
    }


class StandingMetricsTests(unittest.TestCase):
    def test_all_three_bands_are_jointly_required(self):
        control = trace_at([0., 1., 2., 3.])
        self.assertEqual(standing_metrics(control)['in_band_seconds_after_settling'], 2.)
        variants = []
        pitch = deepcopy(control)
        pitch['pitch_rad'][2] += math.radians(11.)
        variants.append(pitch)
        position = deepcopy(control)
        position['base_position_m'][2][0] += .051
        variants.append(position)
        speed = deepcopy(control)
        speed['base_velocity_m_s'][2][1] = .101
        variants.append(speed)
        for case in variants:
            with self.subTest(case=case):
                result = standing_metrics(case)
                self.assertEqual(result['in_band_seconds_after_settling'], 0.)

    def test_longest_segment_does_not_join_across_out_of_band_samples(self):
        case = trace_at(list(map(float, range(9))))
        case['pitch_rad'][4] += math.radians(11.)
        result = standing_metrics(case)
        self.assertEqual(result['in_band_seconds_after_settling'], 5.)
        self.assertEqual(result['longest_in_band_seconds_after_settling'], 3.)
        self.assertEqual(result['in_band_seconds_including_settling'], 6.)

    def test_initial_second_is_excluded(self):
        result = standing_metrics(trace_at([0., .25, .5, 1., 1.5, 2.]))
        self.assertEqual(result['in_band_seconds_after_settling'], 1.)
        self.assertEqual(result['longest_in_band_seconds_after_settling'], 1.)
        self.assertEqual(result['in_band_seconds_including_settling'], 2.)

    def test_straddling_settling_boundary_is_not_interpolated(self):
        result = standing_metrics(trace_at([0., .75, 1.25, 2.]))
        self.assertEqual(result['in_band_seconds_after_settling'], .75)
        self.assertEqual(result['observed_interval_seconds_after_settling'], .75)

    def test_early_death_has_no_unobserved_standing_time(self):
        result = standing_metrics(trace_at([0., .5, .9]))
        self.assertEqual(result['in_band_seconds_after_settling'], 0.)
        self.assertEqual(result['survival_seconds'], .9)
        self.assertFalse(result['reached_horizon'])

    def test_thirty_second_truncation_reaches_horizon_and_failure_does_not(self):
        for terminated, expected in ((False, True), (True, False)):
            with self.subTest(terminated=terminated):
                result = standing_metrics(trace_at([0., 1., 10., 30.], terminated=terminated))
                self.assertEqual(result['in_band_seconds_after_settling'], 29.)
                self.assertEqual(result['survival_seconds'], 30.)
                self.assertEqual(result['reached_horizon'], expected)

    def test_world_translation_and_pitch_reference_are_respected(self):
        case = trace_at([0., 1., 2.], origin=(0., 0.), reference=.2)
        case['base_position_m'][1][0] += .03125
        shifted = deepcopy(case)
        for row in shifted['base_position_m']:
            row[0] += 3.
            row[1] -= 8.
        for key in ('in_band_seconds_after_settling', 'longest_in_band_seconds_after_settling'):
            self.assertEqual(standing_metrics(case)[key], standing_metrics(shifted)[key])
        self.assertEqual(standing_metrics(shifted)['in_band_seconds_after_settling'], 1.)

    def test_limits_are_inclusive(self):
        case = trace_at([0., 1., 2.], reference=0.)
        for index in (1, 2):
            case['pitch_rad'][index] = math.radians(10.)
            case['base_position_m'][index][0] = .05
            case['base_velocity_m_s'][index][1] = .1
        self.assertEqual(standing_metrics(case)['in_band_seconds_after_settling'], 1.)

    def test_trace_reset_tail_and_invalid_parameters_fail(self):
        bad = trace_at([0., 1., 2., 3.])
        bad['done'][2] = True
        with self.assertRaises(ValueError):
            standing_metrics(bad)
        for kwargs in ({'settling_seconds': -1}, {'settling_seconds': True},
                       {'horizon_seconds': 0}, {'horizon_seconds': float('nan')},
                       {'settling_seconds': 31., 'horizon_seconds': 30.}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                standing_metrics(trace_at([0., 1., 2.]), **kwargs)

    def test_horizon_omits_straddling_interval_without_extrapolation(self):
        result = standing_metrics(trace_at([0., 1., 2., 4.]), horizon_seconds=3.)
        self.assertEqual(result['in_band_seconds_after_settling'], 1.)
        self.assertEqual(result['survival_seconds'], 3.)
        self.assertEqual(result['observed_episode_seconds'], 4.)
        self.assertTrue(result['reached_horizon'])


if __name__ == '__main__':
    unittest.main()
