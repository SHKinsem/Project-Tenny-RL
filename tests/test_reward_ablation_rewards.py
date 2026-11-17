from dataclasses import FrozenInstanceError
import math
import unittest

from experiments.reward_ablation.rewards import (
    ARMS, RewardCoefficients, extra_rewards, scalar_rewards,
)


def _map(value, function):
    return [_map(item, function) for item in value] if isinstance(value, list) else function(value)


def _combine(left, right, function):
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise ValueError('Incompatible fake backend shape')
        return [_combine(a, b, function) for a, b in zip(left, right)]
    if isinstance(left, list):
        return [_combine(item, right, function) for item in left]
    if isinstance(right, list):
        return [_combine(left, item, function) for item in right]
    return function(left, right)


class _Array:
    """Minimal vector backend to test broadcasting without importing NumPy/JAX."""
    def __init__(self, values):
        self.values = values.values if isinstance(values, _Array) else values
        self.shape = self._shape(self.values)

    @staticmethod
    def _shape(value):
        return (len(value), *_Array._shape(value[0])) if isinstance(value, list) and value else (() if not isinstance(value, list) else (0,))

    def __truediv__(self, value):
        return _Array(_combine(self.values, value, lambda a, b: a / b))

    def __pow__(self, value):
        return _Array(_combine(self.values, value, lambda a, b: a ** b))

    def __neg__(self):
        return _Array(_map(self.values, lambda a: -a))

    def __rmul__(self, value):
        return _Array(_combine(value, self.values, lambda a, b: a * b))


class _Backend:
    asarray = _Array

    @staticmethod
    def sum(array, axis):
        if axis != -1:
            raise ValueError('Only final-axis reduction is used in this test backend')
        def reduce(value):
            return [reduce(row) for row in value] if isinstance(value[0], list) else sum(value)
        return _Array(reduce(array.values))

    @staticmethod
    def expm1(array):
        return _Array(_map(array.values, math.expm1))


class RewardDefinitionTests(unittest.TestCase):
    def test_four_immutable_arms_differ_only_in_requested_weights(self):
        self.assertEqual(set(ARMS), {'baseline', 'posture', 'position', 'both'})
        with self.assertRaises(TypeError):
            ARMS['extra'] = RewardCoefficients()
        with self.assertRaises(FrozenInstanceError):
            ARMS['both'].posture_weight = 2
        for cfg in ARMS.values():
            self.assertEqual(cfg.pitch_scale_rad, math.radians(10))
            self.assertEqual(cfg.position_scale_m, .05)

    def test_zero_error_adds_no_survival_reward(self):
        for arm in ARMS:
            with self.subTest(arm=arm):
                self.assertEqual(scalar_rewards(0, [0, 0], arm),
                                 {'ablation_posture': 0., 'ablation_position': 0.})

    def test_disabled_term_stays_zero_for_nonzero_error(self):
        self.assertEqual(scalar_rewards(1, [.5, -.5], 'baseline'),
                         {'ablation_posture': 0., 'ablation_position': 0.})
        self.assertEqual(scalar_rewards(1, [.5, -.5], 'posture')['ablation_position'], 0.)
        self.assertEqual(scalar_rewards(1, [.5, -.5], 'position')['ablation_posture'], 0.)

    def test_scales_are_radians_and_metres(self):
        result = scalar_rewards(math.radians(10), [.03, .04], 'both')
        self.assertAlmostEqual(result['ablation_posture'], -4 * (1 - math.exp(-1)))
        self.assertAlmostEqual(result['ablation_position'], -10 * (1 - math.exp(-1)))

    def test_sign_and_xy_axis_symmetry(self):
        reference = scalar_rewards(.2, [.03, -.04], 'both')
        self.assertEqual(reference, scalar_rewards(-.2, [-.03, .04], 'both'))
        self.assertEqual(reference, scalar_rewards(.2, [-.04, .03], 'both'))

    def test_world_translation_does_not_change_relative_position_penalty(self):
        origin_a, point_a = [0., 0.], [.125, -.25]
        origin_b, point_b = [3., -8.], [3.125, -8.25]
        delta = lambda origin, point: [value - start for value, start in zip(point, origin)]
        self.assertEqual(scalar_rewards(.1, delta(origin_a, point_a), 'both'),
                         scalar_rewards(.1, delta(origin_b, point_b), 'both'))

    def test_terms_are_nonpositive_bounded_and_approach_declared_weights(self):
        previous = 0.
        for error in (0., .01, .1, 1., 1e150):
            result = scalar_rewards(error, [error, -error], 'both')
            self.assertTrue(-4 <= result['ablation_posture'] <= 0)
            self.assertTrue(-10 <= result['ablation_position'] <= 0)
            self.assertLessEqual(result['ablation_posture'], previous)
            previous = result['ablation_posture']
        self.assertEqual(result, {'ablation_posture': -4., 'ablation_position': -10.})

    def test_array_backend_reduces_xy_only_and_matches_scalar_reference(self):
        pitches = [0., math.radians(10), -.3]
        xy = [[0., 0.], [.03, .04], [.08, -.06]]
        actual = extra_rewards(pitches, xy, 'both', _Backend)
        for index, (pitch, delta) in enumerate(zip(pitches, xy)):
            expected = scalar_rewards(pitch, delta, 'both')
            for key in actual:
                self.assertAlmostEqual(actual[key].values[index], expected[key])

    def test_array_backend_allows_scalar_pitch_and_batched_positions(self):
        actual = extra_rewards(.2, [[0., 0.], [.03, .04]], 'both', _Backend)
        self.assertEqual(actual['ablation_posture'].shape, ())
        self.assertEqual(actual['ablation_position'].shape, (2,))
        self.assertAlmostEqual(actual['ablation_position'].values[1], -10 * (1 - math.exp(-1)))

    def test_invalid_configuration_and_scalar_inputs_fail(self):
        for kwargs in ({'posture_weight': -1}, {'position_weight': float('inf')},
                       {'pitch_scale_rad': 0}, {'pitch_scale_rad': math.pi + .1},
                       {'position_scale_m': -.1}, {'posture_weight': True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                RewardCoefficients(**kwargs)
        for pitch, xy, arm in ((float('nan'), [0, 0], 'both'),
                               (0, [0], 'both'), (0, [0, float('inf')], 'both'),
                               (0, [0, 0], 'unknown')):
            with self.subTest(pitch=pitch, xy=xy, arm=arm), self.assertRaises(ValueError):
                scalar_rewards(pitch, xy, arm)
        with self.assertRaises(ValueError):
            extra_rewards(0., [0., 0., 0.], 'both', _Backend)


if __name__ == '__main__':
    unittest.main()
