import unittest
from experiments.balance_diagnostics.actuators import summarize_actuator_intervals


class ActuatorIntervalsTest(unittest.TestCase):
    def test_clip_and_near_output_are_separate(self):
        result=summarize_actuator_intervals([0.,.1,.4],[[1.2],[.99]],[[1.],[.99]],
            [[1.],[.99]],[[.8],[1.]],[1.])['actuators'][0]
        self.assertAlmostEqual(result['request_clipped']['fraction'],.25)
        self.assertEqual(result['actual_near_limit']['0.99']['fraction'],1.)
        self.assertAlmostEqual(result['action_near_bound']['0.95']['fraction'],.75)

    def test_window_uses_overlap_and_resets_continuity(self):
        row=summarize_actuator_intervals([0.,1.,2.,3.,4.],[[0.]]*4,[[0.]]*4,
            [[1.],[0.],[1.],[1.]],[[0.]]*4,[1.],start_s=.5,end_s=3.5)['actuators'][0]
        near=row['actual_near_limit']['0.95']
        self.assertAlmostEqual(near['seconds'],2.)
        self.assertAlmostEqual(near['longest_continuous_seconds'],1.5)

    def test_empty_window_has_unknown_fraction(self):
        value=summarize_actuator_intervals([0.,1.],[[0.]],[[0.]],[[0.]],[[0.]],[1.],start_s=.5,end_s=.5)
        self.assertIsNone(value['actuators'][0]['request_clipped']['fraction'])
        self.assertIsNone(value['actuators'][0]['max_abs_actual_nm'])

    def test_reject_invalid_inputs(self):
        with self.assertRaises(ValueError):
            summarize_actuator_intervals([0.,0.],[[0.]],[[0.]],[[0.]],[[0.]],[1.])
        with self.assertRaises(ValueError):
            summarize_actuator_intervals([0.,1.],[[float('nan')]],[[0.]],[[0.]],[[0.]],[1.])
