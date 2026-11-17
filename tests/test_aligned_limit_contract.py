import math
import unittest
from experiments.aligned_limit_ablation.contract import ANCHOR_RATIO,aligned_mean_scalar,aligned_contract


class AlignedContractTest(unittest.TestCase):
    def test_initial_force_requests_align_without_losing_capacity(self):
        one=aligned_contract(1.,[.1,.2]);two=aligned_contract(2.,[.1,.2])
        self.assertEqual(one['policy_leg_force_scales'],two['policy_leg_force_scales'])
        for mu in (-20.,-4.,-.1,0.,.3,3.,20.):
            a=math.tanh(aligned_mean_scalar(mu,mu))
            request=a*two['policy_leg_force_scales'][0]
            self.assertLess(abs(request),one['force_scales_n'][0])
            self.assertAlmostEqual(request,one['force_scales_n'][0]*(2*ANCHOR_RATIO)*math.tanh(mu))
        # Mutable means can move away from the frozen anchor; this is not a
        # permanent half-action multiplier that erases the high-capacity arm.
        request=math.tanh(aligned_mean_scalar(2.,0.))*two['policy_leg_force_scales'][0]
        self.assertGreater(request,one['force_scales_n'][0])
        self.assertLess(request,two['force_scales_n'][0])

    def test_original_normalization_and_wheel_limits_stay_fixed(self):
        for limit in (1.,2.):
            contract=aligned_contract(limit,[.1,.1])
            self.assertEqual(contract['torque_normalization_scales'],[1.,1.,.0295,.0295])
            self.assertEqual(contract['torque_limits_nm'][2:],[.0295,.0295])
