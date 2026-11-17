import copy
import unittest
from experiments.torque_limit_ablation.contract import actuator_contract, experimental_descriptor


class LimitContractTest(unittest.TestCase):
    def test_double_execution_and_force_keep_normalization(self):
        one=actuator_contract(1.,[.1,.2]);two=actuator_contract(2.,[.1,.2])
        self.assertEqual(two['force_scales_n'],[2*v for v in one['force_scales_n']])
        self.assertEqual(two['torque_normalization_scales'],one['torque_normalization_scales'])
        self.assertEqual(two['torque_limits_nm'][2:],one['torque_limits_nm'][2:])

    def test_descriptor_does_not_change_reward_geometry_or_policy(self):
        base=dict(task=dict(type='legwheel',config=dict(action_mode='virtual_leg_force',jmax=[.1,.2],
            torque_scales=[1.,1.,.0295,.0295],force_scales=[9.,4.5],energy_scale=-1.,action_rate_scale=-.5)),
            xml_path='one.xml',policy={'identity':1},metadata={})
        snapshot=copy.deepcopy(base)
        two=experimental_descriptor(base,2.,'two.xml')
        self.assertEqual(base,snapshot)
        self.assertEqual(two['policy'],base['policy'])
        self.assertEqual(two['task']['config']['energy_scale'],-1.)
        self.assertEqual(two['task']['config']['action_rate_scale'],-.5)
        self.assertEqual(two['task']['config']['jmax'],[.1,.2])

    def test_reject_unplanned_limits(self):
        for value in (True,0.,3.,24.,float('nan')):
            with self.assertRaises(ValueError):actuator_contract(value,[.1,.1])
