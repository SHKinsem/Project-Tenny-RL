import copy
import unittest
from experiments.horizon_ablation.contract import horizon_manifest,assert_only_horizon_changed


class HorizonContractTests(unittest.TestCase):
    def setUp(self):
        self.source={'ppo':{'episode_length':600,'discounting':.986,'unroll_length':16},
            'env_config':{'ctrl_dt':.01,'reward_config':{'termination_penalty_enable':False}},
            'seeds':[0,1,2],'initial_array_digest':'same','arms':{'baseline':{}},'steps_per_run':2097152}

    def test_only_wrapper_limit_changes_without_mutating_source(self):
        before=copy.deepcopy(self.source)
        result=horizon_manifest(self.source,30)
        self.assertEqual(self.source,before)
        self.assertEqual(result['ppo']['episode_length'],3000)
        self.assertEqual(result['env_config'],before['env_config'])
        self.assertEqual(result['initial_array_digest'],'same')
        self.assertEqual(result['order_by_seed'],{'0':['baseline'],'1':['baseline'],'2':['baseline']})

    def test_rejects_unrelated_hyperparameter_change(self):
        changed=dict(self.source['ppo'],episode_length=3000,discounting=.99)
        with self.assertRaisesRegex(ValueError,'discounting'):
            assert_only_horizon_changed(self.source['ppo'],changed)

    def test_rejects_enabled_episode_dependent_terminal_penalty(self):
        self.source['env_config']['reward_config']['termination_penalty_enable']=True
        with self.assertRaisesRegex(ValueError,'confound'):
            horizon_manifest(self.source,30)

    def test_only_declared_horizons_and_true_integer_seconds(self):
        for seconds in (True,30.,20,0):
            with self.subTest(seconds=seconds),self.assertRaises(ValueError):
                horizon_manifest(self.source,seconds)
