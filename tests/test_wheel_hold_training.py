from dataclasses import asdict
import importlib.util
from pathlib import Path
import sys
from types import ModuleType,SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.reward_ablation.rewards import ARMS
from experiments.wheel_hold_ablation import WHEEL_HOLD
from experiments.wheel_hold_ablation.train import account_training_objective,validate_manifest


def manifest():
    return dict(condition='weak_wheel',wheel_hold=dict(WHEEL_HOLD),seeds=[0,1,2],
        steps_per_run=2097152,arms={k:asdict(v) for k,v in ARMS.items()},
        model_origin={'run_id':'mature-source'},protected_inputs={'files':{}},
        initial_states=[{'id':str(i)} for i in range(5)],
        env_config=dict(ctrl_dt=.01,sim_dt=.0025),
        ppo=dict(learning_rate=3e-5,normalize_observations=True,normalize_observations_mode='ema',
            normalize_observations_std_eps=0.,normalize_until_count=0,deterministic_eval=True,
            entropy_cost=.001,run_evals=False,episode_length=3000,action_repeat=1,
            num_timesteps=2097152,num_evals=9,batch_size=512,unroll_length=16,num_minibatches=1))


def environment_type():
    """Exercise the real adapter over a controlled reward parent without ML imports."""
    class Parent:
        def __init__(self,arm,config,*,root_body):
            self._config=config
            self.arm=arm
            self.original={'zero_keep':11.,'ablation_position':-4.,'action_penalty':-.3}
        def _get_reward(self,data,action,info,metrics):
            metrics.update({WHEEL_HOLD['metric']:.4,'reward/tracking_sinyaw':.7})
            return self.original
    stub=ModuleType('experiments.reward_ablation.legwheel_env')
    stub.RewardExperimentEnv=Parent
    path=Path(__file__).resolve().parents[1]/'experiments/wheel_hold_ablation/env.py'
    spec=importlib.util.spec_from_file_location('experiments.wheel_hold_ablation._test_env',path)
    module=importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules,{stub.__name__:stub}):
        spec.loader.exec_module(module)
    return module.WheelHoldEnv


class WheelHoldTrainingTest(unittest.TestCase):
    def test_fixed_reward_lr_and_nine_exact_callbacks(self):
        reward,steps=validate_manifest(manifest(),0)
        self.assertEqual(steps,list(range(0,2097153,262144)))
        self.assertEqual(reward['position_weight'],10.)
        self.assertEqual(reward['posture_weight'],0.)
        for key,value in [('training_weight',0.),('original_weight',1.)]:
            wrong=manifest();wrong['wheel_hold'][key]=value
            with self.assertRaises(ValueError):validate_manifest(wrong,0)
        wrong=manifest();wrong['ppo']['learning_rate']=3e-6
        with self.assertRaises(ValueError):validate_manifest(wrong,0)
        wrong=manifest();wrong['ppo']['entropy_cost']=0.
        with self.assertRaises(ValueError):validate_manifest(wrong,0)

    def test_smoke_keeps_full_episode_and_batch_boundaries(self):
        small=manifest();small['steps_per_run']=8192
        small['ppo'].update(num_timesteps=8192,num_evals=2)
        self.assertEqual(validate_manifest(small,0)[1],[0,8192])
        small['ppo']['episode_length']=600
        with self.assertRaises(ValueError):validate_manifest(small,0)

    def test_native_raw_cumulative_score_adjusts_only_training_objective(self):
        measured=dict(base_return=100.,true_position_addition_return=-10.,objective_return=90.,
                      max_xy_displacement_m=.02)
        actual=account_training_objective(measured,{WHEEL_HOLD['metric']:2.},control_dt=.01,reference_dt=.01)
        self.assertNotIn('objective_return',actual)
        self.assertEqual(actual['common_xy_objective_return'],90.)
        self.assertEqual(actual['wheel_position_adjustment_return'],-18.)
        self.assertEqual(actual['training_objective_return'],72.)
        self.assertEqual(actual['base_return'],100.)
        self.assertEqual(actual['max_xy_displacement_m'],.02)
        self.assertEqual(measured['objective_return'],90.)
        original=account_training_objective(measured,{WHEEL_HOLD['metric']:2.},
            control_dt=.01,reference_dt=.01,wheel_position_weight=10.)
        self.assertEqual(original['wheel_position_adjustment_return'],0.)
        self.assertEqual(original['training_objective_return'],90.)

    def test_accounting_uses_time_normalization_once_and_rejects_invalid_scores(self):
        measured=dict(base_return=100.,true_position_addition_return=-10.,objective_return=90.)
        actual=account_training_objective(measured,{WHEEL_HOLD['metric']:2.},control_dt=.01,reference_dt=.005)
        self.assertEqual(actual['wheel_position_adjustment_return'],-36.)
        for score in (float('nan'),float('inf'),-1.):
            with self.assertRaises(ValueError):
                account_training_objective(measured,{WHEEL_HOLD['metric']:score},control_dt=.01,reference_dt=.01)
        with self.assertRaises(KeyError):
            account_training_objective(measured,{},control_dt=.01,reference_dt=.01)

    def test_real_adapter_preserves_yaw_xy_and_raw_metrics(self):
        Env=environment_type()
        cfg=SimpleNamespace(reward_config=SimpleNamespace(zero_keep_scale=10.),wheel_sinpos_start=True,sinyaw=True)
        env=Env(cfg,wheel_position_weight=1.)
        metrics={};rewards=env._get_reward(None,None,None,metrics)
        self.assertEqual(env.arm,'position')
        self.assertAlmostEqual(rewards['zero_keep'],.4*1.+.7*10.)
        self.assertEqual(rewards['ablation_position'],-4.)
        self.assertEqual(rewards['action_penalty'],-.3)
        self.assertEqual(metrics,{WHEEL_HOLD['metric']:.4,'reward/tracking_sinyaw':.7})
        self.assertEqual(cfg.reward_config.zero_keep_scale,10.)
        self.assertEqual(env.original['zero_keep'],11.)

    def test_weight_ten_is_exact_parent_branch_and_bad_shared_weight_fails(self):
        Env=environment_type()
        cfg=SimpleNamespace(reward_config=SimpleNamespace(zero_keep_scale=10.),wheel_sinpos_start=True,sinyaw=True)
        original=Env(cfg,wheel_position_weight=10.)
        self.assertIs(original._get_reward(None,None,None,{}),original.original)
        for value in (True,-1.,11.,float('nan')):
            with self.assertRaises(ValueError):Env(cfg,wheel_position_weight=value)
        cfg.reward_config.zero_keep_scale=1.
        with self.assertRaises(ValueError):Env(cfg)
