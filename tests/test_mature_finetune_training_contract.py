from dataclasses import asdict
import sys
from types import ModuleType,SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.mature_balance_finetune.train import validate_manifest,observed_learning_rate,train_with_ppo
from experiments.reward_ablation.rewards import ARMS


def manifest():
    return dict(seeds=[0,1,2],steps_per_run=2097152,arms={k:asdict(v) for k,v in ARMS.items()},
        model_origin={'run_id':'identified-mature-model'},protected_inputs={'files':{}},
        initial_states=[{'id':str(i)} for i in range(5)],env_config=dict(ctrl_dt=.01,sim_dt=.0025),
        ppo=dict(learning_rate=3e-6,normalize_observations=True,normalize_observations_mode='ema',
            normalize_observations_std_eps=0.,normalize_until_count=0,deterministic_eval=True,
            entropy_cost=.001,run_evals=False,episode_length=3000,action_repeat=1,num_timesteps=2097152,num_evals=9,
            batch_size=512,unroll_length=16,num_minibatches=1))


class MatureTrainingContractTest(unittest.TestCase):
    def test_formal_schedule_and_fixed_reward(self):
        reward,steps=validate_manifest(manifest(),0)
        self.assertEqual(steps,list(range(0,2097153,262144)))
        self.assertEqual(reward['position_weight'],10.)
        self.assertEqual(reward['posture_weight'],0.)

    def test_reject_changed_reward_or_rounded_budget(self):
        wrong=manifest();wrong['arms']['position']['position_weight']=20.
        with self.assertRaises(ValueError):validate_manifest(wrong,0)
        wrong=manifest();wrong['steps_per_run']=2097153;wrong['ppo']['num_timesteps']=2097153
        with self.assertRaises(ValueError):validate_manifest(wrong,0)

    def test_logged_rate_exposes_old_adapter_override(self):
        self.assertEqual(observed_learning_rate({'training/learning_rate':2.9999999e-6},3e-6),[2.9999999e-6])
        with self.assertRaises(ValueError):
            observed_learning_rate({'training/learning_rate':3e-5},3e-6)
        self.assertEqual(observed_learning_rate({'training/policy_loss':.1},3e-6),[])

    def test_smoke_still_requires_matched_batch_and_snapshot_boundaries(self):
        small=manifest();small['steps_per_run']=8192
        small['ppo'].update(num_timesteps=8192,num_evals=2)
        self.assertEqual(validate_manifest(small,0)[1],[0,8192])
        small['ppo']['episode_length']=600
        with self.assertRaises(ValueError):validate_manifest(small,0)

    def test_direct_brax_preserves_low_rate_full_restore_and_original_wiring(self):
        observed={}
        def brax_train(environment,eval_env,wrap_env_fn,network_factory,randomization_fn,
                       progress_fn,restore_params,policy_params_fn,deterministic_eval,
                       normalize_until_count,normalize_observations_mode,learning_rate):
            observed.update(locals())
            return 'unchanged-result'
        def network_factory():pass
        original=('normalizer','actor','critic')
        restored=object();wrapper=object();env=object();eval_env=object();progress=object();callback=object()
        modules={name:ModuleType(name) for name in (
            'brax','brax.training','brax.training.agents','brax.training.agents.ppo',
            'force_teacher','force_teacher.networks','force_teacher.ppo','mujoco_playground')}
        modules['brax.training.agents.ppo'].train=SimpleNamespace(train=brax_train)
        modules['force_teacher.networks'].make_ppo_networks=network_factory
        modules['force_teacher.ppo']._restore_fixed_mode=lambda value:restored if value==original[0] else None
        modules['mujoco_playground'].wrapper=SimpleNamespace(wrap_for_brax_training=wrapper)
        ppo=dict(learning_rate=3e-6,network_factory={'policy_hidden_layer_sizes':(32,32)},
                 policy_params_fn=callback,deterministic_eval=True,normalize_until_count=0,
                 normalize_observations_mode='ema')
        with patch.dict(sys.modules,modules):
            result=train_with_ppo(env=env,eval_env=eval_env,ppo=ppo,params=original,progress_fn=progress)
        self.assertEqual(result,'unchanged-result')
        self.assertEqual(observed['learning_rate'],3e-6)
        self.assertEqual(observed['restore_params'],(restored,original[1],original[2]))
        self.assertIs(observed['network_factory'].func,network_factory)
        self.assertEqual(observed['network_factory'].keywords,ppo['network_factory'])
        self.assertIs(observed['environment'],env)
        self.assertIs(observed['eval_env'],eval_env)
        self.assertIs(observed['wrap_env_fn'],wrapper)
        self.assertIsNone(observed['randomization_fn'])
        self.assertIs(observed['progress_fn'],progress)
        self.assertIs(observed['policy_params_fn'],callback)
        self.assertEqual(ppo['learning_rate'],3e-6)
