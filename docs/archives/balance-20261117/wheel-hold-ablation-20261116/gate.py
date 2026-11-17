"""Metamorphic reward equivalence on real saved states, without new rollouts."""
import json
import os
from pathlib import Path
import sys
import time

os.environ.update(JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',XLA_PYTHON_CLIENT_PREALLOCATE='false')
root=Path('$HOME/projects/tenny-wheel-hold-ablation-20261116')
sys.path[:0]=['$HOME/projects/tenny-native-eval/python','$HOME/projects/tenny-torque200/src',str(root)]
import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx
import numpy as np
from force_teacher.env import student_config
from experiments.reward_ablation.legwheel_env import RewardExperimentEnv
from experiments.wheel_hold_ablation.env import WheelHoldEnv

start=time.monotonic()
base=RewardExperimentEnv('position',student_config())
original=WheelHoldEnv(student_config(),wheel_position_weight=10.)
weak=WheelHoldEnv(student_config(),wheel_position_weight=1.)
assert WheelHoldEnv.step is RewardExperimentEnv.step
assert WheelHoldEnv._get_obs is RewardExperimentEnv._get_obs
assert base._config.to_json_best_effort()==original._config.to_json_best_effort()==weak._config.to_json_best_effort()
states=[env.reset(jax.random.key(42)) for env in [base,original,weak]]
for other in states[1:]:
    for left,right in [(states[0].data.qpos,other.data.qpos),(states[0].data.qvel,other.data.qvel)]:assert np.array_equal(left,right)
    for key in states[0].obs:assert np.array_equal(states[0].obs[key],other.obs[key])
model=base.mj_model;data=mujoco.MjData(model)
weight10_error=0.;adjustment_error=0.;unchanged_error=0.;nonzero=0
with np.load('$HOME/projects/tenny-mature-finetune-20261116/evaluation/policy-0-long.npz') as raw:
    for index in [0,24,49,99,199,299,499,749,999,1499,1999,2999,4999,6999,8999,11999]:
        data.qpos[:]=raw['world_0_qpos'][index];data.qvel[:]=raw['world_0_qvel'][index];data.ctrl[:]=raw['world_0_ctrl'][index]
        mujoco.mj_forward(model,data);state_data=mjx.put_data(model,data)
        rewards=[];metrics=[]
        for env,state in zip([base,original,weak],states):
            metric=dict(state.metrics);info=dict(state.info)
            if 'step' in info:info['step']=jp.asarray(index+1,dtype=jp.asarray(info['step']).dtype)
            rewards.append(env._get_reward(state_data,jp.asarray(raw['world_0_action'][index]),info,metric));metrics.append(metric)
        assert set(rewards[0])==set(rewards[1])==set(rewards[2])
        for key in rewards[0]:
            weight10_error=max(weight10_error,float(np.max(np.abs(np.asarray(rewards[0][key])-np.asarray(rewards[1][key])))))
            if key!='zero_keep':unchanged_error=max(unchanged_error,float(np.max(np.abs(np.asarray(rewards[0][key])-np.asarray(rewards[2][key])))))
        raw_score=np.asarray(metrics[0]['reward/tracking_wheels_sinposition'])
        expected=np.asarray(rewards[0]['zero_keep'])+np.float32(-9)*raw_score
        adjustment_error=max(adjustment_error,float(np.max(np.abs(expected-np.asarray(rewards[2]['zero_keep'])))))
        nonzero+=int(np.any(raw_score>0))
        for name in ['reward/tracking_wheels_sinposition','reward/tracking_sinyaw']:
            assert np.array_equal(metrics[0][name],metrics[1][name]) and np.array_equal(metrics[0][name],metrics[2][name])
assert weight10_error==0 and unchanged_error==0 and adjustment_error<1e-6 and nonzero>0
result=dict(phase='passed',scope='Identical inherited dynamics/observation methods; real saved post-state reward calls on shared data. No claim that this is a full JAX/native trajectory equivalence test.',
            reset_physics_and_actor_observation_exact=True,sampled_states=16,nonzero_wheel_scores=nonzero,
            weight10_all_reward_terms_max_abs_error=weight10_error,weight1_other_terms_max_abs_error=unchanged_error,
            weight1_single_wheel_adjustment_max_abs_error=adjustment_error,shared_zero_keep_scale=10.,yaw_weight=10.,
            actor_uses_world_xy=False,actor_uses_body_local_linear_velocity=False,wall_seconds=time.monotonic()-start)
(root/'gate.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
