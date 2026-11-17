import json
from pathlib import Path
import time
import jax
import jax.numpy as jp
import numpy as np
from force_teacher.env import StudentEnv,student_config
from experiments.reward_ablation.legwheel_env import RewardExperimentEnv
from experiments.reward_ablation.rewards import scalar_rewards,ARMS
from mujoco_playground import wrapper

start=time.monotonic()
out=Path('$HOME/projects/tenny-reward-ablation-20261116')
key=jax.random.PRNGKey(0)
action=jp.array([.1,.1,.02,-.02])
base=StudentEnv(config=student_config())
reference=jax.jit(base.step)(base.reset(key),action)
reference.data.qpos.block_until_ready()
rows=[]
for arm in ARMS:
    env=RewardExperimentEnv(arm,student_config())
    state=env.reset(key)
    nxt=jax.jit(env.step)(state,action)
    nxt.data.qpos.block_until_ready()
    errors=dict(qpos=float(np.max(np.abs(np.asarray(nxt.data.qpos)-np.asarray(reference.data.qpos)))),
                qvel=float(np.max(np.abs(np.asarray(nxt.data.qvel)-np.asarray(reference.data.qvel)))),
                observation=max(float(np.max(np.abs(np.asarray(nxt.obs[k])-np.asarray(reference.obs[k])))) for k in reference.obs),
                done=float(abs(nxt.done-reference.done)))
    q=env._experiment_root_qpos
    w,x,y,z=np.asarray(nxt.data.qpos[q+3:q+7])
    pitch=float(np.arcsin(np.clip(2*(w*y-z*x),-1,1)))-float(state.info['ablation_pitch_reference_rad'])
    terms=scalar_rewards(pitch,(np.asarray(nxt.data.qpos[q:q+2])-np.asarray(state.info['ablation_xy_origin'])).tolist(),arm)
    observed=float(nxt.reward-reference.reward)
    expected=sum(terms.values())
    if max(errors.values())>1e-6 or abs(observed-expected)>2e-5:
        raise ValueError((arm,errors,observed,expected))
    rows.append(dict(arm=arm,max_errors=errors,reward_delta=observed,expected_delta=expected))
env=RewardExperimentEnv('both',student_config())
wrapped=wrapper.wrap_for_brax_training(env,episode_length=2,action_repeat=1)
state=jax.jit(wrapped.reset)(jax.random.split(key,2))
initial_xy=np.asarray(state.info['ablation_xy_origin']).copy()
initial_pitch=np.asarray(state.info['ablation_pitch_reference_rad']).copy()
step=jax.jit(wrapped.step)
for i in range(5):
    state=step(state,jp.tile(action,(2,1)))
    if not np.array_equal(np.asarray(state.info['ablation_xy_origin']),initial_xy) or not np.array_equal(np.asarray(state.info['ablation_pitch_reference_rad']),initial_pitch):
        raise ValueError('Episode anchor followed moving state or changed across auto-reset')
report=dict(passed=True,device=[str(d) for d in jax.devices()],rows=rows,
            auto_reset_fixed_anchors=True,seconds=time.monotonic()-start)
(out/'smoke.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
