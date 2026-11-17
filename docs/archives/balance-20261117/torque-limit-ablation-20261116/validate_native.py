import copy
import hashlib
import json
from pathlib import Path
import time
import mujoco
import numpy as np
from native_eval import Evaluator

start=time.monotonic()
work=Path('$HOME/projects/tenny-torque-limit-ablation-20261116')
old=Path('$HOME/projects/tenny-reward-ablation-20261116')
horizon=Path('$HOME/projects/tenny-horizon-ablation-20261116')
read=lambda p:json.loads(Path(p).read_text())
manifest=read(work/'inputs/manifest2.json')
states=[item['state'] for item in manifest['initial_states']]
one,two=[read(work/f'inputs/initial-limit{limit}.json') for limit in (1,2)]
maximum={key:0. for key in ('observation','action','qpos','qvel','ctrl','reward')}
for identity,path,source in [('initial-limit1',work/'inputs/initial-limit1.json',old/'evaluations/initial')]+[
    (f'limit1-seed-{s}',horizon/f'inputs/evaluation-descriptors/h30-seed-{s}.json',horizon/f'evaluations/h30-seed-{s}') for s in (0,1,2)]:
    descriptor=read(path)
    descriptor['task']['config']['torque_normalization_scales']=[1.,1.,.0295,.0295]
    outcome=Evaluator(descriptor).evaluate(states,num_envs=5,steps=3000,episode_steps=3000,control_dt=.01,seed=0,trace_worlds=list(range(5)))
    prior=read(source/'results.json')
    with np.load(source/'policy-0-long.npz') as arrays:
        for world,case in enumerate(prior['cases']):
            length=len(case['trace']['t_s'])-1
            assert int(outcome['lengths'][world])==length
            actual=next(t for t in outcome['traces'] if t['world']==world)
            for key in maximum:
                error=float(np.max(np.abs(actual[key][:length]-arrays[f'world_{world}_{key}'][:length])))
                maximum[key]=max(maximum[key],error)
                assert error==0,(identity,world,key,error)

action=[1/32,-1/64,1/64,-1/128]
constant=dict(type='mlp',mean=[0.]*24,std=[1.]*24,
    layers=[dict(weights=np.zeros((4,24)).tolist(),bias=action,activation='identity')],
    action_size=4,output_activation='identity')
matched=[]
for index,descriptor in enumerate((one,two)):
    descriptor=copy.deepcopy(descriptor)
    descriptor['policy']=copy.deepcopy(constant)
    if index:descriptor['policy']['layers'][0]['bias'][:2]=[v/2 for v in action[:2]]
    matched.append(Evaluator(descriptor).evaluate(states[:3],num_envs=3,steps=128,episode_steps=128,
        control_dt=.01,seed=0,trace_worlds=[0,1,2]))
physical_errors={key:0. for key in ('observation','qpos','qvel','ctrl','reward')}
for left,right in zip(matched[0]['traces'],matched[1]['traces']):
    for key in physical_errors:
        error=float(np.max(np.abs(left[key]-right[key])))
        physical_errors[key]=max(physical_errors[key],error)
        assert error==0,(key,error)
compiled={}
for limit,descriptor in ((1,one),(2,two)):
    model=mujoco.MjModel.from_xml_path(descriptor['xml_path'])
    assert np.array_equal(model.actuator_ctrlrange[:2],[[-limit,limit]]*2)
    assert np.array_equal(model.actuator_forcerange[:2],[[-limit,limit]]*2)
    compiled[str(limit)]=dict(leg_ctrlrange=model.actuator_ctrlrange[:2].tolist(),
        leg_forcerange=model.actuator_forcerange[:2].tolist(),
        normalization=descriptor['task']['config']['torque_normalization_scales'],
        force_scales=descriptor['task']['config']['force_scales'])
result=dict(reused_limit1_exact_cases=20,limit1_replay_errors=maximum,
    same_physical_torque_state_reward_observation_errors=physical_errors,
    physical_comparison_control_steps=128,physical_comparison_initial_states=3,
    compiled_actuators=compiled,elapsed_seconds=time.monotonic()-start)
(work/'inputs/native-validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
