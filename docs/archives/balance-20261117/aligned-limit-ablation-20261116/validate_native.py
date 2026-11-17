import copy
import json
from pathlib import Path
import time
import mujoco
import numpy as np
from native_eval import Evaluator
from experiments.balance_diagnostics.runner import evaluate_manifest,write_json
from experiments.reward_ablation.evaluation import standing_metrics

started=time.monotonic()
work=Path('$HOME/projects/tenny-aligned-limit-ablation-20261116')
old=Path('$HOME/projects/tenny-reward-ablation-20261116')
read=lambda p:json.loads(Path(p).read_text())
manifest=read(work/'inputs/manifest-limit1.json')
initial=[item['state'] for item in manifest['initial_states']]
bundles=[]
for limit in (1,2):
    bundle=evaluate_manifest(work/f'inputs/initial-plan-limit{limit}.json',work/f'evaluations/initial-limit{limit}',seconds=30.)
    for case in bundle['cases']:case['standing']=standing_metrics(case['trace'])
    write_json(work/f'evaluations/initial-limit{limit}/results.json',bundle)
    bundles.append(bundle)
errors={key:0. for key in ('observation','action','qpos','qvel','ctrl','reward')}
with np.load(work/'evaluations/initial-limit1/policy-0-long.npz') as left,np.load(work/'evaluations/initial-limit2/policy-0-long.npz') as right:
    for world,(one,two) in enumerate(zip(bundles[0]['cases'],bundles[1]['cases'])):
        assert one['trace']['termination']==two['trace']['termination']
        n=len(one['trace']['t_s'])-1
        for key in errors:
            error=float(np.max(np.abs(left[f'world_{world}_{key}'][:n]-right[f'world_{world}_{key}'][:n])))
            errors[key]=max(errors[key],error);assert error==0,(world,key,error)
descriptors=[read(work/f'inputs/initial-limit{n}.json') for n in (1,2)]
with np.load(old/'evaluations/initial/policy-0-long.npz') as data:
    observations=np.concatenate([data[f'world_{w}_observation'][:600:3] for w in range(5)])
policies=[Evaluator(value) for value in descriptors]
actions=[np.asarray(engine.policy_actions(observations)) for engine in policies]
common=np.asarray(manifest['actuator_contract']['policy_leg_force_scales'])
request_error=float(np.max(np.abs(actions[0][:,:2]*common-actions[1][:,:2]*common)))
assert request_error==0
original=Evaluator(old/'inputs/initial-descriptor.json')
old_action=np.asarray(original.policy_actions(observations))
old_error=float(np.max(np.abs(2*actions[0][:,:2]-old_action[:,:2])))
assert old_error<2e-6
assert float(np.max(np.abs(actions[0][:,:2])))<.5
assert np.array_equal(actions[0][:,2:],old_action[:,2:])

probe=[]
for descriptor in descriptors:
    descriptor=copy.deepcopy(descriptor)
    descriptor['policy']=dict(type='mlp',mean=[0.]*24,std=[1.]*24,
        layers=[dict(weights=np.zeros((4,24)).tolist(),bias=[.95,.95,0.,0.],activation='identity')],
        action_size=4,output_activation='identity')
    response=Evaluator(descriptor).evaluate(initial[:1],num_envs=1,steps=1,episode_steps=1,control_dt=.01,trace_worlds=[0])
    probe.append(response['traces'][0]['ctrl'][0,:2].tolist())
assert max(map(abs,probe[0]))<1. and min(map(abs,probe[1]))>1.
compiled={}
for limit,descriptor in zip((1,2),descriptors):
    model=mujoco.MjModel.from_xml_path(descriptor['xml_path'])
    assert np.array_equal(model.actuator_ctrlrange[:2],[[-limit,limit]]*2)
    assert np.array_equal(model.actuator_forcerange[:2],[[-limit,limit]]*2)
    compiled[str(limit)]=dict(ctrlrange=model.actuator_ctrlrange[:2].tolist(),
        forcerange=model.actuator_forcerange[:2].tolist(),capacity=descriptor['task']['config']['force_scales'],
        common_request_scale=descriptor['task']['config']['policy_leg_force_scales'])
result=dict(deterministic_pair_max_abs_errors=errors,common_request_max_abs_error=request_error,
    deterministic_physical_alignment_verified=True,random_common_request_verified=False,
    old_BC_relative_force_reference_max_abs_error=old_error,
    wider_capacity_reachable_probe_nm=probe,compiled_actuators=compiled,
    zero_step_survival_s=[c['standing']['survival_seconds'] for c in bundles[0]['cases']],
    zero_step_primary_in_band_seconds=sum(c['standing']['in_band_seconds_after_settling'] for c in bundles[0]['cases'][:3])/3,
    elapsed_seconds=time.monotonic()-started)
write_json(work/'inputs/native-alignment.json',result)
print(json.dumps(result,indent=2),flush=True)
