import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
root = Path('$HOME/projects/tenny-aligned-limit-ablation-20261116')
sys.path[:0] = [str(root/'native/python'), '$HOME/projects/tenny-torque200/src', str(root)]
import numpy as np
from native_eval import Evaluator
from legwheel.deployment.extract import load

out = Path(__file__).resolve().parent
start = time.monotonic()
bundle = json.loads((root/'comparison/results.json').read_text())
initial_cases = [c for c in bundle['cases'] if c['policy_id'] == 'initial-limit1']
inputs = []
observations = []
reference_actions = []
with np.load(root/'evaluations/initial-limit1/policy-0-long.npz') as raw:
    for world, case in enumerate(initial_cases):
        trace = case['trace']
        count = len(trace['t_s']) - 1
        t = np.asarray(trace['t_s'][:-1])
        p = np.asarray(trace['base_position_m'][:-1])
        v = np.asarray(trace['base_velocity_m_s'][:-1])
        pitch = np.asarray(trace['pitch_rad'][:-1])
        mask = (t >= 1.) & (np.abs(pitch-trace['metadata']['reference_pitch_rad']) <= np.deg2rad(10))
        mask &= (np.linalg.norm(p[:,:2]-p[0,:2],axis=1) <= .05) & (np.linalg.norm(v[:,:2],axis=1) <= .1)
        observations.append(raw[f'world_{world}_observation'][:count][mask])
        reference_actions.append(raw[f'world_{world}_action'][:count][mask])
        inputs.append(dict(initial_state_id=case['initial_state_id'], selected_samples=int(mask.sum())))
obs = np.concatenate(observations)
actions = np.concatenate(reference_actions)
initial_descriptor = json.loads((root/'inputs/initial-limit1.json').read_text())
base = np.asarray(Evaluator(initial_descriptor).policy_actions(obs))
assert np.max(np.abs(base-actions)) < 1e-10
manifest = json.loads((root/'inputs/manifest-limit1.json').read_text())
normalizer, actor, critic = load(manifest['initial_checkpoint'])
def leaves(value, prefix=''):
    if isinstance(value, dict):
        return [item for name, child in value.items() for item in leaves(child,prefix+'/'+str(name))]
    return [dict(name=prefix, shape=list(np.asarray(value).shape), min=float(np.asarray(value).min()), max=float(np.asarray(value).max()))]
rows = []
scale = np.array([*manifest['actuator_contract']['policy_leg_force_scales'], .0295, .0295])
for limit in (1,2):
    for seed in range(3):
        policy_id = f'limit{limit}-seed-{seed}'
        path = root/f'inputs/evaluation-descriptors/{policy_id}.json'
        descriptor = json.loads(path.read_text())
        current = np.asarray(Evaluator(descriptor).policy_actions(obs))
        delta = current-base
        raw_delta = np.arctanh(np.clip(current,-1+1e-12,1-1e-12))-np.arctanh(np.clip(base,-1+1e-12,1-1e-12))
        rows.append(dict(policy_id=policy_id, action_delta_rms=np.sqrt(np.mean(delta**2,axis=0)).tolist(),
                         action_delta_max_abs=np.max(np.abs(delta),axis=0).tolist(),
                         common_physical_request_delta_rms=np.sqrt(np.mean((delta*scale)**2,axis=0)).tolist(),
                         raw_mean_delta_rms=np.sqrt(np.mean(raw_delta**2,axis=0)).tolist(),
                         raw_mean_delta_max_abs=np.max(np.abs(raw_delta),axis=0).tolist()))
result = dict(schema_version=1, kind='same_state_policy_drift', sample_scope='BC rollout states after 1s, meeting simultaneous pitch/XY/speed thresholds; deterministic mean only, not a causal intervention.',
              action_order=initial_descriptor['metadata'].get('action_order'), sample_count=len(obs), inputs=inputs,
              initial_native_action_parity_max_abs=float(np.max(np.abs(base-actions))),
              physical_request_units=['N','N','Nm','Nm'], actor_parameter_leaves=leaves(actor), runs=rows,
              wall_seconds=time.monotonic()-start)
(out/'policy-drift.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
