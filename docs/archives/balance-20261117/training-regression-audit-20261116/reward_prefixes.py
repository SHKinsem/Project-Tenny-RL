"""Re-evaluate frozen policies with short censoring windows; never train or mutate them."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
root = Path('$HOME/projects/tenny-aligned-limit-ablation-20261116')
repo = Path('${REPO}')
sys.path[:0] = [str(root/'native/python'), '$HOME/projects/tenny-torque200/src', str(repo)]
import numpy as np
from native_eval import Evaluator
from experiments.balance_diagnostics.reward_accounting import verify_accounting

out = Path(__file__).resolve().parent
start = time.monotonic()
bundle_path = root/'comparison/results.json'
bundle = json.loads(bundle_path.read_text())
states = [item['state'] for item in bundle['initial_states']]
rows = []
full = []
errors = []
for policy in bundle['policies']:
    identity = policy['id']
    descriptor_path = root/f'inputs/{identity}.json' if identity.startswith('initial-') else root/f'inputs/evaluation-descriptors/{identity}.json'
    descriptor = json.loads(descriptor_path.read_text())
    config = descriptor['task']['config']
    dt = descriptor['eval_defaults']['control_dt']
    engine = Evaluator(descriptor)
    cases = [case for case in bundle['cases'] if case['policy_id'] == identity]
    for case in cases:
        full.append(dict(policy_id=identity, initial_state_id=case['initial_state_id'],
                         seconds=case['trace']['termination']['time_s'],
                         **verify_accounting(case['native_first_episode_metrics'],config,dt)))
    for steps in range(50, 601, 50):
        result = engine.evaluate(states, num_envs=len(states), steps=steps, episode_steps=steps,
                                 control_dt=dt, seed=0, trace_worlds=[])
        assert not result['cancelled'] and result['control_steps'] == steps*len(states)
        for world, case in enumerate(cases):
            metrics = {key: float(values[world]) for key,values in result['episode_metrics'].items()}
            target_return = sum(case['trace']['reward'][1:steps+1])
            error = abs(float(result['returns'][world])-target_return)
            assert error < 1e-6, (identity,steps,world,error)
            errors.append(error)
            rows.append(dict(policy_id=identity, initial_state_id=case['initial_state_id'],
                             prefix_seconds=steps*dt, recorded_seconds=int(result['lengths'][world])*dt,
                             **verify_accounting(metrics,config,dt), raw_metrics=metrics))
    print(json.dumps(dict(policy=identity, prefixes=12)),flush=True)
result = dict(schema_version=1, kind='native_reward_prefix_accounting',
              source_bundle_sha256=hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
              prefix_return_parity_max_abs=max(errors), scope='Censored first episodes, signed native task weights. Prefix time limits leave common dynamics unchanged; reward_scaling in PPO is not included.',
              prefixes=rows, full_episodes=full, wall_seconds=time.monotonic()-start)
(out/'reward-prefixes.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({key:value for key,value in result.items() if key not in ('prefixes','full_episodes')}))
