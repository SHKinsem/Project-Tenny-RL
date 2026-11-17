import csv
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np

from experiments.balance_diagnostics.actuators import summarize_actuator_intervals
from native_eval import Evaluator

start=time.monotonic()
old=Path('$HOME/projects/tenny-reward-ablation-20261116')
new=Path('$HOME/projects/tenny-horizon-ablation-20261116')
dest=Path('$HOME/projects/tenny-torque-limit-audit-20261116')
dest.mkdir(exist_ok=True)
local=Path('${REPO}/tasks/results/torque-limit-audit-20261116')
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
descriptor_path=new/'inputs/evaluation-descriptors/h30-seed-0.json'
descriptor=read(descriptor_path)
bundle_path=new/'comparison/results.json'
bundle=read(bundle_path)
inputs={str(path):sha(path) for path in (descriptor_path,bundle_path,Path(descriptor['xml_path']))}
identities=['initial']+[f'h6-seed-{s}' for s in range(3)]+[f'h30-seed-{s}' for s in range(3)]
cases=[]
rows=[]
for identity in identities:
    source=(old/'evaluations'/('initial' if identity=='initial' else identity.replace('h6','baseline'))
            if not identity.startswith('h30') else new/'evaluations'/identity)
    arrays_path=source/'policy-0-long.npz'
    inputs[str(arrays_path)]=sha(arrays_path)
    policy_descriptor=(old/'inputs/initial-descriptor.json' if identity=='initial' else
        old/'inputs/evaluation-descriptors'/f"{identity.replace('h6','baseline')}.json"
        if identity.startswith('h6') else new/'inputs/evaluation-descriptors'/f'{identity}.json')
    inputs[str(policy_descriptor)]=sha(policy_descriptor)
    policy=read(policy_descriptor)
    assert policy['task']==descriptor['task'] and policy['xml_path']==descriptor['xml_path']
    native=Evaluator(policy)
    native_result=native.evaluate([item['state'] for item in bundle['initial_states']],
        num_envs=5,steps=3000,episode_steps=3000,control_dt=.01,seed=0,trace_worlds=list(range(5)))
    with np.load(arrays_path) as arrays:
        selected=[case for case in bundle['cases'] if case['policy_id']==identity]
        for world,case in enumerate(selected):
            length=len(case['trace']['t_s'])-1
            raw={key:arrays[f'world_{world}_{key}'][:length] for key in ('action','qpos','qvel','ctrl')}
            initial=next(item['state'] for item in bundle['initial_states'] if item['id']==case['initial_state_id'])
            replay=next(item for item in native_result['traces'] if item['world']==world)
            errors={key:float(np.max(np.abs(replay[key][:length]-raw[key]))) for key in raw}
            assert all(value==0 for value in errors.values()),errors
            assert native_result['lengths'][world]==length
            # Native metric sums max-over-substeps of the fraction of all four
            # raw requests clipped each control period. Zero proves no request
            # clipping in any physics substep; a nonzero value is not a duty ratio.
            saturation=float(native_result['episode_metrics']['hybrid/saturation_fraction'][world])
            assert saturation==0.,saturation
            commands=raw['ctrl']
            assert np.all(np.abs(commands[:,:2])<1.)
            # All commands are interior to unit-gain stateless motor limits.
            # Thus raw post-guard request and ideal actual torque equal ctrl.
            records=dict(times_s=np.arange(length+1)*.01,requested_nm=commands,
                command_nm=commands,actual_nm=commands,normalized_action=raw['action'])
            validation=dict(max_abs_errors=errors,control_steps=length,physics_steps=length*4,
                native_saturation_metric_sum=saturation,all_physics_substeps_request_unclipped=True,
                actual_source='Inferred from compiled stateless gain=gear=1 motor law; final-substep ctrl, not separately sampled qfrc.',
                sampling='Torque magnitude: control-weighted final-substep samples. Action: held for the whole control interval.',
                scope='Exact original native policy replay; no Python dynamics or callback instrumentation.')
            trace=case['trace']
            first_departure=trace['t_s'][-1]
            run_start=None
            for t,pitch,pos,vel in zip(trace['t_s'],trace['pitch_rad'],trace['base_position_m'],trace['base_velocity_m_s']):
                if t<1.:
                    continue
                outside=(abs(pitch-trace['metadata']['reference_pitch_rad'])>math.radians(10)
                    or math.hypot(pos[0]-trace['base_position_m'][0][0],pos[1]-trace['base_position_m'][0][1])>.05
                    or math.hypot(vel[0],vel[1])>.1)
                if outside:
                    if run_start is None:run_start=t
                    if t-run_start>=.2-1e-12:
                        first_departure=run_start
                        break
                else:run_start=None
            windows={}
            for label,left,right in [('full',0.,records['times_s'][-1]),
                ('settling',0.,min(1.,first_departure)),
                ('before_departure',1.,first_departure),
                ('after_departure',first_departure,records['times_s'][-1])]:
                windows[label]=summarize_actuator_intervals(records['times_s'],records['requested_nm'],
                    records['command_nm'],records['actual_nm'],records['normalized_action'],
                    descriptor['task']['config']['torque_scales'],start_s=left,end_s=right)
                for axis in (0,1):
                    a=windows[label]['actuators'][axis]
                    rows.append(dict(policy_id=identity,initial_state_id=case['initial_state_id'],window=label,
                        side=('left','right')[axis],duration_s=windows[label]['observed_seconds'],
                        max_abs_actual_nm=a['max_abs_actual_nm'],rms_actual_nm=a['actual_rms_nm'],
                        clipped_fraction=a['request_clipped']['fraction'],
                        actual_ge_90_fraction=a['actual_near_limit']['0.9']['fraction'],
                        actual_ge_99_fraction=a['actual_near_limit']['0.99']['fraction'],
                        action_ge_95_fraction=a['action_near_bound']['0.95']['fraction'],
                        action_ge_99_fraction=a['action_near_bound']['0.99']['fraction'],
                        action_ge_95_longest_s=a['action_near_bound']['0.95']['longest_continuous_seconds']))
            cases.append(dict(policy_id=identity,initial_state_id=case['initial_state_id'],
                departure_s=first_departure,validation=validation,windows=windows))
    print(json.dumps(dict(policy=identity,completed_cases=len(cases),elapsed_seconds=time.monotonic()-start)),flush=True)

assert all(sha(path)==digest for path,digest in inputs.items())
summary=dict(schema_version=1,kind='actuator_limit_audit',cases=cases,
    thresholds=dict(actual_near_limit=[.9,.95,.99],action_near_bound=[.9,.95,.99]),
    phase_definition='After settling 1s, first union-of-bands departure sustained for .2s; band: 10deg, .05m, .1m/s. Timing is descriptive, not causal.',
    scope='35 deterministic first-episode evaluation replays of saved training models, not online stochastic PPO rollout occupancy.',
    input_sha256=inputs,inputs_unchanged=True,elapsed_seconds=time.monotonic()-start)
for directory in (dest,local):
    (directory/'results.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    with (directory/'summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
print(json.dumps(dict(completed_cases=len(cases),elapsed_seconds=summary['elapsed_seconds'],
    max_replay_errors={key:max(c['validation']['max_abs_errors'][key] for c in cases) for key in ('qpos','qvel','ctrl')},
    physics_substeps=sum(c['validation']['physics_steps'] for c in cases))),flush=True)
