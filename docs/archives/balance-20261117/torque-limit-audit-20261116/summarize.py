import json
from pathlib import Path

root=Path(__file__).parent
result=json.loads((root/'results.json').read_text())
groups={}
for identity in dict.fromkeys(case['policy_id'] for case in result['cases']):
    cases=[case for case in result['cases'] if case['policy_id']==identity]
    phases={}
    for phase in ('full','before_departure','after_departure'):
        windows=[case['windows'][phase] for case in cases]
        duration=sum(window['observed_seconds'] for window in windows)
        phases[phase]=dict(observed_seconds_across_five_cases=duration,
            action_ge_95_fraction=[sum(w['actuators'][axis]['action_near_bound']['0.95']['seconds'] for w in windows)/duration if duration else None for axis in (0,1)],
            action_ge_99_fraction=[sum(w['actuators'][axis]['action_near_bound']['0.99']['seconds'] for w in windows)/duration if duration else None for axis in (0,1)],
            max_abs_actual_nm=[max(w['actuators'][axis]['max_abs_actual_nm'] for w in windows if w['actuators'][axis]['max_abs_actual_nm'] is not None) for axis in (0,1)])
    groups[identity]=phases
summary=dict(policies=groups,
    exact_replay_max_errors={key:max(c['validation']['max_abs_errors'][key] for c in result['cases']) for key in ('action','qpos','qvel','ctrl')},
    first_episode_control_steps=sum(c['validation']['control_steps'] for c in result['cases']),
    first_episode_physics_steps=sum(c['validation']['physics_steps'] for c in result['cases']),
    clipped_cases=sum(not c['validation']['all_physics_substeps_request_unclipped'] for c in result['cases']),
    source_scope=result['scope'])
summary['pooled_phases']={}
for phase in ('settling','before_departure','after_departure'):
    windows=[case['windows'][phase] for case in result['cases']]
    duration=sum(w['observed_seconds'] for w in windows)
    summary['pooled_phases'][phase]=dict(observed_seconds=duration,
        action_ge_95_fraction=[sum(w['actuators'][axis]['action_near_bound']['0.95']['seconds'] for w in windows)/duration for axis in (0,1)])
(root/'aggregate.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
for case in result['cases']:
    if case['initial_state_id']=='nominal':
        print(json.dumps(dict(policy=case['policy_id'],departure_s=case['departure_s'],
            before95=[a['action_near_bound']['0.95'] for a in case['windows']['before_departure']['actuators'][:2]],
            after95=[a['action_near_bound']['0.95'] for a in case['windows']['after_departure']['actuators'][:2]])))
