"""Independent signed reward accounting and heading/roll side diagnostics."""
import json
import math
from pathlib import Path
import sys

root=Path('$HOME/projects/tenny-mature-finetune-20261116')
sys.path.insert(0,str(root))
import numpy as np
from experiments.balance_diagnostics.reward_accounting import verify_accounting

bundle=json.loads((root/'evaluation/results.json').read_text())
manifest=bundle['experiment_manifest']
descriptor=json.loads(Path(manifest['base_descriptor']).read_text())
config=descriptor['task']['config']
out=Path(__file__).resolve().parent/'comparison'
rows=[]
for policy in bundle['policies']:
    identity=policy['id'];source=policy['source']
    run=root/'matrix/seed-0/high_lr' if identity=='initial' else root/f"matrix/seed-{source['seed']}/{source['condition']}"
    step=0 if identity=='initial' else 2097152
    evaluation=json.loads((run/f'evaluation-step-{step}.json').read_text())
    selected=[case for case in evaluation['cases'] if case['initial_state_id'] in ('nominal','pitch-plus','pitch-minus')]
    accounts=[verify_accounting(case['native_first_episode_metrics'],config,.01) for case in selected]
    components={name:sum(a['components'][name] for a in accounts)/3 for name in accounts[0]['components']}
    addition=sum(case['metrics']['true_position_addition_return'] for case in selected)/3
    base=sum(a['native_return'] for a in accounts)/3
    rows.append(dict(policy_id=identity,source=source,first_30s_primary_mean_base_return=base,
                     first_30s_primary_mean_position_addition=addition,first_30s_primary_mean_objective_return=base+addition,
                     first_30s_primary_mean_weighted_components=components,
                     native_accounting_max_abs_error=max(abs(a['error']) for a in accounts)))
baseline=rows[0]
for row in rows:
    row['component_deltas_from_initial']={name:value-baseline['first_30s_primary_mean_weighted_components'][name] for name,value in row['first_30s_primary_mean_weighted_components'].items()}
    row['base_delta_from_initial']=row['first_30s_primary_mean_base_return']-baseline['first_30s_primary_mean_base_return']
    row['position_addition_delta_from_initial']=row['first_30s_primary_mean_position_addition']-baseline['first_30s_primary_mean_position_addition']
side=[]
for index,policy in enumerate(bundle['policies']):
    with np.load(root/f'evaluation/policy-{index}-long.npz') as raw:
        for world,state in enumerate(bundle['initial_states']):
            case=next(c for c in bundle['cases'] if c['policy_id']==policy['id'] and c['initial_state_id']==state['id'])
            length=len(case['trace']['t_s'])-1
            q=np.vstack([state['state']['qpos'],raw[f'world_{world}_qpos'][:length]])[:,3:7]
            w,x,y,z=q.T
            yaw=np.unwrap(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))
            yaw=np.rad2deg(yaw-yaw[0])
            roll=np.rad2deg(case['trace']['roll_rad'])
            t=np.asarray(case['trace']['t_s'])
            side.append(dict(policy_id=policy['id'],initial_state_id=state['id'],observed_seconds=float(t[-1]),
                             heading_end_change_deg=float(yaw[-1]),heading_max_abs_change_deg=float(np.abs(yaw).max()),
                             roll_rms_deg=float(np.sqrt(np.sum(np.diff(t)*(roll[:-1]**2+roll[1:]**2)/2)/t[-1])),
                             roll_max_abs_deg=float(np.abs(roll).max())))
result=dict(schema_version=1,scope='First30s native reward partition and post-hoc heading/roll side diagnostics; heading/roll do not replace the prespecified primary drift criteria.',
            first30s_reward_partition=rows,heading_roll_cases=side)
(out/'reward-review.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(first30s=[dict(policy_id=r['policy_id'],base=r['first_30s_primary_mean_base_return'],position=r['first_30s_primary_mean_position_addition'],objective=r['first_30s_primary_mean_objective_return']) for r in rows],
                      nominal_heading=[v for v in side if v['initial_state_id']=='nominal'])))
