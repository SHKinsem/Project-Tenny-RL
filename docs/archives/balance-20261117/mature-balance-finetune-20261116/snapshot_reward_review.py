import json
from pathlib import Path
import sys

root=Path('$HOME/projects/tenny-mature-finetune-20261116')
sys.path.insert(0,str(root))
from experiments.balance_diagnostics.reward_accounting import verify_accounting

manifest=json.loads((root/'inputs/high_lr.json').read_text())
config=json.loads(Path(manifest['base_descriptor']).read_text())['task']['config']
rows=[]
for step in [0,786432,1048576,2097152]:
    data=json.loads((root/f'matrix/seed-1/high_lr/evaluation-step-{step}.json').read_text())
    cases=[c for c in data['cases'] if c['initial_state_id'] in ('nominal','pitch-plus','pitch-minus')]
    accounts=[verify_accounting(c['native_first_episode_metrics'],config,.01) for c in cases]
    parts={name:sum(a['components'][name] for a in accounts)/3 for name in accounts[0]['components']}
    base=sum(a['native_return'] for a in accounts)/3
    addition=sum(c['metrics']['true_position_addition_return'] for c in cases)/3
    rows.append(dict(steps=step,last10_xy_m=sum(c['metrics']['last_10s_mean_xy_displacement_m'] for c in cases)/3,
                     mean_base_return=base,mean_position_addition=addition,mean_objective_return=base+addition,
                     weighted_components=parts))
selected=next(row for row in rows if row['steps']==1048576)
for row in rows:
    row['components_minus_selected']={name:value-selected['weighted_components'][name] for name,value in row['weighted_components'].items()}
result=dict(schema_version=1,scope='Same mature-run seed1, fixed30s primary-state averages, native preweight metrics reconstructed with signed task weights. Shows objective rankings; not a gradient-cause intervention.',stages=rows)
Path(__file__).with_name('comparison').joinpath('stage-reward-review.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
