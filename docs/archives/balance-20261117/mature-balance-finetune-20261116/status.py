import json
from pathlib import Path
import shutil

root=Path('$HOME/projects/tenny-mature-finetune-20261116')
read=lambda p:json.loads(p.read_text())
value={}
smoke=root/'smoke/status.json'
if smoke.exists():
    s=read(smoke);value['smoke']={k:s.get(k) for k in ['phase','steps','elapsed_seconds','requested_learning_rate','observed_learning_rates','checkpoint_reload_exact']}
matrix=root/'matrix/status.json'
if matrix.exists():
    s=read(matrix);value['matrix']={k:s.get(k) for k in ['phase','current_condition','current_seed','elapsed_seconds','total_steps','error']}
    value['matrix']['completed']=[dict(condition=r['condition'],seed=r['seed'],steps=r['steps'],seconds=r['elapsed_seconds']) for r in s.get('completed',[])]
    finished=[]
    for item in s.get('completed',[]):
        path=root/f"matrix/seed-{item['seed']}/{item['condition']}/progress.jsonl"
        row=json.loads(path.read_text().splitlines()[-1])
        cases=[c for c in row['case_metrics'] if c['initial_state_id'] in ('nominal','pitch-plus','pitch-minus')]
        finished.append(dict(condition=item['condition'],seed=item['seed'],
                             primary_lengths=[c['observed_episode_seconds'] for c in cases],
                             primary_mean_rms_xy=sum(c['rms_xy_displacement_m'] for c in cases)/3,
                             primary_mean_last10_xy=(sum(c['last_10s_mean_xy_displacement_m'] for c in cases)/3 if all(c['last_10s_mean_xy_displacement_m'] is not None for c in cases) else None),
                             primary_mean_objective=sum(c['objective_return'] for c in cases)/3))
    value['completed_metrics']=finished
    if s.get('phase')=='training':
        run=root/f"matrix/seed-{s['current_seed']}/{s['current_condition']}"
        p=run/'progress.jsonl'
        if p.exists():
            row=json.loads(p.read_text().splitlines()[-1]);value['current']={k:row.get(k) for k in ['steps','elapsed_seconds','normalizer_drift']}
            value['current']['primary_cases']=[dict(initial_state_id=c['initial_state_id'],observed_episode_seconds=c['observed_episode_seconds'],last10s_xy=c['last_10s_mean_xy_displacement_m'],rms_xy=c['rms_xy_displacement_m'],objective_return=c['objective_return']) for c in row['case_metrics'][:3]]
print(json.dumps(value))
Path(__file__).with_name('latest-status.json').write_text(json.dumps(value,indent=2)+'\n')
local=Path(__file__).resolve().parent
for name in ['preparation.json','resource-check.json']:
    if (root/name).exists():shutil.copy2(root/name,local/name)
for path in (root/'inputs').glob('*.json'):
    (local/'inputs').mkdir(exist_ok=True)
    shutil.copy2(path,local/'inputs'/path.name)
