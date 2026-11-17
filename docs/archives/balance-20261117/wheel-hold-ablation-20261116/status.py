import json
from pathlib import Path
import shutil

root=Path('$HOME/projects/tenny-wheel-hold-ablation-20261116');local=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
result={}
if (root/'smoke/status.json').exists():
    s=read(root/'smoke/status.json');result['smoke']={key:s.get(key) for key in ['phase','steps','elapsed_seconds','observed_learning_rates','checkpoint_reload_exact','callback_steps']}
if (root/'matrix/status.json').exists():
    state=read(root/'matrix/status.json');result['matrix']={k:state.get(k) for k in ['phase','current_seed','elapsed_seconds','total_new_steps','error']}
    result['matrix']['completed']=[dict(seed=r['seed'],seconds=r['elapsed_seconds']) for r in state.get('completed',[])]
    finished=[]
    for row in state.get('completed',[]):
        records=(root/f"matrix/seed-{row['seed']}/weak_wheel/progress.jsonl").read_text().splitlines()
        cases=[c for c in json.loads(records[-1])['case_metrics'] if c['initial_state_id'] in ('nominal','pitch-plus','pitch-minus')]
        finished.append(dict(seed=row['seed'],lengths=[c['observed_episode_seconds'] for c in cases],
                             rms_xy=sum(c['rms_xy_displacement_m'] for c in cases)/3,
                             last10_xy=(sum(c['last_10s_mean_xy_displacement_m'] for c in cases)/3 if all(c['last_10s_mean_xy_displacement_m'] is not None for c in cases) else None)))
    result['completed_metrics']=finished
    if state['phase']=='training':
        path=root/f"matrix/seed-{state['current_seed']}/weak_wheel/progress.jsonl"
        if path.exists():
            p=json.loads(path.read_text().splitlines()[-1]);result['current']={k:p.get(k) for k in ['steps','elapsed_seconds','normalizer_drift']}
print(json.dumps(result));(local/'latest-status.json').write_text(json.dumps(result,indent=2)+'\n')
for name in ['gate.json','preparation.json']:
    if (root/name).exists():shutil.copy2(root/name,local/name)
(local/'inputs').mkdir(exist_ok=True)
for p in (root/'inputs').glob('*.json'):shutil.copy2(p,local/'inputs'/p.name)
