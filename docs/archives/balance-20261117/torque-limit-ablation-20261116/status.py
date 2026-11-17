import json
from pathlib import Path
work=Path('$HOME/projects/tenny-torque-limit-ablation-20261116')
path=work/'matrix2/status.json'
if not path.exists():
    print('Matrix not started');raise SystemExit
state=json.loads(path.read_text())
view={key:state.get(key) for key in ('phase','current_seed','elapsed_seconds','error')}
view['completed_runs']=len(state.get('completed',[]))
seed=state.get('current_seed')
if seed is not None:
    run=work/f'matrix2/seed-{seed}/baseline/status.json'
    if run.exists():view['current']=json.loads(run.read_text())
if state['phase']=='failed':
    view['log_tail']=(work/f'matrix2/seed-{seed}-baseline.log').read_text(errors='replace').splitlines()[-24:]
print(json.dumps(view,indent=2))
