import json
from pathlib import Path
root=Path('$HOME/projects/tenny-horizon-ablation-20261116/matrix30')
if not (root/'status.json').exists():
    print('Not started')
    raise SystemExit
status=json.loads((root/'status.json').read_text())
result={key:status.get(key) for key in ('phase','current_seed','elapsed_seconds','error')}
result['completed_runs']=len(status.get('completed',[]))
seed=status.get('current_seed')
path=root/f'seed-{seed}/baseline/status.json'
if seed is not None and path.exists():
    result['current']=json.loads(path.read_text())
if status['phase']=='failed':
    result['log_tail']=(root/f'seed-{seed}-baseline.log').read_text(errors='replace').splitlines()[-30:]
print(json.dumps(result,indent=2))
