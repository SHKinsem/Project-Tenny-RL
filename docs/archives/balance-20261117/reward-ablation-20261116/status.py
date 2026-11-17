import json
from pathlib import Path

root=Path('$HOME/projects/tenny-reward-ablation-20261116/matrix-v2')
status_file=root/'status.json'
if not status_file.exists():
    print('Matrix has not started')
    raise SystemExit
state=json.loads(status_file.read_text())
result={key:state.get(key) for key in ('phase','current_arm','current_seed','elapsed_seconds','error')}
result['completed_runs']=len(state.get('completed',[]))
arm,seed=state.get('current_arm'),state.get('current_seed')
if arm is not None:
    current=root/f'seed-{seed}'/arm/'status.json'
    if current.exists():
        result['current']=json.loads(current.read_text())
    log=root/f'seed-{seed}-{arm}.log'
    if state.get('phase')=='failed' and log.exists():
        result['log_tail']=log.read_text(errors='replace').splitlines()[-35:]
print(json.dumps(result,indent=2))
