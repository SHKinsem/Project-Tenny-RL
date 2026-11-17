import json
from pathlib import Path
work=Path('$HOME/projects/tenny-aligned-limit-ablation-20261116')
path=work/'matrix/status.json'
if not path.exists():print('Not started');raise SystemExit
state=json.loads(path.read_text())
view={key:state.get(key) for key in ('phase','current_seed','current_condition','elapsed_seconds','error')}
view['completed_runs']=len(state.get('completed',[]))
if state['phase']=='training':
    run=work/f"matrix/seed-{state['current_seed']}/{state['current_condition']}"
    if (run/'status.json').exists():view['current']=json.loads((run/'status.json').read_text())
if state['phase']=='failed':
    log=work/f"matrix/seed-{state['current_seed']}-{state['current_condition']}.log"
    view['log_tail']=log.read_text(errors='replace').splitlines()[-24:]
print(json.dumps(view,indent=2))
