import json
from pathlib import Path
root=Path('/home/jaccob/projects/tenny-torque200/artifacts/console')
a=json.loads((root/'active.json').read_text())
r=Path(a['run_dir'])
s=json.loads((r/'status.json').read_text())
print(json.dumps({k:s.get(k) for k in ('run_id','mode','phase','stage','steps','elapsed_seconds','error','checkpoint','stop_reason','restored_from')}))
print((r/'train.log').read_text(errors='replace')[-1800:])
