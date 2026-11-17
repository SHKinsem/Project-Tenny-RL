import hashlib
import json
from pathlib import Path

source=Path('$HOME/projects/tenny-torque200/src')
out=Path(__file__).resolve().parent/'deployed-source'
old=Path('$HOME/projects/tenny-mature-finetune-20261116/inputs/high_lr.json')
protected=json.loads(old.read_text())['protected_inputs']['files']
rows={}
for name in ['force_teacher/env.py','force_teacher/teacher.py','force_teacher/networks.py','legwheel/env.py','legwheel/rewards.py','hybrid/env.py']:
    path=source/name;data=path.read_bytes();digest=hashlib.sha256(data).hexdigest()
    if str(path) in protected:assert digest==protected[str(path)]
    target=out/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    rows[name]=digest
(out/'identity.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(dict(copied=list(rows),matched_prior_protection=True)))
