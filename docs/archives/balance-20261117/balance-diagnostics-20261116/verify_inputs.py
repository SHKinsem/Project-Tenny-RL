import hashlib
import json
import os
from pathlib import Path
import sys

os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.path.insert(0, '$HOME/projects/tenny-torque200/src')
from force_teacher.contract import tree_digest

work = Path('$HOME/projects/tenny-balance-diagnostics-20261116')
protected = json.loads((work/'inputs/protected-inputs.json').read_text())
files = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in protected['files']}
trees = {path: tree_digest(path) for path in protected['checkpoint_trees']}
if files != protected['files'] or trees != protected['checkpoint_trees']:
    raise ValueError('Saved source/model inputs changed during diagnostics')
result = dict(input_files_unchanged=len(files), checkpoint_trees_unchanged=len(trees),
              source='Exact pre/post hash comparison; no restore or overwrite performed.')
(work/'result/input-preservation.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result))
