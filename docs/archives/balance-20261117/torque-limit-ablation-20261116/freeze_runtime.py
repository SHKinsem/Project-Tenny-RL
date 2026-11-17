import hashlib
import json
from pathlib import Path
import shutil
work=Path('$HOME/projects/tenny-torque-limit-ablation-20261116')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
paths=[*work.joinpath('experiments/torque_limit_ablation').glob('*.py'),
    *work.joinpath('experiments/reward_ablation').glob('*.py'),
    work/'native/src/cpp/legwheel_task.cpp',work/'native/src/CMakeLists.txt',
    *work.joinpath('native/python/native_eval').glob('*.so'),
    work/'assets/torque-limit-2.xml']
receipt={str(p):sha(p) for p in paths}
path=work/'inputs/runtime-freeze.json'
if path.exists():
    assert json.loads(path.read_text())==receipt
else:path.write_text(json.dumps(receipt,indent=2)+'\n')
local=Path('${REPO}/tasks/results/torque-limit-ablation-20261116')
shutil.copytree(work/'inputs',local/'inputs',dirs_exist_ok=True)
print(json.dumps(dict(frozen_runtime_files=len(receipt))))
