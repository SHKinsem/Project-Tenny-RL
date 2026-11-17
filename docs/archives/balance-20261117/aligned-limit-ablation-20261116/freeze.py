import hashlib
import json
from pathlib import Path
import shutil
work=Path('$HOME/projects/tenny-aligned-limit-ablation-20261116')
repo=Path('${REPO}')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
shutil.copy2(repo/'experiments/aligned_limit_ablation/matrix.py',work/'experiments/aligned_limit_ablation/matrix.py')
files=[p for package in ('aligned_limit_ablation','torque_limit_ablation','reward_ablation')
       for p in work.joinpath('experiments',package).glob('*.py')]
files.extend(work.joinpath('native/src/cpp').glob('*.cpp'))
files.extend(work.joinpath('native/src/include').rglob('*.hpp'))
files.extend(work.joinpath('native/python/native_eval').glob('*.so'))
files.extend([work/'assets/torque-limit-2.xml',work/'inputs/anchor-bc.json',
              work/'inputs/alignment-validation.json'])
receipt={str(path):sha(path) for path in files}
target=work/'inputs/runtime-freeze.json'
if target.exists():assert json.loads(target.read_text())==receipt
else:target.write_text(json.dumps(receipt,indent=2)+'\n')
shutil.copytree(work/'inputs',repo/'tasks/results/aligned-limit-ablation-20261116/inputs',dirs_exist_ok=True)
print(json.dumps(dict(frozen_files=len(receipt))),flush=True)
