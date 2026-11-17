import json
from pathlib import Path
import numpy as np

root = Path('$HOME/projects/tenny-aligned-limit-ablation-20261116')
out = []
for folder in sorted((root / 'evaluations').iterdir()):
    path = folder / 'policy-0-long.npz'
    if not path.exists():
        continue
    with np.load(path) as data:
        out.append(dict(id=folder.name, arrays={key: list(data[key].shape) for key in data.files if key.startswith('world_0_')}))
print(json.dumps(out))
descriptor = json.loads((root / 'inputs/initial-limit1.json').read_text())
print(json.dumps(dict(xml_path=descriptor['xml_path'], policy_keys=list(descriptor['policy']), task_keys=list(descriptor['task']['config']), initial_state_keys=list(json.loads((root/'inputs/manifest-limit1.json').read_text())['initial_states'][0]['state']))))
