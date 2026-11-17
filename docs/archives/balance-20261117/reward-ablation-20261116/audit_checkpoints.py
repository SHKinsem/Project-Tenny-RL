import importlib.util
import json
import os
from pathlib import Path
import shutil
os.environ['JAX_PLATFORMS']='cpu'
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
from legwheel.deployment.extract import load

root=Path('$HOME/projects/tenny-reward-ablation-20261116')
repo=Path('${REPO}')
spec=importlib.util.spec_from_file_location('guard_audit',repo/'experiments/reward_ablation/train.py')
guard=importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)
try:
    guard.tree_array_digest(np.asarray([float('nan')]))
except ValueError:
    rejected=True
else:
    raise ValueError('Nonfinite parameter guard did not reject a NaN')
matrix=json.loads((root/'matrix-v2/status.json').read_text())
for row in matrix['completed']:
    params=load(str(root/'matrix-v2'/f'seed-{row["seed"]}'/row['arm']/'checkpoint'))
    guard.tree_array_digest(params)
result=dict(all_12_actor_critic_normalizer_trees_finite=True,nonfinite_parameter_guard_checked=rejected,
            scope='CPU reads of completed experimental checkpoints; no training or model mutation.')
(root/'comparison/checkpoint-finiteness.json').write_text(json.dumps(result,indent=2)+'\n')
shutil.copy2(root/'comparison/checkpoint-finiteness.json',repo/'tasks/results/reward-ablation-20261116/comparison/checkpoint-finiteness.json')
print(json.dumps(result))
