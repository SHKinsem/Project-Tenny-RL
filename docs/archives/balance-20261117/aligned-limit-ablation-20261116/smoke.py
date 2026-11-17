import copy
import json
from pathlib import Path
from experiments.reward_ablation.train import train_one

work=Path('$HOME/projects/tenny-aligned-limit-ablation-20261116')
manifest=copy.deepcopy(json.loads((work/'inputs/manifest-limit2.json').read_text()))
manifest['ppo'].update(num_timesteps=8192,num_evals=2)
manifest['steps_per_run']=8192
path=work/'inputs/smoke-manifest.json';path.write_text(json.dumps(manifest,indent=2)+'\n')
train_one(path,'baseline',0,work/'smoke')
