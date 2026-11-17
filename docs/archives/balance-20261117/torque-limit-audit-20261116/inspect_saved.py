import json
from pathlib import Path
import numpy as np

old = Path('$HOME/projects/tenny-reward-ablation-20261116')
new = Path('$HOME/projects/tenny-horizon-ablation-20261116')
manifest = json.loads((new/'inputs/manifest30.json').read_text())
descriptor = json.loads((new/'inputs/evaluation-descriptors/h30-seed-0.json').read_text())
data = np.load(new/'evaluations/h30-seed-0/policy-0-long.npz')
print(json.dumps(dict(keys={k:list(data[k].shape) for k in data.files if k.startswith('world_0')},
    task=descriptor['task']['config'],xml_path=descriptor['xml_path'],physics_dt=descriptor.get('physics_dt'),
    initial_keys=list(manifest['initial_states'][0]['state'])),indent=2))
for identity in ['initial']+[f'h6-seed-{s}' for s in range(3)]+[f'h30-seed-{s}' for s in range(3)]:
    source=(old/'evaluations'/('initial' if identity=='initial' else identity.replace('h6','baseline'))
            if not identity.startswith('h30') else new/'evaluations'/identity)
    bundle=json.loads((source/'results.json').read_text())
    arrays=np.load(source/'policy-0-long.npz')
    length=[len(case['trace']['t_s'])-1 for case in bundle['cases']]
    ctrl=np.concatenate([arrays[f'world_{w}_ctrl'][:n,:2] for w,n in enumerate(length)])
    action=np.concatenate([arrays[f'world_{w}_action'][:n,:2] for w,n in enumerate(length)])
    print(json.dumps(dict(policy=identity,steps=sum(length),max_leg_command_nm=np.max(np.abs(ctrl),axis=0).tolist(),
        ctrl_ge_99=np.mean(np.abs(ctrl)>=.99,axis=0).tolist(),action_max=np.max(np.abs(action),axis=0).tolist(),
        action_ge_95=np.mean(np.abs(action)>=.95,axis=0).tolist())),flush=True)
