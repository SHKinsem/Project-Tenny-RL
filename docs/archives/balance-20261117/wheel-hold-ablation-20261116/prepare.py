import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile

os.environ.update(JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',XLA_PYTHON_CLIENT_PREALLOCATE='false')
root=Path('$HOME/projects/tenny-wheel-hold-ablation-20261116')
old=Path('$HOME/projects/tenny-mature-finetune-20261116')
sys.path[:0]=['$HOME/projects/tenny-native-eval/python','$HOME/projects/tenny-torque200/src',str(root)]
from force_teacher.contract import tree_digest
from experiments.wheel_hold_ablation import WHEEL_HOLD

read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,data):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')

manifest=read(old/'inputs/high_lr.json')
manifest.update(kind='wheel_hold_ablation_v1',condition='weak_wheel',wheel_hold=dict(WHEEL_HOLD))
inputs=root/'inputs';inputs.mkdir(parents=True,exist_ok=True)
source_descriptor=Path(manifest['base_descriptor'])
shutil.copy2(source_descriptor,inputs/'initial-descriptor.json')
manifest['base_descriptor']=str(inputs/'initial-descriptor.json')
baseline=[]
for seed in [0,1,2]:
    run=old/f'matrix/seed-{seed}/high_lr'
    status=read(run/'status.json');effective=read(run/'effective-config.json')
    assert status['phase']=='completed' and status['steps']==manifest['steps_per_run']
    assert status['initial_array_digest']==manifest['initial_array_digest']==status['ppo_initial_array_digest']
    assert status['requested_learning_rate']==3e-5 and effective['ppo']==dict(manifest['ppo'],seed=seed)
    assert effective['reward']==manifest['reward'] and effective['env']==manifest['env_config']
    assert sha(run/'descriptor.json')==status['descriptor_sha256']
    checkpoint=run/'checkpoint';digest=tree_digest(checkpoint)
    baseline.append(dict(seed=seed,run_dir=str(run),descriptor=str(run/'descriptor.json'),checkpoint=str(checkpoint),
                         checkpoint_sha256=digest,status=dict(status,condition='original_wheel',reused=True,source_run=str(run))))
    manifest['protected_inputs']['checkpoint_trees'][str(checkpoint)]=digest
    for name in ['status.json','effective-config.json','progress.jsonl','training-metrics.jsonl','descriptor.json']:
        manifest['protected_inputs']['files'][str(run/name)]=sha(run/name)
    for descriptor in run.glob('descriptor-step-*.json'):
        manifest['protected_inputs']['files'][str(descriptor)]=sha(descriptor)
manifest['baseline_runs']=baseline
# Wiring and optimizer helpers must remain exactly the matched mature implementation.
def functions(path):
    return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
a=functions(root/'experiments/wheel_hold_ablation/train.py');b=functions(old/'experiments/mature_balance_finetune/train.py')
for name in ['train_with_ppo','observed_learning_rate','verify_protected_inputs','_mean_scalars']:assert a[name]==b[name],name
write(inputs/'manifest.json',manifest)
smoke=copy.deepcopy(manifest);smoke['steps_per_run']=8192;smoke['ppo'].update(num_timesteps=8192,num_evals=2)
write(inputs/'smoke.json',smoke)
files={str(p):sha(p) for package in ['wheel_hold_ablation','mature_balance_finetune','reward_ablation','balance_diagnostics'] for p in (root/'experiments'/package).glob('*.py')}
files.update({str(p):sha(p) for p in inputs.glob('*.json') if p.name!='runtime-freeze.json'})
write(inputs/'runtime-freeze.json',dict(files=files))
with tarfile.open(root/'executed-source.tar.gz','w:gz') as archive:
    for path in files:archive.add(path,arcname=str(Path(path).relative_to(root)))
processes=[]
for p in Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:
        command=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        if 'python' in command and any(v in command for v in ['console_train.py','timed_console_train.py','-m experiments.wheel_hold_ablation']):processes.append(dict(pid=p.name,command=command[:250]))
    except OSError:pass
record=dict(prepared=True,initial_array_digest=manifest['initial_array_digest'],matched_ppo_helpers=True,
            baseline_runs=baseline,source_archive_sha256=sha(root/'executed-source.tar.gz'),matching_processes=processes,runtime_files=len(files))
write(root/'preparation.json',record)
print(json.dumps(dict(prepared=True,initial_array_digest=record['initial_array_digest'],baseline_count=3,matching_processes=processes)))
