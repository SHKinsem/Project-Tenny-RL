import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

os.environ.update(JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',XLA_PYTHON_CLIENT_PREALLOCATE='false')
root=Path('$HOME/projects/tenny-mature-finetune-20261116')
repo=Path('${REPO}')
sys.path[:0]=['$HOME/projects/tenny-native-eval/python','$HOME/projects/tenny-torque200/src',str(root)]
import numpy as np
from dataclasses import asdict
from force_teacher.env import student_config
from force_teacher.contract import tree_digest
from legwheel.deployment.extract import load
from native_eval import Evaluator
from native_eval.legwheel import descriptor_from_run,policy_template
from native_eval.brax_policy import policy_from_brax
from experiments.reward_ablation.train import canonical_initial_params,tree_array_digest
from experiments.reward_ablation.rewards import ARMS

read=lambda p:json.loads(Path(p).read_text())
def write(p,data):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
audit=repo/'tasks/results/training-regression-audit-20261116'
selection=read(audit/'mature-validation-summary.json')
origin=selection['source']
run=Path(origin['run_dir']); checkpoint=Path(origin['checkpoint'])
source=Path('$HOME/projects/tenny-torque200/src')
descriptor=descriptor_from_run(run,source)
params=canonical_initial_params(load(str(checkpoint)))
assert tree_digest(checkpoint)==origin['checkpoint_sha256']
cfg=json.loads(student_config().to_json_best_effort())
assert cfg==read(run/'env_config.json')
metadata=read(run/'policy_metadata.json')
ppo=read(run/'ppo_config.json')
ppo.update(num_timesteps=2097152,num_evals=9,run_evals=False,save_checkpoint_path=None,
           episode_length=3000,num_envs=512,batch_size=512,unroll_length=16,
           num_minibatches=1,num_updates_per_batch=1,clipping_epsilon=.3,num_resets_per_eval=0,
           normalize_observations=True,normalize_observations_mode='ema',normalize_observations_std_eps=0.,
           normalize_until_count=0,deterministic_eval=True,entropy_cost=.001)
ppo.pop('restore_checkpoint_path',None)
template=policy_template(metadata,ppo)
exported=dict(descriptor,policy=policy_from_brax(params[:2],template))
screen_manifest=read(audit/'historical-screen/manifest.json')
selected_index=next(i for i,item in enumerate(screen_manifest['policies']) if item['source']['checkpoint']==str(checkpoint))
with np.load(audit/f'historical-screen/evaluation/policy-{selected_index}-long.npz') as data:
    observations=data['world_0_observation'][:256]
error=float(np.max(np.abs(np.asarray(Evaluator(exported).policy_actions(observations))-np.asarray(Evaluator(descriptor).policy_actions(observations)))))
assert error<1e-5,error
inputs=root/'inputs';inputs.mkdir(parents=True,exist_ok=True)
write(inputs/'initial-descriptor.json',exported)
protected_files=[run/name for name in ['env_config.json','ppo_config.json','policy_metadata.json','policy.cpp.txt','report.json']]
protected_files += [source/name for name in ['force_teacher/env.py','force_teacher/ppo.py','force_teacher/networks.py','legwheel/env.py','legwheel/rewards.py','hybrid/env.py']]
lib=Path('$HOME/projects/tenny-reference/.venv-gpu/lib/python3.12/site-packages')
protected_files += [lib/'brax/training/agents/ppo/train.py',lib/'brax/training/agents/ppo/losses.py']
protected_files += list(Path('$HOME/projects/tenny-native-eval/python/native_eval').glob('*.so'))
protected=dict(files={str(p):sha(p) for p in protected_files},checkpoint_trees={str(checkpoint):tree_digest(checkpoint)})
states=read(audit/'mature-validation-plan.json')['initial_states']
for condition,lr in [('high_lr',3e-5),('low_lr',3e-6)]:
    settings=dict(ppo,learning_rate=lr)
    manifest=dict(schema_version=1,kind='mature_balance_finetune_v1',condition=condition,
                  seeds=[0,1,2],steps_per_run=2097152,ppo=settings,env_config=cfg,
                  arms={key:asdict(value) for key,value in ARMS.items()},reward=asdict(ARMS['position']),
                  initial_checkpoint=str(checkpoint),initial_checkpoint_sha256=origin['checkpoint_sha256'],
                  initial_array_digest=tree_array_digest(params),model_origin=origin,
                  base_descriptor=str(inputs/'initial-descriptor.json'),policy_metadata=metadata,
                  initial_states=states,protected_inputs=protected)
    write(inputs/(condition+'.json'),manifest)
smoke=read(inputs/'low_lr.json')
smoke['steps_per_run']=8192;smoke['ppo'].update(num_timesteps=8192,num_evals=2)
write(inputs/'smoke.json',smoke)
files={str(path):sha(path) for package in ['mature_balance_finetune','reward_ablation','balance_diagnostics'] for path in sorted((root/'experiments'/package).glob('*.py'))}
files.update({str(path):sha(path) for path in inputs.glob('*.json') if path.name!='runtime-freeze.json'})
write(inputs/'runtime-freeze.json',dict(files=files))
with tarfile.open(root/'executed-source.tar.gz','w:gz') as archive:
    for path in files:archive.add(path,arcname=str(Path(path).relative_to(root)))
write(root/'preparation.json',dict(initial_array_digest=smoke['initial_array_digest'],action_export_max_abs_error=error,
                                 model_origin=origin,protected_inputs=protected,runtime_files=len(files),
                                 executed_source_sha256=sha(root/'executed-source.tar.gz')))
active=Path('$HOME/projects/tenny-torque200/artifacts/console/active.json')
live=[]
for process in Path('/proc').iterdir():
    if not process.name.isdigit():continue
    try:
        command=(process/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        if 'python' in command and any(token in command for token in ('console_train.py','timed_console_train.py','experiments.mature_balance_finetune')):
            live.append(dict(pid=process.name,command=command[:300]))
    except OSError:pass
write(root/'resource-check.json',dict(active_record=read(active) if active.exists() else None,matching_processes=live))
print(json.dumps(dict(prepared=True,initial_array_digest=smoke['initial_array_digest'],action_export_max_abs_error=error,matching_processes=live)))
