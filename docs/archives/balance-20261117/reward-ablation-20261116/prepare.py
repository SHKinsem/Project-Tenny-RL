import hashlib
import json
import os
from pathlib import Path
import sys

os.environ['JAX_PLATFORMS']='cpu'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
import mujoco
import numpy as np
from dataclasses import asdict
from legwheel.deployment.extract import load
from force_teacher.env import student_config
from force_teacher.contract import tree_digest
from experiments.reward_ablation.train import tree_array_digest,canonical_initial_params
from experiments.reward_ablation.rewards import ARMS

work=Path('$HOME/projects/tenny-reward-ablation-20261116')
old=Path('$HOME/projects/tenny-balance-diagnostics-20261116/inputs')
run=Path('$HOME/projects/tenny-torque200/artifacts/console/leg_force_bc/20261115T172533.823984Z_0a516875')
read=lambda p:json.loads(Path(p).read_text())
write=lambda p,x:Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
base=read(old/'manifest.json')
pre=read(run/'pretraining-report.json')
cache=Path(pre['bootstrap_receipt']).parent
initials=list(cache.glob('verification-*/checkpoint'))
if len(initials)!=1:
    raise ValueError('Require the unique explicitly verified initializer checkpoint')
initial=canonical_initial_params(load(str(initials[0])))
cfg=json.loads(student_config().to_json_best_effort())
if cfg!=read(run/'env_config.json'):
    raise ValueError('Deployed default standing configuration changed')
ppo=read(run/'ppo_config.json')
ppo.update(num_timesteps=2097152, num_evals=5, run_evals=False, save_checkpoint_path=None,
           num_envs=512,batch_size=512,num_minibatches=1,num_updates_per_batch=1,unroll_length=16,
           clipping_epsilon=.3, num_resets_per_eval=0)
ppo.pop('restore_checkpoint_path',None)
descriptor=read(old/'initial.json')
inputs=work/'inputs'
inputs.mkdir(parents=True,exist_ok=True)
write(inputs/'initial-descriptor.json',descriptor)
states=base['initial_states']
model=mujoco.MjModel.from_xml_path(descriptor['xml_path'])
joint=int(model.body_jntadr[model.body(descriptor['task']['config']['root_body']).id])
q=int(model.jnt_qposadr[joint])
for delta,name in ((.02,'heldout-plus'),(-.02,'heldout-minus')):
    state=json.loads(json.dumps(states[0]['state']))
    result=np.empty(4)
    mujoco.mju_mulQuat(result,np.asarray(state['qpos'][q+3:q+7]),np.array([np.cos(delta/2),0,np.sin(delta/2),0]))
    state['qpos'][q+3:q+7]=result.tolist()
    states.append(dict(id=name,label=f'预留俯仰扰动 {delta:+.2f} rad',state=state))
source=Path('$HOME/projects/tenny-torque200/src')
lib=Path('$HOME/projects/tenny-reference/.venv-gpu/lib/python3.12/site-packages')
protected=read(old/'protected-inputs.json')
for path in (source/'force_teacher/env.py',source/'force_teacher/ppo.py',source/'force_teacher/networks.py',
             source/'legwheel/env.py',source/'legwheel/rewards.py',source/'hybrid/env.py',
             lib/'brax/training/agents/ppo/train.py'):
    protected['files'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
protected['checkpoint_trees'][str(initials[0])]=tree_digest(initials[0])
manifest=dict(schema_version=1,kind='reward_ablation_v1',seeds=[0,1,2],steps_per_run=2097152,
    ppo=ppo,env_config=cfg,arms={key:asdict(value) for key,value in ARMS.items()},
    initial_checkpoint=str(initials[0]),initial_array_digest=tree_array_digest(initial),
    initial_checkpoint_sha256=protected['checkpoint_trees'][str(initials[0])],
    initialization_canonicalization='Only zero std_eps scalar is cast float64 to float32, matching Brax; all values and actor/critic/statistics are unchanged.',
    base_descriptor=str(inputs/'initial-descriptor.json'),policy_metadata=read(run/'policy_metadata.json'),
    initial_states=states,conditions=base['conditions'],protected_inputs=protected,
    order_by_seed={'0':['baseline','posture','position','both'],
                   '1':['posture','position','both','baseline'],
                   '2':['position','both','baseline','posture']},
    comparison_scope='Three paired training seeds; fixed final step budget; five deterministic physical initial conditions; no production deployment.')
if (inputs/'manifest.json').exists() and not (inputs/'manifest-before-std-eps-canonicalization.json').exists():
    (inputs/'manifest-before-std-eps-canonicalization.json').write_bytes((inputs/'manifest.json').read_bytes())
write(inputs/'manifest.json',manifest)
print(json.dumps(dict(manifest=str(inputs/'manifest.json'),initial_array_digest=manifest['initial_array_digest'],
                      steps_per_run=manifest['steps_per_run'],runs=12,initial_states=len(states))))
