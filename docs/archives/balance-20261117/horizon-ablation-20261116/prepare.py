import hashlib
import json
import os
from pathlib import Path
import tarfile
os.environ['JAX_PLATFORMS']='cpu'
os.environ['CUDA_VISIBLE_DEVICES']=''
from experiments.horizon_ablation.contract import horizon_manifest,assert_only_horizon_changed

root=Path('$HOME/projects/tenny-horizon-ablation-20261116')
old=Path('$HOME/projects/tenny-reward-ablation-20261116')
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
write=lambda p,v:Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
source=read(old/'inputs/manifest.json')
verification=read(old/'comparison/matrix-verification.json')
archive=old/'comparison/actual-training-source.tar.gz'
if sha(archive)!=verification['frozen_training_source_sha256']:
    raise ValueError('Prior executed training archive changed')
frozen=read(old/'matrix-v2/frozen-inputs.json')
module=root/'experiments/reward_ablation'
module.mkdir(parents=True,exist_ok=True)
with tarfile.open(archive) as bundle:
    for name,digest in frozen['code_sha256'].items():
        member=bundle.getmember(name)
        if not member.isfile() or Path(name).name!=name:
            raise ValueError('Unexpected source archive member')
        data=bundle.extractfile(member).read()
        if hashlib.sha256(data).hexdigest()!=digest:
            raise ValueError('Archived source differs from executed six-second code')
        (module/name).write_bytes(data)

from force_teacher.contract import tree_digest
from force_teacher.env import student_config
from experiments.reward_ablation.legwheel_env import RewardExperimentEnv
from mujoco_playground import wrapper
from hybrid.core import source_fingerprint
manifest=horizon_manifest(source,30)
if source_fingerprint()!=source['policy_metadata']['source_fingerprint']:
    raise ValueError('Physical/task sources changed')
for path,digest in source['protected_inputs']['files'].items():
    if sha(path)!=digest:
        raise ValueError('Protected source/input changed: '+path)
reuse=[]
for seed in source['seeds']:
    run=old/'matrix-v2'/f'seed-{seed}'/'baseline'
    status=read(run/'status.json')
    effective=read(run/'effective-config.json')
    expected=dict(source['ppo'],seed=seed)
    if (status['phase']!='completed' or status['steps']!=source['steps_per_run'] or
        status['initial_array_digest']!=source['initial_array_digest'] or effective['ppo']!=expected or
        effective['env']!=source['env_config'] or effective['reward']!=source['arms']['baseline']):
        raise ValueError('Six-second comparator does not exactly match: '+str(seed))
    candidate=dict(manifest['ppo'],seed=seed)
    assert_only_horizon_changed(expected,candidate)
    reuse.append(dict(condition='h6',seed=seed,reused=True,source_run=str(run),status=status,
        effective_config_sha256=sha(run/'effective-config.json'),descriptor_sha256=sha(run/'descriptor.json'),
        checkpoint_sha256=tree_digest(run/'checkpoint')))
if json.loads(student_config().to_json_best_effort())!=source['env_config']:
    raise ValueError('Deployed standing config changed')
chains={}
for length in (600,3000):
    env=RewardExperimentEnv('baseline',student_config())
    wrapped=wrapper.wrap_for_brax_training(env,episode_length=length,action_repeat=1)
    chain=[]
    current=wrapped
    for _ in range(10):
        values=vars(current)
        chain.append(dict(type=type(current).__name__,episode_length=values.get('episode_length')))
        current=values.get('env',values.get('_env'))
        if current is None:
            break
    if length not in [item['episode_length'] for item in chain]:
        raise ValueError('Could not verify actual EpisodeWrapper length: '+str(chain))
    chains[str(length)]=chain

src=Path('$HOME/projects/tenny-torque200/src')
references={}
for relative in ('legwheel/env.py','hybrid/env.py','hybrid/timing.py','force_teacher/env.py','force_teacher/ppo.py'):
    references[relative]=[dict(line=i,text=line.strip()) for i,line in enumerate((src/relative).read_text().splitlines(),1)
                          if 'episode_length' in line or 'episode_time' in line]
lib=Path('$HOME/projects/tenny-reference/.venv-gpu/lib/python3.12/site-packages')
brax=(lib/'brax/training/agents/ppo/train.py').read_text().splitlines()
references['brax/train.py']=[dict(line=i,text=line.strip()) for i,line in enumerate(brax,1)
                            if 'episode_length=episode_length' in line]
manifest['reuse_six_seconds']=reuse
manifest['wrapper_verification']=chains
manifest['source_reference_audit']=references
manifest['frozen_source_sha256']=frozen['code_sha256']
manifest['training_time_limit_note']='Only outer Brax EpisodeWrapper changes to 3000; base 600-step field is retained and its terminal penalty branch is disabled.'
manifest['conditions']['training_episode_seconds']=30
(root/'inputs').mkdir(exist_ok=True)
write(root/'inputs/manifest30.json',manifest)
write(root/'inputs/reuse-verification.json',dict(reused=3,only_ppo_difference='episode_length',
    wrapper_chains=chains,source_references=references,old_source_sha256=frozen['code_sha256']))
print(json.dumps(dict(manifest=str(root/'inputs/manifest30.json'),reused_comparators=3,
    actual_wrapper_limits=chains,internal_episode_references=references)),flush=True)
