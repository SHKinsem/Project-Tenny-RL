import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
os.environ['JAX_PLATFORMS']='cpu'
os.environ['CUDA_VISIBLE_DEVICES']=''
from hybrid.core import source_fingerprint

work=Path('$HOME/projects/tenny-reward-ablation-20261116')
local=Path('${REPO}/tasks/results/reward-ablation-20261116')
manifest=json.loads((work/'inputs/manifest.json').read_text())
matrix=json.loads((work/'matrix-v2/status.json').read_text())
if matrix['phase']!='completed' or len(matrix['completed'])!=12:
    raise ValueError('Do not archive an incomplete matrix as complete')
verified=[]
for status in matrix['completed']:
    run=work/'matrix-v2'/f'seed-{status["seed"]}'/status['arm']
    actual=json.loads((run/'effective-config.json').read_text())
    expected=dict(manifest['ppo'],seed=status['seed'])
    if actual['ppo']!=expected or actual['env']!=manifest['env_config'] or actual['reward']!=manifest['arms'][status['arm']]:
        raise ValueError('Recorded effective settings differ from frozen matrix')
    if (status['steps']!=manifest['steps_per_run'] or status['normalizer_drift']!=0 or
        status['initial_array_digest']!=manifest['initial_array_digest'] or
        status['ppo_initial_array_digest']!=manifest['initial_array_digest'] or not status['checkpoint_reload_exact']):
        raise ValueError('Parameter/step verification failed')
    verified.append(dict(arm=status['arm'],seed=status['seed'],steps=status['steps'],actor_changed=status['actor_changed']))
fingerprint=source_fingerprint()
if fingerprint!=manifest['policy_metadata']['source_fingerprint']:
    raise ValueError('Protected physical source/assets fingerprint changed')
archive=work/'comparison/actual-training-source.tar.gz'
with tarfile.open(archive,'w:gz') as output:
    for name in matrix['code_sha256']:
        path=work/'experiments/reward_ablation'/name
        if hashlib.sha256(path.read_bytes()).hexdigest()!=matrix['code_sha256'][name]:
            raise ValueError('Frozen training snapshot changed')
        output.add(path,arcname=name)
record=dict(all_12_effective_configs_exact=True,all_initial_arrays_exact=True,all_normalizers_frozen=True,
    all_checkpoints_reload_exact=True,source_fingerprint_unchanged=fingerprint,completed_runs=verified,
    total_environment_steps=sum(item['steps'] for item in verified),matrix_elapsed_seconds=matrix['elapsed_seconds'],
    frozen_training_source_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
(work/'comparison/matrix-verification.json').write_text(json.dumps(record,indent=2)+'\n')
for source,target in ((work/'comparison',local/'comparison'),(work/'inputs',local/'inputs')):
    shutil.copytree(source,target,dirs_exist_ok=True)
for status in matrix['completed']:
    arm,seed=status['arm'],status['seed']
    source=work/'matrix-v2'/f'seed-{seed}'/arm
    target=local/'run-records'/f'seed-{seed}'/arm
    target.mkdir(parents=True,exist_ok=True)
    for name in ('status.json','effective-config.json','progress.jsonl','training-metrics.jsonl'):
        shutil.copy2(source/name,target/name)
for name in ('smoke.json','array-reward-check.json'):
    shutil.copy2(work/name,local/name)
print(json.dumps(dict(verified_runs=12,total_steps=record['total_environment_steps'],
                      elapsed_seconds=record['matrix_elapsed_seconds'],archived=str(local/'comparison'))))
