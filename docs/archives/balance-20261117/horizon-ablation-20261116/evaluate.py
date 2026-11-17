import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
os.environ['JAX_PLATFORMS']='cpu'
os.environ['CUDA_VISIBLE_DEVICES']=''
from force_teacher.contract import tree_digest
from experiments.balance_diagnostics.runner import evaluate_manifest,write_json
from experiments.reward_ablation.evaluation import standing_metrics

work=Path('$HOME/projects/tenny-horizon-ablation-20261116')
old=Path('$HOME/projects/tenny-reward-ablation-20261116')
manifest=json.loads((work/'inputs/manifest30.json').read_text())
root=work/'evaluations'
root.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
bundles=[]
statuses=[]

# Reuse immutable physical trajectories; change display identities, not data.
for old_id,new_id in [('initial','initial')]+[(f'baseline-seed-{s}',f'h6-seed-{s}') for s in manifest['seeds']]:
    bundle=copy.deepcopy(json.loads((old/'evaluations'/old_id/'results.json').read_text()))
    for policy in bundle['policies']:
        policy['id']=new_id
        policy['label']=new_id
        policy.setdefault('source',{}).update(reused=True,original_policy_id=old_id,training_episode_seconds=6)
    for case in bundle['cases']:
        case['policy_id']=new_id
        case['label']=new_id
        if standing_metrics(case['trace'])!=case['standing']:
            raise ValueError('Shared standing metric or reused trace changed')
    for row in bundle['parity']:
        row['policy_id']=new_id
    bundles.append(bundle)
for row in manifest['reuse_six_seconds']:
    if tree_digest(Path(row['source_run'])/'checkpoint')!=row['checkpoint_sha256']:
        raise ValueError('Reused six-second checkpoint changed')
    statuses.append(dict(row['status'],condition='h6',reused=True))

seen=set()
while True:
    state=json.loads((work/'matrix30/status.json').read_text())
    if state['phase']=='failed':
        raise RuntimeError(state.get('error'))
    for status in state.get('completed',[]):
        seed=status['seed']
        if seed in seen:
            continue
        run=work/'matrix30'/f'seed-{seed}'/'baseline'
        expected=dict(manifest['ppo'],seed=seed)
        effective=json.loads((run/'effective-config.json').read_text())
        if (effective['ppo']!=expected or effective['env']!=manifest['env_config'] or
            effective['reward']!=manifest['arms']['baseline'] or status['steps']!=manifest['steps_per_run'] or
            status['initial_array_digest']!=manifest['initial_array_digest']):
            raise ValueError('Thirty-second run differs from frozen matched settings')
        if sha(run/'descriptor.json')!=status['descriptor_sha256']:
            raise ValueError('Completed policy export changed')
        descriptor=json.loads((run/'descriptor.json').read_text())
        provenance=descriptor['metadata']
        descriptor['metadata']={key:provenance[key] for key in ('source_fingerprint','action_order','task_schema') if key in provenance}
        identity=f'h30-seed-{seed}'
        descriptor['metadata'].update(kind='saved_training_horizon_policy',model_run_id='horizon-ablation-v1/'+identity,
            training_episode_seconds=30,training_steps=status['steps'],training_seed=seed,
            checkpoint_sha256=tree_digest(run/'checkpoint'),initial_array_digest=status['initial_array_digest'],
            initializer_provenance={key:provenance.get(key) for key in ('model_run_id','export_sha256','initializer_params_sha256','kind')})
        path=work/'inputs/evaluation-descriptors'/f'{identity}.json'
        write_json(path,descriptor)
        plan=work/'inputs/evaluation-manifests'/f'{identity}.json'
        write_json(plan,dict(schema_version=1,seed=0,policies=[dict(id=identity,label=identity,descriptor=str(path),
            source=dict(training_episode_seconds=30,reused=False,seed=seed))],initial_states=manifest['initial_states'],
            scope='Matched horizon comparison under common original reward and fixed physical conditions.'))
        bundle=evaluate_manifest(plan,root/identity,seconds=30.)
        for case in bundle['cases']:
            case['standing']=standing_metrics(case['trace'])
        write_json(root/identity/'results.json',bundle)
        bundles.append(bundle)
        statuses.append(dict(status,condition='h30',reused=False))
        seen.add(seed)
        print(json.dumps(dict(event='evaluated',seed=seed,
            nominal_survival=bundle['cases'][0]['standing']['survival_seconds'],
            nominal_standing=bundle['cases'][0]['standing']['in_band_seconds_after_settling'])),flush=True)
    if state['phase']=='completed':
        if seen!=set(manifest['seeds']):
            raise ValueError('Thirty-second group incomplete')
        break
    time.sleep(10)

if any(bundle['initial_states_sha256']!=bundles[0]['initial_states_sha256'] for bundle in bundles):
    raise ValueError('Evaluation initial states differ')
combined=dict(bundles[0],cases=[case for bundle in bundles for case in bundle['cases']],
    policies=[item for bundle in bundles for item in bundle['policies']],
    parity=[item for bundle in bundles for item in bundle['parity']],
    run_statuses=statuses,experiment_manifest=manifest,
    scope='Three paired seeds: six-second original-reward runs reused after exact matching; thirty-second training runs differ only in outer episode limit.')
target=work/'comparison'
target.mkdir(exist_ok=True)
write_json(target/'results.json',combined)
from experiments.horizon_ablation.report import write_horizon_report
print(json.dumps(write_horizon_report(combined,target)),flush=True)
repo=Path('${REPO}/tasks/results/horizon-ablation-20261116')
shutil.copytree(target,repo/'comparison',dirs_exist_ok=True)
shutil.copy2(work/'matrix30/status.json',repo/'matrix-status.json')
