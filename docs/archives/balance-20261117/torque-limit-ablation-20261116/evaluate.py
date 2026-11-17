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

work=Path('$HOME/projects/tenny-torque-limit-ablation-20261116')
old=Path('$HOME/projects/tenny-reward-ablation-20261116')
horizon=Path('$HOME/projects/tenny-horizon-ablation-20261116')
local=Path('${REPO}/tasks/results/torque-limit-ablation-20261116')
manifest=json.loads((work/'inputs/manifest2.json').read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
bundles=[];statuses=[]
for source,identity in [(old/'evaluations/initial','initial-limit1')]+[
    (horizon/f'evaluations/h30-seed-{seed}',f'limit1-seed-{seed}') for seed in manifest['seeds']]:
    bundle=copy.deepcopy(json.loads((source/'results.json').read_text()))
    for policy in bundle['policies']:
        policy.update(id=identity,label=identity)
        policy.setdefault('source',{}).update(reused=True,leg_limit_nm=1.)
    for case in bundle['cases']:
        assert standing_metrics(case['trace'])==case['standing']
        case.update(policy_id=identity,label=identity)
    for row in bundle['parity']:row['policy_id']=identity
    bundles.append(bundle)
for row in manifest['reuse_limit1']:
    assert tree_digest(Path(row['source_run'])/'checkpoint')==row['checkpoint_sha256']
    assert sha(Path(row['source_run'])/'descriptor.json')==row['descriptor_sha256']
    statuses.append(dict(row['status'],condition='limit1',reused=True))

def evaluate(plan,identity):
    bundle=evaluate_manifest(plan,work/'evaluations'/identity,seconds=30.)
    for case in bundle['cases']:case['standing']=standing_metrics(case['trace'])
    write_json(work/'evaluations'/identity/'results.json',bundle)
    print(json.dumps(dict(event='evaluated',policy=identity,
        nominal_standing=bundle['cases'][0]['standing']['in_band_seconds_after_settling'],
        nominal_survival=bundle['cases'][0]['standing']['survival_seconds'])),flush=True)
    return bundle

bundles.append(evaluate(work/'inputs/initial-manifest-limit2.json','initial-limit2'))
seen=set()
while True:
    status_path=work/'matrix2/status.json'
    if not status_path.exists():
        time.sleep(10);continue
    state=json.loads(status_path.read_text())
    if state['phase']=='failed':raise RuntimeError(state.get('error'))
    for status in state.get('completed',[]):
        seed=status['seed']
        if seed in seen:continue
        run=work/'matrix2'/f'seed-{seed}/baseline'
        effective=json.loads((run/'effective-config.json').read_text())
        assert effective['ppo']==dict(manifest['ppo'],seed=seed)
        assert effective['env']==manifest['env_config']
        assert effective['reward']==manifest['arms']['baseline']
        assert effective['actuator_contract']==manifest['actuator_contract']
        assert status['steps']==manifest['steps_per_run']
        assert status['initial_array_digest']==manifest['initial_array_digest']
        assert sha(run/'descriptor.json')==status['descriptor_sha256']
        descriptor=json.loads((run/'descriptor.json').read_text())
        provenance=descriptor['metadata']
        identity=f'limit2-seed-{seed}'
        descriptor['metadata']={key:provenance[key] for key in ('source_fingerprint','action_order','task_schema') if key in provenance}
        descriptor['metadata'].update(kind='experimental_torque_limit_policy',experimental_only=True,
            model_run_id='torque-limit-ablation-v1/'+identity,training_steps=status['steps'],training_seed=seed,
            checkpoint_sha256=tree_digest(run/'checkpoint'),initial_array_digest=status['initial_array_digest'],
            actuator_contract=manifest['actuator_contract'],initializer_provenance=provenance)
        path=work/'inputs/evaluation-descriptors'/f'{identity}.json'
        write_json(path,descriptor)
        plan=work/'inputs/evaluation-manifests'/f'{identity}.json'
        write_json(plan,dict(schema_version=1,seed=0,policies=[dict(id=identity,label=identity,descriptor=str(path),
            source=dict(leg_limit_nm=2.,reused=False,seed=seed))],initial_states=manifest['initial_states']))
        bundles.append(evaluate(plan,identity))
        statuses.append(dict(status,condition='limit2',reused=False));seen.add(seed)
    if state['phase']=='completed':
        assert seen==set(manifest['seeds']);break
    time.sleep(10)
assert all(b['initial_states_sha256']==bundles[0]['initial_states_sha256'] for b in bundles)
combined=dict(bundles[0],cases=[c for b in bundles for c in b['cases']],
    policies=[p for b in bundles for p in b['policies']],parity=[p for b in bundles for p in b['parity']],
    run_statuses=statuses,experiment_manifest=manifest,
    scope='Three paired seeds, original reward/observation normalization fixed. ±1 reused after exact equivalence; ±2 execution/force capacity doubled. Both zero-step physical behaviors reported.')
target=work/'comparison'
write_json(target/'results.json',combined)
from experiments.torque_limit_ablation.report import write_report
print(json.dumps(write_report(combined,target)),flush=True)
shutil.copytree(target,local/'comparison',dirs_exist_ok=True)
shutil.copy2(work/'matrix2/status.json',local/'matrix-status.json')
