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

work=Path('$HOME/projects/tenny-aligned-limit-ablation-20261116')
local=Path('${REPO}/tasks/results/aligned-limit-ablation-20261116')
manifests={limit:json.loads((work/f'inputs/manifest-limit{limit}.json').read_text()) for limit in (1,2)}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
bundles=[json.loads((work/f'evaluations/initial-limit{limit}/results.json').read_text()) for limit in (1,2)]
statuses=[];seen=set()
while True:
    path=work/'matrix/status.json'
    if not path.exists():time.sleep(10);continue
    state=json.loads(path.read_text())
    if state['phase']=='failed':raise RuntimeError(state.get('error'))
    for status in state.get('completed',[]):
        condition,seed=status['condition'],status['seed'];identity=f'{condition}-seed-{seed}'
        if identity in seen:continue
        limit=int(condition[-1]);manifest=manifests[limit]
        run=work/f'matrix/seed-{seed}/{condition}'
        effective=json.loads((run/'effective-config.json').read_text())
        assert effective['ppo']==dict(manifest['ppo'],seed=seed)
        assert effective['env']==manifest['env_config'] and effective['reward']==manifest['arms']['baseline']
        assert effective['actuator_contract']==manifest['actuator_contract']
        assert status['initial_array_digest']==manifest['initial_array_digest']
        assert status['steps']==2097152 and status['reused'] is False
        assert sha(run/'descriptor.json')==status['descriptor_sha256']
        descriptor=json.loads((run/'descriptor.json').read_text());provenance=descriptor['metadata']
        descriptor['metadata']={key:provenance[key] for key in ('source_fingerprint','action_order','task_schema') if key in provenance}
        descriptor['metadata'].update(kind='aligned_limit_policy',experimental_only=True,
            model_run_id='aligned-limit-v1/'+identity,training_steps=status['steps'],training_seed=seed,
            checkpoint_sha256=tree_digest(run/'checkpoint'),initial_array_digest=status['initial_array_digest'],
            actuator_contract=manifest['actuator_contract'],initializer_provenance=provenance)
        destination=work/f'inputs/evaluation-descriptors/{identity}.json';write_json(destination,descriptor)
        plan=work/f'inputs/evaluation-plans/{identity}.json'
        write_json(plan,dict(schema_version=1,seed=0,policies=[dict(id=identity,label=identity,
            descriptor=str(destination),source=dict(leg_limit_nm=limit,reused=False,seed=seed))],initial_states=manifest['initial_states']))
        bundle=evaluate_manifest(plan,work/'evaluations'/identity,seconds=30.)
        for case in bundle['cases']:case['standing']=standing_metrics(case['trace'])
        write_json(work/'evaluations'/identity/'results.json',bundle)
        bundles.append(bundle);statuses.append(status);seen.add(identity)
        print(json.dumps(dict(event='evaluated',policy=identity,
            nominal_standing=bundle['cases'][0]['standing']['in_band_seconds_after_settling'],
            nominal_survival=bundle['cases'][0]['standing']['survival_seconds'])),flush=True)
    if state['phase']=='completed':
        assert len(seen)==6;break
    time.sleep(10)
assert all(b['initial_states_sha256']==bundles[0]['initial_states_sha256'] for b in bundles)
combined=dict(bundles[0],cases=[c for b in bundles for c in b['cases']],policies=[p for b in bundles for p in b['policies']],
    parity=[p for b in bundles for p in b['parity']],run_statuses=statuses,experiment_manifest=manifests[1],
    alignment_validation=json.loads((work/'inputs/alignment-validation.json').read_text()),
    scope='Two fresh capacity arms, common physical requests and frozen BC mean anchor; deterministic initial physical trajectories aligned. Random over-cap clipping is the intended treatment.')
target=work/'comparison';write_json(target/'results.json',combined)
from experiments.aligned_limit_ablation.report import write_report
print(json.dumps(write_report(combined,target)),flush=True)
shutil.copytree(target,local/'comparison',dirs_exist_ok=True)
shutil.copy2(work/'matrix/status.json',local/'matrix-status.json')
