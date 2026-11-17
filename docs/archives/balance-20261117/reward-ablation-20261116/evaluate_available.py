"""Evaluate each completed final policy once; merge only a complete matrix."""
import hashlib
import json
import os
from pathlib import Path
import time
os.environ['JAX_PLATFORMS']='cpu'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

from force_teacher.contract import tree_digest
from experiments.balance_diagnostics.runner import evaluate_manifest,write_json
from experiments.reward_ablation.evaluation import standing_metrics
from experiments.reward_ablation.rewards import scalar_rewards

work=Path('$HOME/projects/tenny-reward-ablation-20261116')
manifest=json.loads((work/'inputs/manifest.json').read_text())
matrix=work/'matrix-v2'
evaluations=work/'evaluations'
evaluations.mkdir(exist_ok=True)
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
ids=['initial']+[f'{arm}-seed-{seed}' for seed in manifest['seeds'] for arm in manifest['arms']]

def evaluate_one(identity,status=None):
    target=evaluations/identity
    if (target/'results.json').exists():
        return
    if identity=='initial':
        descriptor=Path(manifest['base_descriptor'])
        source=dict(kind='untrained_bc_reference',initial_array_digest=manifest['initial_array_digest'])
    else:
        arm,seed=status['arm'],status['seed']
        run=matrix/f'seed-{seed}'/arm
        original=run/'descriptor.json'
        if sha(original)!=status['descriptor_sha256']:
            raise ValueError('Completed descriptor changed: '+identity)
        value=json.loads(original.read_text())
        old=value['metadata']
        provenance={key:old.get(key) for key in ('model_run_id','export_sha256','initializer_params_sha256','kind')}
        value['metadata']={key:old[key] for key in ('source_fingerprint','action_order','task_schema') if key in old}
        value['metadata'].update(kind='saved_reward_ablation_policy',experiment_kind='reward_ablation_v1',
            model_run_id='reward-ablation-v1/'+identity,arm=arm,training_seed=seed,
            training_steps=status['steps'],checkpoint_sha256=tree_digest(run/'checkpoint'),
            initial_array_digest=status['initial_array_digest'],initializer_provenance=provenance,
            reward_additions=manifest['arms'][arm])
        descriptor=work/'inputs/evaluation-descriptors'/f'{identity}.json'
        write_json(descriptor,value)
        source=dict(kind='completed_final_checkpoint',arm=arm,seed=seed,steps=status['steps'],
            original_training_descriptor_sha256=sha(original),checkpoint_sha256=value['metadata']['checkpoint_sha256'])
    local_manifest=work/'inputs/evaluation-manifests'/f'{identity}.json'
    write_json(local_manifest,dict(schema_version=1,seed=0,policies=[dict(id=identity,label=identity,
        descriptor=str(descriptor),source=source)],initial_states=manifest['initial_states'],
        scope='Fixed final policy; common unchanged task and five predefined initial conditions.',
        conditions=manifest['conditions']))
    bundle=evaluate_manifest(local_manifest,target,seconds=30.)
    for case in bundle['cases']:
        case['standing']=standing_metrics(case['trace'])
        if status is not None:
            trace=case['trace']
            origin=trace['base_position_m'][0]
            reference=trace['metadata']['reference_pitch_rad']
            rewards=[]
            for i in range(1,len(trace['t_s'])):
                terms=scalar_rewards(trace['pitch_rad'][i]-reference,
                    [trace['base_position_m'][i][j]-origin[j] for j in (0,1)],status['arm'])
                rewards.append(trace['reward'][i]+sum(terms.values()))
            case['reward_diagnostics']=dict(scope='Reconstructed training objective on deterministic final-policy evaluation, not sampled training rollouts.',
                negative_step_fraction=sum(value<0 for value in rewards)/len(rewards),
                mean_step_reward=sum(rewards)/len(rewards))
    write_json(target/'results.json',bundle)
    print(json.dumps(dict(event='evaluated',policy=identity,
        nominal_standing=bundle['cases'][0]['standing']['in_band_seconds_after_settling'],
        nominal_survival=bundle['cases'][0]['standing']['survival_seconds'])),flush=True)

evaluate_one('initial')
while True:
    state=json.loads((matrix/'status.json').read_text())
    if state['phase']=='failed':
        raise RuntimeError(state.get('error'))
    for status in state.get('completed',[]):
        if status['steps']!=manifest['steps_per_run'] or status['initial_array_digest']!=manifest['initial_array_digest']:
            raise ValueError('Unequal training budget or initializer')
        evaluate_one(f'{status["arm"]}-seed-{status["seed"]}',status)
    if state['phase']=='completed':
        if len(state['completed'])!=12:
            raise ValueError('Incomplete matrix')
        break
    time.sleep(10)

bundles=[json.loads((evaluations/identity/'results.json').read_text()) for identity in ids]
reference=bundles[0]
if any(bundle['initial_states_sha256']!=reference['initial_states_sha256'] for bundle in bundles):
    raise ValueError('Evaluation physical initial states differed')
combined=dict(reference,cases=[case for bundle in bundles for case in bundle['cases']],
    policies=[policy for bundle in bundles for policy in bundle['policies']],
    parity=[row for bundle in bundles for row in bundle['parity']],
    experiment_manifest=manifest,run_statuses=state['completed'],
    scope='Matched reward ablation: four arms, three paired training seeds, five fixed physical conditions. Evaluation under the unchanged base task.')
output=work/'comparison'
output.mkdir(exist_ok=True)
write_json(output/'results.json',combined)
protected=manifest['protected_inputs']
if {path:sha(path) for path in protected['files']}!=protected['files']:
    raise ValueError('Protected production/model/source file changed')
if {path:tree_digest(path) for path in protected['checkpoint_trees']}!=protected['checkpoint_trees']:
    raise ValueError('Protected model checkpoint tree changed')
write_json(output/'input-preservation.json',dict(files_unchanged=len(protected['files']),
    checkpoint_trees_unchanged=len(protected['checkpoint_trees'])))
from experiments.reward_ablation.report import write_ablation_report
print(json.dumps(write_ablation_report(combined,output)),flush=True)
