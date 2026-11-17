import copy
import json
import os
from pathlib import Path
import shutil
import sys

os.environ.update(JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',MPLBACKEND='Agg')
root=Path('$HOME/projects/tenny-wheel-hold-ablation-20261116')
repo=Path('${REPO}')
sys.path[:0]=['$HOME/projects/tenny-native-eval/python','$HOME/projects/tenny-torque200/src',str(root)]
from experiments.balance_diagnostics.runner import evaluate_manifest,write_json
from experiments.mature_balance_finetune.report import measure_case
from experiments.wheel_hold_ablation.train import account_training_objective
from experiments.wheel_hold_ablation.report import write_report

read=lambda p:json.loads(p.read_text())
manifest=read(root/'inputs/manifest.json');state=read(root/'matrix/status.json')
assert state['phase']=='completed' and len(state['completed'])==3
policies=[dict(id='initial',label='Mature reference, no update',descriptor=manifest['base_descriptor'],
               source=dict(condition='initial',seed=None,learning_rate=None,wheel_position_weight=10.,reused=False))]
statuses=[];progress={}
for baseline in manifest['baseline_runs']:
    seed=baseline['seed'];identity=f'original_wheel-seed-{seed}';run=Path(baseline['run_dir'])
    policies.append(dict(id=identity,label=f'Wheel weight 10 / seed {seed}',descriptor=baseline['descriptor'],
                         source=dict(condition='original_wheel',seed=seed,learning_rate=3e-5,wheel_position_weight=10.,reused=True,
                                     source_run=str(run),source_checkpoint=baseline['checkpoint'],original_elapsed_seconds=baseline['status']['elapsed_seconds'])))
    statuses.append(baseline['status'])
    progress[identity]=[json.loads(line) for line in (run/'progress.jsonl').read_text().splitlines() if line.strip()]
for seed in [0,1,2]:
    run=root/f'matrix/seed-{seed}/weak_wheel';status=read(run/'status.json');identity=f'weak_wheel-seed-{seed}'
    assert status['phase']=='completed' and status['steps']==2097152
    policies.append(dict(id=identity,label=f'Wheel weight 1 / seed {seed}',descriptor=str(run/'descriptor.json'),
                         source=dict(condition='weak_wheel',seed=seed,learning_rate=3e-5,wheel_position_weight=1.,reused=False,source_run=str(run))))
    statuses.append(dict(status,reused=False))
    progress[identity]=[json.loads(line) for line in (run/'progress.jsonl').read_text().splitlines() if line.strip()]
plan=dict(schema_version=1,seed=0,policies=policies,initial_states=manifest['initial_states'],scope='Matched mature-model fine tuning: wheel-position weight10 reused controls vs fresh weight1, true XY reward10 and LR3e-5 unchanged. Common native evaluator retains base10; actual objectives are accounted separately.')
write_json(root/'evaluation-plan.json',plan)
bundle=evaluate_manifest(root/'evaluation-plan.json',root/'evaluation',seconds=120.)
for case in bundle['cases']:
    policy=next(p for p in policies if p['id']==case['policy_id']);weight=policy['source']['wheel_position_weight']
    case['measurement']=account_training_objective(measure_case(case['trace']),case['native_first_episode_metrics'],
        control_dt=.01,reference_dt=.01,wheel_position_weight=weight)
bundle.update(kind='wheel_hold_ablation_v1',run_statuses=statuses,experiment_manifest=manifest,progress_records=progress,
              matrix_timing=state,gate=read(root/'gate.json'))
write_json(root/'evaluation/results.json',bundle)
print(json.dumps(write_report(bundle,root/'comparison')),flush=True)
local=repo/'tasks/results/wheel-hold-ablation-20261116'
for name in ['inputs','comparison']:shutil.copytree(root/name,local/name,dirs_exist_ok=True)
shutil.copy2(root/'evaluation/results.json',local/'comparison/results.json')
for name in ['gate.json','preparation.json']:shutil.copy2(root/name,local/name)
shutil.copy2(root/'matrix/status.json',local/'matrix-status.json')
for seed in [0,1,2]:
    run=root/f'matrix/seed-{seed}/weak_wheel';dest=local/f'runs/weak_wheel-seed-{seed}';dest.mkdir(parents=True,exist_ok=True)
    for name in ['status.json','effective-config.json','progress.jsonl','training-metrics.jsonl']:shutil.copy2(run/name,dest/name)
print(json.dumps(dict(event='evaluated',cases=35)),flush=True)
