import json
import os
from pathlib import Path
import shutil
import sys
import time

os.environ.update(JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',MPLBACKEND='Agg')
root=Path('$HOME/projects/tenny-mature-finetune-20261116')
repo=Path('${REPO}')
sys.path[:0]=['$HOME/projects/tenny-native-eval/python','$HOME/projects/tenny-torque200/src',str(root)]
from experiments.balance_diagnostics.runner import evaluate_manifest,write_json
from experiments.mature_balance_finetune.report import measure_case,write_report

manifest=json.loads((root/'inputs/high_lr.json').read_text())
state=json.loads((root/'matrix/status.json').read_text())
assert state['phase']=='completed' and len(state['completed'])==6
policies=[dict(id='initial',label='Recovered mature policy (no update)',descriptor=manifest['base_descriptor'],
               source=dict(condition='initial',seed=None,learning_rate=None,model_origin=manifest['model_origin']))]
for condition in ['high_lr','low_lr']:
    for seed in range(3):
        run=root/f'matrix/seed-{seed}/{condition}'
        status=json.loads((run/'status.json').read_text())
        assert status['phase']=='completed' and status['steps']==2097152
        policies.append(dict(id=f'{condition}-seed-{seed}',label=f'{condition} seed {seed}',descriptor=str(run/'descriptor.json'),
                             source=dict(condition=condition,seed=seed,learning_rate=3e-5 if condition=='high_lr' else 3e-6)))
plan=dict(schema_version=1,seed=0,policies=policies,initial_states=manifest['initial_states'],
          scope='Mature learned policy before and after true XY reward fine tuning; three paired training seeds, original action mapping, common 120s deterministic evaluation.')
write_json(root/'inputs/evaluation-plan.json',plan)
bundle=evaluate_manifest(root/'inputs/evaluation-plan.json',root/'evaluation',seconds=120.)
for case in bundle['cases']:case['measurement']=measure_case(case['trace'])
progress_records={f'{condition}-seed-{seed}':[json.loads(line) for line in (root/f'matrix/seed-{seed}/{condition}/progress.jsonl').read_text().splitlines() if line.strip()]
                  for condition in ['high_lr','low_lr'] for seed in range(3)}
bundle.update(kind='mature_balance_finetune_v1',run_statuses=state['completed'],experiment_manifest=manifest,progress_records=progress_records)
write_json(root/'evaluation/results.json',bundle)
print(json.dumps(write_report(bundle,root/'comparison')),flush=True)
local=repo/'tasks/results/mature-balance-finetune-20261116'
for name in ['comparison','inputs']:
    shutil.copytree(root/name,local/name,dirs_exist_ok=True)
shutil.copy2(root/'evaluation/results.json',local/'comparison/results.json')
shutil.copy2(root/'matrix/status.json',local/'matrix-status.json')
for seed in range(3):
    for condition in ['high_lr','low_lr']:
        source=root/f'matrix/seed-{seed}/{condition}'
        dest=local/f'runs/{condition}-seed-{seed}';dest.mkdir(parents=True,exist_ok=True)
        for name in ['status.json','effective-config.json','progress.jsonl','training-metrics.jsonl']:
            if (source/name).exists():shutil.copy2(source/name,dest/name)
shutil.copy2(root/'preparation.json',local/'preparation.json')
print(json.dumps(dict(event='final_evaluation_and_reports_completed',cases=len(bundle['cases']))),flush=True)
