"""Find pre-comparison learned policies; evaluate their original action semantics."""
import copy
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.update(JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES='', XLA_PYTHON_CLIENT_PREALLOCATE='false')
repo = Path('${REPO}')
sys.path[:0] = ['$HOME/projects/tenny-native-eval/python', '$HOME/projects/tenny-torque200/src',str(repo)]
from legwheel.deployment.extract import load
from force_teacher.contract import tree_digest
from native_eval.legwheel import descriptor_from_run, policy_template
from native_eval.brax_policy import policy_from_brax
from experiments.balance_diagnostics.runner import evaluate_manifest, write_json
from experiments.reward_ablation.evaluation import standing_metrics

start = time.monotonic()
out = Path(__file__).resolve().parent/'historical-screen'
out.mkdir(exist_ok=True)
inventory = json.loads(Path(__file__).with_name('mature-inventory.json').read_text())
source = Path('$HOME/projects/tenny-torque200/src')
old = json.loads((repo/'tasks/results/aligned-limit-ablation-20261116/comparison/results.json').read_text())
fingerprint = '4ca2bc7387038679a16eaedde390b4f4db7b6e678007d83fba268ccc9137960a'
policies = []
excluded = []
for run_info in inventory['runs']:
    if run_info.get('mode') != 'leg_force_bc' or not run_info.get('report',{}).get('checkpoint'):
        continue
    if run_info['report'].get('source_fingerprint') != fingerprint:
        excluded.append(dict(run=run_info['run_dir'], reason='Different source fingerprint; requires its own exact historical source before compatible evaluation.'))
        continue
    run = Path(run_info['run_dir'])
    try:
        final = descriptor_from_run(run,source)
        ppo = json.loads((run/'ppo_config.json').read_text())
        template = policy_template(json.loads((run/'policy_metadata.json').read_text()),ppo)
        for checkpoint_text in run_info['checkpoints']:
            checkpoint = Path(checkpoint_text)
            digest = tree_digest(checkpoint)
            if checkpoint_text == run_info['report']['checkpoint']:
                assert digest == run_info['report']['checkpoint_sha256']
            params = load(checkpoint_text)
            descriptor = copy.deepcopy(final)
            descriptor['policy'] = policy_from_brax(params[:2],template)
            identity = run.name+'-'+checkpoint.name
            descriptor['metadata'].update(model_run_id=run.name, checkpoint_sha256=digest,
                                          checkpoint_steps=int(checkpoint.name), audit_kind='precomparison_original_policy')
            target = out/(identity+'.json')
            write_json(target,descriptor)
            policies.append(dict(id=identity,label=identity,descriptor=target.name,
                                 source=dict(run_dir=str(run),checkpoint=str(checkpoint), checkpoint_sha256=digest,
                                             restored_from=run_info['report'].get('restored_from'),
                                             learning_rate=ppo.get('learning_rate'), episode_length=ppo.get('episode_length'),
                                             original_action_mode=descriptor['task']['config']['action_mode'])))
    except Exception as exc:
        excluded.append(dict(run=str(run),reason=repr(exc)))
write_json(out/'preparation.json',dict(policies=len(policies),excluded=excluded))
plan = dict(schema_version=1,seed=0,policies=policies,initial_states=old['initial_states'][:3],
            scope='Historical learned leg_force_bc policies in their original action mapping, no retraining; fixed nominal and +/-0.01rad resets. Time limits are evaluation censoring only.')
write_json(out/'manifest.json',plan)
result_path=out/'evaluation/results.json'
bundle = json.loads(result_path.read_text()) if result_path.exists() else evaluate_manifest(out/'manifest.json',out/'evaluation',seconds=30.)
for case in bundle['cases']:
    case['standing'] = standing_metrics(case['trace'])
write_json(out/'evaluation/results.json',bundle)
rows=[]
for policy in policies:
    cases=[case for case in bundle['cases'] if case['policy_id']==policy['id']]
    rows.append(dict(policy_id=policy['id'],source=policy['source'],
                     survival_s=[c['standing']['survival_seconds'] for c in cases],
                     in_band_s=[c['standing']['in_band_seconds_after_settling'] for c in cases],
                     max_displacement_m=[max(math.hypot(p[0]-c['trace']['base_position_m'][0][0],p[1]-c['trace']['base_position_m'][0][1]) for p in c['trace']['base_position_m']) for c in cases],
                     reached_horizon=[c['standing']['reached_horizon'] for c in cases]))
summary=dict(schema_version=1,policies=rows,excluded=excluded,wall_seconds=time.monotonic()-start)
write_json(out/'summary.json',summary)
for row in sorted(rows,key=lambda item:min(item['survival_s']),reverse=True):
    print(json.dumps({key:value for key,value in row.items() if key!='source'}),flush=True)
