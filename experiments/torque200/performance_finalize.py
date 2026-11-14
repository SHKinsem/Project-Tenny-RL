"""Publish already-verified A1/B1 artifacts after the user's early stop. No training."""
import json
from pathlib import Path
import shutil

BASE = Path('/home/jaccob/projects/tenny-torque200/artifacts')
RUN = BASE / 'performance-20261114-160724'
WIN = Path('/mnt/c/Users/xusen/OneDrive - HKUST Connect/Workshop/Projects/Project Tenny/experiments/torque200/results')
runs = json.loads((RUN / 'rounds.json').read_text())
assert [r['name'] for r in runs] == ['A1', 'B1']
baseline, web = (r['steady_sps'] for r in runs)
for row in runs:
    events = json.loads((RUN / row['name'] / 'progress.json').read_text())
    row['final_eval_avg_episode_length_steps'] = events[-1]['metrics']['eval/avg_episode_length']
    row['final_eval_avg_episode_length_seconds'] = row['final_eval_avg_episode_length_steps'] * 0.005
    row['final_eval_episode_reward'] = events[-1]['metrics']['eval/episode_reward']
summary = {
    'run_id': RUN.name,
    'completed': True,
    'completed_meaning': 'Selected B1 model checkpoint/export validation completed; not full planned ABBA comparison',
    'comparison_complete': False,
    'stopped_by_user': True,
    'completed_rounds': 2,
    'planned_rounds': 4,
    'checkpoint': runs[-1]['checkpoint'],
    'export_path': str(RUN / 'B1/smoke_params.cpp.txt'),
    'report_path': str(RUN / 'comparison.json'),
    'control_hz': 200,
    'total_environment_steps': sum(r['actual_steps'] for r in runs),
    'total_environment_steps_scope': 'Only completed A1/B1 rounds; interrupted B2 excluded',
    'checkpoint_additional_training_steps': 32768,
    'baseline_sps': baseline,
    'web_sps': web,
    'throughput_change_percent': 100 * (web / baseline - 1),
    'same_start_checkpoint': str(BASE / 'smoke-20261114-154038/resumed/checkpoints/000000000016'),
    'seed_each_round': 0,
    'runs': runs,
    'note': 'Stopped immediately at user request during B2; A2 not started. A1/B1 are a single preliminary comparison, not completed ABBA. Each 32768-step run excludes its first two 8192-step training intervals; final 16384-step timing is used. A retains browser connection but suspends simulation/rendering. B renders to an actual browser. Roughly 13.1% lower throughput is preliminary, not a universal impact estimate. All incomplete B2 and earlier pilot outputs are retained and excluded. Model has only 32768 additional steps, not combined A1+B1 steps; no convergence claim.',
}
for target in (RUN / 'comparison.json', BASE / 'performance-latest.json', BASE / 'performance-status.json'):
    temp = target.with_suffix('.tmp')
    temp.write_text(json.dumps(summary, indent=2))
    temp.replace(target)
target_run = WIN / RUN.name
target_run.mkdir(exist_ok=True)
shutil.copy2(RUN / 'comparison.json', target_run / 'comparison.json')
for name in ('performance-latest.json', 'performance-status.json'):
    shutil.copy2(BASE / name, WIN / name)
print(json.dumps({k: summary[k] for k in ('checkpoint', 'completed_rounds', 'comparison_complete',
                                       'baseline_sps', 'web_sps', 'throughput_change_percent')}, indent=2))
print('B1 eval mean seconds', runs[-1]['final_eval_avg_episode_length_seconds'])
