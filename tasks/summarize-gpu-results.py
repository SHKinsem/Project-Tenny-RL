import csv
import json
from pathlib import Path

root = Path(__file__).parent / 'results' / 'gpu'
benchmarks = []
for file in sorted(root.glob('benchmark-*.json')):
    result = json.loads(file.read_text())
    benchmarks.append({
        'report': file.name, 'backend': result['expected_backend'],
        'batch_size': result['batch_size'], 'status': result['status'],
        'steps_per_second': result.get('median_environment_steps_per_second'),
        'compile_seconds': result.get('rollout_lower_and_compile_seconds'),
        'memory_stats': result.get('memory_stats_after'),
    })
memory = []
for file in sorted(root.glob('*-gpu.csv')):
    with file.open(newline='') as stream:
        rows = list(csv.reader(stream))[1:]
    if rows:
        values = [int(row[1].strip().split()[0]) for row in rows]
        memory.append({'file': file.name, 'samples': len(values),
                       'first_board_MiB': values[0], 'peak_board_MiB': max(values),
                       'peak_minus_first_MiB': max(values)-values[0]})
training = []
for file in sorted(root.glob('train-gpu-*/report.json')):
    result = json.loads(file.read_text())
    training.append({key: result[key] for key in (
        'devices', 'parameter_platforms', 'elapsed_seconds', 'num_envs',
        'observed_progress_steps', 'restore_max_abs_error', 'export_max_abs_error')})
summary = {'benchmarks': benchmarks, 'board_memory_samples': memory, 'training': training,
           'memory_note': 'Board totals include desktop/other apps; difference is not exact process memory.'}
(root / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(summary, indent=2))
