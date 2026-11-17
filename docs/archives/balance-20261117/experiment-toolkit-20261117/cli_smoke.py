import json
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root))
from experiments.experiment_tools.io import load_json, save_json

output = Path(__file__).resolve().parent
started = time.perf_counter()
records = []


def call(arguments, expected=0):
    proc = subprocess.run([sys.executable, '-X', 'utf8', '-m', 'experiments.experiment_tools', *map(str, arguments)],
                          cwd=root, capture_output=True, text=True, encoding='utf-8', timeout=10)
    assert proc.returncode == expected, (arguments, proc.returncode, proc.stderr)
    records.append(dict(arguments=list(map(str, arguments)), returncode=proc.returncode))
    return json.loads(proc.stdout) if proc.stdout.strip() else None


examples = Path('experiments/experiment_tools/examples')
dry_dir = output/'dry-run-must-not-exist'
call(['run', examples/'cpu-jobs.json', '--dry-run', '--output', dry_dir])
assert not dry_dir.exists()
result = call(['run', examples/'cpu-jobs.json', '--output', output/'cpu-example'])
assert result['phase'] == 'completed' and result['completed_jobs'] == 1
result = call(['compare', examples/'paired-study.json'])
assert result['paired_summaries']['heldout']['rms_xy_m']['missing_count'] == 1
result = call(['select', examples/'paired-study.json', '--condition', 'changed', '--metric', 'rms_xy_m'])
assert result['selected']['id'] == 'complete-window'
call(['freeze', examples, '--output', output/'example-inputs.json'])
assert call(['check', output/'example-inputs.json'])['verified'] is True
save_json(output/'failed-job.json', dict(schema_version=1, kind='experiment_jobs', max_total_seconds=3,
    jobs=[dict(id='exit-seven', command=['{python}', '-c', 'raise SystemExit(7)'], timeout_seconds=2)]))
result = call(['run', output/'failed-job.json', '--output', output/'cpu-failed-example'], expected=1)
assert result['jobs'][0]['returncode'] == 7 and result['phase'] == 'failed'
report = dict(phase='verified', commands=records, elapsed_seconds=time.perf_counter()-started,
              scope='Portable examples, real CPU child only; dry-run did not create its output directory.')
save_json(output/'cli-validation.json', report)
print(json.dumps(report, ensure_ascii=False))
