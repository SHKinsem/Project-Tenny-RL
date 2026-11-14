import hashlib
import json
from pathlib import Path
import shutil

base = Path('/home/jaccob/projects/tenny-reference')
win = Path('/mnt/c/Users/xusen/OneDrive - HKUST Connect/Workshop/Projects/Project Tenny')
report_path = max(base.glob('artifacts/train-gpu-*/report.json'), key=lambda p: p.stat().st_mtime)
run = report_path.parent
report = json.loads(report_path.read_text())
assert report['parameter_platforms'] == ['gpu']
assert report['restore_max_abs_error'] == 0
assert report['export_max_abs_error'] <= report['export_atol']
config = json.loads((run / 'env_config.json').read_text())
old = json.loads((win / 'tasks/results/gpu/train-gpu-20261114-145047/env_config.json').read_text())
assert config == old, 'Original environment configuration drifted'
checks = {}
for rel in ('legwheel/config.py', 'legwheel/env.py', 'legwheel/training/ppo.py'):
    windows = win / 'reference/mujoco_legwheel_rl_public' / rel
    linux = base / 'mujoco_legwheel_rl_public' / rel
    a = hashlib.sha256(windows.read_bytes()).hexdigest()
    b = hashlib.sha256(linux.read_bytes()).hexdigest()
    assert a == b, rel
    checks[rel] = a
out = win / 'tasks/results/gpu' / run.name
out.mkdir(parents=True, exist_ok=True)
for name in ('report.json', 'env_config.json', 'ppo_config.json', 'progress.json', 'params_cpu.pkl', 'smoke_params.cpp.txt'):
    source = run / name
    if source.is_file():
        shutil.copy2(source, out / name)
verification = {'run': run.name, 'original_environment_matches_previous_verified_run': True,
    'source_sha256_windows_and_wsl': checks,
    'control_hz': 1 / config['ctrl_dt'], 'actor_observation_size': 26,
    'restore_max_abs_error': report['restore_max_abs_error'],
    'export_max_abs_error': report['export_max_abs_error'],
    'elapsed_seconds': report['elapsed_seconds'],
    'observed_progress_steps': report['observed_progress_steps'],
    'output': str(out), 'checkpoint': report['checkpoint']}
(out / 'mode-verification.json').write_text(json.dumps(verification, indent=2))
print(json.dumps(verification))
