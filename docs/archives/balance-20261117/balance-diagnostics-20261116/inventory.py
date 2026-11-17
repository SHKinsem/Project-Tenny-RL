import importlib.util
import json
from pathlib import Path

root = Path('$HOME/projects/tenny-torque200')
run = root / 'artifacts/console/leg_force_bc/20261115T172533.823984Z_0a516875'
read = lambda path: json.loads(path.read_text())
report = read(run / 'report.json')
pretrain = read(run / 'pretraining-report.json')
selected = {key: report.get(key) for key in (
    'run_id', 'checkpoint', 'checkpoint_sha256', 'source_fingerprint', 'steps',
    'completed', 'checkpoint_validated', 'export_sha256', 'normalizer_change')}
selected['checkpoints'] = [str(path) for path in sorted((run / 'checkpoints').iterdir()) if path.is_dir()]
selected['pretrain_keys'] = list(pretrain)
selected['bootstrap_receipt'] = pretrain.get('bootstrap_receipt')
selected['bootstrap_keys'] = {}
if selected['bootstrap_receipt']:
    receipt = read(Path(selected['bootstrap_receipt']))
    selected['bootstrap_keys'] = receipt
selected['cached_reset'] = str(Path('$HOME/projects/tenny-cpu-gpu-eval-20261115/result/initial.npz'))
selected['cached_reset_exists'] = Path(selected['cached_reset']).is_file()
selected['matplotlib_available'] = importlib.util.find_spec('matplotlib') is not None
selected['native_extension'] = [str(path) for path in Path('$HOME/projects/tenny-native-eval/python/native_eval').glob('*.so')]
print(json.dumps(selected, indent=2))
Path(__file__).with_name('inventory.json').write_text(json.dumps(selected, indent=2) + '\n')
