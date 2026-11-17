import json
from pathlib import Path

root = Path('$HOME/projects/tenny-torque200/artifacts/console')
rows = []
for path in sorted(root.glob('*/*/report.json')):
    try:
        data = json.loads(path.read_text())
        run = path.parent
        ppo = json.loads((run/'ppo_config.json').read_text()) if (run/'ppo_config.json').exists() else {}
        meta = json.loads((run/'policy_metadata.json').read_text()) if (run/'policy_metadata.json').exists() else {}
        row = dict(run_dir=str(run), mode=run.parent.name,
                   report={k:data.get(k) for k in ('steps','checkpoint','checkpoint_sha256','completed','checkpoint_validated','restored_from','source_fingerprint')},
                   ppo={k:ppo.get(k) for k in ('learning_rate','episode_length','num_envs','batch_size','num_minibatches','num_updates_per_batch')},
                   policy={k:meta.get(k) for k in ('mode','action_mode','control_hz','policy_head','training_profile')},
                   checkpoints=[str(p) for p in sorted((run/'checkpoints').glob('*')) if p.is_dir()],
                   report_keys=list(data))
        rows.append(row)
    except Exception as exc:
        rows.append(dict(run_dir=str(path.parent), error=str(exc)))
result = dict(root=str(root), root_files=[p.name for p in root.iterdir() if p.is_file()], runs=rows)
Path(__file__).with_name('mature-inventory.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(runs=len(rows), leg_force_runs=sum(r.get('mode')=='leg_force_bc' for r in rows))))
