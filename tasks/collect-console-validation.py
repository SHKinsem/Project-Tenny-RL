import json, shutil
import os
from pathlib import Path
base=(Path.home() / "projects/tenny-torque200/artifacts/console/original")
win=(Path(os.environ.get("TENNY_ROOT") or Path(__file__).resolve().parents[1]) / "experiments/torque200/results/console-control")
win.mkdir(parents=True,exist_ok=True)
rows=[]
for run in sorted(base.iterdir()):
    report_file=run/'report.json'
    if not report_file.is_file():
        continue
    r=json.loads(report_file.read_text())
    out=win/run.name
    out.mkdir(exist_ok=True)
    for name in ('request.json','status.json','env_config.json','ppo_config.json','report.json','progress.json','metrics.jsonl','train.log','params_cpu.pkl','policy.cpp.txt'):
        source=run/name
        if source.is_file(): shutil.copy2(source,out/name)
    keys=('mode','run_id','phase','completed','stop_reason','actual_timesteps','duration_seconds','training_stop_elapsed_seconds','finalization_seconds','elapsed_seconds','restored_from','parameter_tree_max_abs_error','export_max_abs_error','restore_max_abs_error','all_parameters_finite','finite_training_losses_checked')
    rows.append({k:r.get(k) for k in keys})
(win/'summary.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows))
