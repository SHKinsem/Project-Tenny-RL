import json
import shutil
from pathlib import Path

from experiments.horizon_ablation.report import write_horizon_report

work = Path('$HOME/projects/tenny-horizon-ablation-20261116')
repo = Path('${REPO}')
target = work / 'comparison'
bundle = json.loads((target / 'results.json').read_text())
print(json.dumps(write_horizon_report(bundle, target)), flush=True)
local = repo / 'tasks/results/horizon-ablation-20261116'
shutil.copytree(target, local / 'comparison', dirs_exist_ok=True)
shutil.copy2(work / 'matrix30/status.json', local / 'matrix-status.json')
