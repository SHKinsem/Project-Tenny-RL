import shutil
from pathlib import Path
local=Path('${REPO}/tasks/results/torque-limit-ablation-20261116')
work=Path('$HOME/projects/tenny-torque-limit-ablation-20261116')
for name in ('report.html','paired-summary.json'):
    shutil.copy2(local/'comparison'/name,work/'comparison'/name)
shutil.copy2(local/'FINDINGS.md',work/'FINDINGS.md')
print('Final report notes synchronized; no model or frozen source changed.')
