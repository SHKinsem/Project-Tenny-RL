from pathlib import Path
import shutil

local=Path(__file__).resolve().parent
root=Path('$HOME/projects/tenny-mature-finetune-20261116')
for name in ['FINDINGS.md','PROGRESS.md','statistics-review.md','review-notes.md']:
    shutil.copy2(local/name,root/name)
for name in ['reward-review.json','stage-reward-review.json','replay-seed0.png']:
    shutil.copy2(local/'comparison'/name,root/'comparison'/name)
selected=root/'selected-stage'
if not selected.exists():selected.symlink_to('selected-stage-evaluation',target_is_directory=True)
if not selected.resolve().is_relative_to(root.resolve()):raise ValueError('Report destination outside experiment root')
for name in ['summary.json','descriptor.json','verification.json','comparison.png']:
    shutil.copy2(local/'selected-stage'/name,selected/name)
print('Small reports synchronized; frozen inputs, source and model parameters unchanged.')
