from pathlib import Path
import shutil

local=Path(__file__).resolve().parent;root=Path('$HOME/projects/tenny-wheel-hold-ablation-20261116')
for name in ['FINDINGS.md','PROGRESS.md','observation-boundary.json','review-notes.md','statistics-review.md']:
    shutil.copy2(local/name,root/name)
shutil.copy2(local/'comparison/replay-seed0.png',root/'comparison/replay-seed0.png')
shutil.copy2(root/'executed-source.tar.gz',local/'comparison/executed-source.tar.gz')
selected=root/'selected-snapshot'
if not selected.exists():selected.symlink_to('selected-evaluation',target_is_directory=True)
if not selected.resolve().is_relative_to(root.resolve()):raise ValueError('Unexpected report destination')
for name in ['summary.json','descriptor.json','comparison.png','verification.json']:
    shutil.copy2(local/'selected-snapshot'/name,selected/name)
print('Reports synchronized; frozen source, inputs and model parameters unchanged.')
