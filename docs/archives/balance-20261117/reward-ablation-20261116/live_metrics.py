import json
from pathlib import Path
import statistics
root=Path('$HOME/projects/tenny-reward-ablation-20261116')
rows=[]
for path in sorted((root/'evaluations').glob('*/results.json')):
    bundle=json.loads(path.read_text())
    cases=bundle['cases'][:3]
    rows.append(dict(policy=path.parent.name,
        primary_standing_mean=statistics.mean(case['standing']['in_band_seconds_after_settling'] for case in cases),
        primary_survival_mean=statistics.mean(case['standing']['survival_seconds'] for case in cases),
        nominal_standing=cases[0]['standing']['in_band_seconds_after_settling'],
        maximum_negative_reward_fraction=max(case.get('reward_diagnostics',{}).get('negative_step_fraction',0) for case in bundle['cases'])))
print(json.dumps(rows,indent=2))
