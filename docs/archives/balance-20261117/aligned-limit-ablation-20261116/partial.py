import json
from pathlib import Path
work=Path('$HOME/projects/tenny-aligned-limit-ablation-20261116')
rows=[]
for seed in (0,1,2):
    for limit in (1,2):
        path=work/f'evaluations/limit{limit}-seed-{seed}/results.json'
        if not path.exists():continue
        bundle=json.loads(path.read_text())
        cases=[c for c in bundle['cases'] if c['initial_state_id'] in ('nominal','pitch-plus','pitch-minus')]
        rows.append(dict(seed=seed,limit=limit,
            primary_in_band_seconds=sum(c['standing']['in_band_seconds_after_settling'] for c in cases)/3,
            primary_survival_seconds=sum(c['standing']['survival_seconds'] for c in cases)/3))
print(json.dumps(rows,indent=2))
