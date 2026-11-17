import json
from pathlib import Path

root = Path(__file__).parent
bundle = json.loads((root/'result/results.json').read_text())
rows = []
for case in bundle['cases']:
    full = case['analysis']['windows']['full_episode']
    first = case['analysis']['windows']['first_6s']
    trace = case['trace']
    rows.append(dict(policy=case['policy_id'], initial=case['initial_state_id'],
        seconds=full['observed_seconds'], max_displacement_cm=100*full['max_horizontal_displacement_m'],
        first_6s_displacement_cm=100*first['max_horizontal_displacement_m'],
        pitch_rms_deg=full['pitch_error_rms_deg'], tail=full['tail'],
        events=case['analysis']['events'], termination=trace['termination'],
        wheel_angle_distance_end_m=trace['wheel_angle_distance_m'][-1]))
summary = dict(rows=rows, parity=bundle['parity'], preparation_evidence=bundle['preparation_evidence'],
               timing=bundle['timing'])
(root/'summary-detail.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps(dict(nominal=[row for row in rows if row['initial']=='nominal'],
    initial_reference=bundle['parity'][0], timing=bundle['timing']), indent=2))
