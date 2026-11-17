import hashlib
import json
from pathlib import Path
import time
from hybrid.core import source_fingerprint

started=time.monotonic()
repo=Path('${REPO}')
local=repo/'tasks/results/torque-limit-audit-20261116'
manifest=json.loads(Path('$HOME/projects/tenny-horizon-ablation-20261116/inputs/manifest30.json').read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for path,expected in manifest['protected_inputs']['files'].items():
    assert sha(path)==expected,path
assert source_fingerprint()==manifest['policy_metadata']['source_fingerprint']
result=json.loads((local/'results.json').read_text())
assert all(sha(path)==expected for path,expected in result['input_sha256'].items())
result['actuator_order']=['thigh_left_motor','thigh_right_motor','wheel_left_motor','wheel_right_motor']
result['sampling']={'hard_clipping':'Native raw-request clipping metric includes every physics substep, first episode only. Zero sum proves no clipping; a positive sum is not a physical duty fraction.',
    'magnitude':'100 Hz final-physics-substep ctrl samples; actual torque inferred from unit-gain, unit-gear stateless motor law.',
    'action':'Policy output held over each 10 ms control period; |a| >= .95 is near force action bound, not hard motor clipping.'}
result['protected_files_unchanged']=len(manifest['protected_inputs']['files'])
result['source_fingerprint_unchanged']=source_fingerprint()
result['input_verification_seconds']=time.monotonic()-started
(local/'results.json').write_text(json.dumps(result,indent=2)+'\n')
Path('$HOME/projects/tenny-torque-limit-audit-20261116/results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(protected_files_unchanged=result['protected_files_unchanged'],
    source_fingerprint_unchanged=result['source_fingerprint_unchanged'],
    hashed_replay_inputs=len(result['input_sha256']),input_verification_seconds=result['input_verification_seconds'])))
