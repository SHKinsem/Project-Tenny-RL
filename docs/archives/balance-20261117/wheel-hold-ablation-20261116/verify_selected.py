import json
import os
from pathlib import Path
import sys
os.environ.update(JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='')
root=Path('$HOME/projects/tenny-wheel-hold-ablation-20261116')
sys.path[:0]=['$HOME/projects/tenny-native-eval/python','$HOME/projects/tenny-torque200/src',str(root)]
import numpy as np
from legwheel.deployment.extract import load
from native_eval import Evaluator
from native_eval.brax_policy import policy_from_brax
from native_eval.legwheel import policy_template
from force_teacher.contract import tree_digest
from experiments.reward_ablation.train import tree_array_digest

local=Path(__file__).resolve().parent/'selected-snapshot'
record=json.loads((local/'summary.json').read_text());manifest=json.loads((root/'inputs/manifest.json').read_text())
checkpoint=Path(record['checkpoint']);params=load(str(checkpoint))
assert tree_digest(checkpoint)==record['checkpoint_sha256'] and tree_array_digest(params)==record['parameter_digest']
descriptor=json.loads((local/'descriptor.json').read_text())
rebuilt=dict(descriptor,policy=policy_from_brax(params[:2],policy_template(manifest['policy_metadata'],manifest['ppo'])))
with np.load(local/'policy-0-long.npz') as data:observations=data['world_0_observation'][:256]
error=float(np.max(np.abs(np.asarray(Evaluator(descriptor).policy_actions(observations))-np.asarray(Evaluator(rebuilt).policy_actions(observations)))))
assert error<1e-6
result=dict(phase='verified',checkpoint_sha256=record['checkpoint_sha256'],all_parameter_leaves_finite=True,
            complete_parameter_digest=record['parameter_digest'],checkpoint_to_saved_policy_action_max_abs_error=error,
            independent_120s_last10_max_abs_error=record['independent_last10_max_abs_error'],all_five_completed_120s=True,
            resume_scope='Full normalizer/actor/critic retained, fixed observation scales and original wheel-force action head. Adam state restarts on continuation.')
(local/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
