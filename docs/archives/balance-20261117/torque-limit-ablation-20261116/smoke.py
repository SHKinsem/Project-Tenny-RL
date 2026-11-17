import ast
import copy
import inspect
import json
from pathlib import Path
import textwrap

from experiments.reward_ablation.train import train_one
from experiments.torque_limit_ablation.env import TorqueLimitEnv
from hybrid.env import HybridLegWheelRobot

work=Path('$HOME/projects/tenny-torque-limit-ablation-20261116')
baseline=textwrap.dedent(inspect.getsource(HybridLegWheelRobot._step_control)).replace(
    'jp.mean(jp.abs(raw) > self._torque_scales)','jp.mean(jp.abs(raw) > self._execution_limits)')
changed=textwrap.dedent(inspect.getsource(TorqueLimitEnv._step_control))
assert ast.dump(ast.parse(baseline))==ast.dump(ast.parse(changed)),'Unexpected control-loop math change'
manifest=json.loads((work/'inputs/manifest2.json').read_text())
manifest=copy.deepcopy(manifest)
manifest['ppo'].update(num_timesteps=8192,num_evals=2)
manifest['steps_per_run']=8192
path=work/'inputs/smoke-manifest.json'
path.write_text(json.dumps(manifest,indent=2)+'\n')
train_one(path,'baseline',0,work/'smoke')
