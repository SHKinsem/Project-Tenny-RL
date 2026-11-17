import json
import os
from pathlib import Path
os.environ['JAX_PLATFORMS']='cpu'
os.environ['CUDA_VISIBLE_DEVICES']=''
import jax
import jax.numpy as jp
import numpy as np
from experiments.reward_ablation.rewards import extra_rewards,scalar_rewards,ARMS

pitch=np.asarray([0.,np.deg2rad(10),-np.deg2rad(30)],dtype=np.float32)
xy=np.asarray([[0.,0.],[.03,.04],[-.1,.1]],dtype=np.float32)
errors={}
for arm in ARMS:
    actual=jax.jit(lambda p,x:extra_rewards(p,x,arm,jp))(jp.asarray(pitch),jp.asarray(xy))
    expected=[scalar_rewards(float(p),x.tolist(),arm) for p,x in zip(pitch,xy)]
    errors[arm]=max(float(np.max(np.abs(np.asarray(actual[key])-np.asarray([row[key] for row in expected])))) for key in actual)
if max(errors.values())>2e-6:
    raise ValueError(errors)
result=dict(passed=True,backend='JAX CPU jitted array reward vs standard-library scalar reference',max_abs_errors=errors)
Path('$HOME/projects/tenny-reward-ablation-20261116/array-reward-check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
