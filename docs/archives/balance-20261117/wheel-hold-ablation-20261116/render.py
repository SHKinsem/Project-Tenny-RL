import json
import os
from pathlib import Path
os.environ.update(MUJOCO_GL='egl',MPLBACKEND='Agg')
import numpy as np
import mujoco
import matplotlib.pyplot as plt

root=Path('$HOME/projects/tenny-wheel-hold-ablation-20261116')
bundle=json.loads((root/'evaluation/results.json').read_text());plan=json.loads((root/'evaluation-plan.json').read_text())
descriptor=json.loads(Path(plan['policies'][0]['descriptor']).read_text())
model=mujoco.MjModel.from_xml_path(descriptor['xml_path']);data=mujoco.MjData(model);renderer=mujoco.Renderer(model,height=280,width=400)
camera=mujoco.MjvCamera();camera.lookat[:]=[0,0,.09];camera.distance=.55;camera.azimuth=130;camera.elevation=-18
fig,axes=plt.subplots(2,4,figsize=(13.6,5.4))
for row,(identity,label) in enumerate([('original_wheel-seed-0','Wheel weight 10 / seed 0'),('weak_wheel-seed-0','Wheel weight 1 / seed 0')]):
    index=next(i for i,p in enumerate(bundle['policies']) if p['id']==identity)
    case=next(c for c in bundle['cases'] if c['policy_id']==identity and c['initial_state_id']=='nominal')
    length=len(case['trace']['t_s'])-1
    with np.load(root/f'evaluation/policy-{index}-long.npz') as raw:
        for column,seconds in enumerate([0.,1.,30.,120.]):
            ax=axes[row,column];count=int(round(seconds/.01))
            if count>length:ax.text(.5,.5,f'Ended {length*.01}s',ha='center',transform=ax.transAxes)
            else:
                data.qpos[:]=plan['initial_states'][0]['state']['qpos'] if count==0 else raw['world_0_qpos'][count-1]
                data.qvel[:]=plan['initial_states'][0]['state']['qvel'] if count==0 else raw['world_0_qvel'][count-1]
                mujoco.mj_forward(model,data);renderer.update_scene(data,camera=camera);ax.imshow(renderer.render())
            ax.set_xticks([]);ax.set_yticks([])
            if row==0:ax.set_title(f'{seconds:g} seconds',fontsize=11)
            if column==0:ax.set_ylabel(label,fontsize=10)
fig.suptitle('Matched nominal replay | original versus weak wheel-position reward',fontsize=13)
fig.subplots_adjust(left=.09,right=.99,top=.88,bottom=.02,wspace=.03,hspace=.07)
path=Path(__file__).resolve().parent/'comparison/replay-seed0.png';fig.savefig(path,dpi=130);plt.close(fig);renderer.close()
print(json.dumps(dict(path=str(path))))
