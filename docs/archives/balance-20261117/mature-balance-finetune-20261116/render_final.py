"""Render matched seed-0 replay states, stopping explicitly at episode failure."""
import json
import os
from pathlib import Path

os.environ.update(MUJOCO_GL='egl',MPLBACKEND='Agg')
import numpy as np
import mujoco
import matplotlib.pyplot as plt

root=Path('$HOME/projects/tenny-mature-finetune-20261116')
out=Path(__file__).resolve().parent/'comparison'
bundle=json.loads((root/'evaluation/results.json').read_text())
plan=json.loads((root/'inputs/evaluation-plan.json').read_text())
reference=json.loads(Path(plan['policies'][0]['descriptor']).read_text())
model=mujoco.MjModel.from_xml_path(reference['xml_path']);data=mujoco.MjData(model)
renderer=mujoco.Renderer(model,height=280,width=400)
camera=mujoco.MjvCamera();camera.lookat[:]=[0.,0.,.09];camera.distance=.55;camera.azimuth=130;camera.elevation=-18
fig,axes=plt.subplots(3,4,figsize=(13.6,7.8))
identities=[('initial','Recovered mature reference'),('high_lr-seed-0','3e-5 / seed 0'),('low_lr-seed-0','3e-6 / seed 0')]
for row,(identity,label) in enumerate(identities):
    index=next(i for i,p in enumerate(bundle['policies']) if p['id']==identity)
    case=next(c for c in bundle['cases'] if c['policy_id']==identity and c['initial_state_id']=='nominal')
    length=len(case['trace']['t_s'])-1
    with np.load(root/f'evaluation/policy-{index}-long.npz') as raw:
        for column,seconds in enumerate([0.,1.,30.,120.]):
            ax=axes[row,column]
            count=int(round(seconds/.01))
            if count>length:
                ax.set_facecolor('#eeeeee');ax.text(.5,.5,f'Ended at {length*.01:.2f}s',ha='center',va='center',transform=ax.transAxes)
            else:
                if count==0:
                    data.qpos[:]=plan['initial_states'][0]['state']['qpos'];data.qvel[:]=plan['initial_states'][0]['state']['qvel']
                else:
                    data.qpos[:]=raw['world_0_qpos'][count-1];data.qvel[:]=raw['world_0_qvel'][count-1]
                mujoco.mj_forward(model,data);renderer.update_scene(data,camera=camera);ax.imshow(renderer.render())
            ax.set_xticks([]);ax.set_yticks([])
            if row==0:ax.set_title(f'{seconds:g} seconds',fontsize=11,pad=7)
            if column==0:ax.set_ylabel(label,fontsize=10)
fig.suptitle('Original model and two fine tunes | matched nominal replay, seed 0',fontsize=13)
fig.subplots_adjust(left=.10,right=.99,bottom=.02,top=.92,hspace=.06,wspace=.03)
out.mkdir(exist_ok=True)
fig.savefig(out/'replay-seed0.png',dpi=130);plt.close(fig);renderer.close()
print(json.dumps(dict(path=str(out/'replay-seed0.png'))))
