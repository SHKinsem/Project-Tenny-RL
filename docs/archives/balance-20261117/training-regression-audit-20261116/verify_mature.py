import json
import os
from pathlib import Path
import sys
import time

os.environ.update(JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES='', MUJOCO_GL='egl', MPLBACKEND='Agg')
repo = Path('${REPO}')
sys.path[:0] = ['$HOME/projects/tenny-native-eval/python','$HOME/projects/tenny-torque200/src',str(repo)]
import numpy as np
import mujoco
import matplotlib.pyplot as plt
from native_eval import Evaluator
from experiments.balance_diagnostics.runner import evaluate_manifest, write_json
from experiments.reward_ablation.evaluation import standing_metrics

start=time.monotonic()
out=Path(__file__).resolve().parent
screen=out/'historical-screen'
summary=json.loads((screen/'summary.json').read_text())
eligible=[p for p in summary['policies'] if all(p['reached_horizon'])]
chosen=min(eligible,key=lambda p:max(p['max_displacement_m']))
identity=chosen['policy_id']
old=json.loads((screen/'manifest.json').read_text())
policy=next(p for p in old['policies'] if p['id']==identity)
policy=dict(policy,descriptor=str(screen/policy['descriptor']))
states=json.loads((repo/'tasks/results/aligned-limit-ablation-20261116/comparison/results.json').read_text())['initial_states']
plan=dict(schema_version=1,seed=0,policies=[policy],initial_states=states,
          scope='Selected by 30-second nominal/+/-0.01rad screening; extended evaluation includes held-out +/-0.02rad. Original learned policy, no parameter updates.')
write_json(out/'mature-validation-plan.json',plan)
result_path=out/'mature-validation/results.json'
bundle=json.loads(result_path.read_text()) if result_path.exists() else evaluate_manifest(out/'mature-validation-plan.json',out/'mature-validation',seconds=120.)
for case in bundle['cases']: case['standing']=standing_metrics(case['trace'],horizon_seconds=120.)
write_json(out/'mature-validation/results.json',bundle)
result=dict(selected=chosen,cases=[dict(initial_state_id=c['initial_state_id'],standing=c['standing']) for c in bundle['cases']],
            source=policy['source'],selection_scope='Selected lowest maximum 30s drift among 25 original learned checkpoints, three fixed screening initial states. Held-out states were not used to select.',
            wall_seconds=time.monotonic()-start)
write_json(out/'mature-validation-summary.json',result)
descriptor=json.loads(Path(policy['descriptor']).read_text())
model=mujoco.MjModel.from_xml_path(descriptor['xml_path'])
data=mujoco.MjData(model)
renderer=mujoco.Renderer(model,height=300,width=450)
camera=mujoco.MjvCamera()
camera.lookat[:]=[0.,0.,.09]
camera.distance=.55
camera.azimuth=130
camera.elevation=-18
frames=[0.,1.,5.,10.,20.,30.]
raw=np.load(screen/'evaluation'/f"policy-{next(i for i,p in enumerate(old['policies']) if p['id']==identity)}-long.npz")
fig,axes=plt.subplots(2,3,figsize=(13.5,6.5))
for ax,seconds in zip(axes.flat,frames):
    if seconds==0:
        data.qpos[:]=states[0]['state']['qpos'];data.qvel[:]=states[0]['state']['qvel']
    else:
        index=int(round(seconds/.01))-1
        data.qpos[:]=raw['world_0_qpos'][index];data.qvel[:]=raw['world_0_qvel'][index]
    mujoco.mj_forward(model,data)
    renderer.update_scene(data,camera=camera)
    ax.imshow(renderer.render());ax.set_title(f'{seconds:g} s');ax.axis('off')
fig.suptitle('Recovered learned policy | 2,801,664-step checkpoint | nominal replay',fontsize=13)
fig.tight_layout();fig.savefig(out/'mature-replay.png',dpi=120);plt.close(fig)
renderer.close();raw.close()
fig,axes=plt.subplots(1,3,figsize=(13.5,3.5))
for case in bundle['cases']:
    t=np.asarray(case['trace']['t_s']);p=np.asarray(case['trace']['base_position_m']);v=np.asarray(case['trace']['base_velocity_m_s'])
    pitch=np.rad2deg(np.asarray(case['trace']['pitch_rad'])-case['trace']['metadata']['reference_pitch_rad'])
    axes[0].plot(t,100*np.linalg.norm(p[:,:2]-p[0,:2],axis=1),label=case['initial_state_id'])
    axes[1].plot(t,pitch)
    axes[2].plot(t,np.linalg.norm(v[:,:2],axis=1))
axes[0].set_ylabel('True XY displacement (cm)');axes[1].set_ylabel('Pitch error (deg)');axes[2].set_ylabel('Horizontal speed (m/s)')
for ax in axes:ax.set_xlabel('Time (s)');ax.grid(alpha=.25)
axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(out/'mature-120s.png',dpi=140);plt.close(fig)
print(json.dumps(result),flush=True)
