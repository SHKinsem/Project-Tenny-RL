"""Retrospective selection from existing descriptors; no further training."""
import json
import os
from pathlib import Path
import shutil
import sys

os.environ.update(JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',MPLBACKEND='Agg')
root=Path('$HOME/projects/tenny-mature-finetune-20261116')
sys.path[:0]=['$HOME/projects/tenny-native-eval/python','$HOME/projects/tenny-torque200/src',str(root)]
from experiments.balance_diagnostics.runner import evaluate_manifest,write_json
from experiments.mature_balance_finetune.report import measure_case
import numpy as np
import matplotlib.pyplot as plt

summary=json.loads((root/'comparison/summary.json').read_text())
candidates=[]
for identity,curve in summary['progress_curves'].items():
    for point in curve:
        if point['steps']==0:continue
        primary=point['primary']['windows']['first_30s']
        if primary['complete_set']:
            candidates.append(dict(policy_id=identity,steps=point['steps'],rms=primary['rms_xy_displacement_m'],
                                   last10=primary['last_10s_mean_xy_displacement_m']))
chosen=min(candidates,key=lambda value:value['rms'])
condition,seed_text=chosen['policy_id'].rsplit('-seed-',1);seed=int(seed_text)
descriptor=root/f"matrix/seed-{seed}/{condition}/descriptor-step-{chosen['steps']}.json"
manifest=json.loads((root/'inputs/high_lr.json').read_text())
identity='selected-stage'
plan=dict(schema_version=1,seed=0,initial_states=manifest['initial_states'],
          policies=[dict(id=identity,label='Retrospectively selected saved stage',descriptor=str(descriptor),source=chosen)],
          scope='Exploratory selection of lowest 30-second primary-state XY RMS among 48 nonzero saved stages. Not an additional training seed and not part of the prespecified final-step comparison.')
write_json(root/'selected-stage-plan.json',plan)
bundle=evaluate_manifest(root/'selected-stage-plan.json',root/'selected-stage-evaluation',seconds=120.)
for case in bundle['cases']:case['measurement']=measure_case(case['trace'])
write_json(root/'selected-stage-evaluation/results.json',bundle)
primary=[c for c in bundle['cases'] if c['initial_state_id'] in ('nominal','pitch-plus','pitch-minus')]
heldout=[c for c in bundle['cases'] if c not in primary]
def mean(items,window,key):return sum(c['measurement']['windows'][window][key] for c in items)/len(items)
record=dict(schema_version=1,kind='retrospectively_selected_mature_snapshot',selected=chosen,
            selection_rule='Minimum first-30s true-XY RMS averaged over the three primary initial states, across 48 nonzero saved stages; heldout states excluded from ranking.',
            descriptor=str(descriptor),scope=plan['scope'],
            primary_120s_rms_xy_m=mean(primary,'first_120s','rms_xy_displacement_m'),
            primary_120s_last10_xy_m=mean(primary,'first_120s','last_10s_mean_xy_displacement_m'),
            heldout_120s_rms_xy_m=mean(heldout,'first_120s','rms_xy_displacement_m'),
            heldout_120s_last10_xy_m=mean(heldout,'first_120s','last_10s_mean_xy_displacement_m'),
            cases=[dict(initial_state_id=c['initial_state_id'],measurement=c['measurement']) for c in bundle['cases']],
            recovery_limit='Saved descriptor preserves deterministic actor/normalizer inference; no complete intermediate actor/critic/std checkpoint was saved, so this is not a full PPO resume checkpoint.')
local=Path(__file__).resolve().parent/'selected-stage'
shutil.copytree(root/'selected-stage-evaluation',local,dirs_exist_ok=True)
shutil.copy2(descriptor,local/'descriptor.json')
write_json(local/'summary.json',record)
initial=json.loads((root/'evaluation/results.json').read_text())
fig,axes=plt.subplots(1,3,figsize=(13.5,3.5))
for label,data in [('Mature reference',initial),('Selected saved stage',bundle)]:
    case=next(c for c in data['cases'] if c['initial_state_id']=='nominal' and c['policy_id'] in ('initial','selected-stage'))
    trace=case['trace'];t=np.asarray(trace['t_s']);p=np.asarray(trace['base_position_m']);v=np.asarray(trace['base_velocity_m_s'])
    axes[0].plot(t,1000*np.linalg.norm(p[:,:2]-p[0,:2],axis=1),label=label)
    axes[1].plot(t,np.rad2deg(np.asarray(trace['pitch_rad'])-trace['metadata']['reference_pitch_rad']))
    axes[2].plot(t,np.linalg.norm(v[:,:2],axis=1))
axes[0].set_ylabel('True XY displacement (mm)');axes[1].set_ylabel('Pitch error (deg)');axes[2].set_ylabel('Horizontal speed (m/s)')
axes[0].legend(fontsize=8)
for ax in axes:ax.set_xlabel('Time (s)');ax.grid(alpha=.25)
fig.suptitle('Exploratory saved-stage replay | selection by primary 30s RMS; no new training',fontsize=11)
fig.tight_layout();fig.savefig(local/'comparison.png',dpi=140);plt.close(fig)
print(json.dumps({key:value for key,value in record.items() if key!='cases'}),flush=True)
