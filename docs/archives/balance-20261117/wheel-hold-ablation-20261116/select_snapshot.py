"""Evaluate an existing complete weak-wheel snapshot; no additional training."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

os.environ.update(JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',MPLBACKEND='Agg')
root=Path('$HOME/projects/tenny-wheel-hold-ablation-20261116')
sys.path[:0]=['$HOME/projects/tenny-native-eval/python','$HOME/projects/tenny-torque200/src',str(root)]
from experiments.balance_diagnostics.runner import evaluate_manifest,write_json
from experiments.mature_balance_finetune.report import measure_case
from experiments.wheel_hold_ablation.train import account_training_objective
from force_teacher.contract import tree_digest
from legwheel.deployment.extract import load
from experiments.reward_ablation.train import tree_array_digest
import numpy as np
import matplotlib.pyplot as plt

manifest=json.loads((root/'inputs/manifest.json').read_text())
summary=json.loads((root/'comparison/summary.json').read_text())
candidates=[];stages=[]
for seed in [0,1,2]:
    identity=f'weak_wheel-seed-{seed}'
    for point in summary['progress_curves'][identity]:
        if point['steps']==0:continue
        group=point['primary'];window=group['windows']['first_30s']
        stages.append(dict(seed=seed,steps=point['steps'],completed_primary_cases=group['reached_30s_count'],
                           mean_episode_seconds=group['mean_observed_episode_seconds']))
        if window['complete_set']:
            candidates.append(dict(seed=seed,steps=point['steps'],primary_30s_rms_xy_m=window['rms_xy_displacement_m'],
                                   primary_30s_last10_xy_m=window['last_10s_mean_xy_displacement_m']))
chosen=min(candidates,key=lambda p:p['primary_30s_rms_xy_m'])
run=root/f"matrix/seed-{chosen['seed']}/weak_wheel"
descriptor=run/f"descriptor-step-{chosen['steps']}.json";checkpoint=run/f"checkpoint-step-{chosen['steps']}"
assert checkpoint.exists()
plan=dict(schema_version=1,seed=0,initial_states=manifest['initial_states'],policies=[dict(id='selected',label='Retrospective complete weak-wheel snapshot',descriptor=str(descriptor),source=chosen)],
          scope='Exploratory selection by minimum primary-state 30s XY RMS among complete weak-wheel snapshots; not an additional paired training seed.')
write_json(root/'selected-plan.json',plan)
bundle=evaluate_manifest(root/'selected-plan.json',root/'selected-evaluation',seconds=120.)
for case in bundle['cases']:
    case['measurement']=account_training_objective(measure_case(case['trace']),case['native_first_episode_metrics'],control_dt=.01,reference_dt=.01,wheel_position_weight=1.)
write_json(root/'selected-evaluation/results.json',bundle)
def average(items,key):return sum(c['measurement']['windows']['first_120s'][key] for c in items)/len(items)
primary=[c for c in bundle['cases'] if c['initial_state_id'] in ('nominal','pitch-plus','pitch-minus')]
heldout=[c for c in bundle['cases'] if c['initial_state_id'] in ('heldout-plus','heldout-minus')]
errors=[]
for case in bundle['cases']:
    trace=case['trace'];t=np.asarray(trace['t_s']);p=np.asarray(trace['base_position_m']);d=np.linalg.norm(p[:,:2]-p[0,:2],axis=1)
    assert t[-1]==120 and trace['termination']['truncated']
    tail=t>=110-1e-12;tt=t[tail];dd=d[tail]
    expected=float(np.sum(np.diff(tt)*(dd[1:]+dd[:-1])/2)/(tt[-1]-tt[0]))
    errors.append(abs(expected-case['measurement']['last_10s_mean_xy_displacement_m']))
assert max(errors)<1e-9
record=dict(schema_version=1,selected=chosen,candidate_count=len(candidates),total_nonzero_stages=len(stages),
            noncomplete_primary_stages=[s for s in stages if s['completed_primary_cases']<3],
            checkpoint=str(checkpoint),checkpoint_sha256=tree_digest(checkpoint),parameter_digest=tree_array_digest(load(str(checkpoint))),
            descriptor=str(descriptor),descriptor_sha256=hashlib.sha256(descriptor.read_bytes()).hexdigest(),
            primary_120s_last10_xy_m=average(primary,'last_10s_mean_xy_displacement_m'),
            heldout_120s_last10_xy_m=average(heldout,'last_10s_mean_xy_displacement_m'),
            primary_120s_rms_xy_m=average(primary,'rms_xy_displacement_m'),heldout_120s_rms_xy_m=average(heldout,'rms_xy_displacement_m'),
            all_five_completed_120s=True,independent_last10_max_abs_error=max(errors),
            cases=[dict(initial_state_id=c['initial_state_id'],measurement=c['measurement']) for c in bundle['cases']],
            selection_scope=plan['scope'],resume_scope='Full normalizer/actor/critic parameters retained and reload verified; Adam history absent, so continuation rebuilds optimizer.')
local=Path(__file__).resolve().parent/'selected-snapshot'
shutil.copytree(root/'selected-evaluation',local,dirs_exist_ok=True)
shutil.copy2(descriptor,local/'descriptor.json');write_json(local/'summary.json',record)
initial=json.loads((root/'evaluation/results.json').read_text())
fig,axes=plt.subplots(1,3,figsize=(13.5,3.5))
for label,data,identity in [('Mature reference',initial,'initial'),('Selected weak-wheel snapshot',bundle,'selected')]:
    case=next(c for c in data['cases'] if c['policy_id']==identity and c['initial_state_id']=='nominal')
    trace=case['trace'];t=np.asarray(trace['t_s']);p=np.asarray(trace['base_position_m']);v=np.asarray(trace['base_velocity_m_s'])
    axes[0].plot(t,1000*np.linalg.norm(p[:,:2]-p[0,:2],axis=1),label=label)
    axes[1].plot(t,np.rad2deg(np.asarray(trace['pitch_rad'])-trace['metadata']['reference_pitch_rad']))
    axes[2].plot(t,np.linalg.norm(v[:,:2],axis=1))
axes[0].set_ylabel('True XY displacement (mm)');axes[1].set_ylabel('Pitch error (deg)');axes[2].set_ylabel('Horizontal speed (m/s)')
for ax in axes:ax.set_xlabel('Time (s)');ax.grid(alpha=.25)
axes[0].legend(fontsize=8);fig.suptitle('Exploratory saved weak-wheel snapshot | no new training',fontsize=11);fig.tight_layout();fig.savefig(local/'comparison.png',dpi=140);plt.close(fig)
print(json.dumps({k:v for k,v in record.items() if k not in ('cases','noncomplete_primary_stages')}),flush=True)
