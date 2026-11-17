import json
from pathlib import Path
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

local=Path('${REPO}/tasks/results/torque-limit-audit-20261116')
target=Path('$HOME/projects/tenny-torque-limit-audit-20261116')
result=json.loads((local/'results.json').read_text())
old=Path('$HOME/projects/tenny-reward-ablation-20261116')
new=Path('$HOME/projects/tenny-horizon-ablation-20261116')
nominal=[case for case in result['cases'] if case['initial_state_id']=='nominal']
fig,axes=plt.subplots(7,2,figsize=(14,18),squeeze=False)
for row,case in enumerate(nominal):
    identity=case['policy_id']
    source=(old/'evaluations'/('initial' if identity=='initial' else identity.replace('h6','baseline'))
            if not identity.startswith('h30') else new/'evaluations'/identity)
    n=case['validation']['control_steps']
    with np.load(source/'policy-0-long.npz') as data:
        for axis,color,label in ((0,'#0072B2','Left leg'),(1,'#D55E00','Right leg')):
            axes[row,0].plot(np.arange(1,n+1)*.01,data['world_0_ctrl'][:n,axis],color=color,label=label,lw=.9)
            axes[row,1].stairs(data['world_0_action'][:n,axis],np.arange(n+1)*.01,color=color,label=label,lw=.8)
    for col in (0,1):
        panel=axes[row,col]
        panel.set_title(identity+(' | Final-substep torque samples' if col==0 else ' | Held normalized leg-force action'),fontsize=11)
        panel.set_xlim(0,10);panel.set_ylim(-1.1,1.1)
        panel.axvline(1,color='.6',ls=':',lw=.8)
        panel.axvline(case['departure_s'],color='#9C2B36',ls='--',lw=.9,label='Persistent band departure')
        panel.axhline(1,color='.4',ls='--',lw=.7);panel.axhline(-1,color='.4',ls='--',lw=.7)
        if col==1:
            panel.axhspan(.95,1.,color='.5',alpha=.12);panel.axhspan(-1.,-.95,color='.5',alpha=.12)
        panel.set_ylabel('N m' if col==0 else 'a = F / Fs')
        panel.grid(alpha=.16);panel.spines[['top','right']].set_visible(False)
        if row==0:panel.legend(fontsize=8,loc='lower left')
        if row==6:panel.set_xlabel('Time (s)')
fig.suptitle('Leg actuator audit | nominal initial state | exact original native replays\nTorque: 100 Hz final-substep samples; action: held 10 ms. Zero hard clipping verified separately at every physics substep.',fontsize=13)
fig.tight_layout(rect=(0,0,1,.955))
fig.savefig(target/'nominal-torque-action.png',dpi=150)
plt.close(fig)
shutil.copy2(target/'nominal-torque-action.png',local/'nominal-torque-action.png')
print('Saved nominal-torque-action.png')
