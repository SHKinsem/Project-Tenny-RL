import hashlib
import json
from pathlib import Path
import numpy as np

local=Path(__file__).resolve().parent/'selected-stage'
bundle=json.loads((local/'results.json').read_text())
summary=json.loads((local/'summary.json').read_text())
descriptor=Path(summary['descriptor'])
assert hashlib.sha256(descriptor.read_bytes()).hexdigest()==hashlib.sha256((local/'descriptor.json').read_bytes()).hexdigest()
errors=[];rows=[]
with np.load(local/'policy-0-long.npz') as raw:
    for world,case in enumerate(bundle['cases']):
        trace=case['trace'];m=case['measurement'];t=np.asarray(trace['t_s']);p=np.asarray(trace['base_position_m'])
        assert t[-1]==120 and trace['termination']['truncated'] and not trace['termination']['terminated']
        d=np.linalg.norm(p[:,:2]-p[0,:2],axis=1);dt=np.diff(t)
        rms=float(np.sqrt(np.sum(dt*(d[:-1]**2+d[1:]**2)/2)/t[-1]))
        tail=t>=110-1e-12;tt=t[tail];dd=d[tail]
        mean=float(np.sum(np.diff(tt)*(dd[:-1]+dd[1:])/2)/(tt[-1]-tt[0]))
        errors += [abs(rms-m['rms_xy_displacement_m']),abs(mean-m['last_10s_mean_xy_displacement_m'])]
        # This verified legwheel model's free-root quaternion is qpos[3:7].
        q=np.vstack([bundle['initial_states'][world]['state']['qpos'],raw[f'world_{world}_qpos'][:12000]])[:,3:7]
        w,x,y,z=q.T;yaw=np.unwrap(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))
        rows.append(dict(initial_state_id=case['initial_state_id'],rms_xy_m=rms,last10_xy_m=mean,
                         heading_change_at_120s_deg=float(np.rad2deg(yaw[-1]-yaw[0]))))
assert max(errors)<1e-9
result=dict(phase='verified',cases=5,all_completed_120s=True,descriptor_sha256=hashlib.sha256(descriptor.read_bytes()).hexdigest(),
            independent_xy_max_abs_error=max(errors),measurements=rows,
            scope='Exploratory saved-stage selection; independent XY remeasurement does not convert it into a paired training replicate or a full resume checkpoint.')
(local/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
