import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

os.environ.update(JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='')
root=Path('$HOME/projects/tenny-mature-finetune-20261116')
repo=Path('${REPO}')
sys.path[:0]=['$HOME/projects/tenny-native-eval/python','$HOME/projects/tenny-torque200/src',str(root)]
import numpy as np
from force_teacher.contract import tree_digest
from legwheel.deployment.extract import load
from native_eval import Evaluator
from native_eval.legwheel import policy_template
from native_eval.brax_policy import policy_from_brax
from experiments.reward_ablation.train import tree_array_digest
from experiments.mature_balance_finetune.train import verify_protected_inputs

start=time.monotonic()
read=lambda p:json.loads(Path(p).read_text())
manifests={condition:read(root/f'inputs/{condition}.json') for condition in ['high_lr','low_lr']}
assert {k:v for k,v in manifests['high_lr']['ppo'].items() if k!='learning_rate'}=={k:v for k,v in manifests['low_lr']['ppo'].items() if k!='learning_rate'}
assert manifests['high_lr']['initial_array_digest']==manifests['low_lr']['initial_array_digest']
frozen=read(root/'inputs/runtime-freeze.json')['files']
for path,digest in frozen.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
protected=verify_protected_inputs(manifests['high_lr'])
baseline=repo/'tasks/results/training-regression-audit-20261116/mature-validation/policy-0-long.npz'
run_checks=[]
with np.load(baseline) as old:
    obs=old['world_0_observation'][:256]
    for condition in ['high_lr','low_lr']:
        manifest=manifests[condition]
        for seed in range(3):
            run=root/f'matrix/seed-{seed}/{condition}'
            status=read(run/'status.json');effective=read(run/'effective-config.json')
            assert status['phase']=='completed' and status['steps']==2097152
            assert status['initial_array_digest']==manifest['initial_array_digest']==status['ppo_initial_array_digest']
            assert status['normalizer_drift']==0 and status['checkpoint_reload_exact'] and status['actor_changed']
            assert status['optimizer_learning_rate_verified']
            assert all(np.isclose(rate,manifest['ppo']['learning_rate'],rtol=1e-6,atol=0) for rate in status['observed_learning_rates'])
            assert effective['ppo']==dict(manifest['ppo'],seed=seed)
            assert effective['reward']==manifest['reward'] and effective['env']==manifest['env_config']
            progress=[json.loads(line) for line in (run/'progress.jsonl').read_text().splitlines() if line.strip()]
            assert [row['steps'] for row in progress]==list(range(0,2097153,262144))
            assert len(list(run.glob('descriptor-step-*.json')))==9
            for record in (json.loads(line) for line in (run/'training-metrics.jsonl').read_text().splitlines() if line.strip()):
                assert not record.get('nonfinite_scalar_metrics'),record
            errors={}
            with np.load(run/'trace-step-0.npz') as zero:
                for world in range(5):
                    for field in ['observation','action','qpos','qvel','ctrl','reward']:
                        key=f'world_{world}_{field}'
                        errors[key]=float(np.max(np.abs(zero[key]-old[key][:len(zero[key])])))
            assert max(errors.values())<1e-6,errors
            params=load(str(run/'checkpoint'))
            final_digest=tree_array_digest(params)
            assert final_digest==status['final_array_digest']
            descriptor=read(run/'descriptor.json')
            rebuilt=dict(descriptor,policy=policy_from_brax(params[:2],policy_template(manifest['policy_metadata'],manifest['ppo'])))
            export_error=float(np.max(np.abs(np.asarray(Evaluator(descriptor).policy_actions(obs))-np.asarray(Evaluator(rebuilt).policy_actions(obs)))))
            assert export_error<1e-6
            run_checks.append(dict(condition=condition,seed=seed,callbacks=9,initial_trajectory_max_abs_error=max(errors.values()),
                                   final_export_max_abs_error=export_error,finite_parameter_digest=final_digest,
                                   observed_learning_rates=status['observed_learning_rates'],checkpoint_sha256=tree_digest(run/'checkpoint')))
bundle=read(root/'evaluation/results.json')
measurement_errors=[]
for case in bundle['cases']:
    trace=case['trace'];record=case['measurement']
    t=np.asarray(trace['t_s']);p=np.asarray(trace['base_position_m']);v=np.asarray(trace['base_velocity_m_s'])
    d=np.sqrt(np.sum((p[:,:2]-p[0,:2])**2,axis=1))
    speed=np.sqrt(np.sum(v[:,:2]**2,axis=1));pitch=(np.asarray(trace['pitch_rad'])-trace['metadata']['reference_pitch_rad'])*180/np.pi
    addition=float(np.sum(10*np.expm1(-(d[1:]/.05)**2)*np.diff(t)/.01))
    measurement_errors += [abs(addition-record['true_position_addition_return']),abs(float(np.sum(trace['reward']))-record['base_return'])]
    assert record['observed_episode_seconds']==t[-1]
    for name,horizon in [('first_30s',30.),('first_120s',120.),('full_episode',t[-1])]:
        keep=t<=min(horizon,t[-1])+1e-12;times=t[keep];distance=d[keep];dt=np.diff(times)
        target=record['windows'][name]
        rms=float(np.sqrt(np.sum(dt*(distance[:-1]**2+distance[1:]**2)/2)/times[-1]))
        measurement_errors += [abs(rms-target['rms_xy_displacement_m']),abs(float(distance.max())-target['max_xy_displacement_m'])]
        for threshold,key in [(.05,'in_band_seconds_after_settling'),(.01,'within_1cm_seconds_after_settling')]:
            good=(distance<=threshold)&(np.abs(pitch[keep])<=10)&(speed[keep]<=.1)
            score=float(np.sum(dt[(times[:-1]>=1)&good[:-1]&good[1:]]))
            measurement_errors.append(abs(score-target[key]))
        if times[-1]>=30-1e-9:
            last=times>=times[-1]-10-1e-12;tt=times[last];dd=distance[last]
            tail=float(np.sum(np.diff(tt)*(dd[:-1]+dd[1:])/2)/(tt[-1]-tt[0]))
            measurement_errors.append(abs(tail-target['last_10s_mean_xy_displacement_m']))
        else:assert target['last_10s_mean_xy_displacement_m'] is None
assert len(bundle['cases'])==35 and max(measurement_errors)<1e-6
result=dict(phase='verified',protected_inputs=protected,frozen_runtime_files=len(frozen),runs=run_checks,cases_independently_recomputed=35,
            measurement_max_abs_error=max(measurement_errors),wall_seconds=time.monotonic()-start)
path=root/'comparison/verification.json';path.write_text(json.dumps(result,indent=2)+'\n')
shutil.copy2(path,repo/'tasks/results/mature-balance-finetune-20261116/comparison/verification.json')
shutil.copy2(root/'executed-source.tar.gz',repo/'tasks/results/mature-balance-finetune-20261116/comparison/executed-source.tar.gz')
print(json.dumps(result),flush=True)
