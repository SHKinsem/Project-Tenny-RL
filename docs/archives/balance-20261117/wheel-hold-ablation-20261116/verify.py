"""Post-run CPU audit. Importing this file does not load ML or run evaluation.

Run only after the three training runs and the 35-case evaluation finish.  This
reads saved parameters/trajectories and calls native policy inference only; it
does not launch training or additional physics rollouts.
"""
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time


ROOT=Path('$HOME/projects/tenny-wheel-hold-ablation-20261116')
OLD=Path('$HOME/projects/tenny-mature-finetune-20261116')
REPO=Path('${REPO}')
EXPECTED_STEPS=list(range(0,2097153,262144))
WHEEL_METRIC='reward/tracking_wheels_sinposition'


def require(condition,message):
    if not condition:raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def close(actual,expected,label,errors,*,tolerance=1e-6):
    require(isinstance(actual,(int,float)) and not isinstance(actual,bool) and math.isfinite(actual),
            label+': expected finite saved scalar')
    error=abs(float(actual)-float(expected))
    errors.append(error)
    require(error<=tolerance,label+': numerical discrepancy '+str(error))


def compare_measurement(saved,expected,label,errors):
    """Check helper numeric/boolean fields without conflating explanatory strings."""
    require(isinstance(saved,dict),label+': missing measurement')
    for key,value in expected.items():
        require(key in saved,label+': missing '+key)
        actual=saved[key];name=label+'.'+key
        if isinstance(value,dict):compare_measurement(actual,value,name,errors)
        elif value is None or isinstance(value,bool):
            require(type(actual) is type(value) and actual==value,name+': boundary/unknown mismatch')
        elif isinstance(value,(int,float)):close(actual,value,name,errors)


def compare_arrays(left,right,label,*,exact=True):
    import numpy as np
    left,right=np.asarray(left),np.asarray(right)
    require(left.shape==right.shape,label+': array shapes differ')
    require(np.isfinite(left).all() and np.isfinite(right).all(),label+': nonfinite array')
    error=float(np.max(np.abs(left.astype(float)-right.astype(float)))) if left.size else 0.
    require(np.array_equal(left,right) if exact else error<=1e-6,label+': array discrepancy '+str(error))
    return error


def verify_zero_prefix(path,old_path):
    """Physical arrays match exactly; the final 30s done flag is a new timeout."""
    import numpy as np
    errors={};boundaries=[]
    with np.load(path,allow_pickle=False) as new,np.load(old_path,allow_pickle=False) as old:
        for world in range(5):
            prefix=f'world_{world}_';length=int(new[prefix+'steps'])
            require(length==3000,'Initial mature stage must complete 30 seconds')
            names=[key for key in new.files if key.startswith(prefix) and key!=prefix+'steps']
            require({prefix+field for field in ('qpos','qvel','ctrl','observation','action','reward','done','xmat')}.issubset(names),
                    'Missing zero-stage raw trace fields')
            for key in names:
                require(key in old.files,'Old initial trace missing '+key)
                require(len(new[key])==length and len(old[key])>=length,'Invalid zero-stage prefix length')
                if key==prefix+'done':
                    compare_arrays(new[key][:-1],old[key][:length-1],key+' before timeout')
                    require(bool(new[key][-1]) and not bool(old[key][length-1]),'Expected only the 30s timeout done marker')
                    boundaries.append(dict(world=world,step=length,stage_done=True,old_120s_prefix_done=False))
                else:errors[key]=compare_arrays(new[key],old[key][:length],key)
    return dict(max_abs_error=max(errors.values()),array_fields=len(errors),
                expected_time_limit_marker_differences=boundaries)


def compare_raw_files(new_path,old_path):
    import numpy as np
    with np.load(new_path,allow_pickle=False) as new,np.load(old_path,allow_pickle=False) as old:
        require(set(new.files)==set(old.files),'Old/new same-horizon NPZ fields differ')
        errors={name:compare_arrays(new[name],old[name],name) for name in new.files}
    return dict(new=str(new_path),old=str(old_path),array_fields=len(errors),max_abs_error=max(errors.values()))


def independent_case(case,raw,initial,layout,task,weight,errors):
    """Derive physical summaries from raw qpos/qvel, separately from report code."""
    import numpy as np
    trace=case['trace'];saved=case['measurement'];times=np.asarray(trace['t_s'],dtype=float)
    length=len(times)-1;dt=.01;q,v=layout['root_qpos'],layout['root_qvel']
    require(0<length<=12000,'Invalid first-episode length')
    require(np.array_equal(times,np.arange(length+1)*dt),'Normalized times differ from raw control intervals')
    qpos=np.vstack([initial['qpos'],raw['qpos'][:length]])
    qvel=np.vstack([initial['qvel'],raw['qvel'][:length]])
    require(np.isfinite(qpos).all() and np.isfinite(qvel).all(),'Nonfinite raw first episode')
    positions=qpos[:,q:q+3];velocity=qvel[:,v:v+3]
    quaternion=qpos[:,q+3:q+7]
    quaternion=quaternion/np.sqrt(np.sum(quaternion**2,axis=1,keepdims=True))
    w,x,y,z=quaternion.T
    pitch=np.arcsin(np.clip(2*(w*y-z*x),-1.,1.))
    reference=math.asin(float(initial['extra']['sin_balance_zero'][0]))
    compare_arrays(trace['base_position_m'],positions,'normalized true base position')
    compare_arrays(trace['base_velocity_m_s'],velocity,'normalized true base velocity')
    compare_arrays(trace['pitch_rad'],pitch,'normalized pitch from qpos',exact=False)
    close(trace['metadata']['reference_pitch_rad'],reference,'reset pitch reference',errors,tolerance=1e-12)
    compare_arrays(trace['reward'],np.r_[0.,raw['reward'][:length]],'original base reward sequence')
    compare_arrays(trace['done'],np.r_[False,raw['done'][:length]],'first-episode done sequence')
    done=np.asarray(raw['done'][:length],dtype=bool)
    require(done[-1] and not done[:-1].any(),'Raw first episode includes reset or lacks done')
    termination=trace['termination']
    require(type(termination['terminated']) is bool and type(termination['truncated']) is bool
            and termination['terminated']!=termination['truncated'],'Invalid terminal flags')
    close(termination['time_s'],times[-1],'termination time',errors,tolerance=1e-12)
    displacement=np.sqrt(np.sum((positions[:,:2]-positions[0,:2])**2,axis=1))
    speed=np.sqrt(np.sum(velocity[:,:2]**2,axis=1));pitch_error=np.degrees(pitch-reference)
    addition=float(np.sum(10*np.expm1(-(displacement[1:]/.05)**2)*np.diff(times)/.01))
    base=float(np.sum(np.asarray(raw['reward'][:length],dtype=float)))
    native=case['native_first_episode_metrics']
    weights={
        'tracking_lin_vel':task['vel_scale']*task['enable_lin_vel_x'],
        'tracking_ang_vel':task['vel_scale']*task['enable_ang_vel_yaw'],
        'tracking_wheels_sinposition':task['zero_keep_scale']*task['wheel_sinpos_start'],
        'tracking_sinyaw':task['zero_keep_scale']*task['sinyaw'],
        'pitch_cost':task['pitch_scale'],'roll_cost':task['roll_scale']*task['enable_roll_value'],
        'tracking_height_jnt':task['height_scale']*task['enable_height_value'],
        'energy_cost':task['energy_scale'],'action_rate':task['action_rate_scale'],
        'pitch_velocity_penalty':task['pitch_velocity_penalty_scale'],
    }
    require(task['zero_keep_scale']==10 and not task['termination_penalty'],
            'Common native base must retain original weight10 and disabled terminal penalty')
    reconstruction=math.fsum(scale*float(native['reward/'+name]) for name,scale in weights.items())*dt/task['reference_dt']
    close(base,reconstruction,'base10 reconstructed from preweight native metrics',errors)
    adjustment=(weight-10.)*float(native[WHEEL_METRIC])*dt/task['reference_dt']
    close(saved['base_return'],base,'base10 return',errors)
    close(saved['true_position_addition_return'],addition,'true XY addition',errors)
    close(saved['common_xy_objective_return'],base+addition,'common base10 plus XY',errors)
    close(saved['wheel_position_adjustment_return'],adjustment,'wheel-only objective adjustment',errors)
    close(saved['training_objective_return'],base+addition+adjustment,'actual training objective',errors)
    close(saved['native_wheel_position_raw_score_sum'],native[WHEEL_METRIC],'raw wheel cumulative score',errors)
    close(saved['wheel_position_weight'],weight,'condition wheel weight',errors,tolerance=0.)
    require('objective_return' not in saved,'Ambiguous objective_return must not label the new condition')
    if weight==10:require(adjustment==0.,'Reused original objective must remain the common base10 plus XY')
    def reached(horizon):
        return termination['truncated'] if math.isclose(times[-1],horizon,rel_tol=0,abs_tol=1e-9) else times[-1]>horizon
    require(saved['reached_30s']==reached(30.) and saved['reached_120s']==reached(120.),'Reached-horizon semantics differ')
    close(saved['observed_episode_seconds'],times[-1],'observed first episode',errors,tolerance=1e-12)
    summaries={}
    for name,horizon in [('first_30s',30.),('first_120s',120.),('full_episode',times[-1])]:
        keep=times<=min(horizon,times[-1])+1e-12;tt=times[keep];dd=displacement[keep];interval=np.diff(tt)
        ss=speed[keep];pp=pitch_error[keep];target=saved['windows'][name]
        duration=tt[-1]
        values=dict(max_xy_displacement_m=float(dd.max()),
            rms_xy_displacement_m=float(np.sqrt(np.sum(interval*(dd[:-1]**2+dd[1:]**2)/2)/duration)),
            horizontal_speed_rms_m_s=float(np.sqrt(np.sum(interval*(ss[:-1]**2+ss[1:]**2)/2)/duration)),
            pitch_error_rms_deg=float(np.sqrt(np.sum(interval*(pp[:-1]**2+pp[1:]**2)/2)/duration)))
        for threshold,key in [(.05,'in_band_seconds_after_settling'),(.01,'within_1cm_seconds_after_settling')]:
            good=(dd<=threshold)&(np.abs(pp)<=10)&(ss<=.1)
            values[key]=float(np.sum(interval[(tt[:-1]>=1.)&good[:-1]&good[1:]]))
        tail=None
        if duration>=30.-1e-9:
            keep_tail=tt>=duration-10.-1e-12;tail_t=tt[keep_tail];tail_d=dd[keep_tail]
            tail=float(np.sum(np.diff(tail_t)*(tail_d[:-1]+tail_d[1:])/2)/(tail_t[-1]-tail_t[0]))
            close(target['last_10s_observed_seconds'],tail_t[-1]-tail_t[0],'tail observation coverage',errors)
        else:require(target['last_10s_observed_seconds'] is None,'Short episode tail coverage must be unknown')
        values['last_10s_mean_xy_displacement_m']=tail
        for key,value in values.items():
            if value is None:require(target[key] is None,name+'.'+key+' must be unknown')
            else:close(target[key],value,name+'.'+key,errors)
        close(target['observed_seconds'],duration,name+' duration',errors,tolerance=1e-12)
        require(target['reached_requested_horizon']==reached(horizon),name+' reached status')
        summaries[name]=values
    for key in ('max_xy_displacement_m','rms_xy_displacement_m','horizontal_speed_rms_m_s',
                'pitch_error_rms_deg','last_10s_mean_xy_displacement_m'):
        value=summaries['full_episode'][key]
        if value is None:require(saved[key] is None,'Top-level tail must be unknown')
        else:close(saved[key],value,'full episode '+key,errors)
    for source,target in [('in_band_seconds_after_settling','first30s_in_band_seconds'),
                          ('within_1cm_seconds_after_settling','first30s_within_1cm_seconds')]:
        close(saved[target],summaries['first_30s'][source],target,errors)
    return dict(policy_id=case['policy_id'],initial_state_id=case['initial_state_id'],
        raw_steps=length,wheel_position_weight=weight,base10_return=base,
        common_xy_objective_return=base+addition,wheel_position_adjustment_return=adjustment,
        training_objective_return=base+addition+adjustment)


def verify():
    # These imports occur only after main fixes the backend to CPU.
    import jax
    import mujoco
    import numpy as np
    from force_teacher.contract import tree_digest
    from force_teacher.ppo import normalizer_max_abs_change
    from legwheel.deployment.extract import load
    from native_eval import Evaluator
    from native_eval.brax_policy import policy_from_brax
    from native_eval.legwheel import policy_template
    from experiments.balance_diagnostics.legwheel import model_layout
    from experiments.mature_balance_finetune.report import measure_case
    from experiments.reward_ablation.train import canonical_initial_params,tree_array_digest
    from experiments.wheel_hold_ablation.train import account_training_objective,verify_protected_inputs
    start=time.monotonic();errors=[]
    require(all(device.platform=='cpu' for device in jax.devices()),'Verification must use CPU only')
    manifest=read(ROOT/'inputs/manifest.json');matrix=read(ROOT/'matrix/status.json')
    require(matrix['phase']=='completed' and len(matrix['completed'])==3,'Three new runs must be complete')
    require(matrix['total_new_steps']==3*2097152,'Unexpected new training budget')
    require(manifest['condition']=='weak_wheel' and manifest['seeds']==[0,1,2],'Unexpected experiment matrix')
    require(manifest['wheel_hold']==dict(original_weight=10.,training_weight=1.,metric=WHEEL_METRIC),'Changed wheel-reward contract')
    require(manifest['ppo']['learning_rate']==3e-5 and manifest['ppo']['num_evals']==9
            and manifest['ppo']['episode_length']==3000,'Changed matched PPO settings')
    freeze_path=ROOT/'inputs/runtime-freeze.json';frozen=read(freeze_path)['files']
    require(sha(freeze_path)==matrix['runtime_freeze_sha256'],'Runtime freeze changed after matrix')
    old_frozen=read(OLD/'inputs/runtime-freeze.json')['files']
    for path,digest in {**old_frozen,**frozen}.items():require(sha(path)==digest,'Frozen file changed: '+path)
    preparation=read(ROOT/'preparation.json')
    require(sha(ROOT/'executed-source.tar.gz')==preparation['source_archive_sha256'],'Executed-source archive changed')
    protection=verify_protected_inputs(manifest)
    gate=read(ROOT/'gate.json')
    require(gate['phase']=='passed' and gate['sampled_states']==16 and gate['reset_physics_and_actor_observation_exact'],
            'Missing real-state reward equivalence gate')
    for key in ('weight10_all_reward_terms_max_abs_error','weight1_other_terms_max_abs_error',
                'weight1_single_wheel_adjustment_max_abs_error'):require(gate[key]==0,'Reward gate is not exact: '+key)
    def functions(path):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
    new_functions=functions(ROOT/'experiments/wheel_hold_ablation/train.py')
    old_functions=functions(OLD/'experiments/mature_balance_finetune/train.py')
    for name in ('train_with_ppo','observed_learning_rate','verify_protected_inputs','_mean_scalars'):
        require(new_functions[name]==old_functions[name],'Changed matched PPO helper: '+name)
    initial=canonical_initial_params(load(manifest['initial_checkpoint']))
    initial_digest=tree_array_digest(initial)
    require(initial_digest==manifest['initial_array_digest'] and initial_digest.startswith('7d7'),'Unexpected complete mature initializer')
    require('20261115T081557.290279Z_bd32fd40' in manifest['initial_checkpoint']
            and manifest['initial_checkpoint'].endswith('/000002801664'),'Unexpected mature source checkpoint')
    require(tree_digest(manifest['initial_checkpoint'])==manifest['initial_checkpoint_sha256'],'Original checkpoint bytes changed')
    common=read(manifest['base_descriptor']);task=common['task']['config']
    layout=model_layout(mujoco.MjModel.from_xml_path(common['xml_path']),common['task'])
    require(task['zero_keep_scale']==10 and task['reference_dt']==.01,'Common native task is not the original base10')
    old_initial=OLD/'evaluation/policy-0-long.npz'
    with np.load(old_initial,allow_pickle=False) as original:
        observations=np.concatenate([original[f'world_{world}_observation'][:64] for world in range(5)],axis=0)
    template=policy_template(manifest['policy_metadata'],manifest['ppo'])
    def export_check(params,descriptor,reference=None):
        rebuilt=dict(descriptor,policy=policy_from_brax(params[:2],template))
        actions=np.asarray(Evaluator(descriptor).policy_actions(observations))
        exported=np.asarray(Evaluator(rebuilt).policy_actions(observations))
        error=compare_arrays(actions,exported,'descriptor vs full checkpoint native policy export',exact=False)
        if reference is not None:compare_arrays(actions,Evaluator(reference).policy_actions(observations),'zero-step original native action')
        return error
    baselines=[]
    require([item['seed'] for item in manifest['baseline_runs']]==[0,1,2],'Reused controls must have three ordered seeds')
    for item in manifest['baseline_runs']:
        seed=item['seed'];run=Path(item['run_dir']);status=read(run/'status.json');effective=read(run/'effective-config.json')
        require(item['status']==dict(status,condition='original_wheel',reused=True,source_run=str(run)),
                'Reused baseline status identity changed')
        require(status['phase']=='completed' and status['steps']==2097152
                and status['initial_array_digest']==initial_digest==status['ppo_initial_array_digest'],'Unmatched baseline initializer/budget')
        require(effective['ppo']==dict(manifest['ppo'],seed=seed) and effective['reward']==manifest['reward']
                and effective['env']==manifest['env_config'],'Reused baseline PPO/reward/environment mismatch')
        require(status['normalizer_drift']==0 and status['checkpoint_reload_exact'] and status['optimizer_learning_rate_verified'],
                'Invalid baseline fixed normalizer/checkpoint/rate')
        require(status['observed_learning_rates'] and all(math.isclose(rate,3e-5,rel_tol=1e-6,abs_tol=0)
                for rate in status['observed_learning_rates']),'Reused baseline actual rate differs')
        require(sha(item['descriptor'])==status['descriptor_sha256'] and tree_digest(item['checkpoint'])==item['checkpoint_sha256'],
                'Reused baseline bytes changed')
        old_progress=jsonl(run/'progress.jsonl')
        require([row['steps'] for row in old_progress]==EXPECTED_STEPS
                and old_progress[0]['complete_array_digest']==initial_digest,'Reused baseline full initializer/progress mismatch')
        old_params=load(item['checkpoint']);old_digest=tree_array_digest(old_params)
        require(old_digest==status['final_array_digest'],'Reused baseline full checkpoint changed')
        old_export=export_check(old_params,read(item['descriptor']))
        baselines.append(dict(seed=seed,reused=True,source_run=str(run),source_checkpoint=item['checkpoint'],
            initializer=initial_digest,final_array_digest=old_digest,native_export_max_abs_error=old_export,
            actual_objective='Original base10 plus true XY; wheel adjustment zero.'))
    run_checks=[]
    for seed in range(3):
        run=ROOT/f'matrix/seed-{seed}/weak_wheel';status=read(run/'status.json');effective=read(run/'effective-config.json')
        require(status['phase']=='completed' and status['condition']=='weak_wheel' and status['seed']==seed
                and status['steps']==2097152,'Incomplete new fixed-budget run')
        require(status['initial_array_digest']==initial_digest==status['ppo_initial_array_digest'],'New run did not restore full mature parameters')
        require(status['normalizer_drift']==0 and status['checkpoint_reload_exact'] and status['actor_changed'],'Invalid final parameter guard')
        require(status['optimizer_learning_rate_verified'] and status['observed_learning_rates'],'Missing actual optimizer rate')
        for rate in status['observed_learning_rates']:require(math.isclose(rate,3e-5,rel_tol=1e-6,abs_tol=0),'Actual rate differs from 3e-5')
        require(effective['ppo']==dict(manifest['ppo'],seed=seed) and effective['env']==manifest['env_config']
                and effective['reward']==manifest['reward'] and effective['wheel_hold']==manifest['wheel_hold'],'Effective matched configuration changed')
        require(effective['optimizer_learning_rate_verified'] and effective['brax_learning_rate']==3e-5,'Unverified effective optimizer rate')
        progress=jsonl(run/'progress.jsonl');snapshots=status['snapshot_checkpoints']
        require([row['steps'] for row in progress]==EXPECTED_STEPS==status['callback_steps']
                and [row['steps'] for row in snapshots]==EXPECTED_STEPS,'Missing exact nine-stage callbacks/checkpoints')
        require({p.name for p in run.glob('descriptor-step-*.json')}=={f'descriptor-step-{s}.json' for s in EXPECTED_STEPS},'Unexpected snapshot descriptors')
        require({p.name for p in run.glob('checkpoint-step-*')}=={f'checkpoint-step-{s}' for s in EXPECTED_STEPS},'Unexpected parameter snapshot directories')
        logged_rates=[]
        for record in jsonl(run/'training-metrics.jsonl'):
            require(not record.get('nonfinite_scalar_metrics'),'Nonfinite optimizer/loss metric was recorded')
            for key,value in record['metrics'].items():
                require(math.isfinite(value),'Nonfinite scalar metric')
                if key.rsplit('/',1)[-1]=='learning_rate':logged_rates.append(value)
        require(logged_rates and all(math.isclose(rate,3e-5,rel_tol=1e-6,abs_tol=0) for rate in logged_rates),'Logged actual rate differs')
        checks=[]
        for progress_row,snapshot in zip(progress,snapshots):
            steps=snapshot['steps'];path=run/f'checkpoint-step-{steps}'
            require(Path(snapshot['path'])==path and snapshot['reload_exact'],'Snapshot path/reload metadata mismatch')
            require(snapshot['checkpoint_sha256']==tree_digest(path)==progress_row['checkpoint_sha256'],'Snapshot bytes changed')
            params=load(str(path));digest=tree_array_digest(params)
            require(len(params)==3 and digest==snapshot['complete_array_digest']==progress_row['complete_array_digest'],'Incomplete/nonfinite snapshot parameters')
            require(float(normalizer_max_abs_change(initial[0],params[0]))==0,'Snapshot normalization drifted')
            descriptor_path=run/f'descriptor-step-{steps}.json';descriptor=read(descriptor_path);metadata=descriptor['metadata']
            require(sha(descriptor_path)==progress_row['descriptor_sha256'] and metadata['training_steps']==steps
                    and metadata['finetune_steps']==steps and metadata['checkpoint']==str(path)
                    and metadata['checkpoint_sha256']==snapshot['checkpoint_sha256']
                    and metadata['checkpoint_reload_exact'],'Snapshot descriptor/parameter identity mismatch')
            require(metadata['wheel_hold']==manifest['wheel_hold'] and metadata['initial_array_digest']==initial_digest,'Snapshot objective/initializer mismatch')
            export_error=None
            if steps in (0,2097152):export_error=export_check(params,descriptor,common if steps==0 else None)
            if steps==0:require(digest==initial_digest,'Zero-step full parameters differ')
            if steps==2097152:require(digest==status['final_array_digest'],'Last callback and final parameters differ')
            checks.append(dict(steps=steps,complete_array_digest=digest,reload_exact=True,normalizer_drift=0,
                               checkpoint_sha256=snapshot['checkpoint_sha256'],native_export_max_abs_error=export_error))
        prefix=verify_zero_prefix(run/'trace-step-0.npz',old_initial)
        final_params=load(str(run/'checkpoint'));final_digest=tree_array_digest(final_params)
        final_descriptor=read(run/'descriptor.json')
        require(final_digest==status['final_array_digest'] and sha(run/'descriptor.json')==status['descriptor_sha256'],'Final saved artifacts changed')
        require(final_descriptor['metadata']['checkpoint_sha256']==tree_digest(run/'checkpoint'),'Final descriptor checkpoint bytes mismatch')
        require(tree_array_digest(final_params[1])!=tree_array_digest(initial[1]),'Actor did not update')
        final_export=export_check(final_params,final_descriptor)
        run_checks.append(dict(seed=seed,reused=False,initial_array_digest=initial_digest,
            observed_learning_rates=status['observed_learning_rates'],checkpoint_snapshots=checks,
            zero_step_trajectory=prefix,final_array_digest=final_digest,final_native_export_max_abs_error=final_export))
    bundle=read(ROOT/'evaluation/results.json')
    require(bundle['kind']=='wheel_hold_ablation_v1' and len(bundle['cases'])==35 and len(bundle['policies'])==7,'Incomplete final evaluation')
    require(bundle['initial_states']==manifest['initial_states'] and bundle['horizon_s']==120.,'Changed final physical evaluation conditions')
    expected_ids=['initial']+[f'original_wheel-seed-{s}' for s in range(3)]+[f'weak_wheel-seed-{s}' for s in range(3)]
    require([p['id'] for p in bundle['policies']]==expected_ids,'Final policy index/source identity changed')
    descriptor_paths=[Path(manifest['base_descriptor'])]+[Path(item['descriptor']) for item in manifest['baseline_runs']]+[
        ROOT/f'matrix/seed-{seed}/weak_wheel/descriptor.json' for seed in range(3)]
    state_digest=hashlib.sha256(json.dumps(manifest['initial_states'],sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()
    require(bundle['initial_states_sha256']==state_digest,'Final reset state fingerprint changed')
    for index,policy in enumerate(bundle['policies']):
        source=policy['source'];weight=1. if index>=4 else 10.
        descriptor=read(descriptor_paths[index])
        policy_digest=hashlib.sha256(json.dumps(descriptor['policy'],sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()
        require(policy['descriptor_sha256']==sha(descriptor_paths[index]) and policy['policy_sha256']==policy_digest
                and policy['metadata']==descriptor.get('metadata',{}),'Final evaluated descriptor/policy identity mismatch')
        require(descriptor['task']==common['task'] and descriptor['physics_dt']==common['physics_dt']
                and descriptor['eval_defaults']['control_dt']==.01
                and sha(descriptor['xml_path'])==sha(common['xml_path']),'Compared native task/dynamics changed')
        require(source['wheel_position_weight']==weight,'Policy wheel objective identity mismatch')
        require(source['reused']==(1<=index<=3),'Reused/new provenance mismatch')
        if index:require(source['learning_rate']==3e-5,'Compared learning rates differ')
        if 1<=index<=3:
            item=manifest['baseline_runs'][index-1]
            require(source['source_run']==item['run_dir'] and source['source_checkpoint']==item['checkpoint'],
                    'Final reused policy source changed')
    raw_reuse=[dict(policy_id=expected_ids[index],**compare_raw_files(ROOT/f'evaluation/policy-{index}-long.npz',
               OLD/f'evaluation/policy-{index}-long.npz')) for index in range(4)]
    case_checks=[];state_ids=[item['id'] for item in manifest['initial_states']];seen=set()
    for index,policy in enumerate(bundle['policies']):
        with np.load(ROOT/f'evaluation/policy-{index}-long.npz',allow_pickle=False) as raw:
            for case in [item for item in bundle['cases'] if item['policy_id']==policy['id']]:
                state_id=case['initial_state_id'];key=(policy['id'],state_id)
                require(key not in seen and state_id in state_ids,'Duplicate/unknown final case');seen.add(key)
                require(case['trace']['metadata']['policy_sha256']==policy['policy_sha256']
                        and case['trace']['metadata']['initial_states_sha256']==state_digest
                        and case['trace']['metadata']['initial_state_id']==state_id,'Trace identity differs from evaluated policy/reset')
                world=state_ids.index(state_id);prefix=f'world_{world}_'
                arrays={name[len(prefix):]:raw[name] for name in raw.files if name.startswith(prefix)}
                require(int(arrays['steps'])==12000,'Final native raw workload is not 120 seconds')
                helper=account_training_objective(measure_case(case['trace']),case['native_first_episode_metrics'],
                    control_dt=.01,reference_dt=.01,wheel_position_weight=policy['source']['wheel_position_weight'])
                compare_measurement(case['measurement'],helper,'saved/shared helper measurement',errors)
                case_checks.append(independent_case(case,arrays,manifest['initial_states'][world]['state'],layout,task,
                    policy['source']['wheel_position_weight'],errors))
    require(seen=={(identity,state) for identity in expected_ids for state in state_ids},'Missing physical evaluation cases')
    # Recheck immutable inputs after every parameter and trajectory read.
    require(verify_protected_inputs(manifest)==protection,'Protected input inventory changed')
    for path,digest in {**old_frozen,**frozen}.items():require(sha(path)==digest,'Frozen source changed during audit: '+path)
    require(sha(freeze_path)==matrix['runtime_freeze_sha256'],'Runtime freeze changed during audit')
    return dict(phase='verified',kind='wheel_hold_ablation_verification_v1',
        initial_array_digest=initial_digest,protected_inputs=protection,
        frozen_runtime_files=len(frozen),reused_original_frozen_files=len(old_frozen),
        source_archive_sha256=sha(ROOT/'executed-source.tar.gz'),
        unchanged_ppo_helper_ASTs=True,reward_gate=gate,reused_baseline_runs=baselines,runs=run_checks,
        same_120s_raw_reuse_checks=raw_reuse,raw_reused_baseline_cases_exact=15,raw_unchanged_reference_cases_exact=5,
        cases_independently_recomputed=len(case_checks),case_objective_checks=case_checks,
        measurement_max_abs_error=max(errors),
        objective_accounting='Native base remains wheel/yaw10. Common XY = base10 + true XY; actual = common XY + (wheel_weight-10)*native preweight wheel score*control_dt/reference_dt. No PPO reward scaling or duplicated XY is included.',
        checkpoint_scope='All nine parameter checkpoints contain normalizer, actor and critic. Adam state is absent; later continuation starts new optimizer state.',
        initial_prefix_boundary='30s stage and 120s old initial physical/reward arrays match exactly; only the final 30s timeout done marker differs as expected.',
        devices=[str(device) for device in jax.devices()],wall_seconds=time.monotonic()-start)


def main():
    os.environ.update(JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',XLA_PYTHON_CLIENT_PREALLOCATE='false')
    sys.path[:0]=['$HOME/projects/tenny-native-eval/python','$HOME/projects/tenny-torque200/src',str(ROOT)]
    result=verify()
    path=ROOT/'comparison/verification.json';path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    local=REPO/'tasks/results/wheel-hold-ablation-20261116/comparison/verification.json'
    local.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,local)
    print(json.dumps(result,allow_nan=False),flush=True)


if __name__=='__main__':main()
