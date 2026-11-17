"""Fixed-budget wheel-position reward comparison; original head and complete restore.

The old force PPO wrapper overwrites learning_rate.  This experiment passes the
same fixed settings straight to Brax and verifies its logged optimizer rate.
"""
from dataclasses import asdict

from . import WHEEL_HOLD
from functools import partial
import hashlib
import inspect
import json
import math
from pathlib import Path
import shutil
import time

from experiments.reward_ablation.train import (
    canonical_initial_params, tree_array_digest, write_json,
)


def validate_manifest(manifest,seed):
    """Bind executed reward/settings; smaller matching budgets permit isolated smoke."""
    from experiments.reward_ablation.rewards import ARMS
    expected=asdict(ARMS['position'])
    if manifest.get('arms',{}).get('position')!=expected:
        raise ValueError('Need fixed position weight 10, scale .05, and zero added posture weight')
    if type(seed) is not int or seed not in manifest['seeds']:
        raise ValueError('Seed outside the frozen experiment')
    ppo=manifest['ppo']
    if ppo['learning_rate']!=3e-5:
        raise ValueError('This comparison keeps the original learning rate 3e-5')
    if manifest.get('condition')!='weak_wheel' or manifest.get('wheel_hold')!=dict(WHEEL_HOLD):
        raise ValueError('Need the fixed weak-wheel condition and wheel-position weights 10 to 1')
    fixed=dict(normalize_observations=True,normalize_observations_mode='ema',
        normalize_observations_std_eps=0.,normalize_until_count=0,
        deterministic_eval=True,entropy_cost=.001,run_evals=False,episode_length=3000,action_repeat=1)
    for key,value in fixed.items():
        if ppo.get(key)!=value:
            raise ValueError('Changed fixed PPO behavior: '+key)
    if ppo.get('restore_checkpoint_path') is not None:
        raise ValueError('The complete mature initializer is supplied through restore_params only')
    steps=manifest['steps_per_run']
    if type(steps) is not int or steps<=0 or ppo['num_timesteps']!=steps:
        raise ValueError('Manifest and actual PPO step budgets differ')
    evaluations=ppo['num_evals']
    if type(evaluations) is not int or evaluations<2 or steps%(evaluations-1):
        raise ValueError('Need equally spaced initial and post-training policy callbacks')
    batch_steps=ppo['batch_size']*ppo['unroll_length']*ppo['num_minibatches']*ppo['action_repeat']
    stride=steps//(evaluations-1)
    if stride%batch_steps:
        raise ValueError('Callback stride would be rounded by the real PPO batch budget')
    if manifest['env_config']['ctrl_dt']!=.01 or manifest['env_config']['sim_dt']!=.0025:
        raise ValueError('This comparison preserves 100 Hz control and 400 Hz physics')
    states=manifest['initial_states']
    if len(states)!=5 or len({item['id'] for item in states})!=5:
        raise ValueError('Need the same five explicit physical evaluation states')
    if not manifest.get('model_origin') or not manifest.get('protected_inputs'):
        raise ValueError('Need identified mature-model origin and original byte protection')
    return expected,list(range(0,steps+1,stride))


def observed_learning_rate(metrics,expected):
    """Read actual optimizer metrics, keeping float32 rounding but rejecting overrides."""
    found=[]
    for key,value in metrics.items():
        if key.rsplit('/',1)[-1]=='learning_rate':
            value=float(value)
            if not math.isfinite(value) or not math.isclose(value,expected,rel_tol=1e-6,abs_tol=0.):
                raise ValueError('Brax optimizer learning rate differs from manifest: '+str(value))
            found.append(value)
    return found


def account_training_objective(measurement,native_metrics,*,control_dt,reference_dt,
                              wheel_position_weight=1.):
    """Use native first-episode raw wheel scores, never reconstruct sensor phases.

    Native base_return retains the original wheel/yaw weight ten.  The shared
    physical helper adds true XY, and this function adjusts only wheel position.
    These undiscounted returns precede the unchanged PPO reward_scaling.
    """
    values=(control_dt,reference_dt,wheel_position_weight)
    if any(isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value)
           for value in values) or control_dt<=0 or reference_dt<=0 or not 0<=wheel_position_weight<=10:
        raise ValueError('Need positive finite time scales and wheel weight between zero and ten')
    raw=float(native_metrics[WHEEL_HOLD['metric']])
    if not math.isfinite(raw) or raw<0:
        raise ValueError('Need a finite nonnegative native raw wheel-position score')
    result=dict(measurement)
    common=float(result.pop('objective_return'))
    if not math.isfinite(common) or not math.isclose(common,result['base_return']+result['true_position_addition_return'],
                                                    rel_tol=1e-12,abs_tol=1e-6):
        raise ValueError('Common XY objective must add the true-position term exactly once')
    adjustment=(wheel_position_weight-WHEEL_HOLD['original_weight'])*raw*control_dt/reference_dt
    result.update(common_xy_objective_return=common,wheel_position_adjustment_return=adjustment,
        training_objective_return=common+adjustment,wheel_position_weight=float(wheel_position_weight),
        native_wheel_position_raw_score_sum=raw,
        reward_contract='base_return is the original native base with wheel/yaw weights ten; common_xy_objective_return adds true XY; training_objective_return also applies the wheel-position-only adjustment. Returns are before PPO reward_scaling.')
    return result


def verify_protected_inputs(manifest):
    from force_teacher.contract import tree_digest
    checks=manifest['protected_inputs']
    for path,expected in checks.get('files',{}).items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:
            raise ValueError('Protected original input changed: '+path)
    for path,expected in checks.get('checkpoint_trees',{}).items():
        if tree_digest(path)!=expected:
            raise ValueError('Protected original checkpoint changed: '+path)
    return dict(files_verified=len(checks.get('files',{})),
        checkpoint_trees_verified=len(checks.get('checkpoint_trees',{})))


def train_with_ppo(*,env,eval_env,ppo,params,progress_fn):
    """Original fixed-normalizer PPO behavior, preserving the requested rate."""
    from brax.training.agents.ppo import train
    from force_teacher.networks import make_ppo_networks
    from force_teacher.ppo import _restore_fixed_mode
    from mujoco_playground import wrapper
    options=dict(ppo)
    factory=partial(make_ppo_networks,**dict(options.pop('network_factory')))
    options.pop('restore_checkpoint_path',None)
    supported=inspect.signature(train.train).parameters
    required={'restore_params','eval_env','policy_params_fn','deterministic_eval',
              'normalize_until_count','normalize_observations_mode'}
    if not required.issubset(supported) or set(options)-set(supported):
        raise ValueError('Installed Brax does not match the frozen PPO interface')
    reserved={'environment','eval_env','wrap_env_fn','network_factory','randomization_fn',
              'progress_fn','restore_params'}
    if reserved.intersection(options):
        raise ValueError('Manifest cannot replace the fixed training wiring')
    normalizer=_restore_fixed_mode(params[0])
    options['restore_params']=(normalizer,params[1],params[2])
    return train.train(environment=env,eval_env=eval_env,
        wrap_env_fn=wrapper.wrap_for_brax_training,network_factory=factory,
        randomization_fn=None,progress_fn=progress_fn,**options)


def _mean_scalars(measurements):
    result={}
    for key in measurements[0]:
        if key=='schema_version':
            continue
        values=[measurement.get(key) for measurement in measurements]
        if all(isinstance(v,(int,float)) and not isinstance(v,bool) for v in values):
            result[key]=math.fsum(values)/len(values)
        elif any(v is None for v in values) and all(v is None or isinstance(v,(int,float)) for v in values):
            result[key]=None
    return result


def train_one(manifest_path,seed,output_dir):
    import jax
    import mujoco
    import numpy as np
    from orbax import checkpoint as ocp
    from force_teacher.env import StudentEnv,student_config
    from force_teacher.ppo import normalizer_max_abs_change
    from force_teacher.contract import tree_digest
    from legwheel.deployment.extract import load
    from native_eval import Evaluator
    from native_eval.brax_policy import policy_from_brax
    from native_eval.legwheel import policy_template
    from experiments.balance_diagnostics.legwheel import model_layout,normalized_trace
    from .env import WheelHoldEnv
    from experiments.mature_balance_finetune.report import measure_case

    manifest_path,output_dir=Path(manifest_path).resolve(),Path(output_dir).resolve()
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    reward,expected_callbacks=validate_manifest(manifest,seed)
    if not any(device.platform=='gpu' for device in jax.devices()):
        raise RuntimeError('Mature-policy fine-tuning requires an actual JAX GPU backend')
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError('Use a new or empty output directory; prior experiments are preserved')
    output_dir.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    protection_before=verify_protected_inputs(manifest)
    source_descriptor=Path(manifest['base_descriptor'])
    descriptor=json.loads(source_descriptor.read_text(encoding='utf-8'))
    if 'leg_anchor' in descriptor['policy'] or 'policy_leg_force_scales' in descriptor['task']['config']:
        raise ValueError('Need the original mature control head, not an aligned-capacity policy')
    task_config=descriptor['task']['config']
    if (task_config['zero_keep_scale']!=10 or not task_config['sinyaw']
            or not task_config['wheel_sinpos_start']):
        raise ValueError('Native common base must retain original enabled wheel/yaw rewards with weight ten')
    shutil.copy2(source_descriptor,output_dir/'origin-descriptor.json')
    ppo=dict(manifest['ppo'],seed=seed)
    requested_rate=ppo['learning_rate']
    condition=manifest['condition']
    metadata=manifest['policy_metadata']
    template=policy_template(metadata,ppo)
    cfg=student_config()
    generated_config=json.loads(cfg.to_json_best_effort())
    if generated_config!=manifest['env_config']:
        raise ValueError('Actual standing environment differs from the frozen original configuration')
    if descriptor['eval_defaults']['control_dt']!=cfg.ctrl_dt:
        raise ValueError('Native evaluator and training control periods differ')
    if task_config['reference_dt']!=cfg.reward_config.reference_dt:
        raise ValueError('Native reward time normalization differs from training')
    initial_params=canonical_initial_params(load(manifest['initial_checkpoint']))
    initial_digest=tree_array_digest(initial_params)
    if initial_digest!=manifest['initial_array_digest']:
        raise ValueError('Complete mature actor/critic/normalizer initializer changed')
    env=WheelHoldEnv(cfg,root_body=descriptor['task']['config']['root_body'],
        wheel_position_weight=manifest['wheel_hold']['training_weight'])
    eval_env=StudentEnv(config=student_config())
    layout=model_layout(mujoco.MjModel.from_xml_path(descriptor['xml_path']),descriptor['task'])
    initial_items=manifest['initial_states'];states=[item['state'] for item in initial_items]
    state_digest=hashlib.sha256(json.dumps(initial_items,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    saved_initial=None;seen=[];rates_seen=set();saved_checkpoints=[]
    effective=dict(condition=condition,seed=seed,ppo=ppo,env=generated_config,reward=reward,
        wheel_hold=dict(WHEEL_HOLD),
        model_origin=manifest['model_origin'],original_policy_metadata=metadata,
        original_descriptor_sha256=hashlib.sha256(source_descriptor.read_bytes()).hexdigest(),
        initial_array_digest=initial_digest,initial_states_sha256=state_digest,
        brax_learning_rate=requested_rate,observed_learning_rates=[],optimizer_learning_rate_verified=False,
        brax_driver='Direct Brax train.train; original force factory and fixed-normalizer restoration; no learning-rate override.',
        callback_steps=expected_callbacks,callback_evaluation_seconds=30.,
        evaluation_reward='Native trace.reward retains the common original base with wheel/yaw weights ten; true XY and the wheel-only weight adjustment are added offline using raw first-episode native metrics.',
        snapshot_checkpoint_contents='Complete normalizer, actor and critic parameters at all callbacks; Adam state is not saved.')
    write_json(output_dir/'effective-config.json',effective)
    write_json(output_dir/'status.json',dict(phase='starting',condition=condition,seed=seed,steps=0))

    def snapshot_metadata(steps):
        value={key:descriptor.get('metadata',{})[key] for key in ('source_fingerprint','action_order','task_schema')
               if key in descriptor.get('metadata',{})}
        value.update(kind='wheel_hold_ablation_policy',experiment_kind='wheel_hold_ablation_v1',
            model_run_id=f'wheel-hold-ablation-v1/{condition}-seed-{seed}/step-{steps}',
            condition=condition,training_seed=seed,training_steps=steps,finetune_steps=steps,
            training_steps_semantics='Additional controlled fine-tune steps; original model history is separate.',
            training_learning_rate=requested_rate,initial_array_digest=initial_digest,
            model_origin=manifest['model_origin'],initializer_provenance=descriptor.get('metadata',{}),
            reward_additions=reward,wheel_hold=dict(WHEEL_HOLD),
            native_task_reward='Common original base with wheel/yaw weights ten; true XY and wheel-position-only adjustment are evaluated offline.',
            checkpoint_contents='Complete normalizer, actor and critic parameters; Adam state is not saved.')
        return value

    def on_policy(steps,make_policy,params):
        nonlocal saved_initial
        steps=int(steps)
        if len(seen)>=len(expected_callbacks) or steps!=expected_callbacks[len(seen)]:
            raise ValueError('Actual policy callbacks differ from the matched fixed-step schedule')
        host=jax.device_get(params)
        # This also rejects any nonfinite actor, critic or normalizer leaf.
        parameter_digest=tree_array_digest(host)
        if steps==0:
            saved_initial=parameter_digest
            if saved_initial!=initial_digest:
                write_json(output_dir/'initial-restore-differences.json',dict(
                    expected_array_digest=initial_digest,actual_array_digest=saved_initial,
                    actor_equal=tree_array_digest(host[1])==tree_array_digest(initial_params[1]),
                    critic_equal=tree_array_digest(host[2])==tree_array_digest(initial_params[2]),
                    normalizer_drift=float(normalizer_max_abs_change(initial_params[0],host[0]))))
                raise ValueError('PPO did not restore the complete identical mature initializer')
        drift=float(normalizer_max_abs_change(initial_params[0],host[0]))
        if drift!=0:
            raise ValueError('Frozen mature normalization changed')
        checkpoint_path=output_dir/f'checkpoint-step-{steps}'
        if checkpoint_path.exists():
            raise ValueError('A parameter checkpoint must never overwrite an earlier callback')
        with ocp.PyTreeCheckpointer() as checkpointer:
            checkpointer.save(checkpoint_path,host)
        if tree_array_digest(load(str(checkpoint_path)))!=parameter_digest:
            raise ValueError('Intermediate complete checkpoint save/load changed parameters')
        checkpoint_sha256=tree_digest(checkpoint_path)
        saved_checkpoints.append(dict(steps=steps,path=str(checkpoint_path),
            checkpoint_sha256=checkpoint_sha256,complete_array_digest=parameter_digest,reload_exact=True))
        current=dict(descriptor,policy=policy_from_brax(host[:2],template),metadata=snapshot_metadata(steps))
        current['metadata'].update(checkpoint=str(checkpoint_path),checkpoint_sha256=checkpoint_sha256,
            checkpoint_reload_exact=True)
        descriptor_path=output_dir/f'descriptor-step-{steps}.json'
        if descriptor_path.exists():
            raise ValueError('A policy snapshot must never overwrite an earlier callback')
        write_json(descriptor_path,current)
        native=Evaluator(current).evaluate(states,num_envs=5,steps=3000,episode_steps=3000,
            control_dt=.01,seed=0,trace_worlds=list(range(5)))
        if native['cancelled'] or native['control_steps']!=15000:
            raise ValueError('Incomplete common native evaluation')
        cases=[];raw_arrays={}
        for world,item in enumerate(initial_items):
            raw=next(value for value in native['traces'] if value['world']==world)
            trace=normalized_trace(raw,item['state'],native,world,.01,layout)
            trace['metadata'].update(initial_state_id=item['id'],initial_states_sha256=state_digest,
                source_step=steps,training_seed=seed,condition=condition,
                descriptor_sha256=hashlib.sha256(descriptor_path.read_bytes()).hexdigest())
            measured=measure_case(trace)
            if not math.isclose(measured['base_return'],float(native['returns'][world]),rel_tol=1e-10,abs_tol=1e-6):
                raise ValueError('Measured base return differs from the original native task')
            if not math.isclose(measured['objective_return'],measured['base_return']+measured['true_position_addition_return'],rel_tol=1e-12,abs_tol=1e-6):
                raise ValueError('New objective must add the true-XY reward exactly once')
            native_metrics={key:float(value[world]) for key,value in native['episode_metrics'].items()}
            measured=account_training_objective(measured,native_metrics,control_dt=cfg.ctrl_dt,
                reference_dt=cfg.reward_config.reference_dt,wheel_position_weight=env.wheel_position_weight)
            cases.append(dict(initial_state_id=item['id'],trace=trace,metrics=measured,
                              native_first_episode_metrics=native_metrics))
            length=int(native['lengths'][world])
            raw_arrays[f'world_{world}_steps']=np.asarray(length)
            for key,value in raw.items():
                if key not in ('world','steps'):
                    raw_arrays[f'world_{world}_{key}']=np.asarray(value)[:length]
        np.savez_compressed(output_dir/f'trace-step-{steps}.npz',**raw_arrays)
        write_json(output_dir/f'evaluation-step-{steps}.json',dict(schema_version=1,
            kind='wheel_hold_ablation_evaluation',condition=condition,seed=seed,steps=steps,
            descriptor=str(descriptor_path),cases=cases,initial_states_sha256=state_digest,
            native_wall_seconds=native['wall_seconds']))
        event=dict(condition=condition,seed=seed,steps=steps,elapsed_seconds=time.monotonic()-started,
            normalizer_drift=drift,complete_array_digest=parameter_digest,
            actor_and_normalizer_sha256=tree_array_digest(host[:2]),
            descriptor_sha256=hashlib.sha256(descriptor_path.read_bytes()).hexdigest(),
            checkpoint=str(checkpoint_path),checkpoint_sha256=checkpoint_sha256,checkpoint_reload_exact=True,
            requested_learning_rate=requested_rate,
            case_metrics=[dict(initial_state_id=case['initial_state_id'],**case['metrics']) for case in cases],
            mean_case_metrics=_mean_scalars([case['metrics'] for case in cases]))
        with (output_dir/'progress.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(event,allow_nan=False)+'\n')
        seen.append(steps)
        write_json(output_dir/'status.json',dict(phase='training',**event))
        write_json(output_dir/'latest-descriptor.json',current)
        print(json.dumps(event,allow_nan=False),flush=True)

    def on_progress(steps,metrics):
        scalars={};nonfinite={}
        for key,value in metrics.items():
            if np.ndim(value)==0:
                value=float(value)
                if math.isfinite(value):scalars[key]=value
                else:nonfinite[key]=str(value)
        rates_seen.update(observed_learning_rate(scalars,requested_rate))
        effective['observed_learning_rates']=sorted(rates_seen)
        effective['optimizer_learning_rate_verified']=bool(rates_seen)
        write_json(output_dir/'effective-config.json',effective)
        with (output_dir/'training-metrics.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(dict(steps=int(steps),metrics=scalars,nonfinite_scalar_metrics=nonfinite),allow_nan=False)+'\n')

    live_ppo=dict(ppo,policy_params_fn=on_policy)
    _,params,_=train_with_ppo(env=env,eval_env=eval_env,ppo=live_ppo,
        params=initial_params,progress_fn=on_progress)
    if seen!=expected_callbacks or seen[-1]!=manifest['steps_per_run']:
        raise ValueError('Actual completed training steps differ from the frozen budget')
    if not rates_seen:
        raise ValueError('Brax did not report an actual optimizer learning rate')
    host=jax.device_get(params);final_digest=tree_array_digest(host)
    checkpoint=output_dir/'checkpoint'
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(checkpoint,host)
    if tree_array_digest(load(str(checkpoint)))!=final_digest:
        raise ValueError('Final mature fine-tune checkpoint save/load changed parameters')
    final=dict(descriptor,policy=policy_from_brax(host[:2],template),metadata=snapshot_metadata(seen[-1]))
    final['metadata'].update(checkpoint_sha256=tree_digest(checkpoint))
    write_json(output_dir/'descriptor.json',final)
    protection_after=verify_protected_inputs(manifest)
    if source_descriptor.read_bytes()!=(output_dir/'origin-descriptor.json').read_bytes():
        raise ValueError('Original exported descriptor changed during training')
    write_json(output_dir/'input-protection.json',dict(before=protection_before,after=protection_after,
        original_descriptor_unchanged=True,model_origin=manifest['model_origin']))
    result=dict(phase='completed',condition=condition,seed=seed,steps=seen[-1],
        elapsed_seconds=time.monotonic()-started,initial_array_digest=initial_digest,
        ppo_initial_array_digest=saved_initial,final_array_digest=final_digest,
        actor_changed=tree_array_digest(host[1])!=tree_array_digest(initial_params[1]),
        normalizer_drift=float(normalizer_max_abs_change(initial_params[0],host[0])),
        descriptor_sha256=hashlib.sha256((output_dir/'descriptor.json').read_bytes()).hexdigest(),
        checkpoint_reload_exact=True,requested_learning_rate=requested_rate,
        observed_learning_rates=sorted(rates_seen),optimizer_learning_rate_verified=True,
        callback_steps=seen,devices=[str(device) for device in jax.devices()])
    result.update(wheel_hold=dict(WHEEL_HOLD),snapshot_checkpoints=saved_checkpoints,
        snapshot_checkpoint_contents='Complete normalizer, actor and critic parameters; Adam state is not saved.')
    write_json(output_dir/'status.json',result)
    print(json.dumps(result),flush=True)
    return result
