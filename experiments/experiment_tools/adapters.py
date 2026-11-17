"""Two explicit legacy report adapters; no simulator, weights or training imports."""
import hashlib
from pathlib import Path
from .io import load_json

FORMATS={
    'mature_balance_finetune_summary':('high_lr','low_lr'),
    'wheel_hold_ablation_summary':('original_wheel','weak_wheel'),
}
PARTITIONS={'primary':['nominal','pitch-plus','pitch-minus'],'heldout':['heldout-plus','heldout-minus']}
WINDOWS=('first_30s','first_120s')
METRICS={
    'max_xy_displacement_m':('m','min'),
    'rms_xy_displacement_m':('m','min'),
    'last_10s_mean_xy_displacement_m':('m','min'),
    'horizontal_speed_rms_m_s':('m/s','min'),
    'pitch_error_rms_deg':('deg','min'),
    'in_band_seconds_after_settling':('s','max'),
    'within_1cm_seconds_after_settling':('s','max'),
}


def _summary_path(path):
    path=Path(path).resolve()
    if path.is_file():return path
    options=[p for p in [path/'summary.json',path/'comparison/summary.json'] if p.is_file()]
    if len(options)!=1:raise ValueError('Supply a summary.json file or an unambiguous experiment result directory')
    return options[0]


def _evidence(base,kind,study,policy_map):
    folder=base/('selected-snapshot' if kind=='wheel_hold_ablation_summary' else 'selected-stage')
    summary_path,verification_path,descriptor_path=[folder/name for name in ('summary.json','verification.json','descriptor.json')]
    if not all(p.is_file() for p in [summary_path,verification_path,descriptor_path]):return
    selected=load_json(summary_path);verified=load_json(verification_path)
    if verified.get('phase')!='verified':raise ValueError('Selected snapshot evidence is not verified')
    selection=selected['selected']
    policy_id=(f"weak_wheel-seed-{selection['seed']}" if kind=='wheel_hold_ablation_summary' else selection['policy_id'])
    if policy_id not in policy_map:raise ValueError('Selected snapshot policy is not declared by this report')
    point=next((s for s in study['snapshots'] if s['id']==f"{policy_id}@{selection['steps']}"),None)
    if point is None:raise ValueError('Selected evidence does not match a recorded snapshot')
    digest=hashlib.sha256(descriptor_path.read_bytes()).hexdigest()
    if digest!=selected.get('descriptor_sha256',verified.get('descriptor_sha256')):
        raise ValueError('Selected policy descriptor differs from its verification evidence')
    artifact=dict(descriptor_path=str(descriptor_path),descriptor_sha256=digest,evidence_file=str(verification_path),
                  evidence_sha256=hashlib.sha256(verification_path.read_bytes()).hexdigest(),live_validation_required=True)
    if kind=='wheel_hold_ablation_summary':
        checkpoint_sha=selected.get('checkpoint_sha256')
        parameter_digest=selected.get('parameter_digest')
        action_error=verified.get('checkpoint_to_saved_policy_action_max_abs_error')
        if (verified.get('all_parameter_leaves_finite') is not True or
            type(action_error) not in (int,float) or not 0<=action_error<=1e-6 or
            not isinstance(checkpoint_sha,str) or len(checkpoint_sha)!=64 or
            any(c not in '0123456789abcdef' for c in checkpoint_sha.lower()) or
            not parameter_digest or
            checkpoint_sha!=verified.get('checkpoint_sha256') or
            parameter_digest!=verified.get('complete_parameter_digest')):
            raise ValueError('Selected complete parameters do not match verification evidence')
        artifact.update(full_parameters_verified=True,checkpoint_path=selected['checkpoint'],checkpoint_sha256=selected['checkpoint_sha256'],
                        capability='full_parameters_verified_at_run',optimizer_state='restart')
    else:
        # This specific historical format stored descriptors for intermediate
        # callbacks and complete parameters only at the final step.
        artifact.update(full_parameters_verified=False,capability='descriptor_only',optimizer_state='unavailable')
    point['artifacts']=artifact


def load_study(path):
    """Read canonical paired_study, or a supported small historical summary.

    Never reads results.json/NPZ or infers robot compatibility from dimensions.
    Parameter capability is evidence from the recorded audit, not live access
    to a remote checkpoint. Training must still use the robot's real validator.
    """
    source=_summary_path(path);data=load_json(source)
    if not isinstance(data,dict):raise ValueError('Study must be a JSON object')
    if data.get('kind')=='paired_study':return data
    kind=data.get('kind')
    if kind not in FORMATS:raise ValueError('Unsupported report kind; provide paired_study or a recognized summary.json')
    if data.get('schema_version')!=1:raise ValueError('Unsupported historical report version')
    baseline,treatment=FORMATS[kind];seeds=[r['seed'] for r in data['conditions'][baseline]['seed_results']]
    if seeds!=[r['seed'] for r in data['conditions'][treatment]['seed_results']]:raise ValueError('Historical seed declarations differ')
    definitions=[dict(id=window+'_'+metric,unit=unit,direction=direction)
                 for window in WINDOWS for metric,(unit,direction) in METRICS.items()]
    study=dict(schema_version=1,kind='paired_study',conditions=[dict(id=c,label=data['conditions'][c].get('label',c)) for c in [baseline,treatment]],
               comparison=dict(baseline=baseline,treatment=treatment),seeds=seeds,partitions={k:list(v) for k,v in PARTITIONS.items()},
               metrics=definitions,records=[],reference_records=[],snapshots=[],
               provenance=dict(adapter=kind,summary_path=str(source),summary_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                               recorded_training_contract=data.get('training_statuses'),
                               scope='Historical summary measurements; this import is not a new raw-trajectory or live model-compatibility validation.'))
    for row in data['case_rows']:
        values={m['id']:row[m['id']] for m in definitions};eligible={}
        for window in WINDOWS:
            reached=row['reached_30s' if window=='first_30s' else 'reached_120s']
            if type(reached) is not bool:raise ValueError('Historical completion flag must be boolean')
            for metric in METRICS:
                eligible[window+'_'+metric]=True if metric.endswith('seconds_after_settling') else reached
        item=dict(case_id=row['initial_state_id'],values=values,eligible=eligible)
        if row['policy_id']=='initial':study['reference_records'].append(item)
        else:
            if row['condition'] not in [baseline,treatment]:raise ValueError('Unexpected historical condition')
            study['records'].append(dict(item,condition=row['condition'],seed=row['seed']))
    policies={}
    for condition in [baseline,treatment]:
        for row in data['conditions'][condition]['seed_results']:
            policies[row['policy_id']]=(condition,row['seed'])
    for policy_id,curve in data.get('progress_curves',{}).items():
        if policy_id not in policies:raise ValueError('Progress policy lacks declared condition/seed')
        condition,seed=policies[policy_id]
        for point in curve:
            values={};eligible={}
            for window in WINDOWS:
                group=point['primary']['windows'][window]
                complete=group['complete_set']
                if type(complete) is not bool:raise ValueError('Snapshot completion flag must be boolean')
                for metric in METRICS:
                    key=window+'_'+metric;values[key]=group[metric]
                    eligible[key]=True if metric.endswith('seconds_after_settling') else complete
            study['snapshots'].append(dict(id=f"{policy_id}@{point['steps']}",condition=condition,seed=seed,step=point['steps'],
                                           partition='primary',values=values,eligible=eligible,artifacts={}))
    experiment_root=source.parent.parent if source.parent.name=='comparison' else source.parent
    _evidence(experiment_root,kind,study,policies)
    return study
