"""Select eligible recorded snapshots without rewarding missing horizons."""
import copy
import math
from .paired import summarize


def select_snapshot(study,*,condition,metric,partition='primary',require_full_parameters=False,min_step=1):
    summarize(study)  # Share the public study validation, including typed seed IDs.
    if type(min_step) is not int or min_step<0:raise ValueError('min_step must be a nonnegative integer')
    if condition not in [c['id'] for c in study['conditions']]:raise ValueError('Unknown condition')
    if partition not in study['partitions']:raise ValueError('Unknown partition')
    if type(require_full_parameters) is not bool:raise ValueError('require_full_parameters must be boolean')
    definitions=[m for m in study['metrics'] if m['id']==metric]
    if len(definitions)!=1 or definitions[0]['direction'] not in ('min','max'):raise ValueError('Unknown metric or direction')
    candidates=[];excluded=[];seen=set()
    for snapshot in study.get('snapshots',[]):
        identity=snapshot.get('id')
        if not isinstance(identity,str) or not identity or identity in seen:raise ValueError('Duplicate/missing snapshot id')
        seen.add(identity)
        if snapshot.get('condition') not in [c['id'] for c in study['conditions']]:raise ValueError('Undeclared snapshot condition')
        if not any(type(snapshot.get('seed')) is type(s) and snapshot.get('seed')==s for s in study['seeds']):raise ValueError('Undeclared snapshot seed')
        if snapshot.get('partition') not in study['partitions']:raise ValueError('Undeclared snapshot partition')
        if snapshot['condition']!=condition or snapshot['partition']!=partition:continue
        step=snapshot.get('step')
        if type(step) is not int or step<0:raise ValueError('Invalid snapshot step')
        value=snapshot['values'].get(metric);eligible=snapshot['eligible'].get(metric)
        if type(eligible) is not bool:raise ValueError('Snapshot eligibility must be explicit')
        if value is not None and (type(value) not in (int,float) or not math.isfinite(value)):raise ValueError('Snapshot value must be finite or null')
        artifact=snapshot.get('artifacts',{})
        complete_parameters=(artifact.get('full_parameters_verified') is True and bool(artifact.get('checkpoint_path')) and
                             bool(artifact.get('evidence_file')) and isinstance(artifact.get('checkpoint_sha256'),str) and len(artifact['checkpoint_sha256'])==64 and
                             all(c in '0123456789abcdef' for c in artifact['checkpoint_sha256'].lower()))
        reason=('before_min_step' if step<min_step else 'ineligible_or_missing' if not eligible or value is None else
                'full_parameters_not_verified' if require_full_parameters and not complete_parameters else None)
        if reason:excluded.append(dict(id=identity,reason=reason));continue
        candidates.append(snapshot)
    if not candidates:raise ValueError('No eligible snapshot with the requested evidence; unknown capability is not proof of a complete checkpoint')
    sign=1 if definitions[0]['direction']=='min' else -1
    chosen=min(candidates,key=lambda s:(sign*s['values'][metric],s['step'],s['id']))
    return dict(schema_version=1,kind='snapshot_selection',condition=condition,partition=partition,metric=copy.deepcopy(definitions[0]),
                selected=copy.deepcopy(chosen),eligible_candidates=len(candidates),excluded=excluded,
                retrospective=True,scope='Metric-based retrospective selection, not another training seed or proof of future performance.',
                resume_scope='Parameter evidence is recorded at run time; live path, source, actuator and full-state compatibility must be checked by the robot adapter before continuation.')
