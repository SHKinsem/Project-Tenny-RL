"""Backend-free paired studies: conditions -> seed -> declared case partition.

Every declared row must exist. Within a seed, a metric is unknown if any case
is missing its value or is ineligible. Fixed cases never become extra seeds.
Input mappings are read only; the return value contains JSON-compatible data.
"""
from collections.abc import Mapping
import math


def _list(value, label, *, nonempty=True):
    if not isinstance(value, list) or (nonempty and not value):
        raise ValueError(label+' must be '+('a nonempty list' if nonempty else 'a list'))
    return value


def _identifier(value, label):
    if not isinstance(value, str) or not value:
        raise ValueError(label+' must be a nonempty string')
    return value


def _seed_token(value):
    # JSON object keys would merge integer 0 and string "0". Internal keys keep
    # the type and public seed rows retain the original scalar in a list.
    if type(value) not in (int, str):
        raise ValueError('Each seed must be an integer or string, never a boolean')
    return type(value).__name__, value


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(label+' must be a finite number or explicit null')
    try:
        result=float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(label+' is outside the finite numeric range') from error
    if not math.isfinite(result):
        raise ValueError(label+' must be finite')
    return result


def _mean(values):
    try:
        value=math.fsum(values)/len(values)
    except OverflowError:
        # A finite mean can exist even when the intermediate sum overflows.
        value=math.fsum(item/len(values) for item in values)
    return _number(value, 'Derived mean')


def _stats(values):
    available=sorted(value for value in values if value is not None)
    count=len(available)
    if count:
        middle=count//2
        median=available[middle] if count%2 else _mean(available[middle-1:middle+1])
    else:
        median=None
    return dict(median=median,min=available[0] if count else None,
                max=available[-1] if count else None,count=count,missing_count=len(values)-count)


def _definitions(study):
    conditions=[];condition_ids=set()
    for item in _list(study.get('conditions'), 'conditions'):
        if not isinstance(item, Mapping):raise ValueError('Each condition must be a mapping')
        identity=_identifier(item.get('id'), 'condition id')
        if identity in condition_ids:raise ValueError('Duplicate condition definition: '+identity)
        if not isinstance(item.get('label'), str):raise ValueError('Condition label must be a string')
        condition_ids.add(identity);conditions.append(dict(id=identity,label=item['label']))
    comparison=study.get('comparison')
    if not isinstance(comparison, Mapping):raise ValueError('comparison must be a mapping')
    baseline=_identifier(comparison.get('baseline'), 'baseline condition')
    treatment=_identifier(comparison.get('treatment'), 'treatment condition')
    if baseline==treatment or baseline not in condition_ids or treatment not in condition_ids:
        raise ValueError('Comparison needs two different declared conditions')
    seeds=list(_list(study.get('seeds'), 'seeds'))
    seed_tokens=[_seed_token(seed) for seed in seeds]
    if len(set(seed_tokens))!=len(seed_tokens):raise ValueError('Duplicate seed definition')
    partitions=study.get('partitions')
    if not isinstance(partitions, Mapping) or not partitions:raise ValueError('Need nonempty partitions')
    checked_partitions={};case_ids=set()
    for name,cases in partitions.items():
        _identifier(name, 'partition name')
        checked=[]
        for case in _list(cases, 'partition cases'):
            _identifier(case, 'case id')
            if case in case_ids:raise ValueError('Duplicate case definition across partitions: '+case)
            case_ids.add(case);checked.append(case)
        checked_partitions[name]=checked
    metrics=[];metric_ids=set()
    for item in _list(study.get('metrics'), 'metrics'):
        if not isinstance(item, Mapping):raise ValueError('Each metric must be a mapping')
        identity=_identifier(item.get('id'), 'metric id')
        if identity in metric_ids:raise ValueError('Duplicate metric definition: '+identity)
        if not isinstance(item.get('unit'), str):raise ValueError('Metric unit must be a string; dimensionless may use an empty string')
        if item.get('direction') not in ('min','max'):raise ValueError('Metric direction must be min or max')
        metric_ids.add(identity);metrics.append(dict(id=identity,unit=item['unit'],direction=item['direction']))
    return conditions,dict(baseline=baseline,treatment=treatment),seeds,checked_partitions,metrics


def _payload(record, metric_ids):
    values,eligible=record.get('values'),record.get('eligible')
    expected=set(metric_ids)
    if not isinstance(values, Mapping) or set(values)!=expected:
        raise ValueError('values must explicitly contain every declared metric and no unknown metric')
    if not isinstance(eligible, Mapping) or set(eligible)!=expected:
        raise ValueError('eligible must explicitly contain every declared metric and no unknown metric')
    checked_values={};checked_eligible={}
    for metric in metric_ids:
        value=values[metric]
        checked_values[metric]=None if value is None else _number(value, 'Metric '+metric)
        if type(eligible[metric]) is not bool:raise ValueError('Eligibility must be a boolean for metric '+metric)
        checked_eligible[metric]=eligible[metric]
    return dict(values=checked_values,eligible=checked_eligible)


def _indexes(study, conditions, seeds, partitions, metric_ids):
    condition_ids={item['id'] for item in conditions}
    seed_tokens={_seed_token(seed) for seed in seeds}
    case_ids={case for cases in partitions.values() for case in cases}
    indexed={}
    records=_list(study.get('records'), 'records')
    for record in records:
        if not isinstance(record, Mapping):raise ValueError('Each record must be a mapping')
        condition=_identifier(record.get('condition'),'record condition')
        case=_identifier(record.get('case_id'),'record case id')
        token=_seed_token(record.get('seed'))
        if condition not in condition_ids or token not in seed_tokens or case not in case_ids:
            raise ValueError('Record refers to an undeclared condition, seed or case')
        key=condition,token,case
        if key in indexed:raise ValueError('Duplicate condition/seed/case record')
        indexed[key]=_payload(record,metric_ids)
    expected=len(conditions)*len(seeds)*len(case_ids)
    if len(indexed)!=expected:
        raise ValueError(f'Declared record matrix is incomplete: expected {expected} rows, got {len(indexed)}')
    references={}
    for record in _list(study.get('reference_records', []), 'reference_records', nonempty=False):
        if not isinstance(record, Mapping):raise ValueError('Each reference record must be a mapping')
        case=_identifier(record.get('case_id'),'reference case id')
        if case not in case_ids:raise ValueError('Reference uses an undeclared case')
        if case in references:raise ValueError('Duplicate reference case record')
        references[case]=_payload(record,metric_ids)
    if references and len(references)!=len(case_ids):
        raise ValueError('Nonempty reference_records must cover every declared case')
    return indexed,references


def _partition(cases, records, metric_ids):
    values={};eligible={};missing_cases={};ineligible_cases={};missing_value_cases={}
    for metric in metric_ids:
        unavailable=[case for case in cases if records[case]['values'][metric] is None]
        excluded=[case for case in cases if not records[case]['eligible'][metric]]
        missing=[case for case in cases if case in unavailable or case in excluded]
        missing_cases[metric]=missing;ineligible_cases[metric]=excluded;missing_value_cases[metric]=unavailable
        eligible[metric]=not missing
        values[metric]=None if missing else _mean([records[case]['values'][metric] for case in cases])
    return dict(case_count=len(cases),values=values,eligible=eligible,missing_cases=missing_cases,
                ineligible_cases=ineligible_cases,missing_value_cases=missing_value_cases)


def _delta(seed, left, right, metrics):
    values={};eligible={};improved={}
    for metric in metrics:
        identity=metric['id'];a=left['values'][identity];b=right['values'][identity]
        value=None if a is None or b is None else _number(a-b, 'Derived paired difference')
        values[identity]=value;eligible[identity]=value is not None
        improved[identity]=(None if value is None else value<0 if metric['direction']=='min' else value>0)
    return dict(seed=seed,values=values,eligible=eligible,improved=improved)


def _delta_stats(rows, metrics):
    result={}
    for metric in metrics:
        identity=metric['id'];values=[row['values'][identity] for row in rows]
        known=[row for row in rows if row['values'][identity] is not None]
        result[identity]=dict(**_stats(values),
            improved_seed_count=sum(row['improved'][identity] for row in known),
            degraded_seed_count=sum(not row['improved'][identity] and row['values'][identity]!=0 for row in known),
            tied_seed_count=sum(row['values'][identity]==0 for row in known))
    return result


def summarize(study):
    """Return descriptive paired differences, preserving unknowns and seed types.

    Partitions are disjoint declarations. A nonempty reference must cover every
    declared case. Null values and ineligible values each invalidate that whole
    seed/partition/metric mean; neither drops a case nor substitutes zero.
    Differences use treatment minus baseline; improvement uses metric direction
    and a strict zero sign, without a statistical-significance interpretation.
    """
    if (not isinstance(study, Mapping) or type(study.get('schema_version')) is not int or
            study['schema_version']!=1 or study.get('kind')!='paired_study'):
        raise ValueError('Expected paired_study schema_version=1')
    conditions,comparison,seeds,partitions,metrics=_definitions(study)
    metric_ids=[item['id'] for item in metrics]
    indexed,reference_records=_indexes(study,conditions,seeds,partitions,metric_ids)
    condition_summaries={};means={}
    for condition in conditions:
        identity=condition['id'];condition_summaries[identity]={}
        for partition,cases in partitions.items():
            rows=[]
            for seed in seeds:
                records={case:indexed[identity,_seed_token(seed),case] for case in cases}
                group=_partition(cases,records,metric_ids)
                means[identity,_seed_token(seed),partition]=group
                rows.append(dict(seed=seed,**group))
            condition_summaries[identity][partition]=dict(seed_results=rows,
                metrics={metric:_stats([row['values'][metric] for row in rows]) for metric in metric_ids})
    paired_results={};paired_summaries={}
    for partition in partitions:
        rows=[_delta(seed,means[comparison['treatment'],_seed_token(seed),partition],
            means[comparison['baseline'],_seed_token(seed),partition],metrics) for seed in seeds]
        paired_results[partition]=rows;paired_summaries[partition]=_delta_stats(rows,metrics)
    references=delta_from_reference=None
    if reference_records:
        references={partition:_partition(cases,reference_records,metric_ids) for partition,cases in partitions.items()}
        delta_from_reference={}
        for condition in conditions:
            identity=condition['id'];delta_from_reference[identity]={}
            for partition in partitions:
                rows=[_delta(seed,means[identity,_seed_token(seed),partition],references[partition],metrics) for seed in seeds]
                delta_from_reference[identity][partition]=dict(seed_results=rows,metrics=_delta_stats(rows,metrics))
    return dict(schema_version=1,kind='paired_study_summary',conditions=conditions,comparison=comparison,
        seeds=seeds,partitions=partitions,metrics=metrics,condition_summaries=condition_summaries,
        paired_results=paired_results,paired_summaries=paired_summaries,references=references,
        delta_from_reference=delta_from_reference,
        counts=dict(conditions=len(conditions),seeds=len(seeds),cases=sum(map(len,partitions.values())),
                    records=len(indexed),reference_records=len(reference_records)),
        aggregation='Equal-weight all declared cases within each seed and partition, then pair seeds; fixed cases are not independent training repetitions.',
        difference='Treatment minus baseline; relative-to-reference differences use condition minus the fixed reference.',
        unknown_policy='Any missing value or ineligible case invalidates that seed/partition/metric mean. Unknown values are never zero-filled.',
        scope='Descriptive seed-level statistics and sign counts; no significance, convergence or causal claim is inferred.')
