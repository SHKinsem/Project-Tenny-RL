"""Pure physical measurements and offline reports for mature-policy fine tuning."""
from bisect import bisect_left, bisect_right
from collections.abc import Mapping
import csv
from html import escape
import json
import math
from pathlib import Path
import statistics


CONDITIONS = ('high_lr', 'low_lr')
LEARNING_RATES = {'high_lr': 3e-5, 'low_lr': 3e-6}
SEEDS = (0, 1, 2)
PRIMARY_STATES = ('nominal', 'pitch-plus', 'pitch-minus')
HELDOUT_STATES = ('heldout-plus', 'heldout-minus')
STATES = PRIMARY_STATES+HELDOUT_STATES


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(label+' must be a finite number')
    return float(value)


def _vector_series(value, label, count, width):
    if not isinstance(value, (list, tuple)) or len(value) != count:
        raise ValueError(label+' has an invalid sample count')
    result=[]
    for row in value:
        if not isinstance(row, (list, tuple)) or len(row) != width:
            raise ValueError(label+' has an invalid vector width')
        result.append([_number(v,label) for v in row])
    return result


def _validated(trace):
    if not isinstance(trace, Mapping) or trace.get('schema_version') != 1:
        raise ValueError('Expected normalized trace schema_version=1')
    times=trace.get('t_s')
    if not isinstance(times,(list,tuple)) or len(times)<2:
        raise ValueError('Need reset and at least one post-control sample')
    times=[_number(t,'t_s') for t in times]
    if times[0] != 0 or any(b<=a for a,b in zip(times,times[1:])):
        raise ValueError('Times must start at zero and strictly increase')
    count=len(times)
    pos=_vector_series(trace.get('base_position_m'),'base_position_m',count,3)
    velocity=_vector_series(trace.get('base_velocity_m_s'),'base_velocity_m_s',count,3)
    scalar={}
    for key in ('pitch_rad','reward'):
        values=trace.get(key)
        if not isinstance(values,(list,tuple)) or len(values)!=count:
            raise ValueError(key+' has an invalid sample count')
        scalar[key]=[_number(value,key) for value in values]
    done=trace.get('done')
    if (not isinstance(done,(list,tuple)) or len(done)!=count or
            any(type(value) is not bool for value in done) or done[0] or any(done[:-1]) or not done[-1]):
        raise ValueError('Keep exactly one complete first episode, including its final done')
    if scalar['reward'][0] != 0:
        raise ValueError('Reset reward must be zero')
    termination=trace.get('termination')
    if not isinstance(termination,Mapping):raise ValueError('Need termination metadata')
    flags=[termination.get(name) for name in ('terminated','truncated')]
    if any(type(value) is not bool for value in flags) or sum(flags)!=1:
        raise ValueError('Exactly one termination or truncation flag must be true')
    if not math.isclose(_number(termination.get('time_s'),'termination time'),times[-1],rel_tol=0,abs_tol=1e-9):
        raise ValueError('Termination time differs from the final sample')
    metadata=trace.get('metadata')
    if not isinstance(metadata,Mapping):raise ValueError('Need trace metadata')
    reference=_number(metadata.get('reference_pitch_rad'),'reference_pitch_rad')
    distance=[math.hypot(row[0]-pos[0][0],row[1]-pos[0][1]) for row in pos]
    speed=[math.hypot(row[0],row[1]) for row in velocity]
    error=[math.degrees(value-reference) for value in scalar['pitch_rad']]
    if not all(math.isfinite(value) for value in distance+speed+error):
        raise ValueError('Derived physical measurements must be finite')
    return dict(time=times,displacement=distance,speed=speed,pitch_error=error,
                reward=scalar['reward'],termination=dict(termination))


def _mean_interval(times, values, *, square=False):
    if len(times)<2 or times[-1]<=times[0]:return None
    total=math.fsum((b-a)*((x*x+y*y) if square else (x+y))/2
                   for a,b,x,y in zip(times,times[1:],values,values[1:]))/(times[-1]-times[0])
    return math.sqrt(max(0.,total)) if square else total


def _reached(data, seconds):
    end=data['time'][-1]
    if math.isclose(end,seconds,rel_tol=0,abs_tol=1e-9):
        return data['termination']['truncated']
    return end>seconds


def _window(data, requested):
    end=bisect_right(data['time'],min(requested,data['time'][-1])+1e-12)-1
    times=data['time'][:end+1]
    distance=data['displacement'][:end+1]
    duration=times[-1]
    last_mean=last_coverage=None
    if duration>=30.-1e-9:
        start=bisect_left(times,duration-10.-1e-12)
        last_mean=_mean_interval(times[start:],distance[start:])
        last_coverage=duration-times[start] if last_mean is not None else None
    bands={}
    for threshold,name in ((.05,'in_band_seconds_after_settling'),(.01,'within_1cm_seconds_after_settling')):
        good=[d<=threshold and abs(p)<=10. and v<=.1
              for d,p,v in zip(distance,data['pitch_error'][:end+1],data['speed'][:end+1])]
        total=0.
        for index,(left,right) in enumerate(zip(times,times[1:])):
            if left>=1. and good[index] and good[index+1]:total+=right-left
        bands[name]=total
    return dict(requested_seconds=requested,observed_seconds=duration,reached_requested_horizon=_reached(data,requested),
        max_xy_displacement_m=max(distance),rms_xy_displacement_m=_mean_interval(times,distance,square=True),
        last_10s_mean_xy_displacement_m=last_mean,last_10s_observed_seconds=last_coverage,
        horizontal_speed_rms_m_s=_mean_interval(times,data['speed'][:end+1],square=True),
        pitch_error_rms_deg=_mean_interval(times,data['pitch_error'][:end+1],square=True),**bands)


def measure_case(trace):
    """Measure actual first-episode observations, never padding after failure.

    trace.reward must contain the original base-task rewards, not the XY addition.
    The addition is recomputed at each post-step state using that interval's dt.
    XY summaries use displacement magnitude from reset and trapezoidal means.
    The last-10-second mean is unknown when fewer than 30 seconds were observed.
    """
    data=_validated(trace)
    windows={'first_30s':_window(data,30.),'first_120s':_window(data,120.),
             'full_episode':_window(data,data['time'][-1])}
    full=windows['full_episode'];first=windows['first_30s']
    addition=math.fsum(10.*math.expm1(-(distance/.05)**2)*(right-left)/.01
        for left,right,distance in zip(data['time'],data['time'][1:],data['displacement'][1:]))
    base=math.fsum(data['reward'])
    return dict(schema_version=1,observed_episode_seconds=data['time'][-1],
        reached_30s=_reached(data,30.),reached_120s=_reached(data,120.),
        max_xy_displacement_m=full['max_xy_displacement_m'],rms_xy_displacement_m=full['rms_xy_displacement_m'],
        last_10s_mean_xy_displacement_m=full['last_10s_mean_xy_displacement_m'],
        horizontal_speed_rms_m_s=full['horizontal_speed_rms_m_s'],pitch_error_rms_deg=full['pitch_error_rms_deg'],
        first30s_in_band_seconds=first['in_band_seconds_after_settling'],
        first30s_within_1cm_seconds=first['within_1cm_seconds_after_settling'],
        true_position_addition_return=addition,base_return=base,objective_return=base+addition,
        termination=dict(data['termination']),windows=windows,
        sampling='First-episode recorded states only; trapezoidal state summaries. Band time counts complete post-settling intervals with both endpoints in band. Reward addition uses post-step displacement and dt/.01.',
        reward_contract='trace.reward is the base-task reward; the real XY addition is applied exactly once offline.')


WINDOWS = ('first_30s', 'first_120s')
METRIC_KEYS = ('max_xy_displacement_m', 'rms_xy_displacement_m',
    'last_10s_mean_xy_displacement_m', 'horizontal_speed_rms_m_s', 'pitch_error_rms_deg',
    'in_band_seconds_after_settling', 'within_1cm_seconds_after_settling')
LABELS = {'initial':'成熟模型（不更新参考）', 'high_lr':'固定学习率3e-5', 'low_lr':'固定学习率3e-6'}


def _mean(values):
    return math.fsum(values)/len(values)


def _stats(values):
    available=[value for value in values if value is not None]
    return dict(count=len(available),missing_count=len(values)-len(available),
        median=statistics.median(available) if available else None,
        min=min(available) if available else None,max=max(available) if available else None)


def _group(measurements):
    count=len(measurements)
    result=dict(evaluation_cases=count,
        mean_observed_episode_seconds=_mean([value['observed_episode_seconds'] for value in measurements]),
        reached_30s_count=sum(value['reached_30s'] for value in measurements),
        reached_120s_count=sum(value['reached_120s'] for value in measurements),windows={})
    for name,horizon in (('first_30s',30.),('first_120s',120.)):
        selected=[value['windows'][name] for value in measurements]
        completed=sum(value['reached_requested_horizon'] and value['observed_seconds']>=horizon-1e-9 for value in selected)
        eligible=completed==count
        row=dict(completed_cases=completed,evaluation_cases=count,complete_set=eligible)
        for key in METRIC_KEYS:
            values=[value[key] for value in selected]
            row[key]=_mean(values) if eligible and all(value is not None for value in values) else None
        result['windows'][name]=row
    return result


def _group_stats(groups):
    return {name:{key:_stats([group['windows'][name][key] for group in groups])
                  for key in METRIC_KEYS} for name in WINDOWS}


def _differences(left,right):
    return {name:{key:(left['windows'][name][key]-right['windows'][name][key]
                      if left['windows'][name][key] is not None and right['windows'][name][key] is not None else None)
                  for key in METRIC_KEYS} for name in WINDOWS}


def _policy_metadata(bundle):
    policies=bundle.get('policies')
    if not isinstance(policies,list) or len(policies)!=7:raise ValueError('Need the mature reference and six final policies')
    by_id={};trained={};initial=None
    for policy in policies:
        if not isinstance(policy,Mapping) or not isinstance(policy.get('id'),str) or not policy['id']:
            raise ValueError('Policy identity must be explicit')
        identity=policy['id']
        if identity in by_id:raise ValueError('Duplicate policy identity')
        source=policy.get('source',{})
        if not isinstance(source,Mapping):raise ValueError('Policy source must be a mapping')
        condition=source.get('condition')
        if identity=='initial' and condition in (None,'initial'):
            if initial is not None:raise ValueError('Duplicate mature reference')
            initial=identity
        elif condition in CONDITIONS:
            seed=source.get('seed')
            if type(seed) is not int or seed not in SEEDS or (condition,seed) in trained:
                raise ValueError('Need each condition and training seed exactly once')
            rate=_number(source.get('learning_rate'),'learning_rate')
            if not math.isclose(rate,LEARNING_RATES[condition],rel_tol=0,abs_tol=1e-12):
                raise ValueError('Policy learning rate differs from its condition')
            trained[condition,seed]=identity
        else:raise ValueError('Unexpected policy condition')
        by_id[identity]=dict(policy)
    if initial is None or set(trained)!={(condition,seed) for condition in CONDITIONS for seed in SEEDS}:
        raise ValueError('Incomplete paired mature-policy comparison')
    return by_id,trained,initial


def _status_metadata(statuses,trained,manifest):
    if statuses is None:return dict(verified=False,reason='Training statuses were not supplied')
    if isinstance(statuses,Mapping):records=list(statuses.values())
    elif isinstance(statuses,list):records=statuses
    else:raise ValueError('Run statuses must be a list or mapping')
    if len(records)!=6:raise ValueError('Need six completed training statuses')
    seen=set();steps=set();initials=set()
    for row in records:
        if not isinstance(row,Mapping):raise ValueError('Invalid run status')
        key=row.get('condition'),row.get('seed')
        if key not in trained or key in seen or row.get('phase')!='completed':raise ValueError('Incomplete or duplicate training status')
        if row.get('reused',False) is not False:raise ValueError('Both learning-rate groups must be newly trained')
        count=row.get('steps')
        if type(count) is not int or count<=0:raise ValueError('Need actual positive training steps')
        digest=row.get('initial_array_digest')
        if not isinstance(digest,str) or not digest:raise ValueError('Need complete mature initializer identity')
        seen.add(key);steps.add(count);initials.add(digest)
    if len(steps)!=1 or len(initials)!=1:raise ValueError('Runs differ in training budget or complete initializer')
    actual=next(iter(steps))
    if manifest.get('steps_per_run',actual)!=actual:raise ValueError('Actual steps differ from the frozen budget')
    return dict(verified=True,runs=6,steps_per_run=actual,initial_array_digest=next(iter(initials)),reused_runs=0)


def _check_saved_measurement(saved,actual):
    if not isinstance(saved,Mapping):raise ValueError('measurement must be a mapping')
    for key,value in saved.items():
        if key not in actual:continue
        expected=actual[key]
        if isinstance(expected,Mapping):_check_saved_measurement(value,expected)
        elif isinstance(expected,bool) or expected is None:
            if type(value) is not type(expected) or value!=expected:raise ValueError('Saved measurement differs: '+key)
        elif isinstance(expected,(int,float)):
            if not math.isclose(_number(value,key),expected,rel_tol=1e-10,abs_tol=1e-7):raise ValueError('Saved measurement differs: '+key)


def _progress(records):
    if not isinstance(records,list):raise ValueError('Progress records must be a list')
    output=[];previous=-1
    for row in records:
        step=row.get('steps')
        if type(step) is not int or step<=previous:raise ValueError('Progress steps must strictly increase')
        previous=step
        values=row.get('case_metrics')
        if isinstance(values,Mapping):indexed=dict(values)
        elif isinstance(values,list):
            if len(values)!=5:raise ValueError('Need five progress evaluation cases')
            indexed={value.get('initial_state_id',state):value.get('measurement',value)
                     for state,value in zip(STATES,values)}
        else:raise ValueError('Progress must preserve case_metrics, not only a pooled mean')
        if set(indexed)!=set(STATES):raise ValueError('Progress physical-condition matrix differs')
        primary=_group([indexed[state] for state in PRIMARY_STATES])
        output.append(dict(steps=step,primary=primary,
            mean_base_return=_mean([indexed[state]['base_return'] for state in PRIMARY_STATES]),
            mean_objective_return=_mean([indexed[state]['objective_return'] for state in PRIMARY_STATES])))
    return output


def summarize(bundle,statuses=None):
    """Seed-paired drift summaries; a failed horizon makes its drift comparison unknown."""
    if not isinstance(bundle,Mapping) or bundle.get('kind')!='mature_balance_finetune_v1':
        raise ValueError('Need a mature_balance_finetune_v1 result bundle')
    policies,trained,initial=_policy_metadata(bundle)
    manifest=bundle.get('experiment_manifest',{})
    if not isinstance(manifest,Mapping):raise ValueError('Experiment manifest must be a mapping')
    verified=_status_metadata(statuses if statuses is not None else bundle.get('run_statuses'),trained,manifest)
    indexed={};rows=[];failures=[]
    for case in bundle.get('cases',[]):
        identity,state=case.get('policy_id'),case.get('initial_state_id')
        key=identity,state
        if identity not in policies or state not in STATES or key in indexed:raise ValueError('Unexpected or duplicate physical case')
        value=measure_case(case['trace'])
        if 'measurement' in case:_check_saved_measurement(case['measurement'],value)
        indexed[key]=value
        source=policies[identity].get('source',{})
        row=dict(policy_id=identity,condition=source.get('condition','initial'),seed=source.get('seed'),
            initial_state_id=state,observed_episode_seconds=value['observed_episode_seconds'],
            reached_30s=value['reached_30s'],reached_120s=value['reached_120s'],
            termination_reason=value['termination'].get('reason','unknown'),
            base_return=value['base_return'],true_position_addition_return=value['true_position_addition_return'],
            objective_return=value['objective_return'])
        for name in WINDOWS:
            for metric in METRIC_KEYS:row[name+'_'+metric]=value['windows'][name][metric]
        rows.append(row)
        if not value['reached_120s']:
            failures.append(dict(policy_id=identity,initial_state_id=state,
                observed_episode_seconds=value['observed_episode_seconds'],
                terminated=value['termination']['terminated'],reason=value['termination'].get('reason','unknown')))
    if set(indexed)!={(identity,state) for identity in policies for state in STATES}:
        raise ValueError('Need all seven policies on the same five physical conditions')

    def group(identity,states):return _group([indexed[identity,state] for state in states])

    reference=dict(policy_id=initial,primary=group(initial,PRIMARY_STATES),heldout=group(initial,HELDOUT_STATES))
    conditions={}
    for condition in CONDITIONS:
        seeds=[]
        for seed in SEEDS:
            identity=trained[condition,seed]
            primary,heldout=group(identity,PRIMARY_STATES),group(identity,HELDOUT_STATES)
            seeds.append(dict(seed=seed,policy_id=identity,primary=primary,heldout=heldout,
                primary_delta_from_initial=_differences(primary,reference['primary']),
                heldout_delta_from_initial=_differences(heldout,reference['heldout'])))
        conditions[condition]=dict(label=LABELS[condition],learning_rate=LEARNING_RATES[condition],seed_results=seeds,
            primary_stats=_group_stats([row['primary'] for row in seeds]),
            heldout_stats=_group_stats([row['heldout'] for row in seeds]),
            primary_reached_30s_cases=sum(row['primary']['reached_30s_count'] for row in seeds),
            primary_reached_120s_cases=sum(row['primary']['reached_120s_count'] for row in seeds),
            heldout_reached_30s_cases=sum(row['heldout']['reached_30s_count'] for row in seeds),
            heldout_reached_120s_cases=sum(row['heldout']['reached_120s_count'] for row in seeds))
    pairs=[]
    for high,low in zip(conditions['high_lr']['seed_results'],conditions['low_lr']['seed_results']):
        pairs.append(dict(seed=high['seed'],primary=_differences(low['primary'],high['primary']),
                          heldout=_differences(low['heldout'],high['heldout'])))
    paired_stats={part:{name:{metric:_stats([pair[part][name][metric] for pair in pairs])
        for metric in METRIC_KEYS} for name in WINDOWS} for part in ('primary','heldout')}
    delta=paired_stats['primary']['first_120s']['last_10s_mean_xy_displacement_m']
    direction=('没有完整可比的120秒主工况种子配对，不能按早停的小漂移排名。' if not delta['count'] else
        f"低学习率减高学习率的120秒末10秒平均位移，{delta['count']}/3个完整种子配对的中位差为{delta['median']:+.6f}米；负值表示低学习率组漂移较小。")
    progress={}
    for identity,records in bundle.get('progress_records',{}).items():
        if identity not in policies or identity==initial:raise ValueError('Unexpected progress policy identity')
        progress[identity]=_progress(records)
    return dict(schema_version=1,kind='mature_balance_finetune_summary',training_statuses=verified,
        initial=reference,conditions=conditions,paired_results=pairs,paired_stats=paired_stats,
        case_rows=rows,failures=failures,progress_curves=progress,
        interpretation=direction+' 必须同时核对是否保留平衡及是否优于不更新的成熟参考。',
        aggregation='Average fixed primary states within each training seed, then pair low_lr minus high_lr; heldout states are separate. A seed-level horizon comparison is unknown if any state in that set fails to complete that horizon.',
        metric_direction='Smaller XY max/RMS/last-10-second mean is better only after completing the requested horizon; larger in-band duration is better.',
        limitations=['Both learning-rate groups start from the same mature policy; prior BC-start experiments are not baseline repeats here.',
            'Both trained groups use the true-position addition. Relative to the unchanged reference, reward and parameter-update effects are not separated.',
            'Early termination and missing horizon observations are shown explicitly; smaller displacement before failure is not an improvement.',
            'Three training seeds and five fixed initial states do not establish statistical significance, random robustness or hardware performance.',
            'Base, additional-position and objective returns cover each actual first episode; unequal observed durations prevent ranking them alone.'])


def _plot(bundle,summary,destination,horizon):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'initial':'#777777','high_lr':'#0072B2','low_lr':'#D55E00'}
    figure,axes=plt.subplots(4,3,figsize=(15,13),squeeze=False)
    measures=(('first_30s','rms_xy_displacement_m',100.,'First 30 s: XY RMS (cm)'),
              ('first_120s','last_10s_mean_xy_displacement_m',100.,'120 s: final 10 s mean XY (cm)'),
              ('first_30s','within_1cm_seconds_after_settling',1.,'First 30 s: time within 1 cm (s)'))
    for col,(window,metric,scale,title) in enumerate(measures):
        panel=axes[0,col]
        reference=summary['initial']['primary']['windows'][window][metric]
        if reference is not None:panel.axhline(reference*scale,color=colors['initial'],ls='--',label='Unchanged mature policy')
        for x,condition in enumerate(CONDITIONS):
            stats=summary['conditions'][condition]['primary_stats'][window][metric]
            if stats['count']:
                panel.vlines(x,stats['min']*scale,stats['max']*scale,color=colors[condition],alpha=.5,lw=2)
                panel.plot(x,stats['median']*scale,'_',color=colors[condition],markersize=18,markeredgewidth=2)
            for offset,marker,row in zip((-.09,0.,.09),('o','^','s'),summary['conditions'][condition]['seed_results']):
                value=row['primary']['windows'][window][metric]
                if value is not None:panel.plot(x+offset,value*scale,marker,color=colors[condition],markersize=5)
            panel.text(x,.02,f"{stats['count']}/3 complete seeds",ha='center',va='bottom',fontsize=7,transform=panel.get_xaxis_transform())
        panel.set_xticks((0,1),('3e-5','3e-6'));panel.set_title(title,fontsize=11);panel.set_ylim(bottom=0.)
        if col==0:panel.legend(fontsize=7)
    policies,trained,initial=_policy_metadata(bundle)
    nominal={case['policy_id']:_validated(case['trace']) for case in bundle['cases'] if case['initial_state_id']=='nominal'}
    for row,seed in enumerate(SEEDS,1):
        for condition,identity in [('initial',initial)]+[(condition,trained[condition,seed]) for condition in CONDITIONS]:
            data=nominal[identity]
            for col,(metric,scale) in enumerate((('displacement',100.),('pitch_error',1.),('speed',1.))):
                panel=axes[row,col]
                panel.plot(data['time'],[v*scale for v in data[metric]],color=colors[condition],
                    label='Mature reference' if condition=='initial' else condition,
                    lw=1. if condition=='initial' else 1.3,ls='--' if condition=='initial' else '-')
                panel.plot(data['time'][-1],data[metric][-1]*scale,
                    'x' if data['termination']['terminated'] else '|',color=colors[condition],markersize=5)
        for col,title in enumerate(('True XY displacement (cm)','Pitch error (deg)','Horizontal speed (m/s)')):
            panel=axes[row,col];panel.set_title(f'Seed {seed}: nominal | {title}',fontsize=10)
            panel.set_xlim(0,horizon);panel.set_xlabel('Time (s)')
            if row==1 and col==0:panel.legend(fontsize=8)
    for panel in axes.flat:
        panel.grid(alpha=.18);panel.spines[['top','right']].set_visible(False);panel.tick_params(labelsize=8)
    figure.suptitle(f'Mature-policy true-position fine tuning | 30 s and 120 s outcomes | view: first {horizon:g} s',fontsize=13)
    figure.tight_layout(rect=(0,0,1,.97))
    try:figure.savefig(destination,dpi=160,bbox_inches='tight',facecolor='white')
    finally:plt.close(figure)


def _plot_progress(summary,destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(1,3,figsize=(15,4.3),squeeze=False)
    panels=(('rms_xy_displacement_m',100.,'30 s XY RMS (cm)'),
            ('last_10s_mean_xy_displacement_m',100.,'30 s final 10 s mean XY (cm)'),
            (None,1.,'Observed first-episode duration (s)'))
    for col,(metric,scale,title) in enumerate(panels):
        panel=axes[0,col]
        reference=(summary['initial']['primary']['windows']['first_30s'][metric]
                   if metric else min(30.,summary['initial']['primary']['mean_observed_episode_seconds']))
        if reference is not None:panel.axhline(reference*scale,color='#777777',ls='--',label='Unchanged mature policy')
        for condition,color in (('high_lr','#0072B2'),('low_lr','#D55E00')):
            for seed,style in zip(SEEDS,('-','--',':')):
                identity=summary['conditions'][condition]['seed_results'][seed]['policy_id']
                rows=summary['progress_curves'].get(identity,[])
                if not rows:continue
                values=[(row['primary']['windows']['first_30s'][metric] if metric else
                         min(30.,row['primary']['mean_observed_episode_seconds'])) for row in rows]
                panel.plot([row['steps'] for row in rows],[value*scale if value is not None else None for value in values],
                    color=color,ls=style,marker=('o','^','s')[seed],ms=3,lw=1.,label=f'{condition} seed {seed}')
        panel.set_title(title,fontsize=11);panel.set_xlabel('Training environment steps');panel.set_ylim(bottom=0.)
        panel.grid(alpha=.18);panel.spines[['top','right']].set_visible(False)
        if col==0:panel.legend(fontsize=7)
    figure.suptitle('Deterministic 30 s snapshot evaluations | primary three states averaged within each seed',fontsize=12)
    figure.tight_layout(rect=(0,0,1,.92))
    try:figure.savefig(destination,dpi=160,bbox_inches='tight',facecolor='white')
    finally:plt.close(figure)


def _display(value,scale=1.,signed=False):
    if value is None:return '—'
    sign='+' if signed else ''
    if isinstance(value,Mapping):
        if value['median'] is None:return '— (0/3)'
        return f"{value['median']*scale:{sign}.3f} [{value['min']*scale:{sign}.3f}, {value['max']*scale:{sign}.3f}] ({value['count']}/3)"
    return f'{value*scale:{sign}.3f}'


def _results_table(summary,part,window):
    horizon=30 if window=='first_30s' else 120
    count=3 if part=='primary' else 2
    reference=summary['initial'][part]['windows'][window]
    cells=[LABELS['initial'],f"{reference['completed_cases']}/{count}"]
    columns=(('max_xy_displacement_m',100.),('rms_xy_displacement_m',100.),
        ('last_10s_mean_xy_displacement_m',100.),('within_1cm_seconds_after_settling',1.),
        ('in_band_seconds_after_settling',1.))
    cells += [_display(reference[key],scale) for key,scale in columns]
    rows=['<tr>'+''.join('<td>'+escape(cell)+'</td>' for cell in cells)+'</tr>']
    for condition in CONDITIONS:
        result=summary['conditions'][condition]
        completed=result[f'{part}_reached_{horizon}s_cases']
        cells=[result['label'],f'{completed}/{count*3}']
        cells += [_display(result[part+'_stats'][window][key],scale) for key,scale in columns]
        rows.append('<tr>'+''.join('<td>'+escape(cell)+'</td>' for cell in cells)+'</tr>')
    header='<tr><th>组别</th><th>完成工况</th><th>XY最大位移 cm</th><th>XY RMS cm</th><th>末10秒平均位移 cm</th><th>1cm内合格秒</th><th>5cm内合格秒</th></tr>'
    return '<div class="table"><table><thead>'+header+'</thead><tbody>'+''.join(rows)+'</tbody></table></div>'


def write_report(bundle,output_dir,statuses=None):
    """Create browser-readable JSON/HTML and static scientific figures from saved data."""
    summary=summarize(bundle,statuses)
    output_dir=Path(output_dir).expanduser().resolve();output_dir.mkdir(parents=True,exist_ok=True)
    paths={key:output_dir/name for key,name in (
        ('report_html','report.html'),('comparison_png','comparison.png'),
        ('comparison_detail_png','comparison-detail.png'),('summary_csv','summary.csv'),
        ('summary_json','summary.json'),('paired_summary_json','paired-summary.json'))}
    _plot(bundle,summary,paths['comparison_png'],120.)
    _plot(bundle,summary,paths['comparison_detail_png'],30.)
    if summary['progress_curves']:
        paths['progress_png']=output_dir/'progress.png';_plot_progress(summary,paths['progress_png'])
    text=json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
    for key in ('summary_json','paired_summary_json'):paths[key].write_text(text,encoding='utf-8')
    with paths['summary_csv'].open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summary['case_rows'][0]));writer.writeheader();writer.writerows(summary['case_rows'])
    paired=[]
    for pair in summary['paired_results']:
        values=[str(pair['seed'])]
        for window in WINDOWS:
            values += [_display(pair['primary'][window][key],scale,True) for key,scale in
                (('rms_xy_displacement_m',100.),('last_10s_mean_xy_displacement_m',100.),('within_1cm_seconds_after_settling',1.))]
        paired.append('<tr>'+''.join('<td>'+escape(value)+'</td>' for value in values)+'</tr>')
    failures=''.join('<li>'+escape(row['policy_id'])+' / '+escape(row['initial_state_id'])+
        f"：实际观察{row['observed_episode_seconds']:.3f}秒，"+escape(row['reason'])+
        ('（任务终止）' if row['terminated'] else '（不足120秒的时限截断）')+'</li>' for row in summary['failures'])
    progress_section=('<h2>训练过程的30秒固定回放</h2><p>每个点为主三个固定初态的独立确定性回放平均，非随机训练rollout收益。灰线为不更新的成熟参考。RMS或末10秒均值缺点表示至少一个主初态未完成30秒，不将缺失填成零。</p><img src="progress.png" alt="固定回放的位置漂移和存活随实际训练步变化">'
                      if summary['progress_curves'] else '<p>未提供过程回放记录；本报告不补造训练曲线。</p>')
    evidence=('六次训练完成记录已核对，实际预算与完整成熟初始化一致。' if summary['training_statuses']['verified'] else
              '未提供训练状态记录；实际训练步数与初始化身份仍需外部核对。')
    content=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>成熟平衡模型的真实位置微调</title><style>body{{font:16px/1.65 system-ui,sans-serif;color:#222;margin:32px auto;max-width:1500px;padding:0 24px}}h1{{font-size:27px}}h2{{font-size:21px;margin-top:30px}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:9px 11px;border-bottom:1px solid #ddd;text-align:right}}th:first-child,td:first-child{{text-align:left}}.table{{overflow-x:auto}}img{{display:block;width:100%;height:auto}}small{{color:#666}}</style>
<h1>成熟平衡模型的真实位置微调</h1><p>{escape(summary['interpretation'])}</p>
<p>两组都从同一成熟模型恢复完整actor、critic与固定normalizer，均加入相同真实XY奖励，只比较固定学习率3e-5与3e-6，各三个训练seed；另保留不更新参考。{escape(evidence)}</p>
<p>30秒与120秒分开统计。位移是相对实际reset的世界XY距离模长；RMS与末10秒平均按记录时间加权。末10秒均值在实际观察不足30秒时未知。5cm/1cm合格时间剔除首1秒，并同时要求俯仰误差≤10°、水平速度≤.1m/s。</p>
<p>每seed先平均三个主初态，再显示三个训练seed的中位数[min,max]；括号标出完整种子数。某个初态未撑满对应时域时，该seed的漂移比较记未知，失败仍逐条列出。不能用早倒前的小位移算作改善；max是工况内最大值再平均，末10秒均值用于观察启动瞬态之后的漂移。</p>
<h2>主初态：前30秒</h2>{_results_table(summary,'primary','first_30s')}
<h2>主初态：120秒</h2>{_results_table(summary,'primary','first_120s')}
<h2>预留初态：前30秒</h2>{_results_table(summary,'heldout','first_30s')}
<h2>预留初态：120秒</h2>{_results_table(summary,'heldout','first_120s')}
<h2>主初态的种子配对差：低学习率减高学习率</h2><div class="table"><table><thead><tr><th>seed</th><th>30秒RMS差 cm</th><th>30秒末10秒差 cm</th><th>30秒1cm时间差 s</th><th>120秒RMS差 cm</th><th>120秒末10秒差 cm</th><th>120秒1cm时间差 s</th></tr></thead><tbody>{''.join(paired)}</tbody></table></div>
<p>位移差为负表示低学习率组更小，合格时间差为正表示更长。配对差与两组中位数直接相减不必相同。相对不更新参考的差异同时包含新增奖励和参数更新，不能独立声称位置奖励的因果收益；三个seed和固定初态不证明统计显著、随机鲁棒性或实机表现。</p>
<h2>未完成120秒的工况</h2>{'<ul>'+failures+'</ul>' if failures else '<p>35个首回合均完成120秒；这仍不自动证明长期稳定或实机鲁棒性。</p>'}
{progress_section}
<h2>前30秒曲线</h2><img src="comparison-detail.png" alt="真实位移俯仰速度的前30秒比较">
<h2>完整120秒时域</h2><img src="comparison.png" alt="真实位移俯仰速度的120秒比较，提前终止后留空">
<p>base_return是原任务奖励之和；真实位置新增项逐post-step按−10·(1−exp(−(XY/.05)²))·dt/.01积分，objective_return为两者相加。不同存活长度下不以累计回报独立排名。</p>
<p><a href="results.json">原始轨迹</a> · <a href="summary.json">汇总与过程指标JSON</a> · <a href="summary.csv">逐工况CSV</a></p></html>'''
    paths['report_html'].write_text(content+'\n',encoding='utf-8')
    return {key:str(path) for key,path in paths.items()}
