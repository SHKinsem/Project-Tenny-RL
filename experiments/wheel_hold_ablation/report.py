"""Paired weak-wheel-hold reports; physical measurements share the mature contract."""
from collections.abc import Mapping
import csv
from html import escape
import json
import math
from pathlib import Path

from experiments.mature_balance_finetune.report import (
    measure_case, _number, _validated, _mean, _stats, _group, _group_stats,
    _differences, _display, WINDOWS, METRIC_KEYS,
    PRIMARY_STATES, HELDOUT_STATES, STATES, SEEDS,
)

CONDITIONS = ('original_wheel', 'weak_wheel')
LEARNING_RATE = 3e-5
WHEEL_WEIGHTS = {'original_wheel': 10., 'weak_wheel': 1.}
WHEEL_METRIC = 'reward/tracking_wheels_sinposition'
LABELS = {'initial':'成熟模型（不更新参考）', 'original_wheel':'轮角位置权重10（复用）',
          'weak_wheel':'轮角位置权重1（新训练）'}


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
            if not math.isclose(rate,LEARNING_RATE,rel_tol=0,abs_tol=1e-12):
                raise ValueError('Both wheel-hold groups must use learning rate 3e-5')
            weight=_number(source.get('wheel_position_weight'),'wheel_position_weight')
            if weight!=WHEEL_WEIGHTS[condition]:raise ValueError('Wheel-position weight differs from its condition')
            if type(source.get('reused')) is not bool or source['reused']!=(condition=='original_wheel'):
                raise ValueError('Only the weight-10 trained comparator is reused')
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
    seen=set();steps=set();initials=set();times={condition:[] for condition in CONDITIONS};provenance=[]
    for row in records:
        if not isinstance(row,Mapping):raise ValueError('Invalid run status')
        key=row.get('condition'),row.get('seed')
        if key not in trained or key in seen or row.get('phase')!='completed':raise ValueError('Incomplete or duplicate training status')
        expected_reuse=key[0]=='original_wheel'
        if type(row.get('reused')) is not bool or row['reused']!=expected_reuse:
            raise ValueError('Expected three reused original runs and three newly trained weak-wheel runs')
        count=row.get('steps')
        if type(count) is not int or count<=0:raise ValueError('Need actual positive training steps')
        digest=row.get('initial_array_digest')
        if not isinstance(digest,str) or not digest:raise ValueError('Need complete mature initializer identity')
        seen.add(key);steps.add(count);initials.add(digest)
        elapsed=row.get('elapsed_seconds')
        if elapsed is not None:
            elapsed=_number(elapsed,'elapsed_seconds')
            if elapsed<0:raise ValueError('Run elapsed time cannot be negative')
            times[key[0]].append(elapsed)
        provenance.append({name:row[name] for name in ('condition','seed','reused','source_run','elapsed_seconds','created_at','completed_at') if name in row})
    if len(steps)!=1 or len(initials)!=1:raise ValueError('Runs differ in training budget or complete initializer')
    actual=next(iter(steps))
    if manifest.get('steps_per_run',actual)!=actual:raise ValueError('Actual steps differ from the frozen budget')
    return dict(verified=True,runs=6,steps_per_run=actual,initial_array_digest=next(iter(initials)),
        reused_runs=3,new_runs=3,reused_environment_steps=3*actual,new_environment_steps=3*actual,
        reused_run_elapsed_seconds_sum=sum(times['original_wheel']) if len(times['original_wheel'])==3 else None,
        new_run_elapsed_seconds_sum=sum(times['weak_wheel']) if len(times['weak_wheel'])==3 else None,
        elapsed_time_scope='Sum of each run process elapsed time, not the matrix wall clock; reused training occurred in the prior experiment.',
        run_provenance=provenance)


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


def _objectives(saved,physical,condition,native_metrics=None):
    """Normalize legacy baseline returns without relabeling them as a weak target."""
    common=physical['base_return']+physical['true_position_addition_return']
    for name in ('common_xy_objective_return','objective_return'):
        if name in saved and not math.isclose(_number(saved[name],name),common,rel_tol=1e-10,abs_tol=1e-7):
            raise ValueError(name+' differs from the common base10 plus XY objective')
    adjustment=saved.get('wheel_position_adjustment_return')
    training=saved.get('training_objective_return')
    if condition=='original_wheel':
        if adjustment is not None and abs(_number(adjustment,'wheel adjustment'))>1e-7:
            raise ValueError('Original weight10 has zero wheel adjustment')
        adjustment=0.
        if training is not None and not math.isclose(_number(training,'training objective'),common,rel_tol=1e-10,abs_tol=1e-7):
            raise ValueError('Original training objective differs from common base10 plus XY')
        training=common
    elif condition=='weak_wheel':
        if (adjustment is None)!=(training is None):raise ValueError('Weak training objective needs its wheel adjustment')
        if adjustment is not None:
            adjustment=_number(adjustment,'wheel adjustment');training=_number(training,'training objective')
            if not math.isclose(training,common+adjustment,rel_tol=1e-10,abs_tol=1e-7):
                raise ValueError('Weak training objective differs from common plus wheel adjustment')
            raw_score=saved.get('native_wheel_position_raw_score_sum')
            if raw_score is None and isinstance(native_metrics,Mapping):raw_score=native_metrics.get(WHEEL_METRIC)
            if isinstance(raw_score,(int,float)):
                expected=(WHEEL_WEIGHTS[condition]-10.)*_number(raw_score,WHEEL_METRIC)
                if not math.isclose(adjustment,expected,rel_tol=1e-10,abs_tol=1e-7):
                    raise ValueError('Wheel adjustment differs from native raw wheel-score sum at dt/ref_dt=1')
    else:
        adjustment=training=None
    return dict(common_xy_objective_return=common,wheel_position_adjustment_return=adjustment,
                training_objective_return=training)


def _progress(records,condition):
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
        objectives=[_objectives(indexed[state],indexed[state],condition) for state in PRIMARY_STATES]
        training=[value['training_objective_return'] for value in objectives]
        output.append(dict(steps=step,primary=primary,
            mean_base_return=_mean([indexed[state]['base_return'] for state in PRIMARY_STATES]),
            mean_common_xy_objective_return=_mean([value['common_xy_objective_return'] for value in objectives]),
            mean_training_objective_return=_mean(training) if all(value is not None for value in training) else None))
    return output


def summarize(bundle,statuses=None):
    """Seed-paired drift summaries; a failed horizon makes its drift comparison unknown."""
    if not isinstance(bundle,Mapping) or bundle.get('kind')!='wheel_hold_ablation_v1':
        raise ValueError('Need a wheel_hold_ablation_v1 result bundle')
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
        condition=source.get('condition','initial')
        native_metrics=case.get('native_metrics',case.get('native_first_episode_metrics'))
        objectives=_objectives(case.get('measurement',{}),value,condition,native_metrics)
        row=dict(policy_id=identity,condition=source.get('condition','initial'),seed=source.get('seed'),
            initial_state_id=state,observed_episode_seconds=value['observed_episode_seconds'],
            reached_30s=value['reached_30s'],reached_120s=value['reached_120s'],
            termination_reason=value['termination'].get('reason','unknown'),
            base_return=value['base_return'],true_position_addition_return=value['true_position_addition_return'],
            **objectives)
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
        conditions[condition]=dict(label=LABELS[condition],learning_rate=LEARNING_RATE,
            wheel_position_weight=WHEEL_WEIGHTS[condition],reused=condition=='original_wheel',seed_results=seeds,
            primary_stats=_group_stats([row['primary'] for row in seeds]),
            heldout_stats=_group_stats([row['heldout'] for row in seeds]),
            primary_reached_30s_cases=sum(row['primary']['reached_30s_count'] for row in seeds),
            primary_reached_120s_cases=sum(row['primary']['reached_120s_count'] for row in seeds),
            heldout_reached_30s_cases=sum(row['heldout']['reached_30s_count'] for row in seeds),
            heldout_reached_120s_cases=sum(row['heldout']['reached_120s_count'] for row in seeds))
    pairs=[]
    for original,weak in zip(conditions['original_wheel']['seed_results'],conditions['weak_wheel']['seed_results']):
        pairs.append(dict(seed=original['seed'],primary=_differences(weak['primary'],original['primary']),
                          heldout=_differences(weak['heldout'],original['heldout'])))
    paired_stats={part:{name:{metric:_stats([pair[part][name][metric] for pair in pairs])
        for metric in METRIC_KEYS} for name in WINDOWS} for part in ('primary','heldout')}
    delta=paired_stats['primary']['first_120s']['last_10s_mean_xy_displacement_m']
    direction=('没有完整可比的120秒主工况种子配对，不能按早停的小漂移排名。' if not delta['count'] else
        f"弱轮角约束减原轮角约束的120秒末10秒平均位移，{delta['count']}/3个完整种子配对的中位差为{delta['median']:+.6f}米；负值表示权重1组漂移较小。")
    progress={}
    for identity,records in bundle.get('progress_records',{}).items():
        if identity not in policies or identity==initial:raise ValueError('Unexpected progress policy identity')
        progress[identity]=_progress(records,policies[identity]['source']['condition'])
    return dict(schema_version=1,kind='wheel_hold_ablation_summary',training_statuses=verified,
        model_origin=manifest.get('model_origin'),policy_provenance={identity:policy.get('source',{}) for identity,policy in policies.items()},
        initial=reference,conditions=conditions,paired_results=pairs,paired_stats=paired_stats,
        case_rows=rows,failures=failures,progress_curves=progress,
        interpretation=direction+' 必须同时核对是否保留平衡及是否优于不更新的成熟参考。',
        objective_accounting=dict(common_evaluation='Native base wheel-position weight10 plus identical true-XY addition.',
            weak_training='Common XY objective plus (1-10) times the native raw wheel-position score sum at dt/ref_dt=1.',
            weak_cases_with_training_objective=sum(row['condition']=='weak_wheel' and row['training_objective_return'] is not None for row in rows),
            weak_cases_expected=15,legacy_baseline_objective_normalized=True),
        aggregation='Average fixed primary states within each training seed, then pair weak_wheel minus original_wheel; heldout states are separate. A seed-level horizon comparison is unknown if any state in that set fails to complete that horizon.',
        metric_direction='Smaller XY max/RMS/last-10-second mean is better only after completing the requested horizon; larger in-band duration is better.',
        limitations=['Both wheel-hold groups start from the same complete mature policy, with fixed learning rate 3e-5; only wheel-position contribution changes from weight10 to weight1.',
            'Three matched weight10 runs are reused from the prior mature fine-tuning experiment; only three weight1 runs are newly trained.',
            'Both trained groups use the true-position addition. Relative to the unchanged reference, reward and parameter-update effects are not separated.',
            'Keeping a weak wheel-angle term is a simulation design choice; no IMU-drift, slip or new observation-noise model is tested here.',
            'The common base10-plus-XY evaluation return is not the weak group training objective.',
            'Early termination and missing horizon observations are shown explicitly; smaller displacement before failure is not an improvement.',
            'Three training seeds and five fixed initial states do not establish statistical significance, random robustness or hardware performance.',
            'Base, additional-position, common and training returns cover each actual first episode; unequal observed durations prevent ranking them alone.'])


def _plot(bundle,summary,destination,horizon):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'initial':'#777777','original_wheel':'#0072B2','weak_wheel':'#D55E00'}
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
        panel.set_xticks((0,1),('Wheel weight10 (reused)','Wheel weight1 (new)'));panel.set_title(title,fontsize=11);panel.set_ylim(bottom=0.)
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
    figure.suptitle(f'Wheel-position hold: weight10 vs weight1 | fixed LR 3e-5 | view: first {horizon:g} s',fontsize=13)
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
        for condition,color in (('original_wheel','#0072B2'),('weak_wheel','#D55E00')):
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
    figure.suptitle('Wheel hold weight10 (reused) vs weight1 (new) | fixed LR 3e-5 | deterministic 30 s snapshots',fontsize=12)
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
    receipt=summary['training_statuses']
    if receipt['verified']:
        reuse_note=f"权重10组三个seed复用先前已完成训练，共{receipt['reused_environment_steps']:,}环境步；本轮只新训权重1组三次，新增{receipt['new_environment_steps']:,}步，各{receipt['steps_per_run']:,}步。"
        if receipt['reused_run_elapsed_seconds_sum'] is not None:
            reuse_note+=f"旧三次训练进程用时合计{receipt['reused_run_elapsed_seconds_sum']:.3f}秒，发生在先前实验，不计入本轮新增训练；这不是矩阵墙钟时长。"
        else:reuse_note+='旧训练用时未随状态提供，保留未知，不伪造本轮重新训练时间。'
    else:reuse_note='权重10组为复用基线，权重1组为新训练；未提供状态记录，实际新增预算及旧训练用时仍待核对。'
    known=summary['objective_accounting']['weak_cases_with_training_objective']
    objective_note=f'弱约束组15个终点工况中，{known}个保留了可核对的实际训练目标；其余保留未知。'
    provenance=escape(json.dumps(dict(model_origin=summary['model_origin'],run_provenance=receipt.get('run_provenance',[])),ensure_ascii=False,indent=2))
    content=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>成熟策略的弱轮角位置约束对照</title><style>body{{font:16px/1.65 system-ui,sans-serif;color:#222;margin:32px auto;max-width:1500px;padding:0 24px}}h1{{font-size:27px}}h2{{font-size:21px;margin-top:30px}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:9px 11px;border-bottom:1px solid #ddd;text-align:right}}th:first-child,td:first-child{{text-align:left}}.table{{overflow-x:auto}}img{{display:block;width:100%;height:auto}}small{{color:#666}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}}</style>
<h1>成熟策略的弱轮角位置约束对照</h1><p>{escape(summary['interpretation'])}</p>
<p>两组都从同一成熟模型恢复完整actor、critic与固定normalizer，真实XY项权重均为10、学习率均固定3e-5。唯一训练差异是轮角位置项贡献从10降到1；共享zero_keep的航向项与其他条件保持。各三个配对seed，另保留不更新参考。{escape(evidence)}</p>
<p>{escape(reuse_note)}</p>
<p>保留弱轮角约束是为后续部署考虑，本轮没有增加IMU漂移、轮滑或观测噪声模型，不能据此声称实机漂移问题已解决。</p>
<p>30秒与120秒分开统计。位移是相对实际reset的世界XY距离模长；RMS与末10秒平均按记录时间加权。末10秒均值在实际观察不足30秒时未知。5cm/1cm合格时间剔除首1秒，并同时要求俯仰误差≤10°、水平速度≤.1m/s。</p>
<p>每seed先平均三个主初态，再显示三个训练seed的中位数[min,max]；括号标出完整种子数。某个初态未撑满对应时域时，该seed的漂移比较记未知，失败仍逐条列出。不能用早倒前的小位移算作改善；max是工况内最大值再平均，末10秒均值用于观察启动瞬态之后的漂移。</p>
<h2>主初态：前30秒</h2>{_results_table(summary,'primary','first_30s')}
<h2>主初态：120秒</h2>{_results_table(summary,'primary','first_120s')}
<h2>预留初态：前30秒</h2>{_results_table(summary,'heldout','first_30s')}
<h2>预留初态：120秒</h2>{_results_table(summary,'heldout','first_120s')}
<h2>主初态的种子配对差：弱轮角约束减原轮角约束</h2><div class="table"><table><thead><tr><th>seed</th><th>30秒RMS差 cm</th><th>30秒末10秒差 cm</th><th>30秒1cm时间差 s</th><th>120秒RMS差 cm</th><th>120秒末10秒差 cm</th><th>120秒1cm时间差 s</th></tr></thead><tbody>{''.join(paired)}</tbody></table></div>
<p>位移差为负表示权重1组更小，合格时间差为正表示更长。配对差与两组中位数直接相减不必相同。相对不更新参考的差异同时包含奖励和参数更新，不能独立声称真实XY奖励的因果收益；三个seed和固定初态不证明统计显著、随机鲁棒性或实机表现。</p>
<h2>未完成120秒的工况</h2>{'<ul>'+failures+'</ul>' if failures else '<p>35个首回合均完成120秒；这仍不自动证明长期稳定或实机鲁棒性。</p>'}
{progress_section}
<h2>前30秒曲线</h2><img src="comparison-detail.png" alt="真实位移俯仰速度的前30秒比较">
<h2>完整120秒时域</h2><img src="comparison.png" alt="真实位移俯仰速度的120秒比较，提前终止后留空">
<h2>共同评价回报与实际训练目标</h2><p>base_return始终是原native wheel10任务回报；真实位置新增项逐post-step按−10·(1−exp(−(XY/.05)²))·dt/.01积分。两者之和叫common_xy_objective_return，用于共同评价。原权重10组旧记录的objective_return只在该组规范化为同一数值。</p>
<p>权重1组实际training_objective_return还需加wheel_position_adjustment_return，即(1−10)乘native原始轮角位置score累计（本轮dt/reference_dt=1）。它不是共同wheel10评价回报；缺少该证据时实际目标未知，不能拿旧objective_return冒充。{escape(objective_note)}不同存活长度下不以累计回报独立排名。</p>
<details><summary>成熟起点及复用来源</summary><pre>{provenance}</pre></details>
<p><a href="results.json">原始轨迹</a> · <a href="summary.json">汇总与过程指标JSON</a> · <a href="summary.csv">逐工况CSV</a></p></html>'''
    paths['report_html'].write_text(content+'\n',encoding='utf-8')
    return {key:str(path) for key,path in paths.items()}
