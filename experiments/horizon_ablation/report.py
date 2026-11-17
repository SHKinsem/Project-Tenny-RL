"""Paired reports for one training-time-limit change under a common evaluation.

Only saved trajectories and run records are read. A training seed is a repeat;
fixed evaluation states are conditions within that seed. Plotting is optional at
import time and never loads a policy or a training backend.
"""
from collections.abc import Mapping
import csv
from html import escape
import json
import math
from pathlib import Path
import re

from experiments.reward_ablation.report import _mean, _number, _series, _stats


CONDITIONS = ('h6', 'h30')
SEEDS = (0, 1, 2)
PRIMARY_STATES = ('nominal', 'pitch-plus', 'pitch-minus')
HELDOUT_STATES = ('heldout-plus', 'heldout-minus')
STATES = PRIMARY_STATES + HELDOUT_STATES
LABELS = {'initial': 'BC初始化（未继续训练）', 'h6': '6秒训练回合', 'h30': '30秒训练回合'}


def _identity(policy_id):
    if policy_id == 'initial':
        return 'initial', None
    match = re.fullmatch(r'(h6|h30)-seed-([012])', str(policy_id))
    if match is None:
        raise ValueError('Unexpected horizon policy identity: ' + str(policy_id))
    return match[1], int(match[2])


def _matrix_metadata(bundle):
    manifest = bundle.get('experiment_manifest') or {}
    if not isinstance(manifest, Mapping):
        raise ValueError('experiment_manifest must be a mapping')
    seeds = manifest.get('seeds', list(SEEDS))
    if not isinstance(seeds, list) or any(type(seed) is not int for seed in seeds) or sorted(seeds) != list(SEEDS):
        raise ValueError('Expected paired training seeds 0, 1, 2')
    steps = manifest.get('steps_per_run', 2097152)
    if type(steps) is not int or steps <= 0:
        raise ValueError('steps_per_run must be a positive integer')
    statuses = bundle.get('run_statuses')
    if statuses is not None:
        if isinstance(statuses, Mapping):
            records = dict(statuses)
        elif isinstance(statuses, list):
            records = {}
            for status in statuses:
                if not isinstance(status, Mapping):
                    raise ValueError('Run statuses must contain mappings')
                identity = status.get('policy_id', f"{status.get('condition')}-seed-{status.get('seed')}")
                if identity in records:
                    raise ValueError('Duplicate run status: ' + str(identity))
                records[identity] = status
        else:
            raise ValueError('run_statuses must be a list or mapping')
        expected = {f'{condition}-seed-{seed}' for condition in CONDITIONS for seed in SEEDS}
        if set(records) != expected:
            raise ValueError('Statuses need the complete two-by-three horizon matrix')
        hashes = set()
        for identity, status in records.items():
            if not isinstance(status, Mapping) or status.get('phase') != 'completed' or status.get('steps') != steps:
                raise ValueError('Incomplete or unequal-step run: ' + identity)
            condition, seed = _identity(identity)
            if status.get('condition') != condition or status.get('seed') != seed:
                raise ValueError('Run status identity differs from policy: ' + identity)
            if type(status.get('reused')) is not bool or status['reused'] != (condition == 'h6'):
                raise ValueError('Expected reused 6-second runs and newly trained 30-second runs')
            digest = status.get('initial_array_digest')
            if not isinstance(digest, str) or not digest:
                raise ValueError('Each run needs its complete initializer identity')
            hashes.add(digest)
        if len(hashes) != 1:
            raise ValueError('Runs did not use the same complete initializer')
    return dict(seeds=list(SEEDS), runs=6, steps_per_run=steps, statuses_verified=statuses is not None,
                reused_condition='h6', new_condition='h30', policy_hz=100, physics_hz=400,
                sole_planned_variable='Outer training episode time limit: 600 versus 3000 control steps.')


def _case_row(case):
    from experiments.balance_diagnostics.metrics import analyze_trace
    from experiments.reward_ablation.evaluation import standing_metrics
    condition, seed = _identity(case['policy_id'])
    trace = case['trace']
    series = _series(trace)
    analysis = case.get('analysis')
    if analysis is None:
        analysis = analyze_trace(trace)
    standing = case.get('standing')
    if standing is None:
        standing = standing_metrics(trace)
    full = analysis['windows']['full_episode']
    duration = _number(standing['survival_seconds'], 'survival_seconds')
    if not 0 < duration <= 30 or not math.isclose(duration, series['time'][-1], rel_tol=0, abs_tol=1e-9):
        raise ValueError('Evaluation trace duration differs from its 30-second outcome')
    if not math.isclose(_number(full['observed_seconds'], 'observed_seconds'), duration, rel_tol=0, abs_tol=1e-9):
        raise ValueError('Analysis window duration differs from the trace')
    total = _number(standing['in_band_seconds_after_settling'], 'in-band seconds')
    longest = _number(standing['longest_in_band_seconds_after_settling'], 'longest in-band seconds')
    if not 0 <= longest <= total + 1e-9 or total > max(0., duration-1.) + 1e-9:
        raise ValueError('Invalid cumulative or continuous post-settling standing time')
    reached = standing['reached_horizon']
    expected_reached = math.isclose(duration, 30., rel_tol=0, abs_tol=1e-9) and trace['termination']['truncated']
    if type(reached) is not bool or reached != expected_reached:
        raise ValueError('Horizon status must distinguish a time limit from task failure')
    row = dict(policy_id=case['policy_id'], policy_label=str(case.get('label', LABELS[condition])),
        condition=condition, seed=seed, initial_state_id=case['initial_state_id'],
        in_band_seconds_after_settling=total, longest_in_band_seconds_after_settling=longest,
        survival_seconds=duration, reached_horizon=reached,
        max_displacement_m=_number(full['max_horizontal_displacement_m'], 'max displacement'),
        pitch_error_rms_deg=_number(full['pitch_error_rms_deg'], 'pitch error RMS'),
        horizontal_speed_rms_m_s=_number(full['horizontal_speed_rms_m_s'], 'horizontal speed RMS'),
        termination_reason=str(trace['termination']['reason']))
    if any(row[key] < 0 for key in ('max_displacement_m', 'pitch_error_rms_deg', 'horizontal_speed_rms_m_s')):
        raise ValueError('Physical magnitudes must be nonnegative')
    return row, series


def paired_summary(bundle):
    """Summarize complete final policies, pairing training seeds before medians."""
    if not isinstance(bundle, Mapping) or bundle.get('schema_version') != 1 or bundle.get('kind') != 'balance_diagnostics':
        raise ValueError('Expected balance_diagnostics version 1 results')
    if _number(bundle.get('horizon_s'), 'horizon_s') != 30.:
        raise ValueError('Both groups must use the same 30-second evaluation')
    matrix = _matrix_metadata(bundle)
    policies = {'initial'} | {f'{condition}-seed-{seed}' for condition in CONDITIONS for seed in SEEDS}
    indexed, rows, series = {}, [], {}
    for case in bundle.get('cases', []):
        identity, state = case['policy_id'], case['initial_state_id']
        if identity not in policies or state not in STATES or (identity, state) in indexed:
            raise ValueError('Unexpected or duplicate policy/physical-condition case')
        row, values = _case_row(case)
        indexed[identity, state] = row
        rows.append(row)
        series[identity, state] = values
    if set(indexed) != {(policy, state) for policy in policies for state in STATES}:
        raise ValueError('Need BC and all six final policies on the same five initial states')

    def group(identity, states):
        selected = [indexed[identity, state] for state in states]
        return dict(mean_in_band_seconds=_mean(row['in_band_seconds_after_settling'] for row in selected),
            mean_longest_in_band_seconds=_mean(row['longest_in_band_seconds_after_settling'] for row in selected),
            mean_survival_seconds=_mean(row['survival_seconds'] for row in selected),
            mean_max_displacement_m=_mean(row['max_displacement_m'] for row in selected),
            mean_pitch_error_rms_deg=_mean(row['pitch_error_rms_deg'] for row in selected),
            reached_horizon_count=sum(row['reached_horizon'] for row in selected), evaluation_conditions=len(selected))

    initial = dict(primary=group('initial', PRIMARY_STATES), heldout=group('initial', HELDOUT_STATES))
    conditions = {}
    for condition in CONDITIONS:
        seed_results = [dict(seed=seed, primary=group(f'{condition}-seed-{seed}', PRIMARY_STATES),
                             heldout=group(f'{condition}-seed-{seed}', HELDOUT_STATES)) for seed in SEEDS]
        conditions[condition] = dict(label=LABELS[condition], training_episode_seconds=6 if condition == 'h6' else 30,
            reused=condition == 'h6', seed_results=seed_results,
            primary_in_band_seconds=_stats(item['primary']['mean_in_band_seconds'] for item in seed_results),
            primary_longest_in_band_seconds=_stats(item['primary']['mean_longest_in_band_seconds'] for item in seed_results),
            primary_survival_seconds=_stats(item['primary']['mean_survival_seconds'] for item in seed_results),
            heldout_in_band_seconds=_stats(item['heldout']['mean_in_band_seconds'] for item in seed_results))
    pairs = []
    for short, long in zip(conditions['h6']['seed_results'], conditions['h30']['seed_results']):
        pairs.append(dict(seed=short['seed'],
            delta_in_band_seconds=long['primary']['mean_in_band_seconds']-short['primary']['mean_in_band_seconds'],
            delta_longest_in_band_seconds=long['primary']['mean_longest_in_band_seconds']-short['primary']['mean_longest_in_band_seconds'],
            delta_survival_seconds=long['primary']['mean_survival_seconds']-short['primary']['mean_survival_seconds'],
            delta_to_initial_in_band_seconds=long['primary']['mean_in_band_seconds']-initial['primary']['mean_in_band_seconds']))
    delta = _stats(pair['delta_in_band_seconds'] for pair in pairs)
    to_bc = _stats(pair['delta_to_initial_in_band_seconds'] for pair in pairs)
    if math.isclose(delta['median'], 0., abs_tol=1e-9):
        interpretation = '累计合格时间的配对中位差为零，本轮未显示改善。'
    elif delta['median'] > 0 and to_bc['median'] <= 0:
        interpretation = '30秒组累计合格时间在配对描述中增加，但仍未超过BC初始化；应解读为减缓退化。'
    elif delta['median'] > 0:
        interpretation = '30秒组累计合格时间在配对描述中同时高于6秒组与BC；三种子结果仍不足以认定收敛或实机鲁棒性。'
    else:
        interpretation = '累计合格时间的配对中位差为负，本轮未显示延长回合的一致优势；差值大小需与种子范围共同阅读。'
    summary = dict(schema_version=1, kind='paired_training_horizon_summary', matrix=matrix,
        primary_states=list(PRIMARY_STATES), heldout_states=list(HELDOUT_STATES), initial=initial,
        conditions=conditions, paired_results=pairs, paired_delta_in_band_seconds=delta,
        paired_delta_longest_in_band_seconds=_stats(pair['delta_longest_in_band_seconds'] for pair in pairs),
        paired_delta_survival_seconds=_stats(pair['delta_survival_seconds'] for pair in pairs),
        improved_seed_counts={name: sum(pair[key] > 0 for pair in pairs) for name, key in (
            ('cumulative', 'delta_in_band_seconds'), ('longest', 'delta_longest_in_band_seconds'), ('survival', 'delta_survival_seconds'))},
        interpretation=interpretation,
        primary_measure='Cumulative in-band time after 1-second settling; average primary conditions within each training seed, then pair seeds.',
        scope='Three paired training seeds; five fixed evaluation conditions are not independent training repetitions.',
        limitations=['The six-second comparator reuses matched completed runs; it was not retrained in this experiment.',
            'Only the outer training time limit changes; base task timing/source stays unchanged.',
            'Survival, cumulative in-band time, and longest continuous interval are distinct outcomes.',
            'Earlier failure with smaller displacement is not better balance; training reward is not the ranking score.',
            'No policy or inner-loop frequency was varied, so this comparison cannot establish or exclude a frequency bottleneck.',
            'Fixed initial states and three seeds do not establish significance, general robustness, or convergence.'])
    return summary, rows, series


def _plot(summary, series, destination, *, horizon_seconds):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'initial': '#777777', 'h6': '#0072B2', 'h30': '#D55E00'}
    figure, axes = plt.subplots(4, 3, figsize=(15, 13), squeeze=False)
    measures = (('mean_in_band_seconds', 'primary_in_band_seconds', 'Cumulative in-band time (s)'),
                ('mean_longest_in_band_seconds', 'primary_longest_in_band_seconds', 'Longest in-band interval (s)'),
                ('mean_survival_seconds', 'primary_survival_seconds', 'First-episode survival (s)'))
    for column, (raw, stats, title) in enumerate(measures):
        panel = axes[0, column]
        for x, condition in enumerate(CONDITIONS):
            values = summary['conditions'][condition]
            distribution = values[stats]
            panel.vlines(x, distribution['min'], distribution['max'], color=colors[condition], alpha=.6, linewidth=2)
            panel.plot(x, distribution['median'], '_', color=colors[condition], markersize=18, markeredgewidth=2)
            for offset, marker, item in zip((-.09, 0, .09), ('o', '^', 's'), values['seed_results']):
                panel.plot(x+offset, item['primary'][raw], marker, color=colors[condition], markersize=5)
        panel.axhline(summary['initial']['primary'][raw], color=colors['initial'], linestyle='--', label='Untrained BC')
        panel.set_xticks(range(2), ['6 s training (reused)', '30 s training'])
        panel.set_title(title, fontsize=11)
        panel.set_ylabel('Seconds')
        panel.set_ylim(bottom=0)
        panel.legend(fontsize=8)
    for row, seed in enumerate(SEEDS, 1):
        for condition in ('initial',)+CONDITIONS:
            identity = 'initial' if condition == 'initial' else f'{condition}-seed-{seed}'
            values = series[identity, 'nominal']
            for column, key in enumerate(('displacement', 'pitch_error', 'speed')):
                panel = axes[row, column]
                panel.plot(values['time'], values[key], color=colors[condition], label=condition,
                           linewidth=1.3, linestyle='--' if condition == 'initial' else '-')
                panel.plot(values['time'][-1], values[key][-1], 'x' if values['terminated'] else '|',
                           color=colors[condition], markersize=5)
        for column, title in enumerate(('Body displacement (m)', 'Pitch error (deg)', 'Horizontal body speed (m/s)')):
            panel = axes[row, column]
            panel.set_title(f'Seed {seed}: nominal | {title}', fontsize=10)
            panel.set_xlabel('Time (s)')
            panel.set_xlim(0, horizon_seconds)
            panel.axvline(6, color='.6', linestyle=':', linewidth=1)
            for limit in ((-10., 10.) if column == 1 else ((.05,) if column == 0 else (.1,))):
                panel.axhline(limit, color='.8', linestyle=':', linewidth=.8)
            if row == 1 and column == 0:
                panel.legend(fontsize=8)
    for panel in axes.flat:
        panel.grid(alpha=.18)
        panel.tick_params(labelsize=8)
        panel.spines[['top', 'right']].set_visible(False)
    view = f'early first {horizon_seconds:g} s' if horizon_seconds < 30 else 'common full 30 s evaluation'
    figure.suptitle('Training episode horizon: 6 s vs 30 s | '+view+' | dots: training seeds', fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, .97))
    try:
        figure.savefig(destination, dpi=160, bbox_inches='tight', facecolor='white')
    finally:
        plt.close(figure)


def _display(values, signed=False):
    sign = '+' if signed else ''
    return f"{values['median']:{sign}.3f} [{values['min']:{sign}.3f}, {values['max']:{sign}.3f}]"


def write_horizon_report(bundle, output_dir):
    """Render a complete saved comparison; caller writes results.json alongside."""
    summary, rows, series = paired_summary(bundle)
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {key: output_dir/name for key, name in (
        ('report_html', 'report.html'), ('comparison_png', 'comparison.png'),
        ('comparison_detail_png', 'comparison-detail.png'), ('summary_csv', 'summary.csv'),
        ('paired_summary_json', 'paired-summary.json'))}
    _plot(summary, series, paths['comparison_png'], horizon_seconds=30.)
    _plot(summary, series, paths['comparison_detail_png'], horizon_seconds=10.)
    paths['paired_summary_json'].write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    with paths['summary_csv'].open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    initial = summary['initial']['primary']
    result_rows = ['<tr><td>BC初始化（未继续训练）</td><td>'+f"{initial['mean_in_band_seconds']:.3f}"+
        '</td><td>'+f"{initial['mean_longest_in_band_seconds']:.3f}"+'</td><td>'+f"{initial['mean_survival_seconds']:.3f}"+'</td></tr>']
    for condition in CONDITIONS:
        result = summary['conditions'][condition]
        cells = [result['label'], _display(result['primary_in_band_seconds']),
                 _display(result['primary_longest_in_band_seconds']), _display(result['primary_survival_seconds'])]
        result_rows.append('<tr>'+''.join('<td>'+escape(cell)+'</td>' for cell in cells)+'</tr>')
    pair_rows = []
    for short, long, pair in zip(summary['conditions']['h6']['seed_results'], summary['conditions']['h30']['seed_results'], summary['paired_results']):
        cells = [str(pair['seed']), f"{short['primary']['mean_in_band_seconds']:.3f}",
                 f"{long['primary']['mean_in_band_seconds']:.3f}", f"{pair['delta_in_band_seconds']:+.3f}",
                 f"{pair['delta_longest_in_band_seconds']:+.3f}", f"{pair['delta_survival_seconds']:+.3f}",
                 f"{long['heldout']['mean_in_band_seconds']-short['heldout']['mean_in_band_seconds']:+.3f}"]
        pair_rows.append('<tr>'+''.join('<td>'+escape(cell)+'</td>' for cell in cells)+'</tr>')
    labels = ''.join('<li>'+escape(row['policy_label'])+' · '+escape(row['policy_id'])+'</li>'
                     for row in rows if row['initial_state_id'] == 'nominal')
    evidence = ('已核对六份完成记录、相同实际步数与完整初始化身份；6秒组标记为复用，30秒组为新训练。'
                if summary['matrix']['statuses_verified'] else
                '策略和初态已核对齐全；未提供训练状态清单，实际步数与初始化一致性尚需结合运行记录核对。')
    content = f'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>6秒与30秒训练回合对照</title>
<style>body{{font:16px/1.65 system-ui,sans-serif;color:#222;background:#fff;margin:32px auto;max-width:1500px;padding:0 24px}}
h1{{font-size:27px}}h2{{font-size:21px;margin-top:30px}}p{{max-width:1150px}}table{{border-collapse:collapse;width:100%;font-size:14px}}
th,td{{padding:9px 12px;border-bottom:1px solid #ddd;text-align:right;vertical-align:top}}th:first-child,td:first-child{{text-align:left}}
.table{{overflow-x:auto}}img{{display:block;width:100%;height:auto}}small{{color:#666}}a{{color:#24577d}}</style>
<h1>6秒与30秒训练回合对照</h1>
<p>{escape(summary['interpretation'])}</p>
<p>两个训练回合上限、三个配对训练种子，每次目标 {summary['matrix']['steps_per_run']:,} 环境步；另保留未训练BC。6秒组三个结果复用上一轮匹配的原奖励训练，并非本轮重新训练。{escape(evidence)}</p>
<p>本实验的计划唯一变量是Brax外层训练截断长度600→3000步；底层任务时序合同与原奖励保持，俯仰/位移新增项均不启用。策略100 Hz、物理400 Hz保持不变；本次没有改变内层反馈或策略频率，不能据此证明或排除控制频率瓶颈。</p>
<p>两组模型统一在原任务中做最长30秒首回合评价。累计合格时间剔除起始1秒，要求俯仰误差≤10°、真实机身XY偏移≤5厘米、平面速度≤0.1米/秒同时满足。每seed先平均标称与±0.01弧度三个主要初态，再对三个训练seed显示中位数[min,max]；这些是预先声明的描述性带宽。</p>
<div class="table"><table><thead><tr><th>组别</th><th>累计合格秒</th><th>最长连续合格秒</th><th>存活秒</th></tr></thead><tbody>{''.join(result_rows)}</tbody></table></div>
<p>30秒相对6秒的同seed配对中位差：累计合格 {_display(summary['paired_delta_in_band_seconds'], True)} 秒，最长连续合格 {_display(summary['paired_delta_longest_in_band_seconds'], True)} 秒。配对差与两组各自中位数之差不必相同；应优先阅读配对结果。存活增加但合格时间不增加，不应称为原地平衡改善；只胜过6秒组但未超过BC时，应称减缓退化。</p>
<h2>逐种子配对结果</h2><div class="table"><table><thead><tr><th>训练seed</th><th>6秒累计合格秒</th><th>30秒累计合格秒</th><th>累计配对差秒</th><th>最长段配对差秒</th><th>存活配对差秒</th><th>预留初态累计差秒</th></tr></thead><tbody>{''.join(pair_rows)}</tbody></table></div>
<p><small>训练seed是重复单位，五个固定初态不是额外独立训练重复。±0.02弧度是预留固定工况，不是随机鲁棒性验证。提前终止后不累积合格时间、不补造轨迹；早倒造成的小位移不能当作更好位置保持。达到30秒仅表示截断，不自动证明稳定或收敛。三种子短训不支持统计显著、长期收敛或实机结论。</small></p>
<h2>早期前10秒</h2><p>只缩放视窗，不截断原始数据或摘要；10秒后的终止点见完整图。</p><img src="comparison-detail.png" alt="三个训练seed的前10秒实际位移、俯仰和速度对照">
<h2>共同30秒评价时域</h2><img src="comparison.png" alt="共同30秒评价时域，首回合终止后留空">
<p><a href="results.json">完整轨迹JSON</a> · <a href="paired-summary.json">配对汇总JSON</a> · <a href="summary.csv">逐工况CSV</a></p>
<details><summary>策略产物标签</summary><ul>{labels}</ul></details></html>'''
    paths['report_html'].write_text(content+'\n', encoding='utf-8')
    return {key: str(path) for key, path in paths.items()}
