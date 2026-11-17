"""Paired, fixed-budget reward-ablation reports from already saved trajectories.

The training seed is the repeat unit. Fixed evaluation states remain conditions
within a seed; their averages are never presented as independent repetitions.
No policy, service, or training framework is loaded by the report.
"""
from collections.abc import Mapping
import csv
from html import escape
import json
import math
from pathlib import Path
import re
import statistics


ARMS = ('baseline', 'posture', 'position', 'both')
SEEDS = (0, 1, 2)
PRIMARY_STATES = ('nominal', 'pitch-plus', 'pitch-minus')
HELDOUT_STATES = ('heldout-plus', 'heldout-minus')
STATES = PRIMARY_STATES + HELDOUT_STATES
LABELS = {'initial': 'BC初始化（未继续训练）', 'baseline': '原奖励',
          'posture': '只加俯仰', 'position': '只加真实位移', 'both': '两项同时加入'}


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(name + ' must be a finite number')
    return float(value)


def _identity(policy_id):
    if policy_id == 'initial':
        return 'initial', None
    match = re.fullmatch(r'(baseline|posture|position|both)-seed-([012])', str(policy_id))
    if match is None:
        raise ValueError('Unexpected policy identity: ' + str(policy_id))
    return match[1], int(match[2])


def _stats(values):
    values = list(values)
    return {'median': statistics.median(values), 'min': min(values), 'max': max(values)}


def _mean(values):
    values = list(values)
    return math.fsum(values) / len(values)


def _matrix_metadata(bundle):
    manifest = bundle.get('experiment_manifest') or {}
    if not isinstance(manifest, Mapping):
        raise ValueError('experiment_manifest must be a mapping')
    seeds = manifest.get('seeds', list(SEEDS))
    if sorted(seeds) != list(SEEDS) or any(type(seed) is not int for seed in seeds):
        raise ValueError('Expected three paired training seeds 0, 1, 2')
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
                    raise ValueError('run_statuses needs run mappings')
                identity = status.get('policy_id', f"{status.get('arm')}-seed-{status.get('seed')}")
                if identity in records:
                    raise ValueError('Duplicate run status: ' + identity)
                records[identity] = status
        else:
            raise ValueError('run_statuses must be a list or policy-id mapping')
        expected = {f'{arm}-seed-{seed}' for arm in ARMS for seed in SEEDS}
        if set(records) != expected:
            raise ValueError('Run statuses must contain the complete 4-by-3 matrix')
        for identity, status in records.items():
            if not isinstance(status, Mapping) or status.get('phase') != 'completed' or status.get('steps') != steps:
                raise ValueError('Incomplete or unequal-step run: ' + identity)
            arm, seed = _identity(identity)
            if status.get('arm', arm) != arm or status.get('seed', seed) != seed:
                raise ValueError('Run status identity mismatch: ' + identity)
        initial_hashes = {status['initial_array_digest'] for status in records.values()
                          if status.get('initial_array_digest') is not None}
        if len(initial_hashes) > 1:
            raise ValueError('Training runs did not use the same complete initializer')
    return {'seeds': list(SEEDS), 'runs': 12, 'steps_per_run': steps,
            'statuses_verified': statuses is not None}


def _series(trace):
    times = [_number(value, 't_s') for value in trace['t_s']]
    if len(times) < 2 or times[0] != 0 or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError('Trace must include time zero and increasing post-step times')
    count = len(times)
    done = trace['done']
    if len(done) != count or any(type(value) is not bool for value in done) or done[0] or any(done[:-1]) or not done[-1]:
        raise ValueError('Trace must stop exactly at its first done')
    reference = _number(trace['metadata']['reference_pitch_rad'], 'reference_pitch_rad')
    position, velocity = trace['base_position_m'], trace['base_velocity_m_s']
    if len(position) != count or len(velocity) != count or len(trace['pitch_rad']) != count:
        raise ValueError('Trace series dimensions differ')
    for rows in (position, velocity):
        for row in rows:
            if len(row) != 3:
                raise ValueError('Body vectors need three SI coordinates')
            for value in row:
                _number(value, 'body coordinate')
    origin = position[0]
    displacement = [math.hypot(row[0]-origin[0], row[1]-origin[1]) for row in position]
    speed = [math.hypot(row[0], row[1]) for row in velocity]
    pitch = [math.degrees(_number(value, 'pitch_rad')-reference) for value in trace['pitch_rad']]
    termination = trace['termination']
    flags = [termination.get(name) for name in ('terminated', 'truncated')]
    if any(type(value) is not bool for value in flags) or sum(flags) != 1:
        raise ValueError('Completed traces must distinguish termination and time-limit truncation')
    return {'time': times, 'displacement': displacement, 'speed': speed, 'pitch_error': pitch,
            'terminated': flags[0]}


def _case_row(case):
    from experiments.balance_diagnostics.metrics import analyze_trace
    arm, seed = _identity(case['policy_id'])
    series = _series(case['trace'])
    analysis = case.get('analysis')
    if analysis is None:
        analysis = analyze_trace(case['trace'])
    standing = case.get('standing')
    if standing is None:
        from .evaluation import standing_metrics
        standing = standing_metrics(case['trace'])
    full = analysis['windows']['full_episode']
    duration = _number(standing['survival_seconds'], 'survival_seconds')
    if not math.isclose(duration, series['time'][-1], abs_tol=1e-9, rel_tol=0):
        raise ValueError('Standing measurements differ from trace duration')
    if not math.isclose(_number(full['observed_seconds'], 'observed_seconds'), duration, abs_tol=1e-9, rel_tol=0):
        raise ValueError('Window measurements differ from trace duration')
    in_band = _number(standing['in_band_seconds_after_settling'], 'in_band_seconds_after_settling')
    longest = _number(standing['longest_in_band_seconds_after_settling'], 'longest_in_band_seconds_after_settling')
    if not 0 <= longest <= in_band + 1e-9 or in_band > max(0, duration-1) + 1e-9:
        raise ValueError('Invalid post-settling standing durations')
    if type(standing['reached_horizon']) is not bool:
        raise ValueError('reached_horizon must be boolean')
    termination = case['trace']['termination']
    row = {'policy_id': case['policy_id'], 'policy_label': str(case.get('label', LABELS[arm])),
        'arm': arm, 'seed': seed, 'initial_state_id': case['initial_state_id'],
        'in_band_seconds_after_settling': in_band,
        'longest_in_band_seconds_after_settling': longest, 'survival_seconds': duration,
        'reached_horizon': standing['reached_horizon'],
        'max_displacement_m': _number(full['max_horizontal_displacement_m'], 'max displacement'),
        'pitch_error_rms_deg': _number(full['pitch_error_rms_deg'], 'pitch error RMS'),
        'horizontal_speed_rms_m_s': _number(full['horizontal_speed_rms_m_s'], 'horizontal speed RMS'),
        'termination_reason': str(termination['reason'])}
    return row, series


def paired_summary(bundle):
    """Validate the complete matrix and return seed-level descriptive summaries."""
    if not isinstance(bundle, Mapping) or bundle.get('schema_version') != 1 or bundle.get('kind') != 'balance_diagnostics':
        raise ValueError('Expected balance_diagnostics version 1 results')
    if _number(bundle.get('horizon_s'), 'horizon_s') != 30.:
        raise ValueError('This reward-ablation comparison uses a fixed 30-second evaluation')
    matrix = _matrix_metadata(bundle)
    expected = {'initial'} | {f'{arm}-seed-{seed}' for arm in ARMS for seed in SEEDS}
    indexed, rows, series = {}, [], {}
    for case in bundle.get('cases', []):
        identity, state = case['policy_id'], case['initial_state_id']
        if identity not in expected or state not in STATES or (identity, state) in indexed:
            raise ValueError('Unexpected or duplicate policy/initial-state case')
        row, values = _case_row(case)
        indexed[identity, state] = row
        rows.append(row)
        series[identity, state] = values
    if set(indexed) != {(policy, state) for policy in expected for state in STATES}:
        raise ValueError('Comparison needs BC and all 12 runs on the same five initial states')

    def group(identity, states):
        selected = [indexed[identity, state] for state in states]
        return {'mean_in_band_seconds': _mean(row['in_band_seconds_after_settling'] for row in selected),
                'mean_longest_in_band_seconds': _mean(row['longest_in_band_seconds_after_settling'] for row in selected),
                'mean_survival_seconds': _mean(row['survival_seconds'] for row in selected),
                'mean_max_displacement_m': _mean(row['max_displacement_m'] for row in selected),
                'mean_pitch_error_rms_deg': _mean(row['pitch_error_rms_deg'] for row in selected),
                'reached_horizon_count': sum(row['reached_horizon'] for row in selected),
                'evaluation_conditions': len(selected)}

    initial = {'primary': group('initial', PRIMARY_STATES), 'heldout': group('initial', HELDOUT_STATES)}
    seed_results = {}
    for arm in ARMS:
        seed_results[arm] = [dict(seed=seed, primary=group(f'{arm}-seed-{seed}', PRIMARY_STATES),
                                  heldout=group(f'{arm}-seed-{seed}', HELDOUT_STATES)) for seed in SEEDS]
    arms = {}
    for arm in ARMS:
        pairs = []
        for item, baseline in zip(seed_results[arm], seed_results['baseline']):
            pairs.append(dict(seed=item['seed'],
                delta_in_band_seconds=item['primary']['mean_in_band_seconds']-baseline['primary']['mean_in_band_seconds'],
                delta_survival_seconds=item['primary']['mean_survival_seconds']-baseline['primary']['mean_survival_seconds'],
                delta_to_initial_in_band_seconds=item['primary']['mean_in_band_seconds']-initial['primary']['mean_in_band_seconds']))
        median_delta = statistics.median(pair['delta_in_band_seconds'] for pair in pairs)
        median_initial_delta = statistics.median(pair['delta_to_initial_in_band_seconds'] for pair in pairs)
        if arm == 'baseline':
            interpretation = '同预算原奖励参照；同时与未继续训练的BC比较，以识别PPO退化。'
        elif median_delta > 0 and median_initial_delta <= 0:
            interpretation = '本轮有效站立时间较同预算原奖励组增加，但尚未超过BC初始化；应解读为减缓退化。'
        elif median_delta > 0 and median_initial_delta > 0:
            interpretation = '本轮有效站立时间的配对描述同时高于原奖励组与BC初始化；仍需更多种子和训练预算验证。'
        elif median_delta == 0:
            interpretation = '本轮有效站立时间的配对中位差为零，未显示一致改善。'
        else:
            interpretation = '本轮有效站立时间的配对中位差为负，未优于同预算原奖励组。'
        arms[arm] = {'label': LABELS[arm], 'seed_results': seed_results[arm], 'paired_results': pairs,
            'primary_in_band_seconds': _stats(item['primary']['mean_in_band_seconds'] for item in seed_results[arm]),
            'primary_longest_in_band_seconds': _stats(item['primary']['mean_longest_in_band_seconds'] for item in seed_results[arm]),
            'primary_survival_seconds': _stats(item['primary']['mean_survival_seconds'] for item in seed_results[arm]),
            'heldout_in_band_seconds': _stats(item['heldout']['mean_in_band_seconds'] for item in seed_results[arm]),
            'paired_delta_in_band_seconds': _stats(pair['delta_in_band_seconds'] for pair in pairs),
            'paired_improved_seed_count': sum(pair['delta_in_band_seconds'] > 0 for pair in pairs),
            'interpretation': interpretation}
    summary = {'schema_version': 1, 'kind': 'paired_reward_ablation_summary', 'matrix': matrix,
        'primary_states': list(PRIMARY_STATES), 'heldout_states': list(HELDOUT_STATES),
        'initial': initial, 'arms': arms,
        'primary_measure': 'Mean observed in-band time after 1-second settling, averaged within each training seed over the three primary fixed conditions.',
        'scope': 'Three paired training seeds only; fixed evaluation conditions are not independent training repetitions. Descriptive medians/ranges, no significance claim.',
        'limitations': ['Earlier death cannot be ranked better because its displacement is smaller.',
                        'Training rewards and full-episode reward rates are not cross-arm ranking scores.',
                        'Reaching 30 seconds means censoring, not proven convergence.',
                        'Fixed held-out pitch conditions do not prove random or real-world robustness.']}
    return summary, rows, series


def _plot(summary, series, destination, *, horizon_seconds):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    colors = {'initial': '#777777', 'baseline': '#0072B2', 'posture': '#D55E00',
              'position': '#009E73', 'both': '#CC79A7'}
    figure, axes = plt.subplots(4, 3, figsize=(15, 13), squeeze=False)
    measures = (('mean_in_band_seconds', 'primary_in_band_seconds', 'In-band time after settling (s)'),
                ('mean_longest_in_band_seconds', 'primary_longest_in_band_seconds', 'Longest in-band interval (s)'),
                ('mean_survival_seconds', 'primary_survival_seconds', 'First-episode survival (s)'))
    for column, (raw_key, stats_key, title) in enumerate(measures):
        panel = axes[0, column]
        for x, arm in enumerate(ARMS):
            values = [item['primary'][raw_key] for item in summary['arms'][arm]['seed_results']]
            data = summary['arms'][arm][stats_key]
            panel.vlines(x, data['min'], data['max'], color=colors[arm], alpha=.6, linewidth=2)
            panel.plot(x, data['median'], '_', color=colors[arm], markersize=17, markeredgewidth=2)
            for offset, marker, value in zip((-.1, 0, .1), ('o', '^', 's'), values):
                panel.plot(x+offset, value, marker, color=colors[arm], markersize=5)
        panel.axhline(summary['initial']['primary'][raw_key], color=colors['initial'], linestyle='--', label='BC initial')
        panel.set_xticks(range(4), ARMS, rotation=12)
        panel.set_title(title, fontsize=11)
        panel.set_ylabel('Seconds')
        panel.set_ylim(bottom=0)
        panel.legend(fontsize=8)
    for row, seed in enumerate(SEEDS, 1):
        panels = axes[row]
        for arm in ('initial',)+ARMS:
            identity = 'initial' if arm == 'initial' else f'{arm}-seed-{seed}'
            values = series[identity, 'nominal']
            time = values['time']
            for column, key in enumerate(('displacement', 'pitch_error', 'speed')):
                panels[column].plot(time, values[key], label=arm, color=colors[arm],
                                    linewidth=1.3, linestyle='--' if arm == 'initial' else '-')
                panels[column].plot(time[-1], values[key][-1], 'x' if values['terminated'] else '|',
                                    color=colors[arm], markersize=5)
        for column, title in enumerate(('Body displacement (m)', 'Pitch error (deg)', 'Horizontal body speed (m/s)')):
            panel = panels[column]
            panel.set_title(f'Seed {seed}: nominal | {title}', fontsize=10)
            panel.set_xlabel('Time (s)')
            panel.set_xlim(0, horizon_seconds)
            panel.axvline(6, color='.6', linestyle=':', linewidth=1)
            if column == 1:
                for limit in (-10, 10):
                    panel.axhline(limit, color='.8', linestyle=':', linewidth=.8)
            else:
                panel.axhline(.05 if column == 0 else .1, color='.8', linestyle=':', linewidth=.8)
            if row == 1 and column == 0:
                panel.legend(fontsize=8)
    for panel in axes.flat:
        panel.grid(alpha=.18)
        panel.tick_params(labelsize=8)
        panel.spines[['top', 'right']].set_visible(False)
    view_label = f'early first {horizon_seconds:g} s' if horizon_seconds < 30 else 'full 30 s horizon'
    figure.suptitle('Matched reward ablation | '+view_label+' | dots: training seeds | dash: untrained BC', fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, .97))
    try:
        figure.savefig(destination, dpi=160, bbox_inches='tight', facecolor='white')
    finally:
        plt.close(figure)


def _display_stats(values, *, signed=False):
    prefix = '+' if signed else ''
    return f"{values['median']:{prefix}.2f} [{values['min']:{prefix}.2f}, {values['max']:{prefix}.2f}]"


def write_ablation_report(bundle, output_dir):
    """Write standalone views of a complete matrix, without running any models."""
    summary, rows, series = paired_summary(bundle)
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    paths = {key: destination/name for key, name in (
        ('comparison_png', 'comparison.png'), ('comparison_detail_png', 'comparison-detail.png'),
        ('report_html', 'report.html'), ('summary_csv', 'summary.csv'), ('paired_summary_json', 'paired-summary.json'))}
    _plot(summary, series, paths['comparison_png'], horizon_seconds=30.)
    _plot(summary, series, paths['comparison_detail_png'], horizon_seconds=min(10., float(bundle['horizon_s'])))
    paths['paired_summary_json'].write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    with paths['summary_csv'].open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    table_rows = []
    initial = summary['initial']['primary']
    table_rows.append('<tr><td>BC初始化（未训练参考）</td><td>'+f"{initial['mean_in_band_seconds']:.2f}"+
        '</td><td>—</td><td>—</td><td>'+f"{initial['mean_survival_seconds']:.2f}"+'</td><td>固定参考，不作为额外训练seed。</td></tr>')
    for arm in ARMS:
        result = summary['arms'][arm]
        cells = [result['label'], _display_stats(result['primary_in_band_seconds']),
                 _display_stats(result['paired_delta_in_band_seconds'], signed=True),
                 f"{result['paired_improved_seed_count']}/3", _display_stats(result['primary_survival_seconds']),
                 result['interpretation']]
        table_rows.append('<tr>'+''.join('<td>'+escape(cell)+'</td>' for cell in cells)+'</tr>')
    detail_rows = []
    for arm in ARMS:
        for result, pair in zip(summary['arms'][arm]['seed_results'], summary['arms'][arm]['paired_results']):
            cells = [LABELS[arm], str(result['seed']), f"{result['primary']['mean_in_band_seconds']:.3f}",
                     f"{pair['delta_in_band_seconds']:+.3f}", f"{result['primary']['mean_survival_seconds']:.3f}",
                     f"{result['heldout']['mean_in_band_seconds']:.3f}"]
            detail_rows.append('<tr>'+''.join('<td>'+escape(cell)+'</td>' for cell in cells)+'</tr>')
    notes = ''
    for arm in ARMS:
        labels = sorted({row['policy_label'] for row in rows if row['arm'] == arm})
        notes += '<li>'+escape(LABELS[arm])+': '+escape('；'.join(labels))+'</li>'
    steps = summary['matrix']['steps_per_run']
    evidence = '已核对12次训练状态均完成且步数相同。' if summary['matrix']['statuses_verified'] else '已核对全部策略和初态齐全；未提供训练状态清单，需结合训练产物核对实际完成步数。'
    content = f'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>姿态与真实位移奖励独立对照</title>
<style>body{{font:16px/1.65 system-ui,sans-serif;color:#222;background:#fff;margin:32px auto;max-width:1500px;padding:0 24px}}
h1{{font-size:27px}}h2{{font-size:21px;margin-top:30px}}p{{max-width:1100px}}table{{border-collapse:collapse;width:100%;font-size:14px}}
th,td{{padding:9px 12px;border-bottom:1px solid #ddd;text-align:right;vertical-align:top}}th:first-child,td:first-child,td:last-child{{text-align:left}}
.table{{overflow-x:auto}}img{{display:block;width:100%;height:auto}}small{{color:#666}}a{{color:#24577d}}</style>
<h1>姿态与真实位移奖励独立对照</h1>
<p>四组 × 三个配对训练种子，每次目标 {steps:,} 环境步；另保留未继续训练的BC初始化。{escape(evidence)}</p>
<p>主指标是开始1秒后，同时满足俯仰误差≤10°、实际机身XY偏移≤5厘米、平面速度≤0.1米/秒的有效站立时间。每个训练种子先平均标称及±0.01弧度三个固定初态，再对三个训练种子显示中位数与[min,max]。这些带宽是预先声明的描述性范围，不是已标定的成功标准。</p>
<div class="table"><table><thead><tr><th>组别</th><th>有效站立秒</th><th>相对原奖励配对差（秒）</th><th>改善种子</th><th>存活秒</th><th>描述性判断</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table></div>
<p>训练seed是重复单位，固定五个初态不构成十五个独立训练样本。只胜过退化的原奖励组、未超过BC时，结论是减缓退化。提前失败会限制位移大小，不能因此认定位置保持更好；不同训练奖励总分和整回合奖励/秒均不作跨组排名。</p>
<h2>配对种子结果</h2><div class="table"><table><thead><tr><th>组别</th><th>训练seed</th><th>主初态有效秒</th><th>配对差秒</th><th>主初态存活秒</th><th>预留初态有效秒</th></tr></thead><tbody>{''.join(detail_rows)}</tbody></table></div>
<p><small>±0.02弧度是额外固定工况，不能证明随机初态或实机鲁棒性。30秒存活表示达到截断上限，不能自动认定收敛。有效站立只累积首回合实际观测区间，死亡后不再累积，且没有补造终止后的状态。一次权重设置和三种子短训不能排除其他系数或更长训练的结果。</small></p>
<h2>早期前10秒（early first 10 s）</h2><p>细节图固定显示0～10秒，以便比较起步和早期失稳；原始轨迹与摘要仍使用完整首回合，10秒后的终止点可在下方完整图查看。</p><img src="comparison-detail.png" alt="早期前10秒的三种子标称初态位移、俯仰、速度曲线">
<h2>完整30秒上限</h2><img src="comparison.png" alt="完整评估时域，首回合终止后曲线留空">
<p><a href="results.json">完整轨迹与诊断JSON</a> · <a href="summary.csv">各策略各初态CSV</a> · <a href="paired-summary.json">配对汇总JSON</a></p>
<details><summary>策略产物标签</summary><ul>{notes}</ul></details>
</html>'''
    paths['report_html'].write_text(content+'\n', encoding='utf-8')
    return {key: str(path) for key, path in paths.items()}
