"""Offline paired actuator-capacity reports, with each group's zero-step reference."""
from collections.abc import Mapping
import csv
import hashlib
from html import escape
import json
import math
from pathlib import Path
import re

from experiments.reward_ablation.report import _mean, _number, _series, _stats


CONDITIONS = ('limit1', 'limit2')
SEEDS = (0, 1, 2)
PRIMARY_STATES = ('nominal', 'pitch-plus', 'pitch-minus')
HELDOUT_STATES = ('heldout-plus', 'heldout-minus')
STATES = PRIMARY_STATES + HELDOUT_STATES
NORMALIZATION_SCALES = [1., 1., .0295, .0295]
LABELS = {'limit1': '±1 N·m基线（复用）', 'limit2': '±2 N·m（新训练）',
          'initial-limit1': '±1 N·m的0步BC参考', 'initial-limit2': '±2 N·m的0步BC参考'}


def _identity(identity):
    if identity in ('initial-limit1', 'initial-limit2'):
        return identity.removeprefix('initial-'), None
    match = re.fullmatch(r'(limit[12])-seed-([012])', str(identity))
    if match is None:
        raise ValueError('Unexpected torque-limit policy identity: ' + str(identity))
    return match[1], int(match[2])


def _matrix_metadata(bundle):
    manifest = bundle.get('experiment_manifest')
    if not isinstance(manifest, Mapping):
        raise ValueError('Need the frozen experiment_manifest')
    seeds = manifest.get('seeds')
    if not isinstance(seeds, list) or any(type(seed) is not int for seed in seeds) or sorted(seeds) != list(SEEDS):
        raise ValueError('Expected paired training seeds 0, 1, 2')
    steps = manifest.get('steps_per_run')
    if type(steps) is not int or steps != 2097152:
        raise ValueError('This comparison requires 2097152 steps per run')
    if manifest.get('training_episode_seconds', 30) != 30:
        raise ValueError('Training episode duration must stay 30 seconds')
    if manifest.get('torque_normalization_scales', NORMALIZATION_SCALES) != NORMALIZATION_SCALES:
        raise ValueError('Reward and observation torque normalization must remain fixed')
    statuses = bundle.get('run_statuses')
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
        raise ValueError('Need complete run_statuses for the two-by-three matrix')
    expected = {f'{condition}-seed-{seed}' for condition in CONDITIONS for seed in SEEDS}
    if set(records) != expected:
        raise ValueError('Need all six completed runs')
    hashes = set()
    for identity, status in records.items():
        condition, seed = _identity(identity)
        if (status.get('phase') != 'completed' or status.get('steps') != steps or
                status.get('condition') != condition or status.get('seed') != seed):
            raise ValueError('Incomplete, unequal-budget, or mismatched run: ' + identity)
        if type(status.get('reused')) is not bool or status['reused'] != (condition == 'limit1'):
            raise ValueError('Only the matched ±1 comparator is reused')
        digest = status.get('initial_array_digest')
        if not isinstance(digest, str) or not digest:
            raise ValueError('Each run needs its full initializer identity')
        hashes.add(digest)
    if len(hashes) != 1:
        raise ValueError('Runs did not restore the same complete initializer')
    states = bundle.get('initial_states')
    if (not isinstance(states, list) or len(states) != 5 or
            {item.get('id') for item in states if isinstance(item, Mapping)} != set(STATES)):
        raise ValueError('Need the same five explicit initial states')
    digest = hashlib.sha256(json.dumps(states, sort_keys=True, allow_nan=False,
                                     separators=(',', ':')).encode()).hexdigest()
    if digest != bundle.get('initial_states_sha256'):
        raise ValueError('Physical initial-state fingerprint differs from the saved states')
    return dict(seeds=list(SEEDS), runs=6, steps_per_run=steps, statuses_verified=True,
        initial_states_sha256=digest, reused_condition='limit1', new_condition='limit2',
        training_episode_seconds=30, policy_hz=100, physics_hz=400,
        torque_normalization_scales=list(NORMALIZATION_SCALES),
        sole_planned_variable='Leg motor limits ±1 to ±2 N m, with force scales recalibrated by the same formula.',
        initializer_note='Identical actor/critic/normalizer arrays do not imply identical initial closed-loop behavior at a wider force range.')


def _case_row(case):
    from experiments.balance_diagnostics.metrics import analyze_trace
    from experiments.reward_ablation.evaluation import standing_metrics
    condition, seed = _identity(case['policy_id'])
    trace = case['trace']
    values = _series(trace)
    analysis = case.get('analysis') or analyze_trace(trace)
    standing = case.get('standing') or standing_metrics(trace)
    full = analysis['windows']['full_episode']
    duration = _number(standing['survival_seconds'], 'survival_seconds')
    if not 0 < duration <= 30 or not math.isclose(duration, values['time'][-1], rel_tol=0, abs_tol=1e-9):
        raise ValueError('First-episode duration differs from its trace')
    if not math.isclose(_number(full['observed_seconds'], 'observed_seconds'), duration, rel_tol=0, abs_tol=1e-9):
        raise ValueError('Analysis duration differs from the trace')
    total = _number(standing['in_band_seconds_after_settling'], 'in-band seconds')
    longest = _number(standing['longest_in_band_seconds_after_settling'], 'longest interval')
    if not 0 <= longest <= total+1e-9 or total > max(0., duration-1.)+1e-9:
        raise ValueError('Invalid cumulative or continuous in-band duration')
    reached = standing['reached_horizon']
    expected = math.isclose(duration, 30., rel_tol=0, abs_tol=1e-9) and trace['termination']['truncated']
    if type(reached) is not bool or reached != expected:
        raise ValueError('A task failure must not be called horizon success')
    row = dict(policy_id=case['policy_id'], policy_label=str(case.get('label', LABELS.get(case['policy_id'], LABELS[condition]))),
        condition=condition, seed=seed, initial_state_id=case['initial_state_id'],
        in_band_seconds_after_settling=total, longest_in_band_seconds_after_settling=longest,
        survival_seconds=duration, reached_horizon=reached,
        max_displacement_m=_number(full['max_horizontal_displacement_m'], 'max displacement'),
        pitch_error_rms_deg=_number(full['pitch_error_rms_deg'], 'pitch RMS'),
        horizontal_speed_rms_m_s=_number(full['horizontal_speed_rms_m_s'], 'speed RMS'),
        termination_reason=str(trace['termination']['reason']))
    if any(row[name] < 0 for name in ('max_displacement_m', 'pitch_error_rms_deg', 'horizontal_speed_rms_m_s')):
        raise ValueError('Physical magnitudes must be nonnegative')
    return row, values


def paired_summary(bundle):
    """Pair final policies by training seed and retain two distinct zero-step outcomes."""
    if not isinstance(bundle, Mapping) or bundle.get('schema_version') != 1 or bundle.get('kind') != 'balance_diagnostics':
        raise ValueError('Expected balance_diagnostics version 1')
    if _number(bundle.get('horizon_s'), 'horizon_s') != 30.:
        raise ValueError('Both limits require the same 30-second evaluation')
    matrix = _matrix_metadata(bundle)
    policies = {f'initial-{condition}' for condition in CONDITIONS} | {
        f'{condition}-seed-{seed}' for condition in CONDITIONS for seed in SEEDS}
    indexed, rows, series = {}, [], {}
    for case in bundle.get('cases', []):
        identity, state = case['policy_id'], case['initial_state_id']
        if identity not in policies or state not in STATES or (identity, state) in indexed:
            raise ValueError('Unexpected or duplicate policy/initial-state case')
        if case['trace'].get('metadata', {}).get('initial_states_sha256') != matrix['initial_states_sha256']:
            raise ValueError('Case was evaluated with different physical initial states')
        row, values = _case_row(case)
        indexed[identity, state] = row
        rows.append(row)
        series[identity, state] = values
    if set(indexed) != {(policy, state) for policy in policies for state in STATES}:
        raise ValueError('Need both zero-step references and all six final policies on five states')

    def group(identity, states):
        selected = [indexed[identity, state] for state in states]
        return dict(mean_in_band_seconds=_mean(row['in_band_seconds_after_settling'] for row in selected),
            mean_longest_in_band_seconds=_mean(row['longest_in_band_seconds_after_settling'] for row in selected),
            mean_survival_seconds=_mean(row['survival_seconds'] for row in selected),
            mean_max_displacement_m=_mean(row['max_displacement_m'] for row in selected),
            mean_pitch_error_rms_deg=_mean(row['pitch_error_rms_deg'] for row in selected),
            reached_horizon_count=sum(row['reached_horizon'] for row in selected), evaluation_conditions=len(selected))

    initial = {condition: dict(policy_id=f'initial-{condition}', primary=group(f'initial-{condition}', PRIMARY_STATES),
        heldout=group(f'initial-{condition}', HELDOUT_STATES)) for condition in CONDITIONS}
    conditions = {}
    for condition in CONDITIONS:
        seeds = []
        for seed in SEEDS:
            primary, heldout = group(f'{condition}-seed-{seed}', PRIMARY_STATES), group(f'{condition}-seed-{seed}', HELDOUT_STATES)
            gain = {name: primary[name]-initial[condition]['primary'][name] for name in (
                'mean_in_band_seconds', 'mean_longest_in_band_seconds', 'mean_survival_seconds')}
            seeds.append(dict(seed=seed, primary=primary, heldout=heldout, delta_from_initial=gain))
        conditions[condition] = dict(label=LABELS[condition], leg_limit_nm=1 if condition == 'limit1' else 2,
            reused=condition == 'limit1', seed_results=seeds,
            primary_in_band_seconds=_stats(item['primary']['mean_in_band_seconds'] for item in seeds),
            primary_longest_in_band_seconds=_stats(item['primary']['mean_longest_in_band_seconds'] for item in seeds),
            primary_survival_seconds=_stats(item['primary']['mean_survival_seconds'] for item in seeds),
            heldout_in_band_seconds=_stats(item['heldout']['mean_in_band_seconds'] for item in seeds),
            delta_from_initial_in_band_seconds=_stats(item['delta_from_initial']['mean_in_band_seconds'] for item in seeds))
    pairs = []
    for one, two in zip(conditions['limit1']['seed_results'], conditions['limit2']['seed_results']):
        gain1, gain2 = one['delta_from_initial']['mean_in_band_seconds'], two['delta_from_initial']['mean_in_band_seconds']
        pairs.append(dict(seed=one['seed'],
            delta_in_band_seconds=two['primary']['mean_in_band_seconds']-one['primary']['mean_in_band_seconds'],
            delta_longest_in_band_seconds=two['primary']['mean_longest_in_band_seconds']-one['primary']['mean_longest_in_band_seconds'],
            delta_survival_seconds=two['primary']['mean_survival_seconds']-one['primary']['mean_survival_seconds'],
            limit1_delta_from_initial_in_band_seconds=gain1, limit2_delta_from_initial_in_band_seconds=gain2,
            delta_training_gain_in_band_seconds=gain2-gain1,
            delta_heldout_in_band_seconds=two['heldout']['mean_in_band_seconds']-one['heldout']['mean_in_band_seconds']))
    delta = _stats(pair['delta_in_band_seconds'] for pair in pairs)
    direction = ('为零，本轮没有显示最终累计合格时间增加' if math.isclose(delta['median'], 0., abs_tol=1e-9)
                 else '为正，2 N·m组最终累计合格时间在配对描述中增加' if delta['median'] > 0
                 else '为负，本轮未显示2 N·m组最终累计合格时间的优势')
    summary = dict(schema_version=1, kind='paired_torque_limit_summary', matrix=matrix, initial=initial,
        primary_states=list(PRIMARY_STATES), heldout_states=list(HELDOUT_STATES), conditions=conditions,
        paired_results=pairs, paired_delta_in_band_seconds=delta,
        paired_delta_longest_in_band_seconds=_stats(pair['delta_longest_in_band_seconds'] for pair in pairs),
        paired_delta_survival_seconds=_stats(pair['delta_survival_seconds'] for pair in pairs),
        paired_delta_training_gain_in_band_seconds=_stats(pair['delta_training_gain_in_band_seconds'] for pair in pairs),
        improved_seed_counts={name: sum(pair[key] > 0 for pair in pairs) for name, key in (
            ('cumulative', 'delta_in_band_seconds'), ('longest', 'delta_longest_in_band_seconds'),
            ('survival', 'delta_survival_seconds'), ('own_initial_gain', 'delta_training_gain_in_band_seconds'))},
        interpretation='最终累计合格时间的配对中位差'+direction+'。应同时阅读各自0步参考；相同权重在扩大腿力范围后不代表相同初始闭环行为。',
        scope='Three paired training seeds; five fixed initial states are conditions within a seed, not independent training repetitions.',
        limitations=['Same parameter initialization is not the same physical zero-step policy after action scaling changes.',
            'Differences between gains from each own zero-step reference are descriptive, not a causal decomposition.',
            'The ±1 comparator reuses matched completed runs; it was not retrained here.',
            'Motor capacity and the force envelope change together by the declared calibration formula.',
            'Reward and previous-torque observation normalization remain at the original physical scales.',
            'Survival, cumulative in-band time and longest continuous interval are distinct outcomes.',
            'Three fixed seeds and fixed evaluation states do not establish significance, convergence or hardware robustness.'])
    return summary, rows, series


def _plot(summary, series, destination, *, horizon_seconds):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'limit1': '#0072B2', 'limit2': '#D55E00'}
    figure, axes = plt.subplots(4, 3, figsize=(15, 13), squeeze=False)
    measures = (('mean_in_band_seconds', 'primary_in_band_seconds', 'Cumulative in-band time (s)'),
                ('mean_longest_in_band_seconds', 'primary_longest_in_band_seconds', 'Longest in-band interval (s)'),
                ('mean_survival_seconds', 'primary_survival_seconds', 'First-episode survival (s)'))
    for column, (raw, stats, title) in enumerate(measures):
        panel = axes[0, column]
        for x, condition in enumerate(CONDITIONS):
            result = summary['conditions'][condition]
            distribution = result[stats]
            panel.vlines(x, distribution['min'], distribution['max'], color=colors[condition], alpha=.6, linewidth=2)
            panel.plot(x, distribution['median'], '_', color=colors[condition], markersize=18, markeredgewidth=2)
            for offset, marker, item in zip((-.09, 0, .09), ('o', '^', 's'), result['seed_results']):
                panel.plot(x+offset, item['primary'][raw], marker, color=colors[condition], markersize=5)
            panel.axhline(summary['initial'][condition]['primary'][raw], color=colors[condition],
                          linestyle='--', alpha=.6, label=condition+' zero-step BC')
        panel.set_xticks(range(2), ['1 N m (reused)', '2 N m'])
        panel.set_title(title, fontsize=11); panel.set_ylabel('Seconds'); panel.set_ylim(bottom=0)
        panel.legend(fontsize=8)
    for row, seed in enumerate(SEEDS, 1):
        for condition in CONDITIONS:
            for initial in (True, False):
                identity = f'initial-{condition}' if initial else f'{condition}-seed-{seed}'
                values = series[identity, 'nominal']
                for column, key in enumerate(('displacement', 'pitch_error', 'speed')):
                    panel = axes[row, column]
                    panel.plot(values['time'], values[key], color=colors[condition],
                        label=condition+(' zero-step' if initial else ' trained'),
                        linewidth=1. if initial else 1.3, linestyle='--' if initial else '-', alpha=.6 if initial else 1.)
                    panel.plot(values['time'][-1], values[key][-1], 'x' if values['terminated'] else '|', color=colors[condition], markersize=5)
        for column, title in enumerate(('Body displacement (m)', 'Pitch error (deg)', 'Horizontal body speed (m/s)')):
            panel = axes[row, column]
            panel.set_title(f'Seed {seed}: nominal | {title}', fontsize=10)
            panel.set_xlabel('Time (s)'); panel.set_xlim(0, horizon_seconds)
            for limit in ((-10., 10.) if column == 1 else ((.05,) if column == 0 else (.1,))):
                panel.axhline(limit, color='.8', linestyle=':', linewidth=.8)
            if row == 1 and column == 0: panel.legend(fontsize=7)
    for panel in axes.flat:
        panel.grid(alpha=.18); panel.tick_params(labelsize=8); panel.spines[['top', 'right']].set_visible(False)
    view = f'early first {horizon_seconds:g} s' if horizon_seconds < 30 else 'common full 30 s evaluation'
    figure.suptitle('Leg capacity: 1 vs 2 N m | '+view+' | both own zero-step references shown', fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, .97))
    try: figure.savefig(destination, dpi=160, bbox_inches='tight', facecolor='white')
    finally: plt.close(figure)


def _display(values, signed=False):
    sign = '+' if signed else ''
    return f"{values['median']:{sign}.3f} [{values['min']:{sign}.3f}, {values['max']:{sign}.3f}]"


def write_report(bundle, output_dir):
    """Render from saved trajectories only; the caller keeps results.json alongside."""
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
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    result_rows = []
    for condition in CONDITIONS:
        initial = summary['initial'][condition]['primary']
        cells = [LABELS['initial-'+condition], f"{initial['mean_in_band_seconds']:.3f}",
                 f"{initial['mean_longest_in_band_seconds']:.3f}", f"{initial['mean_survival_seconds']:.3f}", '—']
        result_rows.append('<tr>'+''.join('<td>'+escape(cell)+'</td>' for cell in cells)+'</tr>')
    for condition in CONDITIONS:
        result = summary['conditions'][condition]
        cells = [result['label'], _display(result['primary_in_band_seconds']),
                 _display(result['primary_longest_in_band_seconds']), _display(result['primary_survival_seconds']),
                 _display(result['delta_from_initial_in_band_seconds'], True)]
        result_rows.append('<tr>'+''.join('<td>'+escape(cell)+'</td>' for cell in cells)+'</tr>')
    pair_rows = []
    for pair in summary['paired_results']:
        cells = [str(pair['seed'])]+[f"{pair[key]:+.3f}" for key in ('delta_in_band_seconds',
            'delta_longest_in_band_seconds', 'delta_survival_seconds',
            'limit1_delta_from_initial_in_band_seconds', 'limit2_delta_from_initial_in_band_seconds',
            'delta_training_gain_in_band_seconds', 'delta_heldout_in_band_seconds')]
        pair_rows.append('<tr>'+''.join('<td>'+escape(cell)+'</td>' for cell in cells)+'</tr>')
    labels = ''.join('<li>'+escape(row['policy_label'])+' · '+escape(row['policy_id'])+'</li>' for row in rows if row['initial_state_id'] == 'nominal')
    content = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>腿力矩上限±1与±2 N·m对照</title><style>body{{font:16px/1.65 system-ui,sans-serif;color:#222;margin:32px auto;max-width:1500px;padding:0 24px}}h1{{font-size:27px}}h2{{font-size:21px;margin-top:30px}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:9px 12px;border-bottom:1px solid #ddd;text-align:right}}th:first-child,td:first-child{{text-align:left}}.table{{overflow-x:auto}}img{{display:block;width:100%;height:auto}}small{{color:#666}}</style>
<h1>腿力矩上限±1与±2 N·m对照</h1><p>{escape(summary['interpretation'])}</p>
<p>±1组复用上一轮匹配结果，±2组新训练；已核对六份完成记录、三个配对seed、每次{summary['matrix']['steps_per_run']:,}步和相同完整参数初始化。两组训练回合均30秒，策略100Hz、物理400Hz。</p>
<p>只扩大腿电机上限，并按同一公式重新标定虚拟腿力范围。原奖励不加姿态或真实位移项；能耗、动作变化惩罚和前拍扭矩观测的归一化尺度固定为[1,1,.0295,.0295]，不能随2 N·m上限一起变松。</p>
<p>相同网络权重在更宽腿力范围下不代表相同的0步闭环行为。因此分别保留两组0步BC参考，显示最终原始配对差及相对各自0步的变化；增量之差也只是描述，不构成训练效果的因果分解。</p>
<p>两组统一五个实际物理初态，首回合最长30秒。累计合格时间剔除开始1秒，要求俯仰误差≤10°、真实XY位移≤5cm、平面速度≤.1m/s同时满足。每seed先平均标称与±.01rad三个主要初态，再对三个训练seed显示中位数[min,max]。</p>
<div class="table"><table><thead><tr><th>组别</th><th>累计合格秒</th><th>最长连续合格秒</th><th>环境终止前秒</th><th>累计相对各自0步变化秒</th></tr></thead><tbody>{''.join(result_rows)}</tbody></table></div>
<p>±2相对±1的最终配对中位差：累计合格{_display(summary['paired_delta_in_band_seconds'],True)}秒，最长连续段{_display(summary['paired_delta_longest_in_band_seconds'],True)}秒。各自0步增量之差的中位数为{_display(summary['paired_delta_training_gain_in_band_seconds'],True)}秒。配对差与各组中位数相减不必相同。</p>
<h2>逐种子配对差</h2><div class="table"><table><thead><tr><th>seed</th><th>最终累计差</th><th>最长段差</th><th>终止前时长差</th><th>±1相对其0步</th><th>±2相对其0步</th><th>两组增量之差</th><th>预留初态累计差</th></tr></thead><tbody>{''.join(pair_rows)}</tbody></table></div>
<p><small>固定五初态不是额外独立训练重复，±.02rad仅为预留固定工况。首回合提前终止后不补造状态；较早失败的小位移、较长存活或较高训练奖励不能单独代表更好原地平衡。三种子短训不证明统计显著、长期收敛或实机能力，±2结果不能外推±24。</small></p>
<h2>前10秒</h2><p>只缩放视窗，不改变摘要；虚线分别为两组各自0步参考。</p><img src="comparison-detail.png" alt="三个seed前10秒真实位移俯仰速度与各自0步参考">
<h2>完整30秒评价时域</h2><img src="comparison.png" alt="共同30秒首回合时域，终止后留空">
<p><a href="results.json">完整轨迹JSON</a> · <a href="paired-summary.json">配对汇总JSON</a> · <a href="summary.csv">逐工况CSV</a></p><details><summary>策略产物标签</summary><ul>{labels}</ul></details></html>'''
    paths['report_html'].write_text(content+'\n', encoding='utf-8')
    return {key: str(path) for key, path in paths.items()}
