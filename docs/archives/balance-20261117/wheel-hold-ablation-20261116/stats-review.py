"""Independent CSV aggregation of wheel-hold weights; no policy or replay."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).parent
SOURCE = ROOT / 'comparison'
STATES = {'primary': ('nominal', 'pitch-plus', 'pitch-minus'),
          'heldout': ('heldout-plus', 'heldout-minus')}
CONDITIONS = ('original_wheel', 'weak_wheel')
METRICS = {
    'max_xy_displacement_m': ('最大XY位移（mm）', 1000),
    'rms_xy_displacement_m': ('XY位移RMS（mm）', 1000),
    'last_10s_mean_xy_displacement_m': ('末10秒平均XY位移（mm）', 1000),
    'horizontal_speed_rms_m_s': ('平面速度RMS（mm/s）', 1000),
    'pitch_error_rms_deg': ('俯仰误差RMS（度）', 1),
    'in_band_seconds_after_settling': ('累计合格时间（秒）', 1),
    'within_1cm_seconds_after_settling': ('1cm内时间（秒）', 1),
}

with (SOURCE/'summary.csv').open(encoding='utf-8', newline='') as stream:
    rows = list(csv.DictReader(stream))
all_states = tuple(state for states in STATES.values() for state in states)
expected = {('initial', '', state) for state in all_states} | {
    (condition, str(seed), state) for condition in CONDITIONS
    for seed in range(3) for state in all_states}
actual = {(row['condition'], row['seed'], row['initial_state_id']) for row in rows}
assert len(rows) == len(actual) == 35 and actual == expected
assert all(row['reached_30s'] == row['reached_120s'] == 'True'
           and row['termination_reason'] == 'time_limit'
           and float(row['observed_episode_seconds']) == 120 for row in rows)
summary = json.loads((SOURCE/'summary.json').read_text())
paired_summary = json.loads((SOURCE/'paired-summary.json').read_text())
assert summary == paired_summary, 'summary.json and paired-summary.json differ'
aggregates, errors = {}, []
checks = 0

def check(value, expected_value, label):
    global checks
    checks += 1
    if not math.isclose(value, expected_value, rel_tol=0, abs_tol=1e-12):
        errors.append(dict(label=label, csv=value, json=expected_value))

for group, states in STATES.items():
    aggregates[group] = {}
    for horizon in (30, 120):
        window = f'first_{horizon}s'
        table = {}
        for metric in METRICS:
            field = window + '_' + metric
            initial = statistics.mean(float(row[field]) for row in rows
                                      if row['condition'] == 'initial' and row['initial_state_id'] in states)
            check(initial, summary['initial'][group]['windows'][window][metric], f'{group}/{window}/{metric}/initial')
            values = {}
            for condition in CONDITIONS:
                values[condition] = []
                for seed in range(3):
                    selected = [row for row in rows if row['condition'] == condition
                                and int(row['seed']) == seed and row['initial_state_id'] in states]
                    assert len(selected) == len(states)
                    result = statistics.mean(float(row[field]) for row in selected)
                    values[condition].append(result)
                    saved_seed = next(item for item in summary['conditions'][condition]['seed_results'] if item['seed'] == seed)
                    check(result, saved_seed[group]['windows'][window][metric], f'{group}/{window}/{metric}/{condition}/{seed}')
                    check(result-initial, saved_seed[group+'_delta_from_initial'][window][metric],
                          f'{group}/{window}/{metric}/{condition}/{seed}/initial-delta')
                stats = summary['conditions'][condition][group+'_stats'][window][metric]
                for name, fn in (('median', statistics.median), ('min', min), ('max', max)):
                    check(fn(values[condition]), stats[name], f'{group}/{window}/{metric}/{condition}/{name}')
            original, weak = values['original_wheel'], values['weak_wheel']
            deltas = [w-o for w, o in zip(weak, original)]
            for seed, delta in enumerate(deltas):
                saved = next(item for item in summary['paired_results'] if item['seed'] == seed)
                check(delta, saved[group][window][metric], f'{group}/{window}/{metric}/paired/{seed}')
            saved_stats = summary['paired_stats'][group][window][metric]
            for name, fn in (('median', statistics.median), ('min', min), ('max', max)):
                check(fn(deltas), saved_stats[name], f'{group}/{window}/{metric}/paired/{name}')
            table[metric] = dict(initial=initial, original=original, weak=weak,
                original_median=statistics.median(original), weak_median=statistics.median(weak),
                paired=deltas, paired_median=statistics.median(deltas),
                difference_of_medians=statistics.median(weak)-statistics.median(original),
                original_delta_initial=[value-initial for value in original],
                weak_delta_initial=[value-initial for value in weak])
        aggregates[group][window] = table
assert not errors, errors

def triples(values, scale):
    return '/'.join(f'{value*scale:.6f}' for value in values)

def stats(values, scale):
    return f'{statistics.median(values)*scale:.6f} [{min(values)*scale:.6f}, {max(values)*scale:.6f}]'

lines = [
    '# 弱轮角约束对照：独立统计复核', '',
    '本复核仅从summary.csv的35行独立计算，未加载模型/原始results.json，未训练或重做物理评估。原始轨迹、奖励、输入保护和完整checkpoint检查由主代理另行完成。', '',
    '本轮对照从同一个成熟完整初始化出发，学习率均3e-5、真实XY项权重10/尺度0.05m、30秒训练回合、每seed训练2,097,152步。唯一对照因子是轮角保持权重10与1。权重10复用以前3个配对seed，权重1为本轮新增3个seed；这些不是6次本轮新增训练。历史baseline中间产物仍按descriptor对待，新weak组完整stage checkpoint验证的性质与范围以主代理记录为准。', '',
    f'CSV身份矩阵完整唯一，35/35最终首回合均完成120秒、原因time_limit。summary.json与paired-summary.json内容完全相同；独立计算与JSON的{checks}项初始均值、seed内均值、相对initial差、组median/min/max和配对差/统计全部一致（绝对容差1e-12）。', '',
    '## 结论边界', '',
    '- 本轮固定设置下，降低轮角保持权重从10到1，对真实位置保持有一致收益：主要初态和预留初态的30秒、120秒位移RMS及末10秒平均位移在全部3个配对seed上均降低。',
    '- 保留了本次固定工况的120秒平衡：全部最终回放都达到120秒time_limit；剔除起始1秒后的合格时间，30秒窗口均29秒、120秒窗口均119秒。该指标已封顶，不证明稳定性进一步增强，也不代表无限时域、抗噪声、抗滑或实机鲁棒性。',
    '- 与原权重组相比，主要初态的最大位移和平面速度RMS在全部3个seed上也降低；俯仰误差RMS则轻微升高。不能写成所有姿态指标均改善。CSV只有窗口最大值，不能单凭它确定最大位移发生在启动时刻。',
    '- 与未更新成熟参考相比，weak组主要初态的末段位移、位移RMS、速度RMS均在3个seed上降低；最大位移有2/3 seed高于参考，俯仰误差RMS亦略高，因此依然存在取舍。',
    '- 权重1组的1cm内时间明显增加，但并非全部工况达标：15个回放中13个在1cm内超过可观测119秒的一半。seed2的pitch-minus仅38.21秒，末段偏移10.106mm；heldout-minus为0秒，末段偏移11.384mm。不能用组平均或中位数声称所有回放始终处于1cm内。',
    '- 两组训练目标本身不同，不能直接用训练objective return跨组排名。应以共同物理指标为主，共同base10+XY评估回报只是辅助量。',
    '- 相对同预算weight10，本次仅改变轮角项，支持“本设置下减弱轮角项改善位置保持”；相对不更新参考同时还包含训练、XY项及30秒训练设置，不能把全部收益单独归因于某一项。三配对seed不构成统计显著性、全局最优或实机结论。', '',
]

for group, label in (('primary', '主要初态'), ('heldout', '预留初态')):
    lines += [f'## {label}：组汇总', '',
        '3个主要初态或2个预留初态先在每个训练seed内部平均，再对3个seed取中位数；固定初态不作为独立训练重复。每格为中位数[最小,最大]，初始参考为单个固定成熟模型的对应初态平均。配对差为weak−original；位移/速度/姿态误差负值较好，时间类指标方向相反。', '',
        '| 窗口 | 指标 | initial | original median [range] | weak median [range] | 配对差median [range] | 组中位数之差 |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for window, table in aggregates[group].items():
        for metric, item in table.items():
            name, scale = METRICS[metric]
            lines.append(f'| {window} | {name} | {item["initial"]*scale:.6f} | {stats(item["original"],scale)} | {stats(item["weak"],scale)} | {stats(item["paired"],scale)} | {item["difference_of_medians"]*scale:.6f} |')
    lines += ['', f'### {label}：关键指标seed明细', '',
        '序列均按seed0/1/2排列；相对initial差使用对应初态集合的成熟参考均值。', '',
        '| 窗口 | 指标 | original seed0/1/2 | weak seed0/1/2 | weak−original seed0/1/2 | original−initial seed0/1/2 | weak−initial seed0/1/2 |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for window, table in aggregates[group].items():
        for metric in ('last_10s_mean_xy_displacement_m', 'rms_xy_displacement_m', 'within_1cm_seconds_after_settling'):
            item = table[metric]
            name, scale = METRICS[metric]
            lines.append(f'| {window} | {name} | {triples(item["original"],scale)} | {triples(item["weak"],scale)} | {triples(item["paired"],scale)} | {triples(item["original_delta_initial"],scale)} | {triples(item["weak_delta_initial"],scale)} |')
    lines.append('')

main = aggregates['primary']['first_120s']
tail = main['last_10s_mean_xy_displacement_m']
rms = main['rms_xy_displacement_m']
one = main['within_1cm_seconds_after_settling']
pitch = main['pitch_error_rms_deg']
lines += ['## 可用于汇报的核心数值', '',
    f'主要初态120秒末10秒真实位移模长：成熟参考{tail["initial"]*1000:.3f}mm，权重10组中位{tail["original_median"]*1000:.3f}mm，权重1组中位{tail["weak_median"]*1000:.3f}mm；3个seed的配对变化为{triples(tail["paired"],1000)}mm，配对中位差{tail["paired_median"]*1000:.3f}mm，3/3 seed均改善。组中位数之差另为{tail["difference_of_medians"]*1000:.3f}mm，不与配对中位差混写。', '',
    f'对应XY位移RMS：参考{rms["initial"]*1000:.3f}mm，权重10中位{rms["original_median"]*1000:.3f}mm，权重1中位{rms["weak_median"]*1000:.3f}mm。1cm内时间：参考{one["initial"]:.3f}秒，权重10中位{one["original_median"]:.3f}秒，权重1中位{one["weak_median"]:.3f}秒（3个seed平均值范围{min(one["weak"]):.3f}～{max(one["weak"]):.3f}秒，可观测区间119秒）。', '',
    f'俯仰误差RMS的配对中位变化为+{pitch["paired_median"]:.6f}度（seed范围+{min(pitch["paired"]):.6f}～+{max(pitch["paired"]):.6f}度）；这是小幅姿态代价，合格带与存活未受损。结论仅限本次短训、三配对seed和固定初态。', '',
    '## 输入身份', '',
]
for name in ('summary.csv', 'summary.json', 'paired-summary.json'):
    lines.append(f'- `{name}` SHA256：`{hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()}`')
(ROOT/'statistics-review.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
print(json.dumps(dict(rows=len(rows), checked_statistics=checks, mismatches=len(errors),
    tail_mm={key: [value*1000 for value in tail[key]] if isinstance(tail[key],list) else tail[key]*1000
             for key in ('initial','original','weak','paired','paired_median','difference_of_medians')},
    report=str(ROOT/'statistics-review.md')), indent=2))
