"""Independent CSV-level aggregation audit; no policies or physical replay."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).parent
SOURCE = ROOT / 'comparison'
PRIMARY = ('nominal', 'pitch-plus', 'pitch-minus')
HELDOUT = ('heldout-plus', 'heldout-minus')
METRICS = (
    'max_xy_displacement_m', 'rms_xy_displacement_m',
    'last_10s_mean_xy_displacement_m', 'horizontal_speed_rms_m_s',
    'pitch_error_rms_deg', 'in_band_seconds_after_settling',
    'within_1cm_seconds_after_settling',
)
DISPLAY = {
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
assert len(rows) == 35, len(rows)
expected = {('initial', '', state) for state in PRIMARY+HELDOUT} | {
    (condition, str(seed), state) for condition in ('high_lr', 'low_lr')
    for seed in range(3) for state in PRIMARY+HELDOUT
}
actual = {(row['condition'], row['seed'], row['initial_state_id']) for row in rows}
assert actual == expected and len(actual) == len(rows)
assert all(row['reached_30s'] == row['reached_120s'] == 'True'
           and row['termination_reason'] == 'time_limit'
           and float(row['observed_episode_seconds']) == 120 for row in rows)
summary = json.loads((SOURCE/'summary.json').read_text())
paired_summary = json.loads((SOURCE/'paired-summary.json').read_text())
assert summary == paired_summary, 'summary.json and paired-summary.json differ'
aggregates = {}
errors = []

def check(value, expected_value, label):
    if not math.isclose(value, expected_value, rel_tol=0, abs_tol=1e-12):
        errors.append(dict(label=label, csv=value, json=expected_value))

for group, states in (('primary', PRIMARY), ('heldout', HELDOUT)):
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
            for condition in ('high_lr', 'low_lr'):
                values[condition] = []
                for seed in range(3):
                    selected = [row for row in rows if row['condition'] == condition
                                and int(row['seed']) == seed and row['initial_state_id'] in states]
                    assert len(selected) == len(states)
                    result = statistics.mean(float(row[field]) for row in selected)
                    values[condition].append(result)
                    saved_seed = next(item for item in summary['conditions'][condition]['seed_results'] if item['seed'] == seed)
                    check(result, saved_seed[group]['windows'][window][metric], f'{group}/{window}/{metric}/{condition}/{seed}')
                stats = summary['conditions'][condition][group+'_stats'][window][metric]
                for name, fn in (('median', statistics.median), ('min', min), ('max', max)):
                    check(fn(values[condition]), stats[name], f'{group}/{window}/{metric}/{condition}/{name}')
            deltas = [low-high for low, high in zip(values['low_lr'], values['high_lr'])]
            for seed, delta in enumerate(deltas):
                saved = next(item for item in summary['paired_results'] if item['seed'] == seed)
                check(delta, saved[group][window][metric], f'{group}/{window}/{metric}/paired/{seed}')
            saved_stats = summary['paired_stats'][group][window][metric]
            for name, fn in (('median', statistics.median), ('min', min), ('max', max)):
                check(fn(deltas), saved_stats[name], f'{group}/{window}/{metric}/paired/{name}')
            table[metric] = dict(initial=initial, high=values['high_lr'], low=values['low_lr'],
                high_median=statistics.median(values['high_lr']), low_median=statistics.median(values['low_lr']),
                paired=deltas, paired_median=statistics.median(deltas),
                difference_of_medians=statistics.median(values['low_lr'])-statistics.median(values['high_lr']),
                high_delta_initial=[value-initial for value in values['high_lr']],
                low_delta_initial=[value-initial for value in values['low_lr']])
        aggregates[group][window] = table

assert not errors, errors

def triple(values, scale):
    return '/'.join(f'{value*scale:.6f}' for value in values)

def stats(values, scale):
    return f'{statistics.median(values)*scale:.6f} [{min(values)*scale:.6f}, {max(values)*scale:.6f}]'

lines = [
    '# 成熟模型微调：独立统计复核', '',
    '本复核仅从summary.csv的35行独立计算组内平均、训练seed中位数和配对差，未加载模型或原始results.json、未运行物理或训练。原始轨迹物理指标与输入哈希由主代理另行复核。', '',
    '数据起点为成熟081557/2,801,664步完整模型，不是先前BC初始化。两组均增加真实位置项（权重10、尺度0.05m）、使用30秒训练回合，分别采用学习率3e-5和3e-6。统计单位为3个训练seed；3个主要固定初态先在每个seed内部平均，2个预留初态单列，不能把固定初态当独立训练重复。', '',
    '检查结果：CSV身份矩阵完整且唯一，35/35首回合均为120秒time_limit；summary.json与paired-summary.json内容完全相同。下列独立重算与JSON中初始值、seed内平均、组median/min/max、配对seed差和配对median/min/max全部相符（绝对容差1e-12）。', '',
    '## 判读', '',
    '- 两个学习率组都保留了本次固定工况下的120秒平衡；累计合格时间30秒窗口均29秒、120秒窗口均119秒，已封顶，不能据此判断哪个学习率更稳定。120秒截断也不是无限时域或实机稳定性证明。',
    '- 当前结果不支持“降低学习率更好”。在主要初态的30秒与120秒窗口，低学习率的末段平均真实位移及位移RMS在全部3个配对seed上都大于高学习率；高学习率这两项在全部3个seed上都小于未更新成熟参考，低学习率则均大于参考。',
    '- 改善有指标取舍：高学习率的最大位移和平面速度RMS较成熟参考更高，不能写成“所有漂移指标都改善”。低学习率的最大位移组中位数小于高学习率（配对2/3 seed更小），但这没有转化为更小的末段偏移或RMS。',
    '- 两组大多数回放仍停在1cm范围外；不能把约1.1～2.4cm的持久偏移描述为回到起点。1cm内时间与5cm/10度/0.1m/s合格带时间是不同指标。',
    '- high-vs-low可用于本次固定设置下的学习率对照；相对不更新参考的变化同时包含新增位置奖励、参数更新及30秒训练设置，不能单独归因于真实位移奖励。三seed只作描述性证据，不宣称统计显著或普遍最优。', '',
]

for group, group_label in (('primary', '主要初态'), ('heldout', '预留初态')):
    lines += [f'## {group_label}：组汇总', '',
        '每格组统计为“3个seed的中位数 [最小, 最大]”；初始参考只有一份固定模型。配对差统一为low−high；XY/速度/姿态误差正值表示低学习率更差，时间类指标方向相反。', '',
        '| 窗口 | 指标 | 初始 | high median [range] | low median [range] | 配对差median [range] | 组中位数之差 |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for window, table in aggregates[group].items():
        for metric, item in table.items():
            label, scale = DISPLAY[metric]
            lines.append(f'| {window} | {label} | {item["initial"]*scale:.6f} | {stats(item["high"],scale)} | {stats(item["low"],scale)} | {stats(item["paired"],scale)} | {item["difference_of_medians"]*scale:.6f} |')
    lines += ['', f'### {group_label}：关键指标seed明细', '',
        '以下seed序列均按0/1/2排列；相对初始参考差也是先在每个seed内对固定初态平均再计算。', '',
        '| 窗口 | 指标 | high seed0/1/2 | low seed0/1/2 | low−high seed0/1/2 | high−initial seed0/1/2 | low−initial seed0/1/2 |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for window, table in aggregates[group].items():
        for metric in ('last_10s_mean_xy_displacement_m', 'rms_xy_displacement_m', 'within_1cm_seconds_after_settling'):
            item = table[metric]
            label, scale = DISPLAY[metric]
            lines.append(f'| {window} | {label} | {triple(item["high"],scale)} | {triple(item["low"],scale)} | {triple(item["paired"],scale)} | {triple(item["high_delta_initial"],scale)} | {triple(item["low_delta_initial"],scale)} |')
    lines.append('')

p = aggregates['primary']['first_120s']['last_10s_mean_xy_displacement_m']
r = aggregates['primary']['first_120s']['rms_xy_displacement_m']
lines += ['## 可直接用于汇报的结论', '',
    f'在本次成熟起点、相同位置奖励和相同训练预算下，两种学习率都在35组固定回放中保留了120秒存活。主要初态120秒末10秒平均真实位移：初始参考{p["initial"]*1000:.3f}mm，高学习率组中位{p["high_median"]*1000:.3f}mm，低学习率组中位{p["low_median"]*1000:.3f}mm；低减高的配对中位差为+{p["paired_median"]*1000:.3f}mm，3/3 seed均指向高学习率较小。', '',
    f'位移RMS同向：初始{r["initial"]*1000:.3f}mm，高学习率{r["high_median"]*1000:.3f}mm，低学习率{r["low_median"]*1000:.3f}mm。高学习率减小了稳态偏移和RMS，但其最大位移、速度RMS较参考更高；尚未实现稳定回到1cm以内。不能据此将收益单独归因于新增奖励，也不能把低学习率称为普遍更优。', '',
    '汇报组中位数与配对中位差时必须分别命名，两者不要求相等。完整主要/预留30秒、120秒统计和seed差见上表。', '',
    '## 输入身份', '',
]
for name in ('summary.csv', 'summary.json', 'paired-summary.json'):
    lines.append(f'- `{name}` SHA256：`{hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()}`')
(ROOT/'statistics-review.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
print(json.dumps({
    'rows': len(rows), 'json_checks_mismatched': len(errors),
    'primary_120s_tail_mm': {key: [value*1000 for value in p[key]] if isinstance(p[key],list) else p[key]*1000
                            for key in ('initial','high','low','paired','paired_median','difference_of_medians')},
    'review': str(ROOT/'statistics-review.md'),
}, indent=2))
