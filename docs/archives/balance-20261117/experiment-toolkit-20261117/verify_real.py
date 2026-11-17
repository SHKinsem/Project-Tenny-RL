"""One-off read-only regression against two recorded experiment reports."""
import hashlib
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.experiment_tools import freeze_inputs, verify_inputs, load_study, summarize, select_snapshot
from experiments.experiment_tools.adapters import WINDOWS, METRICS
from experiments.experiment_tools.io import load_json, save_json

started = time.perf_counter()
output = Path(__file__).resolve().parent
results = output.parent
roots = [results/'mature-balance-finetune-20261116', results/'wheel-hold-ablation-20261116']
protected = []
for root in roots:
    protected.append(root/'comparison/summary.json')
    folder = root/('selected-snapshot' if root.name.startswith('wheel') else 'selected-stage')
    protected.extend(folder/name for name in ('summary.json', 'verification.json', 'descriptor.json'))
frozen = freeze_inputs(protected)
checks = 0
max_error = 0.0


def same(actual, expected):
    global checks, max_error
    checks += 1
    if expected is None:
        assert actual is None, (actual, expected)
    else:
        assert actual is not None
        error = abs(actual-expected)
        max_error = max(max_error, error)
        assert math.isfinite(actual) and error <= 1e-12, (actual, expected, error)


reports = []
for root in roots:
    old = load_json(root/'comparison/summary.json')
    study = load_study(root)
    new = summarize(study)
    for condition in new['comparison'].values():
        for partition in new['partitions']:
            new_group = new['condition_summaries'][condition][partition]
            for old_seed in old['conditions'][condition]['seed_results']:
                seed = old_seed['seed']
                new_seed = next(row for row in new_group['seed_results'] if row['seed'] == seed)
                new_delta = next(row for row in new['delta_from_reference'][condition][partition]['seed_results'] if row['seed'] == seed)
                for window in WINDOWS:
                    for metric in METRICS:
                        field = window+'_'+metric
                        same(new_seed['values'][field], old_seed[partition]['windows'][window][metric])
                        same(new_delta['values'][field], old_seed[partition+'_delta_from_initial'][window][metric])
            for window in WINDOWS:
                for metric in METRICS:
                    for statistic, value in old['conditions'][condition][partition+'_stats'][window][metric].items():
                        same(new_group['metrics'][window+'_'+metric][statistic], value)
    for partition in new['partitions']:
        for old_pair in old['paired_results']:
            new_pair = next(row for row in new['paired_results'][partition] if row['seed'] == old_pair['seed'])
            for window in WINDOWS:
                for metric in METRICS:
                    same(new_pair['values'][window+'_'+metric], old_pair[partition][window][metric])
        for window in WINDOWS:
            for metric in METRICS:
                same(new['references'][partition]['values'][window+'_'+metric], old['initial'][partition]['windows'][window][metric])
                for statistic, value in old['paired_stats'][partition][window][metric].items():
                    same(new['paired_summaries'][partition][window+'_'+metric][statistic], value)
    wheel = root.name.startswith('wheel')
    condition = 'weak_wheel' if wheel else 'high_lr'
    selection = select_snapshot(study, condition=condition, metric='first_30s_rms_xy_displacement_m',
                                require_full_parameters=wheel)
    expected_id = 'weak_wheel-seed-0@786432' if wheel else 'high_lr-seed-1@1048576'
    assert selection['selected']['id'] == expected_id
    assert selection['selected']['artifacts']['full_parameters_verified'] is wheel
    save_json(output/(root.name+'-report.json'), new)
    save_json(output/(root.name+'-selection.json'), selection)
    metric = 'first_120s_last_10s_mean_xy_displacement_m'
    reports.append(dict(source=root.name, counts=new['counts'], selected=expected_id,
                        capability=selection['selected']['artifacts']['capability'],
                        primary_last10_paired=new['paired_summaries']['primary'][metric]))

result = dict(phase='verified', scalar_comparisons=checks, max_abs_error=max_error,
              input_verification=verify_inputs(frozen), reports=reports,
              elapsed_seconds=time.perf_counter()-started,
              scope='Read-only comparison of small historical summaries and recorded evidence; no simulator or training.')
save_json(output/'historical-inputs.json', frozen)
save_json(output/'real-validation.json', result)
print(result)
