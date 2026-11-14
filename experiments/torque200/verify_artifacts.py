"""Verify trusted local training outputs, including critic and exploration scales."""
import json
from dataclasses import replace
from pathlib import Path
import pickle
import sys

import jax
import numpy as np

from legwheel.deployment.extract import load


def check_parameter_trees(before, restored):
    def canonical(params):
        result = list(params)
        # The original Orbax loader reconstructs RunningStatisticsState, but its
        # UInt64 count is a {hi,lo} dict. Compare both words, not Python wrappers.
        count = result[0].count
        words = count if isinstance(count, dict) else {'hi': count.hi, 'lo': count.lo}
        if set(words) != {'hi', 'lo'}:
            raise AssertionError('Unexpected normalization count representation')
        result[0] = replace(result[0], count=words)
        return result

    left = jax.tree_util.tree_flatten_with_path(canonical(before))
    right = jax.tree_util.tree_flatten_with_path(canonical(restored))
    if left[1] != right[1]:
        raise AssertionError('Checkpoint parameter tree structure changed')
    max_error = 0.0
    for (path, original), (_, recovered) in zip(left[0], right[0], strict=True):
        a, b = np.asarray(original), np.asarray(recovered)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise AssertionError(f'Invalid parameter leaf: {path}')
        np.testing.assert_array_equal(a, b, err_msg=str(path))
        if a.size:
            max_error = max(max_error, float(np.max(np.abs(a.astype(float) - b.astype(float)))))
    return {'parameter_leaves_checked': len(left[0]), 'parameter_tree_max_abs_error': max_error,
            'all_parameters_finite': True}


def check_losses(events):
    checked = []
    for event in events:
        for name, value in event['metrics'].items():
            if name.startswith('training/') and 'loss' in name:
                if not np.isfinite(value):
                    raise AssertionError(f'Nonfinite training metric: {name}')
                checked.append(name)
    # Some Brax configurations omit train metrics; state this rather than assume.
    return {'finite_training_losses_checked': sorted(set(checked))}


def verify_directory(directory):
    directory = Path(directory)
    report = json.loads((directory / 'report.json').read_text())
    # This pickle was created by our own run, not an external untrusted checkpoint.
    with (directory / 'params_cpu.pkl').open('rb') as stream:
        before = pickle.load(stream)
    restored = load(report['checkpoint'])
    result = check_parameter_trees(before, restored)
    result.update(check_losses(json.loads((directory / 'progress.json').read_text())))
    result['checkpoint'] = report['checkpoint']
    (directory / 'parameter-verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    return before


if __name__ == '__main__':
    first, second = (verify_directory(p) for p in sys.argv[1:3])
    before = jax.tree.leaves(list(first)[1:])
    after = jax.tree.leaves(list(second)[1:])
    max_change = max(float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
                     for a, b in zip(before, after, strict=True))
    if not np.isfinite(max_change) or max_change <= 0:
        raise AssertionError('Resumed training did not change policy/value parameters')
    summary = {'both_parameter_trees_verified': True, 'resumed_network_max_abs_change': max_change}
    (Path(sys.argv[1]).parent / 'parameter-verification.json').write_text(json.dumps(summary, indent=2))
    print('TORQUE200_PARAMETER_TREES_OK', json.dumps(summary), flush=True)
