"""Fixed-descriptor native evaluation, with no training or production writes."""
import hashlib
import json
import math
from pathlib import Path
import time


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(',', ':')).encode()).hexdigest()


def evaluate_manifest(manifest_path, output_dir, *, seconds=30., reference_seconds=6.):
    import mujoco
    import numpy as np
    from native_eval import Evaluator
    from native_eval.api import load_descriptor
    from .legwheel import model_layout, normalized_trace
    from .metrics import analyze_trace

    start = time.monotonic()
    manifest_path, output_dir = Path(manifest_path).resolve(), Path(output_dir).resolve()
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 1 or not manifest.get('policies') or not manifest.get('initial_states'):
        raise ValueError('Expected version 1 manifest with policies and initial_states')
    if not (math.isfinite(seconds) and reference_seconds == 6. and reference_seconds <= seconds <= 120):
        raise ValueError('Version 1 reports use a 6-second reference; require 6 <= seconds <= 120')
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError('Output directory must be new or empty; existing results are preserved')
    output_dir.mkdir(parents=True, exist_ok=True)
    initial_items = manifest['initial_states']
    states = [item['state'] for item in initial_items]
    if len({item['id'] for item in initial_items}) != len(initial_items):
        raise ValueError('Initial-state ids must be unique')
    if len({item['id'] for item in manifest['policies']}) != len(manifest['policies']):
        raise ValueError('Policy ids must be unique')
    state_hash = _digest(initial_items)
    cases, parity, identities = [], [], []
    first_task = None
    for index, policy in enumerate(manifest['policies']):
        descriptor_path = (manifest_path.parent / policy['descriptor']).resolve()
        descriptor_bytes = descriptor_path.read_bytes()
        descriptor = load_descriptor(descriptor_path)
        task_identity = dict(task=descriptor['task'], physics_dt=descriptor.get('physics_dt'),
                             control_dt=descriptor['eval_defaults']['control_dt'],
                             xml_sha256=hashlib.sha256(Path(descriptor['xml_path']).read_bytes()).hexdigest())
        if first_task is not None and task_identity != first_task:
            raise ValueError('Compared policies must use the same model and task settings')
        first_task = task_identity
        dt = descriptor['eval_defaults']['control_dt']
        counts = [int(round(value/dt)) for value in (reference_seconds, seconds)]
        if any(not math.isclose(count*dt, value, rel_tol=0, abs_tol=1e-9)
               for count, value in zip(counts, (reference_seconds, seconds))):
            raise ValueError('Evaluation durations must be exact control-step multiples')
        engine = Evaluator(descriptor)
        model = mujoco.MjModel.from_xml_path(descriptor['xml_path'])
        layout = model_layout(model, descriptor['task'])
        results = []
        for count, label in zip(counts, ('reference', 'long')):
            result = engine.evaluate(states, num_envs=len(states), steps=count, episode_steps=count,
                                     control_dt=dt, seed=manifest.get('seed', 0), trace_worlds=list(range(len(states))))
            if result['cancelled'] or result['control_steps'] != count*len(states):
                raise ValueError('Incomplete native workload')
            trace_arrays = {f'world_{trace["world"]}_{key}': value for trace in result['traces']
                            for key, value in trace.items() if key != 'world'}
            np.savez_compressed(output_dir / f'policy-{index}-{label}.npz', **trace_arrays)
            results.append(result)
        short, long = results
        identities.append(dict(id=policy['id'], label=policy['label'],
            descriptor_sha256=hashlib.sha256(descriptor_bytes).hexdigest(),
            policy_sha256=_digest(descriptor['policy']), metadata=descriptor.get('metadata', {}),
            source=policy.get('source', {})))
        for world, initial_item in enumerate(initial_items):
            short_raw = next(trace for trace in short['traces'] if trace['world'] == world)
            long_raw = next(trace for trace in long['traces'] if trace['world'] == world)
            prefix = min(int(long['lengths'][world]), counts[0])
            errors = {key: float(np.max(np.abs(np.asarray(short_raw[key][:prefix], dtype=float) -
                                             np.asarray(long_raw[key][:prefix], dtype=float))))
                      for key in ('observation', 'action', 'qpos', 'qvel', 'ctrl', 'reward')}
            if any(value != 0 for value in errors.values()):
                raise ValueError('Changing the time limit changed the shared trajectory prefix')
            if min(int(long['lengths'][world]), counts[0]) != int(short['lengths'][world]):
                raise ValueError('Short and long first-episode lengths disagree')
            parity.append(dict(policy_id=policy['id'], initial_state_id=initial_item['id'],
                               compared_steps=prefix, max_abs_errors=errors, exact_prefix_match=True,
                               reference_return=float(short['returns'][world]),
                               reference_length_seconds=int(short['lengths'][world])*dt))
            trace = normalized_trace(long_raw, initial_item['state'], long, world, dt, layout)
            trace['metadata'].update(policy_sha256=identities[-1]['policy_sha256'],
                                     initial_states_sha256=state_hash, initial_state_id=initial_item['id'])
            analysis = analyze_trace(trace, windows_s=(reference_seconds,))
            cases.append(dict(policy_id=policy['id'], label=policy['label'], plot_label=policy.get('plot_label', policy['id']),
                              initial_state_id=initial_item['id'],
                              initial_state_label=initial_item.get('label', initial_item['id']),
                              trace=trace, analysis=analysis,
                              native_first_episode_metrics={key: float(values[world])
                                  for key, values in long.get('episode_metrics', {}).items()}))
        # Reject a policy descriptor changed concurrently while being evaluated.
        if descriptor_path.read_bytes() != descriptor_bytes:
            raise ValueError('Policy descriptor changed during evaluation')
        print(json.dumps(dict(policy=policy['id'], lengths_s=[int(v)*dt for v in long['lengths']],
                              reasons=long['reasons'], native_seconds=long['wall_seconds'])), flush=True)
    bundle = dict(schema_version=1, kind='balance_diagnostics', horizon_s=seconds,
        reference_horizon_s=reference_seconds, scope=manifest.get('scope', 'Fixed deterministic initial states only.'),
        conditions=manifest.get('conditions', {}), initial_states=initial_items,
        initial_states_sha256=state_hash, policies=identities, parity=parity, cases=cases,
        preparation_evidence=manifest.get('preparation_evidence', {}),
        timing=dict(evaluation_and_analysis_seconds=time.monotonic()-start),
        limitations=['No new training or parameter changes.',
                     'Fixed initial states are not a random robustness study.',
                     'A time limit is censored survival, not proof of stable equilibrium.',
                     'Descriptive threshold order does not establish causality.'])
    write_json(output_dir / 'results.json', bundle)
    return bundle
