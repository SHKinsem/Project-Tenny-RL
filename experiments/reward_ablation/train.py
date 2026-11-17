"""One fixed-step experimental run using the deployed force PPO implementation."""
import hashlib
import json
from pathlib import Path
import time


def validate_reward_manifest(manifest):
    """The current runner supports only the explicitly recorded fixed arms."""
    from dataclasses import asdict
    from .rewards import ARMS
    if manifest.get('arms') != {name: asdict(value) for name, value in ARMS.items()}:
        raise ValueError('Recorded reward coefficients differ from the executed fixed ARMS')


def final_policy_metadata(initial_metadata, arm, seed, steps, checkpoint_sha256, initial_array_digest, additions):
    """Keep initializer history separate from the newly trained model identity."""
    original_keys = ('model_run_id', 'export_sha256', 'initializer_params_sha256', 'kind')
    value = {key: initial_metadata[key] for key in ('source_fingerprint', 'action_order', 'task_schema')
             if key in initial_metadata}
    value.update(kind='saved_reward_ablation_policy', experiment_kind='reward_ablation_v1',
        model_run_id=f'reward-ablation-v1/{arm}-seed-{seed}', arm=arm, training_seed=seed,
        training_steps=steps, checkpoint_sha256=checkpoint_sha256, initial_array_digest=initial_array_digest,
        initializer_provenance={key: initial_metadata.get(key) for key in original_keys}, reward_additions=additions)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def tree_array_digest(value):
    import jax
    import numpy as np
    digest = hashlib.sha256()
    for leaf in jax.tree.leaves(jax.device_get(value)):
        array = np.asarray(leaf)
        if array.dtype.kind in 'fc' and not np.isfinite(array).all():
            raise ValueError('Checkpoint parameters contain nonfinite values')
        digest.update(str(array.dtype).encode() + repr(array.shape).encode() + array.tobytes())
    return digest.hexdigest()


def canonical_initial_params(params):
    """Match Brax's float32 zero epsilon without changing any inference values."""
    import numpy as np
    if float(params[0].std_eps) != 0.:
        raise ValueError('This fixed-normalizer contract requires zero std_eps')
    values = list(params)
    values[0] = values[0].replace(std_eps=np.asarray(0., dtype=np.float32))
    return values


def train_one(manifest_path, arm, seed, output_dir):
    import jax
    import numpy as np
    from orbax import checkpoint as ocp
    from force_teacher.env import StudentEnv, student_config
    from force_teacher.ppo import train_student, normalizer_max_abs_change
    from legwheel.deployment.extract import load
    from native_eval import Evaluator
    from native_eval.brax_policy import policy_from_brax
    from native_eval.legwheel import policy_template
    from .legwheel_env import RewardExperimentEnv
    from .rewards import ARMS

    manifest_path, output_dir = Path(manifest_path).resolve(), Path(output_dir).resolve()
    manifest = json.loads(manifest_path.read_text())
    validate_reward_manifest(manifest)
    if arm not in ARMS or seed not in manifest['seeds']:
        raise ValueError('Arm/seed outside frozen experiment matrix')
    if not any(device.platform == 'gpu' for device in jax.devices()):
        raise RuntimeError('This experiment requires an actual JAX GPU backend')
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError('Run output must be new or empty')
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    descriptor = json.loads(Path(manifest['base_descriptor']).read_text())
    ppo = dict(manifest['ppo'])
    ppo.update(seed=seed, policy_params_fn=None)
    metadata = manifest['policy_metadata']
    template = policy_template(metadata, ppo)
    cfg = student_config()
    generated_config = json.loads(cfg.to_json_best_effort())
    if generated_config != manifest['env_config']:
        raise ValueError('Generated deployed environment differs from frozen configuration')
    initial_params = canonical_initial_params(load(manifest['initial_checkpoint']))
    initial_digest = tree_array_digest(initial_params)
    if initial_digest != manifest['initial_array_digest']:
        raise ValueError('Initializer actor/critic/normalizer arrays changed')
    env = RewardExperimentEnv(arm, cfg, root_body=descriptor['task']['config']['root_body'])
    eval_env = StudentEnv(config=student_config())
    states = [item['state'] for item in manifest['initial_states'][:3]]
    progress_path = output_dir/'progress.jsonl'
    latest = {'steps': 0}
    saved_initial = None

    def on_policy(steps, make_policy, params):
        nonlocal saved_initial
        steps = int(steps)
        host = jax.device_get(params)
        policy = policy_from_brax(host[:2], template)
        if steps == 0:
            saved_initial = tree_array_digest(host)
            if saved_initial != initial_digest:
                before = jax.tree.flatten_with_path(jax.device_get(initial_params))[0]
                after = jax.tree.flatten_with_path(host)[0]
                differences = []
                for (left_path, left), (right_path, right) in zip(before, after):
                    left, right = np.asarray(left), np.asarray(right)
                    if left.dtype != right.dtype or left.shape != right.shape or not np.array_equal(left, right):
                        differences.append(dict(before_path=str(left_path),after_path=str(right_path),
                            before_dtype=str(left.dtype),after_dtype=str(right.dtype),
                            before_shape=list(left.shape),after_shape=list(right.shape),
                            max_abs=float(np.max(np.abs(left.astype(float)-right.astype(float))))
                                if left.shape == right.shape and left.dtype.kind in 'fiu' and right.dtype.kind in 'fiu' else None))
                write_json(output_dir/'initial-restore-differences.json',dict(before_leaves=len(before),
                    after_leaves=len(after),differences=differences,
                    actor_equal=tree_array_digest(initial_params[1])==tree_array_digest(host[1]),
                    critic_equal=tree_array_digest(initial_params[2])==tree_array_digest(host[2]),
                    normalizer_max_abs_change=float(normalizer_max_abs_change(initial_params[0],host[0]))))
                raise ValueError('PPO failed to restore the complete identical initializer')
        drift = float(normalizer_max_abs_change(initial_params[0], host[0]))
        if drift != 0:
            raise ValueError('Frozen normalizer changed')
        current = dict(descriptor, policy=policy)
        result = Evaluator(current).evaluate(states, num_envs=len(states), steps=600, episode_steps=600,
                                             control_dt=.01, seed=0, trace_worlds=[])
        event = dict(arm=arm, seed=seed, steps=steps, elapsed_seconds=time.monotonic()-start,
            normalizer_drift=drift, common_base_reward_mean=float(np.mean(result['returns'])),
            common_episode_seconds=[int(length)*.01 for length in result['lengths']],
            actor_and_normalizer_sha256=tree_array_digest(host[:2]))
        with progress_path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(event, allow_nan=False)+'\n')
        latest.update(steps=steps)
        write_json(output_dir/'status.json', dict(phase='training', **event))
        write_json(output_dir/'latest-descriptor.json', current)
        print(json.dumps(event), flush=True)

    def on_progress(steps, metrics):
        record = {key: float(value) for key, value in metrics.items()
                  if np.ndim(value) == 0 and np.isfinite(value)}
        with (output_dir/'training-metrics.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(dict(steps=int(steps), metrics=record), allow_nan=False)+'\n')

    ppo['policy_params_fn'] = on_policy
    write_json(output_dir/'effective-config.json', dict(arm=arm, seed=seed,
        ppo={key: value for key, value in ppo.items() if key != 'policy_params_fn'},
        env=generated_config, initial_array_digest=initial_digest,
        reward=manifest['arms'][arm], run_evaluation_reward='Common unchanged base task; not training objective'))
    write_json(output_dir/'status.json', dict(phase='starting', arm=arm, seed=seed, steps=0))
    _, params, metrics = train_student(env=env, eval_env=eval_env, env_cfg=cfg,
        ppo_params=ppo, progress_fn=on_progress, params=initial_params)
    if latest['steps'] != manifest['steps_per_run']:
        raise ValueError('Actual completed steps differ from the frozen matched budget')
    host = jax.device_get(params)
    final_descriptor = dict(descriptor, policy=policy_from_brax(host[:2], template))
    checkpoint = output_dir/'checkpoint'
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(checkpoint, host)
    recovered = load(str(checkpoint))
    if tree_array_digest(recovered) != tree_array_digest(host):
        raise ValueError('Experimental checkpoint save/load changed parameters')
    from force_teacher.contract import tree_digest
    final_descriptor['metadata'] = final_policy_metadata(descriptor.get('metadata', {}), arm, seed,
        latest['steps'], tree_digest(checkpoint), initial_digest, manifest['arms'][arm])
    write_json(output_dir/'descriptor.json', final_descriptor)
    result = dict(phase='completed', arm=arm, seed=seed, steps=latest['steps'],
        elapsed_seconds=time.monotonic()-start, initial_array_digest=initial_digest,
        ppo_initial_array_digest=saved_initial, final_array_digest=tree_array_digest(host),
        actor_changed=tree_array_digest(host[1]) != tree_array_digest(initial_params[1]),
        normalizer_drift=float(normalizer_max_abs_change(initial_params[0], host[0])),
        descriptor_sha256=hashlib.sha256((output_dir/'descriptor.json').read_bytes()).hexdigest(),
        devices=[str(device) for device in jax.devices()], checkpoint_reload_exact=True)
    write_json(output_dir/'status.json', result)
    print(json.dumps(result), flush=True)
    return result
