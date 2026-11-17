"""Post-run checks only: saved checkpoints, configs, traces, and protected inputs."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import time

os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import jax
import numpy as np
from force_teacher.contract import tree_digest
from hybrid.core import source_fingerprint
from legwheel.deployment.extract import load
from experiments.horizon_ablation.contract import assert_only_horizon_changed

started = time.monotonic()
work = Path('$HOME/projects/tenny-horizon-ablation-20261116')
old = Path('$HOME/projects/tenny-reward-ablation-20261116')
repo = Path('${REPO}')
local = repo / 'tasks/results/horizon-ablation-20261116'
read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest = read(work / 'inputs/manifest30.json')
matrix = read(work / 'matrix30/status.json')
bundle = read(work / 'comparison/results.json')
assert matrix['phase'] == 'completed' and len(matrix['completed']) == 3
assert len(bundle['cases']) == len(bundle['parity']) == 35
assert all(row['exact_prefix_match'] for row in bundle['parity'])
assert all(not case['standing']['reached_horizon'] for case in bundle['cases'])
assert source_fingerprint() == manifest['policy_metadata']['source_fingerprint']
for path, expected in manifest['protected_inputs']['files'].items():
    assert sha(path) == expected, path
for path, expected in manifest['protected_inputs']['checkpoint_trees'].items():
    assert tree_digest(path) == expected, path
for name, expected in manifest['frozen_source_sha256'].items():
    assert sha(work / 'experiments/reward_ablation' / name) == expected, name

def array_digest(params):
    digest = hashlib.sha256()
    arrays = jax.tree.leaves(jax.device_get(params))
    for leaf in arrays:
        value = np.asarray(leaf)
        assert value.dtype.kind not in 'fc' or np.isfinite(value).all()
        digest.update(str(value.dtype).encode() + repr(value.shape).encode() + value.tobytes())
    return digest.hexdigest(), len(arrays)

records = []
for seed in manifest['seeds']:
    short = old / 'matrix-v2' / f'seed-{seed}/baseline'
    long = work / 'matrix30' / f'seed-{seed}/baseline'
    cfg6, cfg30 = read(short / 'effective-config.json'), read(long / 'effective-config.json')
    assert_only_horizon_changed(cfg6['ppo'], cfg30['ppo'])
    assert cfg30['ppo'] == dict(manifest['ppo'], seed=seed)
    for field in ('env', 'reward', 'initial_array_digest', 'arm', 'seed'):
        assert cfg6[field] == cfg30[field], (seed, field)
    old_policy = read(old / f'evaluations/baseline-seed-{seed}/results.json')['policies'][0]
    old_checkpoint = tree_digest(short / 'checkpoint')
    assert old_checkpoint == old_policy['source']['checkpoint_sha256']
    assert sha(short / 'descriptor.json') == old_policy['source']['original_training_descriptor_sha256']
    for condition, run in (('h6', short), ('h30', long)):
        status = read(run / 'status.json')
        assert status['phase'] == 'completed' and status['steps'] == manifest['steps_per_run']
        assert status['initial_array_digest'] == status['ppo_initial_array_digest'] == manifest['initial_array_digest']
        assert status['normalizer_drift'] == 0 and status['checkpoint_reload_exact'] and status['actor_changed']
        assert sha(run / 'descriptor.json') == status['descriptor_sha256']
        finite = None
        if condition == 'h30':
            digest, leaves = array_digest(load(str(run / 'checkpoint')))
            assert digest == status['final_array_digest']
            finite = dict(all_leaves_finite=True, leaf_count=leaves, array_digest=digest)
            evaluated = next(p for p in bundle['policies'] if p['id'] == f'h30-seed-{seed}')
            assert tree_digest(run / 'checkpoint') == evaluated['metadata']['checkpoint_sha256']
        dest = local / 'runs' / f'{condition}-seed-{seed}'
        dest.mkdir(parents=True, exist_ok=True)
        for filename in ('status.json', 'effective-config.json', 'progress.jsonl', 'training-metrics.jsonl'):
            shutil.copy2(run / filename, dest / filename)
        records.append(dict(condition=condition, seed=seed, steps=status['steps'],
                            elapsed_seconds=status['elapsed_seconds'], finite_check=finite,
                            reused_evaluation_checkpoint_verified=condition == 'h6'))

archive = work / 'comparison/actual-training-source.tar.gz'
with tarfile.open(archive, 'w:gz') as stream:
    for name in manifest['frozen_source_sha256']:
        stream.add(work / 'experiments/reward_ablation' / name, arcname=name)
report = dict(all_effective_configs_match=True, sole_ppo_difference='episode_length:600->3000',
    all_initial_arrays_exact=True, all_normalizers_frozen=True, all_checkpoint_reloads_exact=True,
    all_3_new_checkpoints_finite=True, reused_6s_checkpoints_match_prior_evaluation=True,
    frozen_training_files_verified=len(manifest['frozen_source_sha256']),
    source_fingerprint_unchanged=source_fingerprint(),
    protected_files_unchanged=len(manifest['protected_inputs']['files']),
    protected_checkpoint_trees_unchanged=len(manifest['protected_inputs']['checkpoint_trees']),
    common_prefix_exact_cases=35, reached_30_seconds=0, evaluated_first_episodes=35,
    new_environment_steps=3*manifest['steps_per_run'],
    new_matrix_elapsed_seconds=matrix['elapsed_seconds'],
    frozen_training_source_archive_sha256=sha(archive), run_checks=records,
    verification_seconds=time.monotonic()-started)
(work / 'comparison/verification.json').write_text(json.dumps(report, indent=2)+'\n')
shutil.copytree(work / 'inputs', local / 'inputs', dirs_exist_ok=True)
shutil.copy2(work / 'actuator-frequency-evidence.json', local / 'actuator-frequency-evidence.json')
for filename in ('verification.json', 'actual-training-source.tar.gz'):
    shutil.copy2(work / 'comparison' / filename, local / 'comparison' / filename)
print(json.dumps(report, indent=2), flush=True)
