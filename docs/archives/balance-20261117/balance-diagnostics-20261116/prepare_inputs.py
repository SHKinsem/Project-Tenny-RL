"""Freeze existing policy artifacts and cached resets; never sample or train."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['JAX_DISABLE_JIT'] = 'true'

import mujoco
import numpy as np

from native_eval import Evaluator
from native_eval.brax_policy import policy_from_brax
from native_eval.legwheel import descriptor_from_run, policy_descriptor, policy_template
from legwheel.deployment.extract import load
from force_teacher.contract import tree_digest

start = time.monotonic()
root = Path('$HOME/projects/tenny-torque200')
run = root / 'artifacts/console/leg_force_bc/20261115T172533.823984Z_0a516875'
out = Path('$HOME/projects/tenny-balance-diagnostics-20261116/inputs')
out.mkdir(parents=True, exist_ok=True)
read = lambda file: json.loads(Path(file).read_text())
sha = lambda file: hashlib.sha256(Path(file).read_bytes()).hexdigest()
write = lambda file, value: Path(file).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
report, ppo, cfg, pre = [read(run / name) for name in (
    'report.json', 'ppo_config.json', 'env_config.json', 'pretraining-report.json')]
receipt_path = Path(pre['bootstrap_receipt'])
receipt = read(receipt_path)
cache = receipt_path.parent
protected = {str(path): sha(path) for path in (
    run/'env_config.json', run/'ppo_config.json', run/'policy_metadata.json', run/'policy.cpp.txt',
    run/'report.json', Path(pre['initial_params_path']), cache/'params.msgpack', cache/'policy.cpp.txt',
    cache/'policy_metadata.json', receipt_path)}
checkpoints = sorted(path for path in (run / 'checkpoints').iterdir() if path.is_dir())
checkpoint_hashes = {str(path): tree_digest(path) for path in checkpoints}
if checkpoint_hashes[report['checkpoint']] != report['checkpoint_sha256']:
    raise ValueError('Final checkpoint differs from its validated report')
if sha(pre['initial_params_path']) != pre['initial_params_sha256'] or sha(cache/'params.msgpack') != pre['initial_params_sha256']:
    raise ValueError('Initializer cached export is not bound to this run initializer')
for name in ('params.msgpack', 'policy.cpp.txt', 'policy_metadata.json'):
    if sha(cache/name) != receipt['files'][name]:
        raise ValueError('Initializer artifact differs from its verification receipt: ' + name)

final_descriptor = descriptor_from_run(run, root/'src')
cache_meta = read(cache/'policy_metadata.json')
if cache_meta['source_fingerprint'] != final_descriptor['metadata']['source_fingerprint']:
    raise ValueError('Initializer and final task source differ')
initial_descriptor = dict(final_descriptor)
initial_descriptor['policy'] = policy_descriptor(cache/'policy.cpp.txt', cache_meta, ppo)
initial_descriptor['metadata'] = dict(final_descriptor['metadata'],
    checkpoint_sha256=None, export_sha256=sha(cache/'policy.cpp.txt'),
    initializer_params_sha256=pre['initial_params_sha256'], kind='run_initializer')
descriptors = [('initial', '模仿初始化', initial_descriptor,
                dict(kind='verified_initializer', params_sha256=pre['initial_params_sha256']))]
template = policy_template(read(run/'policy_metadata.json'), ppo)
for number, checkpoint in enumerate(checkpoints[:-1], 1):
    params = load(str(checkpoint))
    descriptor = dict(final_descriptor)
    descriptor['policy'] = policy_from_brax(params[:2], template)
    descriptor['metadata'] = dict(final_descriptor['metadata'], checkpoint_sha256=checkpoint_hashes[str(checkpoint)],
        export_sha256=None, kind='saved_checkpoint', checkpoint_steps=int(checkpoint.name))
    descriptors.append((f'mid-{number}', f'中间 {int(checkpoint.name):,} 步', descriptor,
                        dict(kind='saved_checkpoint', steps=int(checkpoint.name), checkpoint_sha256=checkpoint_hashes[str(checkpoint)])))
descriptors.append(('final', f'最终 {report["steps"]:,} 步', final_descriptor,
                    dict(kind='validated_export', steps=report['steps'], checkpoint_sha256=report['checkpoint_sha256'])))

# Compare final exported inference with the actual checkpoint, without compiling a policy or using a GPU.
final_params = load(report['checkpoint'])
checkpoint_descriptor = dict(final_descriptor, policy=policy_from_brax(final_params[:2], template))
reference = Path('$HOME/projects/tenny-cpu-gpu-eval-20261115/result')
with np.load(reference/'common-policy.npz', allow_pickle=False) as archive:
    observations = archive['observation'][:256]
export_error = float(np.max(np.abs(Evaluator(final_descriptor).policy_actions(observations) -
                                    Evaluator(checkpoint_descriptor).policy_actions(observations))))
if export_error > 1e-5:
    raise ValueError('Final exported inference differs from actual checkpoint')

contract = read(reference/'contract.json')
if contract['source_fingerprint'] != final_descriptor['metadata']['source_fingerprint'] or contract['config'] != cfg:
    raise ValueError('Cached reset source/configuration differs from this run')
with np.load(reference/'initial.npz', allow_pickle=False) as archive:
    arrays = {key: archive[key].copy() for key in archive.files}
nominal = dict(qpos=arrays['qpos'][0].tolist(), qvel=arrays['qvel'][0].tolist(), command=[0.,0.,0.,0.],
               extra={name: [float(arrays[name][0])] for name in ('position_start','sinyaw_start','sin_balance_zero')})
model = mujoco.MjModel.from_xml_path(final_descriptor['xml_path'])
body = model.body(final_descriptor['task']['config']['root_body']).id
root_joint = int(model.body_jntadr[body])
root_q = int(model.jnt_qposadr[root_joint])
states = [dict(id='nominal', label='标称重置', state=nominal)]
for delta, label in ((.01,'plus'),(-.01,'minus')):
    state = json.loads(json.dumps(nominal))
    perturbation = np.array([np.cos(delta/2),0.,np.sin(delta/2),0.])
    rotated = np.empty(4)
    mujoco.mju_mulQuat(rotated, np.asarray(nominal['qpos'][root_q+3:root_q+7]), perturbation)
    state['qpos'][root_q+3:root_q+7] = rotated.tolist()
    states.append(dict(id='pitch-'+label, label=f'固定俯仰扰动 {delta:+.2f} rad', state=state))

policies = []
normalizers = {}
for identity, label, descriptor, source in descriptors:
    write(out/f'{identity}.json', descriptor)
    policies.append(dict(id=identity, label=label, descriptor=f'{identity}.json', source=source))
    normalizers[identity] = dict(mean=descriptor['policy']['mean'], std=descriptor['policy']['std'])
normalizer_changes = {identity: max(float(np.max(np.abs(np.asarray(values[key])-np.asarray(normalizers['initial'][key]))))
                                   for key in ('mean','std')) for identity, values in normalizers.items()}
evidence = dict(run_id=run.name, protected_files=protected, checkpoint_trees=checkpoint_hashes,
    initializer_receipt_sha256=sha(receipt_path), cached_reset_sha256=sha(reference/'initial.npz'),
    native_extension_sha256=sha(next(Path('$HOME/projects/tenny-native-eval/python/native_eval').glob('*.so'))),
    final_export_checkpoint_action_max_abs=export_error, normalizer_max_changes=normalizer_changes,
    preparation_seconds=time.monotonic()-start,
    initial_state_scope='Nominal reset from cached real production reset; local pitch perturbations retain the same equilibrium/position/yaw reference.')
manifest = dict(schema_version=1, seed=0, policies=policies, initial_states=states,
    scope='Nominal standing with three fixed initial states; deterministic CPU/native evaluation, no training.',
    conditions=dict(command=[0.,0.,0.,0.], noise=False, pushes=False, domain_randomization=False,
                    physics_hz=400, control_hz=100, time_limit_override='Evaluation only; task and source unchanged.'),
    preparation_evidence=evidence)
write(out/'manifest.json', manifest)
write(out/'protected-inputs.json', dict(files=protected, checkpoint_trees=checkpoint_hashes))
print(json.dumps(dict(manifest=str(out/'manifest.json'), policies=len(policies), initial_states=len(states),
                      preparation_seconds=evidence['preparation_seconds'], export_error=export_error,
                      normalizer_max_changes=normalizer_changes)), flush=True)
