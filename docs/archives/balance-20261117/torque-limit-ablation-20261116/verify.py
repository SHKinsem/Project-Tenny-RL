import hashlib
import ast
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tarfile
import time
os.environ['JAX_PLATFORMS']='cpu'
os.environ['CUDA_VISIBLE_DEVICES']=''
import jax
import numpy as np
from force_teacher.contract import tree_digest
from hybrid.core import source_fingerprint
from legwheel.deployment.extract import load
from experiments.reward_ablation.train import tree_array_digest
from native_eval.brax_policy import policy_from_brax
from native_eval.legwheel import policy_template
from force_teacher.env import student_config
from experiments.torque_limit_ablation.env import TorqueLimitEnv
from mujoco_playground import wrapper

started=time.monotonic()
work=Path('$HOME/projects/tenny-torque-limit-ablation-20261116')
local=Path('${REPO}/tasks/results/torque-limit-ablation-20261116')
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest=read(work/'inputs/manifest2.json')
matrix=read(work/'matrix2/status.json')
assert matrix['phase']=='completed' and len(matrix['completed'])==3
assert source_fingerprint()==manifest['source_identity']['base']
for path,expected in manifest['protected_inputs']['files'].items():assert sha(path)==expected,path
for path,expected in manifest['protected_inputs']['checkpoint_trees'].items():assert tree_digest(path)==expected,path
runtime=read(work/'inputs/runtime-freeze.json')
for path,expected in runtime.items():assert sha(path)==expected,path
for path,expected in manifest['source_identity']['experiment_files'].items():assert sha(work/path)==expected,path
original_assets=Path(read('$HOME/projects/tenny-reward-ablation-20261116/inputs/initial-descriptor.json')['xml_path']).parent
copied_asset_count=0
for path in original_assets.rglob('*'):
    if path.is_file():
        assert sha(path)==sha(work/'assets'/path.relative_to(original_assets)),str(path)
        copied_asset_count+=1
native_sources={str(path):sha(path) for path in (work/'native/src').rglob('*')
    if path.is_file() and (path.suffix in ('.cpp','.hpp','.py') or path.name=='CMakeLists.txt')}
checks=[]
actual_env=TorqueLimitEnv('baseline',student_config(),leg_limit_nm=2.)
assert np.array_equal(actual_env.mj_model.actuator_ctrlrange[:2],[[-2.,2.]]*2)
assert np.array_equal(actual_env.mj_model.actuator_forcerange[:2],[[-2.,2.]]*2)
assert np.array_equal(np.asarray(actual_env._torque_scales),np.asarray([1.,1.,.0295,.0295],dtype=np.float32))
wrapped=wrapper.wrap_for_brax_training(actual_env,episode_length=manifest['ppo']['episode_length'],action_repeat=1)
chain=[]
for _ in range(10):
    values=vars(wrapped)
    chain.append(dict(type=type(wrapped).__name__,episode_length=values.get('episode_length')))
    wrapped=values.get('env',values.get('_env'))
    if wrapped is None:break
assert any(item['type']=='EpisodeWrapper' and item['episode_length']==3000 for item in chain)
public_path=local.parents[2]/'experiments/torque_limit_ablation/env.py'
def without_metadata_update(path):
    tree=ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node,ast.FunctionDef) and node.name=='__init__':
            for i,statement in enumerate(node.body):
                if (isinstance(statement,ast.Expr) and isinstance(statement.value,ast.Call)
                    and isinstance(statement.value.func,ast.Attribute) and statement.value.func.attr=='update'
                    and ast.unparse(statement.value.func.value)=='self.policy_metadata'):
                    node.body[i]=ast.Pass()
    return ast.dump(tree)
assert without_metadata_update(public_path)==without_metadata_update(work/'experiments/torque_limit_ablation/env.py')
spec=importlib.util.spec_from_file_location('experiments.torque_limit_ablation.checked_public_env',public_path)
public_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(public_module)
for limit in (1.,2.):
    public_env=public_module.TorqueLimitEnv('baseline',student_config(),leg_limit_nm=limit)
    for field in ('torque_limits_nm','force_scales_n','torque_normalization_scales'):
        assert public_env.policy_metadata[field]==public_env.actuator_contract[field]
for old in manifest['reuse_limit1']:
    seed=old['seed'];short=Path(old['source_run']);long=work/f'matrix2/seed-{seed}/baseline'
    assert tree_digest(short/'checkpoint')==old['checkpoint_sha256']
    assert sha(short/'descriptor.json')==old['descriptor_sha256']
    left,right=read(short/'effective-config.json'),read(long/'effective-config.json')
    for field in ('ppo','env','reward','initial_array_digest','arm','seed'):assert left[field]==right[field],field
    assert right['actuator_contract']==manifest['actuator_contract']
    status=read(long/'status.json')
    assert status['steps']==2097152 and status['phase']=='completed'
    assert status['initial_array_digest']==status['ppo_initial_array_digest']==manifest['initial_array_digest']
    assert status['normalizer_drift']==0 and status['actor_changed'] and status['checkpoint_reload_exact']
    assert sha(long/'descriptor.json')==status['descriptor_sha256']
    params=load(str(long/'checkpoint'))
    assert tree_array_digest(params)==status['final_array_digest']
    leaves=jax.tree.leaves(jax.device_get(params))
    assert all(np.isfinite(np.asarray(value)).all() for value in leaves)
    exported=policy_from_brax(params[:2],policy_template(manifest['policy_metadata'],manifest['ppo']))
    assert exported==read(long/'descriptor.json')['policy']
    dest=local/f'runs/limit2-seed-{seed}';dest.mkdir(parents=True,exist_ok=True)
    for filename in ('status.json','effective-config.json','progress.jsonl','training-metrics.jsonl'):
        shutil.copy2(long/filename,dest/filename)
    checks.append(dict(seed=seed,steps=status['steps'],elapsed_seconds=status['elapsed_seconds'],
        all_parameter_leaves_finite=True,leaf_count=len(leaves),descriptor_matches_loaded_actor=True,
        checkpoint_sha256=tree_digest(long/'checkpoint')))

bundle=read(work/'comparison/results.json')
assert len(bundle['cases'])==len(bundle['parity'])==40
assert all(p['exact_prefix_match'] for p in bundle['parity'])
usage=[]
for case in bundle['cases']:
    identity=case['policy_id'];state_id=case['initial_state_id'];trace=case['trace']
    if identity=='initial-limit1':source=Path('$HOME/projects/tenny-reward-ablation-20261116/evaluations/initial')
    elif identity.startswith('limit1-seed-'):source=Path('$HOME/projects/tenny-horizon-ablation-20261116/evaluations')/identity.replace('limit1','h30')
    else:source=work/'evaluations'/identity
    n=len(trace['t_s'])-1
    world=next(i for i,item in enumerate(bundle['initial_states']) if item['id']==state_id)
    with np.load(source/'policy-0-long.npz') as arrays:
        commands=arrays[f'world_{world}_ctrl'][:n,:2]
        actions=arrays[f'world_{world}_action'][:n,:2]
        native=case.get('native_first_episode_metrics',{})
        if 'reward' in native:assert abs(native['reward']-sum(trace['reward']))<1e-7
        # Old baseline clipping was already independently checked across all
        # physical substeps in the preceding actuator audit, not filled by guess.
        clip=native.get('hybrid/saturation_fraction')
        if identity.startswith('limit1') or identity=='initial-limit1':
            prior=read('${REPO}/tasks/results/torque-limit-audit-20261116/results.json')
            old_id='initial' if identity=='initial-limit1' else identity.replace('limit1','h30')
            matched=next(c for c in prior['cases'] if c['policy_id']==old_id and c['initial_state_id']==state_id)
            clip=matched['validation']['native_saturation_metric_sum']
        usage.append(dict(policy_id=identity,initial_state_id=state_id,
            control_steps=n,physics_steps=4*n,native_raw_clipping_metric_sum=clip,
            max_abs_leg_command_nm=np.max(np.abs(commands),axis=0).tolist(),
            sampled_command_over_original_1nm_fraction=np.mean(np.abs(commands)>1.,axis=0).tolist(),
            held_force_action_ge95_fraction=np.mean(np.abs(actions)>=.95,axis=0).tolist(),
            scope='Command magnitude: 100Hz final-substep samples. Action: held controls. Raw clipping: full-physics native aggregate; nonzero is not a duty ratio.'))
target=work/'comparison'
(target/'actuator-usage.json').write_text(json.dumps(dict(schema_version=1,cases=usage),indent=2)+'\n')
archive=target/'actual-experiment-source.tar.gz'
with tarfile.open(archive,'w:gz') as stream:
    added=set()
    for path in runtime:
        item=Path(path)
        if item.suffix in ('.py','.cpp','.txt','.xml'):
            stream.add(item,arcname=item.relative_to(work).as_posix())
            added.add(str(item))
    for path in native_sources:
        if path not in added:
            stream.add(path,arcname=Path(path).relative_to(work).as_posix())
    stream.add(work/'inputs/manifest2.json',arcname='inputs/manifest2.json')
result=dict(reused_limit1_verified=True,new_effective_configs_matched=True,all_initial_arrays_exact=True,
    all_normalizers_frozen=True,all_3_new_checkpoints_finite=True,all_checkpoint_reloads_exact=True,
    protected_files_unchanged=len(manifest['protected_inputs']['files']),
    protected_checkpoint_trees_unchanged=len(manifest['protected_inputs']['checkpoint_trees']),
    frozen_runtime_files_unchanged=len(runtime),base_source_fingerprint_unchanged=source_fingerprint(),
    copied_source_assets_identical=copied_asset_count,native_build_source_sha256=native_sources,
    experiment_fingerprint=manifest['experimental_source_fingerprint'],
    formal_new_environment_steps=6291456,formal_matrix_elapsed_seconds=matrix['elapsed_seconds'],
    separate_smoke_steps=8192,smoke_elapsed_seconds=read(work/'smoke/status.json')['elapsed_seconds'],
    exact_common_prefix_cases=40,reached_30_seconds=sum(c['standing']['reached_horizon'] for c in bundle['cases']),
    actual_frozen_env_wrapper_chain=chain,public_metadata_correction_only=True,
    public_limit1_limit2_metadata_verified=True,source_archive_sha256=sha(archive),runs=checks,
    verification_seconds=time.monotonic()-started)
(target/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
shutil.copytree(work/'inputs',local/'inputs',dirs_exist_ok=True)
for filename in ('actuator-usage.json','verification.json','actual-experiment-source.tar.gz'):
    shutil.copy2(target/filename,local/'comparison'/filename)
print(json.dumps(result,indent=2),flush=True)
