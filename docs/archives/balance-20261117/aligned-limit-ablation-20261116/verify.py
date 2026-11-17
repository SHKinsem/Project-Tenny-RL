import copy
import hashlib
import json
import math
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
from force_teacher.env import student_config
from hybrid.core import source_fingerprint
from legwheel.deployment.extract import load
from native_eval.brax_policy import policy_from_brax
from mujoco_playground import wrapper
from experiments.aligned_limit_ablation.env import AlignedLimitEnv
from experiments.aligned_limit_ablation.export import policy_template
from experiments.reward_ablation.train import tree_array_digest

started=time.monotonic()
work=Path('$HOME/projects/tenny-aligned-limit-ablation-20261116')
local=Path('${REPO}/tasks/results/aligned-limit-ablation-20261116')
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifests={limit:read(work/f'inputs/manifest-limit{limit}.json') for limit in (1,2)}
manifest=manifests[1];matrix=read(work/'matrix/status.json')
assert matrix['phase']=='completed' and matrix['total_runs']==6
assert matrix['total_environment_steps']==12582912
assert source_fingerprint()==manifest['source_identity']['base_source_fingerprint']
for field in ('ppo','env_config','arms','initial_checkpoint','initial_array_digest','initial_states'):
    assert manifests[1][field]==manifests[2][field],field
assert manifests[1]['actuator_contract']['policy_leg_force_scales']==manifests[2]['actuator_contract']['policy_leg_force_scales']
for path,expected in manifest['protected_inputs']['files'].items():assert sha(path)==expected,path
for path,expected in manifest['protected_inputs']['checkpoint_trees'].items():assert tree_digest(path)==expected,path
runtime=read(work/'inputs/runtime-freeze.json')
for path,expected in runtime.items():assert sha(path)==expected,path
for path,expected in manifest['source_identity']['experiment_files'].items():assert sha(work/path)==expected,path
anchor=read(work/'inputs/anchor-bc.json')['policy'];anchor['output_activation']='identity'
assert sha(work/'inputs/anchor-bc.json')==manifest['source_identity']['anchor_descriptor_sha256']
alignment=read(work/'inputs/alignment-validation.json')
assert all(value==0 for value in alignment['deterministic_pair_max_abs_errors'].values())
assert alignment['random_common_request_verified'] and alignment['common_request_max_abs_error']==0
wrappers={};runs=[]
for limit in (1,2):
    environment=AlignedLimitEnv('baseline',student_config(),leg_limit_nm=float(limit))
    assert np.array_equal(environment.mj_model.actuator_ctrlrange[:2],[[-limit,limit]]*2)
    assert np.array_equal(environment.mj_model.actuator_forcerange[:2],[[-limit,limit]]*2)
    assert np.array_equal(np.asarray(environment._torque_scales),np.asarray([1.,1.,.0295,.0295],dtype=np.float32))
    current=wrapper.wrap_for_brax_training(environment,episode_length=3000,action_repeat=1)
    chain=[]
    for _ in range(10):
        values=vars(current);chain.append(dict(type=type(current).__name__,episode_length=values.get('episode_length')))
        current=values.get('env',values.get('_env'))
        if current is None:break
    assert any(c['type']=='EpisodeWrapper' and c['episode_length']==3000 for c in chain)
    wrappers[str(limit)]=chain
    for seed in (0,1,2):
        path=work/f'matrix/seed-{seed}/limit{limit}'
        status=read(path/'status.json');effective=read(path/'effective-config.json')
        assert status['phase']=='completed' and status['steps']==2097152
        assert status['initial_array_digest']==status['ppo_initial_array_digest']==manifest['initial_array_digest']
        assert status['normalizer_drift']==0 and status['actor_changed'] and status['checkpoint_reload_exact']
        assert sha(path/'descriptor.json')==status['descriptor_sha256']
        assert effective['ppo']==dict(manifest['ppo'],seed=seed)
        assert effective['env']==manifest['env_config'] and effective['reward']==manifest['arms']['baseline']
        assert effective['actuator_contract']==manifests[limit]['actuator_contract']
        params=load(str(path/'checkpoint'))
        assert tree_array_digest(params)==status['final_array_digest']
        leaves=jax.tree.leaves(jax.device_get(params))
        assert all(np.isfinite(np.asarray(value)).all() for value in leaves)
        descriptor=read(path/'descriptor.json')
        generated=policy_from_brax(params[:2],policy_template(manifests[limit]['policy_metadata'],manifest['ppo']))
        assert generated==descriptor['policy']
        assert descriptor['policy']['leg_anchor']==anchor
        evaluation=read(work/f'inputs/evaluation-descriptors/limit{limit}-seed-{seed}.json')
        assert evaluation['metadata']['checkpoint_sha256']==tree_digest(path/'checkpoint')
        dest=local/f'runs/limit{limit}-seed-{seed}';dest.mkdir(parents=True,exist_ok=True)
        for name in ('status.json','effective-config.json','progress.jsonl','training-metrics.jsonl'):
            shutil.copy2(path/name,dest/name)
        runs.append(dict(limit=limit,seed=seed,steps=status['steps'],elapsed_seconds=status['elapsed_seconds'],
            all_parameter_leaves_finite=True,parameter_leaf_count=len(leaves),frozen_anchor_exact=True,
            checkpoint_sha256=tree_digest(path/'checkpoint')))
bundle=read(work/'comparison/results.json')
assert len(bundle['cases'])==len(bundle['parity'])==40
assert all(p['exact_prefix_match'] for p in bundle['parity'])
usage=[]
for case in bundle['cases']:
    identity=case['policy_id'];state_id=case['initial_state_id'];trace=case['trace']
    limit=int(identity.split('-')[1][-1] if identity.startswith('initial') else identity.split('-')[0][-1])
    n=len(trace['t_s'])-1;world=next(i for i,s in enumerate(bundle['initial_states']) if s['id']==state_id)
    path=work/f'evaluations/{identity}/policy-0-long.npz'
    with np.load(path) as arrays:
        actions=arrays[f'world_{world}_action'][:n,:2]
        commands=arrays[f'world_{world}_ctrl'][:n,:2]
    dt=np.diff(trace['t_s'])
    origin=trace['base_position_m'][0];reference=trace['metadata']['reference_pitch_rad']
    in_band=[abs(p-reference)<=math.radians(10.) and math.hypot(x[0]-origin[0],x[1]-origin[1])<=.05
             and math.hypot(v[0],v[1])<=.1 for p,x,v in zip(trace['pitch_rad'],trace['base_position_m'],trace['base_velocity_m_s'])]
    qualifying=np.asarray([trace['t_s'][i]>=1. and in_band[i] and in_band[i+1] for i in range(n)])
    qualified_seconds=float(np.sum(dt[qualifying]))
    native=case['native_first_episode_metrics']
    assert abs(native['reward']-sum(trace['reward']))<1e-7
    usage.append(dict(policy_id=identity,initial_state_id=state_id,leg_limit_nm=limit,
        control_steps=n,physics_steps=n*4,qualified_seconds=qualified_seconds,
        native_raw_clipping_metric_sum=native['hybrid/saturation_fraction'],
        max_abs_leg_command_nm=np.max(np.abs(commands),axis=0).tolist(),
        request_over_low_Fs1_fraction=np.mean(np.abs(actions)>.5,axis=0).tolist(),
        request_over_own_force_capacity_fraction=np.mean(np.abs(actions)>.5*limit,axis=0).tolist(),
        sampled_command_over_1nm_fraction=np.mean(np.abs(commands)>1.,axis=0).tolist(),
        qualified_request_over_low_capacity_fraction=(np.sum(dt[qualifying,None]*(np.abs(actions[qualifying])>.5),axis=0)/qualified_seconds).tolist() if qualified_seconds else None,
        qualified_sampled_command_over_1nm_fraction=(np.sum(dt[qualifying,None]*(np.abs(commands[qualifying])>1.),axis=0)/qualified_seconds).tolist() if qualified_seconds else None,
        scope='Actions are common Fs2 pre-cap requests held for 10ms. Command magnitudes are last physics-substep samples, not full-400Hz amplitudes. Native clipping is a max-over-substeps sum, not a duty ratio.'))
original_assets=Path('$HOME/projects/tenny-torque200/src/assets')
assets=0
for path in original_assets.rglob('*'):
    if path.is_file():assert sha(path)==sha(work/'assets'/path.relative_to(original_assets));assets+=1
target=work/'comparison'
(target/'actuator-usage.json').write_text(json.dumps(dict(schema_version=1,cases=usage),indent=2)+'\n')
native_sources={str(p):sha(p) for p in work.joinpath('native/src').rglob('*')
    if p.is_file() and (p.suffix in ('.cpp','.hpp','.py') or p.name=='CMakeLists.txt')}
archive=target/'actual-experiment-source.tar.gz'
with tarfile.open(archive,'w:gz') as stream:
    files={str(p) for p in map(Path,runtime) if p.suffix in ('.py','.cpp','.hpp','.xml')}
    files.update(native_sources)
    for path in sorted(files):stream.add(path,arcname=Path(path).relative_to(work).as_posix())
    for name in ('manifest-limit1.json','manifest-limit2.json','alignment-validation.json'):
        stream.add(work/'inputs'/name,arcname='inputs/'+name)
result=dict(both_arms_newly_trained=True,reused_runs=0,formal_runs=6,formal_environment_steps=12582912,
    formal_matrix_elapsed_seconds=matrix['elapsed_seconds'],all_effective_configs_exact=True,
    all_initial_arrays_exact=True,all_normalizers_frozen=True,all_6_checkpoints_finite_reload_exact=True,
    frozen_anchor_unchanged=True,zero_step_physical_alignment=alignment,
    actual_wrapper_chains=wrappers,protected_files_unchanged=len(manifest['protected_inputs']['files']),
    protected_old_checkpoint_trees_unchanged=len(manifest['protected_inputs']['checkpoint_trees']),
    frozen_runtime_files_unchanged=len(runtime),copied_assets_identical=assets,
    base_source_fingerprint_unchanged=source_fingerprint(),experiment_fingerprint=manifest['experimental_source_fingerprint'],
    exact_common_prefix_cases=40,reached_30_seconds=sum(c['standing']['reached_horizon'] for c in bundle['cases']),
    separate_smoke_steps=8192,smoke_elapsed_seconds=read(work/'smoke/status.json')['elapsed_seconds'],
    native_source_sha256=native_sources,source_archive_sha256=sha(archive),runs=runs,
    verification_seconds=time.monotonic()-started)
(target/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
shutil.copytree(work/'inputs',local/'inputs',dirs_exist_ok=True)
for name in ('verification.json','actuator-usage.json','actual-experiment-source.tar.gz'):
    shutil.copy2(target/name,local/'comparison'/name)
print(json.dumps(result,indent=2),flush=True)
