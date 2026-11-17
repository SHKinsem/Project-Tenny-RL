import copy
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import xml.etree.ElementTree as ET
from experiments.aligned_limit_ablation.contract import aligned_contract,aligned_descriptor
from experiments.aligned_limit_ablation.native_patch import patch_sources

repo=Path('${REPO}')
work=Path('$HOME/projects/tenny-aligned-limit-ablation-20261116')
old=Path('$HOME/projects/tenny-reward-ablation-20261116')
horizon=Path('$HOME/projects/tenny-horizon-ablation-20261116')
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
if (work/'inputs/manifest-limit1.json').exists():raise ValueError('Preserve the prepared experiment')
baseline=read(horizon/'inputs/manifest30.json')
for path,expected in baseline['protected_inputs']['files'].items():assert sha(path)==expected,path
archive=old/'comparison/actual-training-source.tar.gz'
assert sha(archive)==read(old/'comparison/matrix-verification.json')['frozen_training_source_sha256']
frozen=read(old/'matrix-v2/frozen-inputs.json')['code_sha256']
module=work/'experiments/reward_ablation';module.mkdir(parents=True,exist_ok=True)
with tarfile.open(archive) as stream:
    for name,expected in frozen.items():
        data=stream.extractfile(name).read();assert hashlib.sha256(data).hexdigest()==expected
        (module/name).write_bytes(data)
trainer=(module/'train.py').read_text()
replacements=[
    ('from force_teacher.ppo import train_student, normalizer_max_abs_change','from experiments.aligned_limit_ablation.ppo import train_student, normalizer_max_abs_change'),
    ('from native_eval.legwheel import policy_template','from experiments.aligned_limit_ablation.export import policy_template'),
    ('from .legwheel_env import RewardExperimentEnv','from experiments.aligned_limit_ablation.env import AlignedLimitEnv as RewardExperimentEnv'),
    ("root_body=descriptor['task']['config']['root_body'])","root_body=descriptor['task']['config']['root_body'], leg_limit_nm=manifest['leg_limit_nm'])"),
    ("reward=manifest['arms'][arm], run_evaluation_reward=","reward=manifest['arms'][arm], actuator_contract=manifest['actuator_contract'], run_evaluation_reward=")]
for before,after in replacements:
    assert trainer.count(before)==1,before
    trainer=trainer.replace(before,after)
(module/'train.py').write_text(trainer)
for package in ('aligned_limit_ablation','torque_limit_ablation','balance_diagnostics'):
    shutil.copytree(repo/'experiments'/package,work/'experiments'/package,dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__'))
for filename in ('evaluation.py','report.py'):shutil.copy2(repo/'experiments/reward_ablation'/filename,module/filename)
ppo_path=Path('$HOME/projects/tenny-torque200/src/force_teacher/ppo.py')
ppo=ppo_path.read_text()
before='from force_teacher.networks import make_ppo_networks'
assert ppo.count(before)==1
(work/'experiments/aligned_limit_ablation/ppo.py').write_text(ppo.replace(before,'from experiments.aligned_limit_ablation.networks import make_ppo_networks'))
native=work/'native'
shutil.copytree('$HOME/projects/tenny-native-eval/src',native/'src',dirs_exist_ok=True,
    ignore=shutil.ignore_patterns('__pycache__','build','results','.git'))
keys=('cpp/legwheel_task.cpp','cpp/policy.cpp','cpp/bindings.cpp','include/native_eval/runtime.hpp')
native_original={name:sha(native/'src'/name) for name in keys}
patched=patch_sources({name:(native/'src'/name).read_text() for name in keys})
for name,text in patched.items():(native/'src'/name).write_text(text)
shutil.copytree(native/'src/python',native/'python',dirs_exist_ok=True)
base=read(old/'inputs/initial-descriptor.json')
assets=Path(base['xml_path']).parent
shutil.copytree(assets,work/'assets',dirs_exist_ok=True)
xml=ET.parse(base['xml_path'])
for node in xml.getroot().find('actuator'):
    if node.get('name') in ('thigh_left_motor','thigh_right_motor'):
        assert node.tag=='motor';node.set('ctrlrange','-2 2');node.set('forcerange','-2 2')
xml2=work/'assets/torque-limit-2.xml';xml.write(xml2,encoding='utf-8')
write(work/'inputs/anchor-bc.json',base)
experiment_files={str(p.relative_to(work)):sha(p) for package in ('aligned_limit_ablation','torque_limit_ablation','reward_ablation')
    for p in (work/'experiments'/package).glob('*.py')}
identity=dict(base_source_fingerprint=baseline['policy_metadata']['source_fingerprint'],
    experiment_files=experiment_files,native_original=native_original,
    native_patched={name:sha(native/'src'/name) for name in keys},xml2_sha256=sha(xml2),
    anchor_descriptor_sha256=sha(work/'inputs/anchor-bc.json'))
fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
for limit in (1,2):
    descriptor=aligned_descriptor(base,float(limit),base['xml_path'] if limit==1 else xml2)
    descriptor['metadata'].update(source_fingerprint=fingerprint,base_source_fingerprint=identity['base_source_fingerprint'],
        model_run_id=f'aligned-limit-v1/initial-limit{limit}',kind='aligned_physical_initializer')
    write(work/f'inputs/initial-limit{limit}.json',descriptor)
    manifest=copy.deepcopy(baseline)
    manifest.update(kind='aligned_torque_limit_ablation_v1',leg_limit_nm=float(limit),training_episode_seconds=30,
        torque_normalization_scales=[1.,1.,.0295,.0295],
        actuator_contract=aligned_contract(float(limit),descriptor['task']['config']['jmax']),
        base_descriptor=str(work/f'inputs/initial-limit{limit}.json'),source_identity=identity,
        experimental_source_fingerprint=fingerprint,reuse_limit1=[],both_arms_retrained=True,
        initialization_alignment='common_Fs2_request_with_frozen_BC_mean_anchor_v1')
    manifest.pop('reuse_six_seconds',None)
    manifest['policy_metadata'].update(source_fingerprint=fingerprint,base_source_fingerprint=identity['base_source_fingerprint'],
        torque_limits_nm=manifest['actuator_contract']['torque_limits_nm'],
        force_scales_n=manifest['actuator_contract']['force_scales_n'],experimental_only=True,
        policy_leg_force_scales=manifest['actuator_contract']['policy_leg_force_scales'])
    write(work/f'inputs/manifest-limit{limit}.json',manifest)
    write(work/f'inputs/initial-plan-limit{limit}.json',dict(schema_version=1,seed=0,
        policies=[dict(id=f'initial-limit{limit}',label=f'initial-limit{limit}',descriptor=str(work/f'inputs/initial-limit{limit}.json'),
            source=dict(kind='aligned_BC_initializer',leg_limit_nm=limit))],initial_states=baseline['initial_states']))
write(work/'inputs/source-identity.json',identity)
local=repo/'tasks/results/aligned-limit-ablation-20261116'
shutil.copytree(work/'inputs',local/'inputs',dirs_exist_ok=True)
print(json.dumps(dict(work=str(work),fingerprint=fingerprint,formal_runs=6,
    formal_steps=6*baseline['steps_per_run'],common_force_scales=descriptor['task']['config']['policy_leg_force_scales'])),flush=True)
