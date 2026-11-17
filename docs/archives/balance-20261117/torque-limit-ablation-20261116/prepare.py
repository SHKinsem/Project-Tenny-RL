import copy
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import xml.etree.ElementTree as ET

from experiments.torque_limit_ablation.contract import actuator_contract,experimental_descriptor,patch_frozen_trainer
from experiments.torque_limit_ablation.native_patch import patch_legwheel_source

repo=Path('${REPO}')
work=Path('$HOME/projects/tenny-torque-limit-ablation-20261116')
old=Path('$HOME/projects/tenny-reward-ablation-20261116')
horizon=Path('$HOME/projects/tenny-horizon-ablation-20261116')
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
if (work/'inputs/manifest2.json').exists():
    raise ValueError('Prepared experiment already exists; preserve frozen inputs')
baseline=read(horizon/'inputs/manifest30.json')
for path,expected in baseline['protected_inputs']['files'].items():
    assert sha(path)==expected,path
archive=old/'comparison/actual-training-source.tar.gz'
assert sha(archive)==read(old/'comparison/matrix-verification.json')['frozen_training_source_sha256']
frozen=read(old/'matrix-v2/frozen-inputs.json')['code_sha256']
module=work/'experiments/reward_ablation'
module.mkdir(parents=True,exist_ok=True)
with tarfile.open(archive) as stream:
    for name,expected in frozen.items():
        data=stream.extractfile(name).read()
        assert hashlib.sha256(data).hexdigest()==expected
        (module/name).write_bytes(data)
original_train=(module/'train.py').read_text()
(module/'train.py').write_text(patch_frozen_trainer(original_train))
for package in ('torque_limit_ablation','balance_diagnostics'):
    shutil.copytree(repo/'experiments'/package,work/'experiments'/package,dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__'))
for filename in ('evaluation.py','report.py'):
    shutil.copy2(repo/'experiments/reward_ablation'/filename,module/filename)

native=work/'native'
shutil.copytree(Path('$HOME/projects/tenny-native-eval/src'),native/'src',dirs_exist_ok=True,
    ignore=shutil.ignore_patterns('__pycache__','build','results','.git'))
cpp=native/'src/cpp/legwheel_task.cpp'
original_cpp=cpp.read_text()
cpp.write_text(patch_legwheel_source(original_cpp))
shutil.copytree(native/'src/python',native/'python',dirs_exist_ok=True)
base=read(old/'inputs/initial-descriptor.json')
assets=Path(base['xml_path']).parent
shutil.copytree(assets,work/'assets',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
xml=ET.parse(assets/Path(base['xml_path']).name)
for node in xml.getroot().find('actuator'):
    if node.get('name') in ('thigh_left_motor','thigh_right_motor'):
        assert node.tag=='motor'
        node.set('ctrlrange','-2 2');node.set('forcerange','-2 2')
xml2=work/'assets/torque-limit-2.xml'
xml.write(xml2,encoding='utf-8')
one=experimental_descriptor(base,1.,base['xml_path'])
two=experimental_descriptor(base,2.,xml2)
experiment_files={p.relative_to(work).as_posix():sha(p) for p in (work/'experiments/torque_limit_ablation').glob('*.py')}
source_identity=dict(base=baseline['policy_metadata']['source_fingerprint'],experiment_files=experiment_files,
    native_original_sha256=hashlib.sha256(original_cpp.encode()).hexdigest(),native_patched_sha256=sha(cpp),
    modified_xml_sha256=sha(xml2),trainer_original_sha256=hashlib.sha256(original_train.encode()).hexdigest(),
    trainer_patched_sha256=sha(module/'train.py'))
fingerprint=hashlib.sha256(json.dumps(source_identity,sort_keys=True).encode()).hexdigest()
for limit,descriptor in ((1,one),(2,two)):
    descriptor['metadata'].update(source_fingerprint=fingerprint,
        base_source_fingerprint=baseline['policy_metadata']['source_fingerprint'],
        model_run_id=f'torque-limit-ablation-v1/initial-limit{limit}',kind='experimental_torque_limit_initializer')
    write(work/f'inputs/initial-limit{limit}.json',descriptor)

manifest=copy.deepcopy(baseline)
manifest.update(kind='torque_limit_ablation_v1',leg_limit_nm=2.,training_episode_seconds=30,
    torque_normalization_scales=[1.,1.,.0295,.0295],actuator_contract=actuator_contract(2.,two['task']['config']['jmax']),
    base_descriptor=str(work/'inputs/initial-limit2.json'),source_identity=source_identity,
    experimental_source_fingerprint=fingerprint)
manifest['policy_metadata'].update(source_fingerprint=fingerprint,base_source_fingerprint=source_identity['base'],
    torque_limits_nm=manifest['actuator_contract']['torque_limits_nm'],
    force_scales_n=manifest['actuator_contract']['force_scales_n'],experimental_only=True,
    torque_normalization_scales=manifest['torque_normalization_scales'])
manifest.pop('reuse_six_seconds',None)
manifest['reuse_limit1']=[]
for seed in baseline['seeds']:
    run=horizon/'matrix30'/f'seed-{seed}/baseline'
    status=read(run/'status.json');effective=read(run/'effective-config.json')
    assert status['phase']=='completed' and status['steps']==baseline['steps_per_run']
    assert effective['ppo']==dict(baseline['ppo'],seed=seed)
    assert effective['env']==baseline['env_config'] and effective['reward']==baseline['arms']['baseline']
    assert status['initial_array_digest']==baseline['initial_array_digest']
    evaluated=read(horizon/f'evaluations/h30-seed-{seed}/results.json')['policies'][0]
    assert evaluated['metadata']['training_steps']==status['steps']
    manifest['reuse_limit1'].append(dict(condition='limit1',seed=seed,reused=True,source_run=str(run),
        status=status,descriptor_sha256=sha(run/'descriptor.json'),
        checkpoint_sha256=evaluated['metadata']['checkpoint_sha256']))
manifest['frozen_source_sha256']={name:sha(module/name) for name in frozen}
write(work/'inputs/manifest2.json',manifest)
write(work/'inputs/preparation-source.json',source_identity)
write(work/'inputs/initial-manifest-limit2.json',dict(schema_version=1,seed=0,
    policies=[dict(id='initial-limit2',label='initial-limit2',descriptor=str(work/'inputs/initial-limit2.json'),
        source=dict(kind='same_BC_arrays_at_doubled_force_range',leg_limit_nm=2.))],initial_states=baseline['initial_states']))
shutil.copytree(work/'inputs',repo/'tasks/results/torque-limit-ablation-20261116/inputs',dirs_exist_ok=True)
print(json.dumps(dict(work=str(work),fingerprint=fingerprint,actuator_contract=manifest['actuator_contract'],
    training_episode_length=manifest['ppo']['episode_length'],reused_runs=len(manifest['reuse_limit1']))),flush=True)
