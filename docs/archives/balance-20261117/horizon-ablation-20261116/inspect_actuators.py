import json
import os
from pathlib import Path
import sys
os.environ['JAX_PLATFORMS']='cpu'
os.environ['CUDA_VISIBLE_DEVICES']=''
import mujoco

root=Path('$HOME/projects/tenny-horizon-ablation-20261116')
base=json.loads((root/'inputs/manifest30.json').read_text())
xml=Path(base['env_config']['_XML_PATH'].removeprefix('unserializable object: '))
student=mujoco.MjModel.from_xml_path(str(xml))
original=Path('$HOME/projects/tenny-reference/mujoco_legwheel_rl_public')
sys.path.insert(0,str(original))
from legwheel import config as reference_config
if not Path(reference_config.__file__).resolve().is_relative_to(original):
    raise ValueError('Wrong original position-controller config')
cfg=reference_config.default_config()
reference=mujoco.MjModel.from_xml_path(str(cfg._XML_PATH))

def describe(model):
    return dict(physics_dt=float(model.opt.timestep),activation_states=int(model.na),actuators=[dict(
        name=model.actuator(i).name,dyntype=int(model.actuator_dyntype[i]),
        gaintype=int(model.actuator_gaintype[i]),biastype=int(model.actuator_biastype[i]),
        gain_first3=model.actuator_gainprm[i,:3].tolist(),bias_first3=model.actuator_biasprm[i,:3].tolist(),
        gear=model.actuator_gear[i].tolist(),force_limited=bool(model.actuator_forcelimited[i]),
        force_range=model.actuator_forcerange[i].tolist(),ctrl_limited=bool(model.actuator_ctrllimited[i]),
        ctrl_range=model.actuator_ctrlrange[i].tolist(),
        joint_force_limited=bool(model.jnt_actfrclimited[model.actuator_trnid[i,0]]) if hasattr(model,'jnt_actfrclimited') else None,
        joint_force_range=model.jnt_actfrcrange[model.actuator_trnid[i,0]].tolist() if hasattr(model,'jnt_actfrcrange') else None)
        for i in range(model.nu)])
src=Path('$HOME/projects/tenny-torque200/src')
snippets={}
for relative,words in (
    ('wheel_teacher/teacher.py',('for _ in range(4)','vel_pid.kp','last_target_wheel_vel','cfg.ctrl_dt','cfg.sim_dt')),
    ('hybrid/env.py',('def substep','torque_command(safe_a','mjx_env.step(self.mjx_model, d, tau, 1)','length=self.n_substeps')),
    ('force_teacher/env.py',('cfg.control_hz','hybrid_config(100)'))):
    snippets[relative]=[dict(line=i,text=line.strip()) for i,line in enumerate((src/relative).read_text().splitlines(),1)
                        if any(word in line for word in words)]
from native_eval import Evaluator
native=Evaluator(base['base_descriptor'])
student_info,reference_info=describe(student),describe(reference)
student_info['raw_xml_physics_dt']=student_info.pop('physics_dt')
reference_info['raw_xml_physics_dt']=reference_info.pop('physics_dt')
student_info['actual_native_evaluator_physics_dt']=native.model_info['physics_dt']
student_info['configured_training_physics_dt']=base['env_config']['sim_dt']
reference_info['configured_teacher_physics_dt']=float(cfg.sim_dt)
snippets['legwheel/env.py']=[dict(line=i,text=line.strip()) for i,line in enumerate((src/'legwheel/env.py').read_text().splitlines(),1)
                             if 'timestep' in line or 'sim_dt' in line]
report=dict(student=student_info,reference=reference_info,
    reference_config=dict(control_dt=float(cfg.ctrl_dt),physics_dt=float(cfg.sim_dt),wheel_velocity_feedback=bool(cfg.vel_pid.enable)),
    deployed_source_snippets=snippets,scope='Compiled actuator/static source audit, no control-rate experiment or hardware measurement.')
(root/'actuator-frequency-evidence.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(student_hips=student_info['actuators'][:2],reference_hips=reference_info['actuators'][:2],
    actual_native_physics_dt=student_info['actual_native_evaluator_physics_dt'],
    training_model_timestep_source=snippets['legwheel/env.py']),indent=2))
