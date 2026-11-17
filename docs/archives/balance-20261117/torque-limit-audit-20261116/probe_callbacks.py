import json
from pathlib import Path
import mujoco
from native_eval import Evaluator
root=Path('$HOME/projects/tenny-horizon-ablation-20261116')
engine=Evaluator(root/'inputs/evaluation-descriptors/h30-seed-0.json')
initial=json.loads((root/'inputs/manifest30.json').read_text())['initial_states'][0]['state']
control=[];sensor=[]
def record_control(model,data):
    control.append(dict(t=data.time,ctrl=data.ctrl.copy().tolist(),actual=data.qfrc_actuator.copy().tolist()))
def record_sensor(model,data,stage):
    sensor.append(dict(t=data.time,stage=stage,ctrl=data.ctrl.copy().tolist(),actual=data.qfrc_actuator.copy().tolist()))
try:
    mujoco.set_mjcb_control(record_control)
    mujoco.set_mjcb_sensor(record_sensor)
    engine.evaluate([initial],num_envs=1,steps=1,episode_steps=1,control_dt=.01,trace_worlds=[0])
finally:
    mujoco.set_mjcb_control(None)
    mujoco.set_mjcb_sensor(None)
print(json.dumps(dict(control=control,sensor=sensor)))
