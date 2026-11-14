"""Read-only geometry check; removes rendering assets, retains inertials/contacts."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import numpy as np

root = Path(__file__).resolve().parents[1]
path = root / 'experiments/torque200/assets/legwheel_robot2_stl_nolinkage6_terrain.xml'
xml = ET.parse(path).getroot()
for parent in list(xml.iter()):
    for child in list(parent):
        if child.tag in ('include', 'asset', 'visual') or (child.tag == 'geom' and child.get('type') == 'mesh'):
            parent.remove(child)
    parent.attrib.pop('material', None)
m = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding='unicode'))
d = mujoco.MjData(m)
rows = []
for side in ('left', 'right'):
    hip, knee, wheel = [m.joint(f'{name}_{side}_joint') for name in ('thigh', 'calf', 'wheel')]
    h, k, w = [int(j.dofadr[0]) for j in (hip, knee, wheel)]
    hp, kp = [int(j.qposadr[0]) for j in (hip, knee)]
    body = m.body(f'wheel_{side}').id
    geom = m.geom(f'wheel_{side}_geom_shell').id
    for q in (-0.5, -0.25, 0.0):
        g = lambda t: -0.0004-2.1405*t+0.2637*t*t-1.2386*t**3-0.3494*t**4
        gp = -2.1405+0.5274*q-3.7158*q*q-1.3976*q**3
        d.qpos[:] = m.qpos0
        d.qpos[hp], d.qpos[kp] = q, g(q)
        mujoco.mj_forward(m, d)
        point = d.geom_xpos[geom].copy() + np.array([0, 0, -0.0195])
        local = d.xmat[body].reshape(3, 3).T @ (point-d.xpos[body])
        jac = np.zeros((3, m.nv))
        mujoco.mj_jac(m, d, jac, None, point, body)
        effective = jac[:, h]+gp*jac[:, k]
        # Track this material point, not the moving geometric lowest point.
        pts = []
        for eps in (-1e-6, 1e-6):
            d.qpos[hp], d.qpos[kp] = q+eps, g(q+eps)
            mujoco.mj_forward(m, d)
            pts.append(d.xpos[body]+d.xmat[body].reshape(3, 3) @ local)
        error = float(np.max(np.abs((pts[1]-pts[0])/2e-6-effective)))
        A = -np.array([[effective[0], effective[2]], [jac[0, w], jac[2, w]]])
        rows.append(dict(side=side, hip=q, knee=g(q), map_Nm_per_N=A.tolist(), fd_error=error, determinant=float(np.linalg.det(A))))
        assert error < 1e-8
print(json.dumps(dict(mass_kg=float(m.body_mass.sum()), support_per_wheel_N=float(m.body_mass.sum()*9.81/2), hip_armature=float(m.dof_armature[h]), wheel_force_cap_N=0.0295/0.0195, samples=rows), indent=2))
