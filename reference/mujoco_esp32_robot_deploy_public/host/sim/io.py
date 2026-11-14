"""Sim 支路：MJX 状态 → SimFrame。固件没有对应目录（仿真只在主机上跑）。

已是仿真系，不再改符号。qpos 下标来自 env._qpos_wanted_joints，不要写死 9/12。
"""

from __future__ import annotations

import math

from controller.sim_frame import SimFrame


def scalar(x) -> float:
    import numpy as np

    return float(np.asarray(x).reshape(-1)[0])


def _mat3(x):
    import numpy as np

    return np.asarray(x).reshape(3, 3)


def _vec3(x) -> list[float]:
    import numpy as np

    a = np.asarray(x).reshape(-1)
    return [float(a[0]), float(a[1]), float(a[2])]


def yaw_from_xmat(env, data) -> float:
    # 与训练合同同一套：atan2(R10, R00)，body 是 env._body_id。
    R = _mat3(data.xmat[int(env._body_id)])
    return float(math.atan2(R[1, 0], R[0, 0]))


def _euler_from_xmat(env, data) -> tuple[float, float, float]:
    # pitch/roll 让 sin(pitch)= -R[2,0]、sin(roll)= R[2,1]/cos(pitch)，与训练 sinpitch/sinroll 同值。
    R = _mat3(data.xmat[int(env._body_id)])
    sinpitch = max(-1.0, min(1.0, float(-R[2, 0])))
    pitch = math.asin(sinpitch)
    cospitch = math.sqrt(max(1.0 - sinpitch * sinpitch, 1e-12))
    sinroll = max(-1.0, min(1.0, float(R[2, 1]) / cospitch))
    roll = math.asin(sinroll)
    yaw = float(math.atan2(R[1, 0], R[0, 0]))
    return pitch, roll, yaw


def wheels(env, data) -> tuple[float, float]:
    import numpy as np

    # wanted joints 里 1、3 是左右轮；0、2 是大腿。
    q = np.asarray(data.qpos)
    idx = [int(i) for i in env._qpos_wanted_joints]
    return float(q[idx[1]]), float(q[idx[3]])


def from_mjx(env, state) -> SimFrame:
    import numpy as np

    data = state.data
    q = np.asarray(data.qpos)
    v = np.asarray(data.qvel)
    iq = [int(i) for i in env._qpos_wanted_joints]
    iv = [int(i) for i in env._qvel_wanted_joints]
    pitch, roll, yaw = _euler_from_xmat(env, data)
    return SimFrame(
        q_l=float(q[iq[1]]),
        q_r=float(q[iq[3]]),
        yaw_rad=yaw,
        pitch_rad=pitch,
        roll_rad=roll,
        gyro=_vec3(env.get_gyro_newest(data)),
        thigh_l=float(q[iq[0]]),
        thigh_r=float(q[iq[2]]),
        thigh_vel_l=float(v[iv[0]]),
        thigh_vel_r=float(v[iv[2]]),
        wheel_vel_l=float(v[iv[1]]),
        wheel_vel_r=float(v[iv[3]]),
        pitch_zero_rad=scalar(env.get_balance_angle(data)),
    )


def command_from_state(state) -> list[float]:
    return [scalar(c) for c in state.info["command"]]
