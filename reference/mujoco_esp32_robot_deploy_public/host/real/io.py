"""Real 支路：真机传感器 → SimFrame。对应 src/real/io.cpp。

只做单位、符号、舵机映射。不要在这里刷新锚点。
deg2rad 必须用 3.1415/180，与固件历史尺度一致，不要改成 math.pi。
"""

from __future__ import annotations

from controller.sim_frame import SimFrame

DEG2RAD = 3.1415 / 180.0


def mapping_thighs_position(thigh_left: float, thigh_right: float) -> tuple[float, float]:
    # STS3032 计数端点是实机标定的；仿真下蹲 0、站立 -0.5。右腿计数方向与左腿相反。
    thigh_left_squad, thigh_left_stand = 2136.0, 2676.0
    thigh_right_squad, thigh_right_stand = 2019.0, 1479.0
    sim_left_squad, sim_left_stand = 0.0, -0.5
    sim_right_squad, sim_right_stand = 0.0, -0.5
    sim_l = sim_left_squad + (sim_left_stand - sim_left_squad) * (
        thigh_left - thigh_left_squad
    ) / (thigh_left_stand - thigh_left_squad)
    sim_r = sim_right_stand + (sim_right_squad - sim_right_stand) * (
        thigh_right - thigh_right_stand
    ) / (thigh_right_squad - thigh_right_stand)
    return sim_l, sim_r


def mapping_thighs_velocity(
    thigh_left_vel: float, thigh_right_vel: float
) -> tuple[float, float]:
    # 4096 计数/圈。左腿从蹲到站在实机是正速度，仿真是负速度，所以左腿取负。
    scale = 2 * 3.1415 / 4096
    return -thigh_left_vel * scale, thigh_right_vel * scale


def to_sim_frame(sensors) -> SimFrame:
    # sensors 下标与 .ino get_sensors_data 一致。轮角必须取负才与仿真 qpos 同号。
    s = list(sensors)
    thigh_l, thigh_r = mapping_thighs_position(s[14], s[15])
    tvl, tvr = mapping_thighs_velocity(s[16], s[17])
    return SimFrame(
        q_l=-s[10],
        q_r=-s[11],
        yaw_rad=s[3] * DEG2RAD,
        pitch_rad=s[2] * DEG2RAD,
        roll_rad=s[1] * DEG2RAD,
        gyro=[s[4] * DEG2RAD, s[5] * DEG2RAD, s[6] * DEG2RAD],
        thigh_l=thigh_l,
        thigh_r=thigh_r,
        thigh_vel_l=tvl,
        thigh_vel_r=tvr,
        wheel_vel_l=-s[12],
        wheel_vel_r=-s[13],
    )
