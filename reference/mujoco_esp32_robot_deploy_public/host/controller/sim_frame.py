"""仿真系本体状态。real I/O 与 sim I/O 都填这一份。对应 src/controller/sim_frame.h。

HoldObs 只吃这些字段；ObsPack 用同一份拼 26 维。单位：弧度、弧度/秒。轮角是未饱和累计角，与仿真 XML 同号。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SimFrame:
    q_l: float = 0.0          # 左轮累计角 [rad]
    q_r: float = 0.0          # 右轮累计角 [rad]
    yaw_rad: float = 0.0      # 与 atan2(R10, R00) 同一套
    pitch_rad: float = 0.0
    roll_rad: float = 0.0
    gyro: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    thigh_l: float = 0.0      # 仿真系：下蹲约 0，站立约 -0.5
    thigh_r: float = 0.0
    thigh_vel_l: float = 0.0
    thigh_vel_r: float = 0.0
    wheel_vel_l: float = 0.0
    wheel_vel_r: float = 0.0
    # 训练 _get_obs 的 sin_balance_zero = sin(get_balance_angle)。
    # sim 能算质心几何，填这个；真机没有 COM，保持 None，ObsPack 走大腿曲线。
    pitch_zero_rad: float | None = None
