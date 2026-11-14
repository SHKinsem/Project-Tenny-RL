"""共享 26 维观测拼装。对应 src/controller/obs.cpp。

HoldObs 只填 [7] 和 [11]/[13]。pack() 把 SimFrame + 锚点误差拼成策略输入。
note_policy_action() 更新轮速历史；不跑 PID，也不调用 env.step。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .sim_frame import SimFrame

OBS_DIM = 26


def clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


@dataclass
class ObsPackConfig:
    wheel_radius: float = 0.0195
    thigh_lo: float = -0.4
    thigh_hi: float = -0.06
    pitch_zero_low: float = 0.22
    pitch_zero_high: float = 0.05
    pitch_power: float = 1.0
    sinpitch_scale: float = 1.0
    sinroll_scale: float = 1.0
    wheel_vel_smoothing_alpha: float = 0.0
    vel_pid_target_scale: float = 30.0
    vel_pid_smoothing: float = 0.5
    transform_action_bias: list = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    transform_action_scale: list = field(default_factory=lambda: [0.25, 0.25, 1.0, 1.0])


class ObsPack:
    def __init__(self, cfg: ObsPackConfig | None = None) -> None:
        self.cfg = cfg or ObsPackConfig()
        self.reset()

    def reset(self) -> None:
        self.smoothed_wheel_vel = [0.0, 0.0]
        self.wheels_commands_history = [0.0, 0.0, 0.0, 0.0]
        self.last_target_wheel_vel = [0.0, 0.0]

    def pack(self, command, frame: SimFrame, hold: dict) -> list[float]:
        cfg = self.cfg
        if frame.pitch_zero_rad is not None:
            # 与训练 get_balance_angle / view.sh 同一套质心几何零点。
            pitch_zero_now = frame.pitch_zero_rad
        else:
            avg_thigh = (frame.thigh_l + frame.thigh_r) * 0.5 - cfg.thigh_hi
            thigh_len = cfg.thigh_lo - cfg.thigh_hi
            ratio = avg_thigh / thigh_len
            factor = ratio if cfg.pitch_power == 1.0 else math.pow(ratio, cfg.pitch_power)
            pitch_zero_now = cfg.pitch_zero_low + (
                cfg.pitch_zero_high - cfg.pitch_zero_low
            ) * factor

        a = cfg.wheel_vel_smoothing_alpha
        self.smoothed_wheel_vel[0] = a * self.smoothed_wheel_vel[0] + (1.0 - a) * frame.wheel_vel_l
        self.smoothed_wheel_vel[1] = a * self.smoothed_wheel_vel[1] + (1.0 - a) * frame.wheel_vel_r

        return [
            float(command[0]),
            float(command[1]),
            float(command[2]),
            float(command[3]),
            frame.gyro[0],
            frame.gyro[1],
            frame.gyro[2],
            float(hold["obs_dpsi"]),
            (math.sin(frame.pitch_rad) - math.sin(pitch_zero_now)) * cfg.sinpitch_scale,
            (math.sin(frame.roll_rad) - math.sin(command[3])) * cfg.sinroll_scale,
            frame.thigh_l,
            float(hold["obs_ds_l"]),
            frame.thigh_r,
            float(hold["obs_ds_r"]),
            # 训练 _get_obs 把大腿速度改成零均值噪声；noise=0 时就是 0。
            # 喂真实大腿速度会让这两维出分布，前后/左右跟踪都会偏。
            0.0,
            self.smoothed_wheel_vel[0],
            0.0,
            self.smoothed_wheel_vel[1],
            self.smoothed_wheel_vel[0] * cfg.wheel_radius,
            self.smoothed_wheel_vel[1] * cfg.wheel_radius,
            self.wheels_commands_history[0],
            self.wheels_commands_history[1],
            self.wheels_commands_history[2],
            self.wheels_commands_history[3],
            self.last_target_wheel_vel[0] - frame.wheel_vel_l,
            self.last_target_wheel_vel[1] - frame.wheel_vel_r,
        ]

    def note_policy_action(self, last_act, frame: SimFrame) -> None:
        cfg = self.cfg
        residual_l = (
            (last_act[2] + cfg.transform_action_bias[2])
            * cfg.transform_action_scale[2]
            * cfg.vel_pid_target_scale
        )
        residual_r = (
            (last_act[3] + cfg.transform_action_bias[3])
            * cfg.transform_action_scale[3]
            * cfg.vel_pid_target_scale
        )
        self.wheels_commands_history[0] = self.wheels_commands_history[2]
        self.wheels_commands_history[1] = self.wheels_commands_history[3]
        self.wheels_commands_history[2] = residual_l
        self.wheels_commands_history[3] = residual_r
        tgt_l = clip(
            residual_l + frame.wheel_vel_l,
            -cfg.vel_pid_target_scale,
            cfg.vel_pid_target_scale,
        )
        tgt_r = clip(
            residual_r + frame.wheel_vel_r,
            -cfg.vel_pid_target_scale,
            cfg.vel_pid_target_scale,
        )
        s = cfg.vel_pid_smoothing
        self.last_target_wheel_vel[0] = tgt_l * s + self.last_target_wheel_vel[0] * (1.0 - s)
        self.last_target_wheel_vel[1] = tgt_r * s + self.last_target_wheel_vel[1] * (1.0 - s)
