"""Web 操作员通道：摇杆 / 滑条 / K → command[4]。对应 src/web/command.cpp。

不负责 HTTP，也不刷新锚点。仿真和真机都把这里的 command 喂给 HoldObs。
"""

from __future__ import annotations

import math

# 与 src/web/command.cpp kCommandRange 同一套。
COMMAND_RANGE = (
    (-0.2, 0.2),   # lin_vel_x
    (-1.5, 1.5),   # ang_vel_yaw
    (-0.5, 0.0),   # height_value
    (-0.2, 0.2),   # roll_value
)

# 历史固件摇杆归一化用这个 π，不要改成 math.pi。
JOYSTICK_PI = 3.1415926
JOYSTICK_DEAD = 0.1
JOYSTICK_TURN_DEAD = 0.3
JOYSTICK_TIMEOUT_MS = 1000


def clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def linear_map(x: float, x0: float, x1: float, y0: float, y1: float) -> float:
    return (x - x0) / (x1 - x0) * (y1 - y0) + y0


def map_height(height_bar: float, command: list[float], range_=COMMAND_RANGE) -> None:
    # 100 → 下蹲 -0.5，0 → 站立 0。
    command[2] = linear_map(height_bar, 100.0, 0.0, range_[2][0], range_[2][1])


def map_roll(oscillation_bar: float, command: list[float], range_=COMMAND_RANGE) -> None:
    command[3] = linear_map(oscillation_bar, -50.0, 50.0, range_[3][0], range_[3][1])


def map_joystick(
    x: float,
    y: float,
    command: list[float],
    range_=COMMAND_RANGE,
) -> tuple[bool, bool]:
    enable = False
    turn_enable = False
    magnitude = math.sqrt(x * x + y * y)
    if magnitude > JOYSTICK_DEAD:
        enable = True
    else:
        magnitude = 0.0

    yaw_value = -math.atan2(abs(x), abs(y))
    yaw_value = yaw_value * 2.0 / JOYSTICK_PI
    if abs(yaw_value) > JOYSTICK_TURN_DEAD:
        turn_enable = True
    else:
        yaw_value = 0.0
    if not enable:
        yaw_value = 0.0
    if y * x < 0.0:
        yaw_value = -yaw_value

    # 无窗口闭环实测（同一套 ObsPack+策略+env）：
    #   vx=+0.20 → dx=+0.48 前进
    #   vx=-0.10 → dx=+0.08 仍前进
    #   vx=-0.15 → dx=+0.01 仍前进
    #   vx=-0.18 → dx=-0.03 才后退
    #   vx=-0.20 → dx=-0.06 后退，幅度约是 +0.20 的 1/8
    # 所以下拉必须尽快给到 -0.2，不能按 y 线性慢爬。
    vx_scale = max(abs(range_[0][0]), abs(range_[0][1]))
    full_y = 0.2
    if (not enable) or abs(y) <= JOYSTICK_DEAD:
        x_vel = 0.0
    else:
        t = clip((abs(y) - JOYSTICK_DEAD) / (full_y - JOYSTICK_DEAD), 0.0, 1.0)
        x_vel = (1.0 if y > 0.0 else -1.0) * t * vx_scale
    command[0] = clip(x_vel, range_[0][0], range_[0][1])
    command[1] = clip(yaw_value, range_[1][0], range_[1][1])
    return enable, turn_enable


def apply_control_page(
    joy_x: float,
    joy_y: float,
    height_bar: float,
    oscillation_bar: float,
    command: list[float],
    range_=COMMAND_RANGE,
) -> tuple[bool, bool]:
    map_roll(oscillation_bar, command, range_)
    map_height(height_bar, command, range_)
    return map_joystick(joy_x, joy_y, command, range_)
