"""操作员状态：摇杆、滑条、开关、K。对应 src/web/lqrwebserver.cpp 里的全局量。

HTTP 只读写这里。command.py 再把它变成 command[4]。
"""

from __future__ import annotations

from .command import JOYSTICK_TIMEOUT_MS, apply_control_page

# 与 src/web/lqrwebserver.cpp 的 Kweb 初始表同一套。
DEFAULT_K = {
    "wheel_ctrl_scale": 100.0,
    "wheel_Kp": 0.0001,
    "wheel_Ki": 0.0,
    "wheel_Kd": 0.0,
    "sinwheel_scale": 1.0,
    "sinpitch_scale": 1.3,
    "sinyaw_scale": 0.2,
    "hold_error_clip_enable": 1.0,
    "hold_error_clip_ds": 3.0,
    "hold_error_clip_dpsi": 1.0,
    "pitch_max": 0.6,
    "send_interval": 1000.0,
    "run_interval": 4.0,
    "thigh_Kp": 0.40,
    "thigh_Kd": 0.1,
    "thigh_smoothing_alpha(0,1)": 0.0,
    "wheel_vel_smoothing_alpha(0,1)": 0.0,
    "control_smooth(0, 1)": 0.0,
    "cmd_lin_vel_x(-0.2,0.2)": 0.0,
    "cmd_ang_vel_yaw(-1.5,1.5)": 0.0,
    "cmd_height_value(-0.5,0)": -0.1,
    "cmd_roll_value(-0.2,0.2)": 0.0,
    "command_vx_rate": 0.2,
    "pitch_zero_low": 0.22,
    "pitch_zero_high": 0.10,
    "sinroll_scale": 1.0,
}

CMD_K_KEYS = (
    "cmd_lin_vel_x(-0.2,0.2)",
    "cmd_ang_vel_yaw(-1.5,1.5)",
    "cmd_height_value(-0.5,0)",
    "cmd_roll_value(-0.2,0.2)",
)


class Operator:
    def __init__(self) -> None:
        self.joystick = [0.0, 0.0]
        self.last_joystick_ms = 0
        self.height_bar = 0.0
        self.oscillation_bar = 0.0
        self.switcher = False
        self.switcher_last = False
        self.page = "control"
        self.k = dict(DEFAULT_K)
        self.debug = ""

    def get_k(self, key: str, default: float) -> float:
        return float(self.k[key]) if key in self.k else default

    def set_joystick(self, x: float, y: float, now_ms: int) -> None:
        self.joystick = [float(x), float(y)]
        self.last_joystick_ms = now_ms

    def live_joystick(self, now_ms: int) -> list[float]:
        # 与固件 getJoyStickSize2 相同：超过 1s 没更新就当回中。
        if now_ms - self.last_joystick_ms > JOYSTICK_TIMEOUT_MS:
            return [0.0, 0.0]
        return list(self.joystick)

    def turn_on_edge(self) -> bool:
        turn_on = (not self.switcher_last) and self.switcher
        self.switcher_last = self.switcher
        return turn_on

    def command(self, now_ms: int = 0) -> list[float]:
        # 1) K 的 cmd_* 是底。2) control 页再用摇杆/滑条覆盖。
        cmd = [self.get_k(k, 0.0) for k in CMD_K_KEYS]
        if self.page == "control":
            joy = self.live_joystick(now_ms)
            apply_control_page(
                joy[0], joy[1], self.height_bar, self.oscillation_bar, cmd
            )
        return cmd
