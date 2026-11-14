"""HoldObs = Hold（停车保持）+ Obs。只算双锚点 / Δs / Δψ，不是整份 26 维。

对应 src/controller/hold_obs.cpp。名字解释见 docs/sim_verify_framework.md §2。
real / sim 都把 SimFrame 喂到这里。不要在这里做单位换算或取负。
step() 顺序必须与固件相同：死区 → 按指令刷锚点 → 开机/摔倒覆盖 → 算误差 → clip 观测。
"""

from __future__ import annotations

from .sim_frame import SimFrame


def clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


class HoldObs:
    def __init__(self, reset_on_fallen: bool = True) -> None:
        # 朝向编码 sin(yaw * 0.5) / 0.5。内层 0.5 硬编码；sinyaw_scale 只乘在观测上。
        self.yaw_sin_scale = 0.5
        self.sinyaw_scale = 1.0
        # 与训练 sample_command 死区相同：低于此视为停车。
        self.cmd_lin_vel_x_zero_margin = 0.000195
        self.cmd_ang_vel_zero_margin = 0.01
        # 摔倒：pitch < -pitch_max 或 > pitch_max+0.5。
        self.pitch_max = 0.6
        self.hold_error_clip_enable = True
        self.hold_error_clip_ds = 3.0
        self.hold_error_clip_dpsi = 1.0
        self.reset_on_fallen = reset_on_fallen
        self.position_start = 0.0  # s0
        self.yaw_sin_zero = 0.0  # ψ0，已是 sinyaw 编码
        self.hold_anchors_inited = False
        self.ds_true = 0.0
        self.dpsi_true = 0.0

    def sinyaw(self, yaw_rad: float) -> float:
        import math

        # 不用裸 sin(yaw)：大角度饱和太快。除以 scale 后小角度斜率仍是 1。
        return math.sin(yaw_rad * self.yaw_sin_scale) / self.yaw_sin_scale

    @staticmethod
    def hold_position(q_l: float, q_r: float) -> float:
        # 左右轮共模 = 沿轨迹弧长。铰链同号前进，不能写成差动。
        return 0.5 * (q_l + q_r)

    @staticmethod
    def fallen(pitch_rad: float, pitch_max: float = 0.6) -> bool:
        return pitch_rad < -pitch_max or pitch_rad > pitch_max + 0.5

    def step(
        self,
        command,
        frame: SimFrame | None = None,
        *,
        q_l: float = 0.0,
        q_r: float = 0.0,
        yaw_rad: float = 0.0,
        pitch_rad: float = 0.0,
        turn_on: bool = False,
    ) -> dict:
        # q_l= 等必须用关键字。第二个位置参数是 frame；写成 step(cmd, q_l, q_r)
        # 会把 float 当成 SimFrame。
        if frame is None:
            frame = SimFrame(q_l=q_l, q_r=q_r, yaw_rad=yaw_rad, pitch_rad=pitch_rad)

        cmd = list(command)
        yaw_sin = self.sinyaw(frame.yaw_rad)
        hold_s = self.hold_position(frame.q_l, frame.q_r)

        # 1) 死区：低于阈值的 vx/yaw 当成 0，再决定要不要刷锚点。
        pos_moving = abs(cmd[0]) > self.cmd_lin_vel_x_zero_margin
        yaw_moving = abs(cmd[1]) > self.cmd_ang_vel_zero_margin
        if not pos_moving:
            cmd[0] = 0.0
        if not yaw_moving:
            cmd[1] = 0.0

        # 2) 按指令分别刷新。前进只刷 s0；转向只刷 ψ0。
        if pos_moving:
            self.position_start = hold_s
        if yaw_moving:
            self.yaw_sin_zero = yaw_sin

        # 3) 开机 / 第一帧 / 摔倒：两套锚点一起取当前值。
        if not self.hold_anchors_inited or turn_on or (
            self.reset_on_fallen and self.fallen(frame.pitch_rad, self.pitch_max)
        ):
            self.position_start = hold_s
            self.yaw_sin_zero = yaw_sin
            self.hold_anchors_inited = True

        # 4) 相对冻结锚点的误差。差动轮转不进 Δs。
        dpsi_true = yaw_sin - self.yaw_sin_zero
        ds_true = hold_s - self.position_start
        dpsi = dpsi_true
        ds = ds_true
        # 5) 只 clip 观测，锚点不动。
        if self.hold_error_clip_enable:
            dpsi = clip(dpsi, -self.hold_error_clip_dpsi, self.hold_error_clip_dpsi)
            ds = clip(ds, -self.hold_error_clip_ds, self.hold_error_clip_ds)

        self.ds_true = ds_true
        self.dpsi_true = dpsi_true
        return {
            "cmd": cmd,
            "obs_dpsi": dpsi * self.sinyaw_scale,
            "obs_ds_l": ds,
            "obs_ds_r": ds,
            "ds_true": ds_true,
            "dpsi_true": dpsi_true,
            "s0": self.position_start,
            "yaw0": self.yaw_sin_zero,
        }
