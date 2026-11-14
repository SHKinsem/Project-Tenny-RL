#include "hold_obs.h"
#include <cmath>

float HoldObs::clip(float x, float lo, float hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

bool HoldObs::fallen(float pitch_rad, float pitch_max) {
    // +0.5 让前倾比后仰更晚判定摔倒。
    return pitch_rad < -pitch_max || pitch_rad > pitch_max + 0.5f;
}

float HoldObs::sinyaw(float yaw_rad) const {
    // 不用裸 sin(yaw)：大角度饱和太快。除以 scale 后小角度斜率仍是 1。
    return sinf(yaw_rad * yaw_sin_scale) / yaw_sin_scale;
}

float HoldObs::hold_position(float q_l, float q_r) {
    // 未饱和左右轮角共模 = 沿轨迹弧长（差一个轮半径，训练里也不乘 R）。
    // 本机左右铰链同号前进，必须是 (q_l+q_r)/2，不能写成差动。
    return 0.5f * (q_l + q_r);
}

HoldObsOut HoldObs::step(float command[4], const SimFrame& frame, bool turn_on) {
    const float yaw_sin = sinyaw(frame.yaw_rad);
    const float hold_s = hold_position(frame.q_l, frame.q_r);

    // 1) 死区：低于阈值的 vx/yaw 当成 0，再决定要不要刷锚点。
    const bool pos_moving = fabsf(command[0]) > cmd_lin_vel_x_zero_margin;
    const bool yaw_moving = fabsf(command[1]) > cmd_ang_vel_zero_margin;
    if (!pos_moving) {
        command[0] = 0.f;
    }
    if (!yaw_moving) {
        command[1] = 0.f;
    }

    // 2) 按指令分别刷新。前进只刷 s0（直行保向）；转向只刷 ψ0（原地转不丢位置）。
    if (pos_moving) {
        position_start = hold_s;
    }
    if (yaw_moving) {
        yaw_sin_zero = yaw_sin;
    }

    // 3) 开机 / 第一帧 / 摔倒：两套锚点一起取当前值（覆盖上面的单侧刷新）。
    if (!hold_anchors_inited || turn_on ||
        (reset_on_fallen && fallen(frame.pitch_rad, pitch_max))) {
        position_start = hold_s;
        yaw_sin_zero = yaw_sin;
        hold_anchors_inited = true;
    }

    // 4) 相对冻结锚点的误差。差动轮转不进 Δs（左右一正一负，共模不变）。
    const float dpsi_unclip = yaw_sin - yaw_sin_zero;
    const float ds_unclip = hold_s - position_start;
    float dpsi = dpsi_unclip;
    float ds = ds_unclip;

    // 5) 只 clip 观测。锚点保持原值，真值仍可从 ds_true 读出。
    if (hold_error_clip_enable) {
        dpsi = clip(dpsi, -hold_error_clip_dpsi, hold_error_clip_dpsi);
        ds = clip(ds, -hold_error_clip_ds, hold_error_clip_ds);
    }

    HoldObsOut out;
    for (int i = 0; i < 4; ++i) {
        out.cmd[i] = command[i];
    }
    out.obs_dpsi = dpsi * sinyaw_scale;
    // 两轮槽填同一个 Δs：看均值是位置，看左右差恒为 0。
    out.obs_ds_l = ds;
    out.obs_ds_r = ds;
    out.ds_true = ds_unclip;
    out.dpsi_true = dpsi_unclip;
    out.s0 = position_start;
    out.yaw0 = yaw_sin_zero;
    ds_true = ds_unclip;
    dpsi_true = dpsi_unclip;
    return out;
}
