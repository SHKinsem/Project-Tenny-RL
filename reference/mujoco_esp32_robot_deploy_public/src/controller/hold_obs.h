#ifndef CONTROLLER_HOLD_OBS_H
#define CONTROLLER_HOLD_OBS_H

#include "sim_frame.h"

// =============================================================================
// HoldObs = Hold（停车保持）+ Obs。只算双锚点 s0 / ψ0 和 Δs / Δψ，不是整份 26 维。
// 26 维其余槽位在 ObsPack。名字解释：docs/sim_verify_framework.md §2。
//
// 不要在这里做 deg→rad 或轮角取负——那是 src/real 的事。
// =============================================================================

// 一步输出。ds_true / dpsi_true 是 clip 前的真值，obs_* 是送进网络的。
struct HoldObsOut {
    float cmd[4];       // 死区清零后的命令，写入 obs[0:4]
    float obs_dpsi;     // → input[7]
    float obs_ds_l;     // → input[11]，与 obs_ds_r 相同
    float obs_ds_r;     // → input[13]
    float ds_true;      // 未 clip 的 Δs，串口 debug 用
    float dpsi_true;
    float s0;           // 当前位置锚点
    float yaw0;         // 当前朝向锚点（sinyaw 编码后）
};

class HoldObs {
public:
    // 朝向编码：sin(yaw * 0.5) / 0.5，与训练 _getSinYaw_newest 一致。
    // 内层 0.5 硬编码；Kweb 的 sinyaw_scale 只乘在观测上。
    float yaw_sin_scale = 0.5f;
    float sinyaw_scale = 1.0f;

    // 与训练 sample_command 死区相同：低于此视为停车，命令写成 0 再进观测。
    float cmd_lin_vel_x_zero_margin = 0.000195f;
    float cmd_ang_vel_zero_margin = 0.01f;

    // 摔倒：pitch < -pitch_max 或 > pitch_max+0.5。固件当场重锚；训练走 episode reset。
    float pitch_max = 0.6f;
    bool reset_on_fallen = true;

    // 推理时只 clip 送进网络的误差，锚点不动。半幅与训练 hold_error_clip 对齐。
    bool hold_error_clip_enable = true;
    float hold_error_clip_ds = 3.0f;
    float hold_error_clip_dpsi = 1.0f;

    // 双锚点状态。第一帧 / 开机 / 摔倒时都写成当前 s、ψ。
    float position_start = 0.f;   // s0
    float yaw_sin_zero = 0.f;     // ψ0，已是 sinyaw 编码
    bool hold_anchors_inited = false;
    float ds_true = 0.f;
    float dpsi_true = 0.f;

    float sinyaw(float yaw_rad) const;
    static float hold_position(float q_l, float q_r);
    static bool fallen(float pitch_rad, float pitch_max);

    // command[] 会被就地死区清零（与写入 obs 的 command 一致）。
    HoldObsOut step(float command[4], const SimFrame& frame, bool turn_on = false);

private:
    static float clip(float x, float lo, float hi);
};

#endif
