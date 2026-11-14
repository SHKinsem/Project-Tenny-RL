#include "obs.h"
#include <cmath>

float ObsPack::clip(float x, float lo, float hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

void ObsPack::reset() {
    smoothed_wheel_vel[0] = 0.f;
    smoothed_wheel_vel[1] = 0.f;
    wheels_commands_history[0] = 0.f;
    wheels_commands_history[1] = 0.f;
    wheels_commands_history[2] = 0.f;
    wheels_commands_history[3] = 0.f;
    last_target_wheel_vel[0] = 0.f;
    last_target_wheel_vel[1] = 0.f;
}

void ObsPack::pack(float input[kObsDim], const float command[4],
                   const SimFrame &frame, const HoldObsOut &hold) {
    float avg_thigh = (frame.thigh_l + frame.thigh_r) * 0.5f - cfg.thigh_hi;
    float thigh_len = cfg.thigh_lo - cfg.thigh_hi;
    float pitch_zero_now = cfg.pitch_zero_low +
        (cfg.pitch_zero_high - cfg.pitch_zero_low) *
        powf(avg_thigh / thigh_len, cfg.pitch_power);

    input[0] = command[0];
    input[1] = command[1];
    input[2] = command[2];
    input[3] = command[3];
    input[4] = frame.gyro[0];
    input[5] = frame.gyro[1];
    input[6] = frame.gyro[2];
    input[7] = hold.obs_dpsi;
    input[8] = (sinf(frame.pitch_rad) - sinf(pitch_zero_now)) * cfg.sinpitch_scale;
    input[9] = (sinf(frame.roll_rad) - sinf(command[3])) * cfg.sinroll_scale;
    input[10] = frame.thigh_l;
    input[11] = hold.obs_ds_l;
    input[12] = frame.thigh_r;
    input[13] = hold.obs_ds_r;
    // 训练 _get_obs 大腿速度维是零均值噪声；无噪声时为 0。不要喂真实舵机速度。
    input[14] = 0.f;
    smoothed_wheel_vel[0] = cfg.wheel_vel_smoothing_alpha * smoothed_wheel_vel[0] +
        (1.0f - cfg.wheel_vel_smoothing_alpha) * frame.wheel_vel_l;
    smoothed_wheel_vel[1] = cfg.wheel_vel_smoothing_alpha * smoothed_wheel_vel[1] +
        (1.0f - cfg.wheel_vel_smoothing_alpha) * frame.wheel_vel_r;
    input[15] = smoothed_wheel_vel[0];
    input[16] = 0.f;
    input[17] = smoothed_wheel_vel[1];
    input[18] = smoothed_wheel_vel[0] * cfg.wheel_radius;
    input[19] = smoothed_wheel_vel[1] * cfg.wheel_radius;
    input[20] = wheels_commands_history[0];
    input[21] = wheels_commands_history[1];
    input[22] = wheels_commands_history[2];
    input[23] = wheels_commands_history[3];
    input[24] = last_target_wheel_vel[0] - frame.wheel_vel_l;
    input[25] = last_target_wheel_vel[1] - frame.wheel_vel_r;
}

void ObsPack::note_policy_action(const float last_act[4], const SimFrame &frame) {
    // 轮残差进观测历史。大腿 bias/scale 仍由 Control 做 PID 目标，不在这里。
    float residual_l = (last_act[2] + cfg.transform_action_bias[2]) *
        cfg.transform_action_scale[2] * cfg.vel_pid_target_scale;
    float residual_r = (last_act[3] + cfg.transform_action_bias[3]) *
        cfg.transform_action_scale[3] * cfg.vel_pid_target_scale;
    wheels_commands_history[0] = wheels_commands_history[2];
    wheels_commands_history[1] = wheels_commands_history[3];
    wheels_commands_history[2] = residual_l;
    wheels_commands_history[3] = residual_r;
    float tgt_l = clip(residual_l + frame.wheel_vel_l,
                       -cfg.vel_pid_target_scale, cfg.vel_pid_target_scale);
    float tgt_r = clip(residual_r + frame.wheel_vel_r,
                       -cfg.vel_pid_target_scale, cfg.vel_pid_target_scale);
    tgt_l = tgt_l * cfg.vel_pid_smoothing +
        last_target_wheel_vel[0] * (1.0f - cfg.vel_pid_smoothing);
    tgt_r = tgt_r * cfg.vel_pid_smoothing +
        last_target_wheel_vel[1] * (1.0f - cfg.vel_pid_smoothing);
    last_target_wheel_vel[0] = tgt_l;
    last_target_wheel_vel[1] = tgt_r;
}
