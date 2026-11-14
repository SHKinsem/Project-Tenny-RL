#ifndef CONTROLLER_OBS_H
#define CONTROLLER_OBS_H

#include "hold_obs.h"

// =============================================================================
// ObsPack = 共享 26 维观测拼装（sim / real 同一套）
//
// HoldObs 只填 [7] 和 [11]/[13]。其余槽位（gyro、pitch/roll、大腿、轮速历史）
// 对着 host/controller/obs.py。不要在 Control 或 viewer 里再抄一份。
//
// note_policy_action 只更新观测用的轮速历史 / 目标残差。
// 真机 PID、仿真 env.step 仍是各自的执行器，不在这里。
// =============================================================================

static const int kObsDim = 26;

struct ObsPackConfig {
    float wheel_radius = 0.0195f;
    float thigh_lo = -0.4f;
    float thigh_hi = -0.06f;
    float pitch_zero_low = 0.22f;
    float pitch_zero_high = 0.05f;
    float pitch_power = 1.0f;
    float sinpitch_scale = 1.0f;
    float sinroll_scale = 1.0f;
    float wheel_vel_smoothing_alpha = 0.0f;
    float vel_pid_target_scale = 30.0f;
    float vel_pid_smoothing = 0.5f;
    float transform_action_bias[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float transform_action_scale[4] = {0.25f, 0.25f, 1.0f, 1.0f};
};

class ObsPack {
public:
    ObsPackConfig cfg;
    float smoothed_wheel_vel[2] = {0.f, 0.f};
    float wheels_commands_history[4] = {0.f, 0.f, 0.f, 0.f};
    float last_target_wheel_vel[2] = {0.f, 0.f};

    void reset();
    void pack(float input[kObsDim], const float command[4],
              const SimFrame &frame, const HoldObsOut &hold);
    void note_policy_action(const float last_act[4], const SimFrame &frame);

private:
    static float clip(float x, float lo, float hi);
};

#endif
