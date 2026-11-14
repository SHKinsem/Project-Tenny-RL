#ifndef REAL_IO_H
#define REAL_IO_H

#include "controller/sim_frame.h"

// =============================================================================
// Real 支路：真机传感器 → SimFrame
//
// 只做单位、符号、舵机映射。不要在这里刷新锚点或拼 26 维观测。
// 仿真测不了这里的符号；填反了 HoldObs 仍会「算对」，机器人会反着走。
// 主机对照：real/io.py
// =============================================================================

namespace real_io {

struct RealAttitude {
    float yaw_rad = 0.0f;
    float pitch_rad = 0.0f;
    float roll_rad = 0.0f;
};

extern RealAttitude real_attitude;

// 把 .ino / robot 填的 raw sensors[] 变成仿真系 SimFrame。
void to_sim_frame(const float *sensors, int dim, SimFrame *frame);

// 大腿舵机计数 ↔ 仿真关节角 [rad]。正变换给观测，逆变换给舵机指令。
void mapping_thighs_position(float thigh_left, float thigh_right,
                             float &sim_thigh_left, float &sim_thigh_right);
void inv_mapping_thighs_position(float sim_thigh_left, float sim_thigh_right,
                                 float &thigh_left, float &thigh_right,
                                 float thigh_lo, float thigh_hi);
void mapping_thighs_velocity(float thigh_left_vel, float thigh_right_vel,
                             float &sim_thigh_left_vel, float &sim_thigh_right_vel);

}  // namespace real_io

#endif
