#ifndef CONTROLLER_SIM_FRAME_H
#define CONTROLLER_SIM_FRAME_H

// =============================================================================
// SimFrame = 仿真坐标系下的本体状态（两条支路的共同语言）
//
//   真机传感器 -- src/real     --> SimFrame -- HoldObs + ObsPack --> 26 维观测
//   MJX 状态   -- 主机 sim/io  --> SimFrame -- HoldObs + ObsPack --> 同上
//
// HoldObs / ObsPack 只吃 SimFrame，不吃 IMU 度数、舵机计数、MJX qpos 下标。
// 单位：弧度、弧度/秒。轮角是未饱和累计角（FOC shaft_angle 取负之后）。
// 左右铰链与仿真 XML 同号：前进时 q_l 与 q_r 一起增加。
// 主机对照：controller/sim_frame.py
// =============================================================================

struct SimFrame {
    float q_l = 0.f;          // 左轮累计角 [rad]，仿真系
    float q_r = 0.f;          // 右轮累计角 [rad]，仿真系
    float yaw_rad = 0.f;      // 航向 [rad]，与 atan2(R10, R00) 同一套
    float pitch_rad = 0.f;    // 俯仰 [rad]，摔倒判断用
    float roll_rad = 0.f;     // 横滚 [rad]
    float gyro[3] = {0.f, 0.f, 0.f};  // 角速度 [rad/s]，已 deg→rad
    float thigh_l = 0.f;      // 左大腿关节角 [rad]（下蹲约 0，站立约 -0.5）
    float thigh_r = 0.f;
    float thigh_vel_l = 0.f;
    float thigh_vel_r = 0.f;
    float wheel_vel_l = 0.f;  // 轮速 [rad/s]，仿真系（已随轮角取负）
    float wheel_vel_r = 0.f;
};

#endif
