#include "io.h"
#include <cmath>

namespace real_io {

RealAttitude real_attitude;

static float clip(float x, float lo, float hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

// 历史固件用 3.1415 不是 M_PI。改了观测会整体偏一丁点，和已部署尺度不一致。
static float deg2rad(float deg) { return deg * 3.1415f / 180.f; }

void mapping_thighs_position(float thigh_left, float thigh_right,
                             float &sim_thigh_left, float &sim_thigh_right) {
    // STS3032 计数：下蹲 / 站立两端点是实机标定的。
    // 仿真里下蹲约 0 rad、站立约 -0.5 rad（大腿向下为负）。
    const float thigh_left_squad = 2136;
    const float thigh_left_stand = 2676;
    const float thigh_right_squad = 2019;
    const float thigh_right_stand = 1479;  // 右腿计数方向与左腿相反
    const float sim_thigh_left_squad = 0;
    const float sim_thigh_left_stand = -0.5f;
    const float sim_thigh_right_squad = 0;
    const float sim_thigh_right_stand = -0.5f;
    sim_thigh_left = sim_thigh_left_squad +
        (sim_thigh_left_stand - sim_thigh_left_squad) *
        (thigh_left - thigh_left_squad) / (thigh_left_stand - thigh_left_squad);
    sim_thigh_right = sim_thigh_right_stand +
        (sim_thigh_right_squad - sim_thigh_right_stand) *
        (thigh_right - thigh_right_stand) / (thigh_right_squad - thigh_right_stand);
}

void inv_mapping_thighs_position(float sim_thigh_left, float sim_thigh_right,
                                 float &thigh_left, float &thigh_right,
                                 float thigh_lo, float thigh_hi) {
    const float thigh_left_squad = 2136;
    const float thigh_left_stand = 2676;
    const float thigh_right_squad = 2019;
    const float thigh_right_stand = 1479;
    const float sim_thigh_left_squad = 0;
    const float sim_thigh_left_stand = -0.5f;
    const float sim_thigh_right_squad = 0;
    const float sim_thigh_right_stand = -0.5f;

    // 先限在策略大腿范围内，再映回舵机计数，避免写出机械行程。
    sim_thigh_left = clip(sim_thigh_left, thigh_lo, thigh_hi);
    sim_thigh_right = clip(sim_thigh_right, thigh_lo, thigh_hi);

    thigh_left = (sim_thigh_left - sim_thigh_left_squad) *
        (thigh_left_stand - thigh_left_squad) /
        (sim_thigh_left_stand - sim_thigh_left_squad) + thigh_left_squad;
    thigh_right = (sim_thigh_right - sim_thigh_right_stand) *
        (thigh_right_squad - thigh_right_stand) /
        (sim_thigh_right_squad - sim_thigh_right_stand) + thigh_right_stand;
}

void mapping_thighs_velocity(float thigh_left_vel, float thigh_right_vel,
                             float &sim_thigh_left_vel, float &sim_thigh_right_vel) {
    // 4096 计数 / 圈。左腿从蹲到站在实机是正速度，仿真是负速度，所以左腿再取负。
    const float scale = 2 * 3.1415f / 4096;
    sim_thigh_left_vel = -thigh_left_vel * scale;
    sim_thigh_right_vel = thigh_right_vel * scale;
}

void to_sim_frame(const float *sensors, int dim, SimFrame *frame) {
    // raw sensors 由 wl_pro_robot.ino 的 get_sensors_data 填写：
    // 角度仍是度、轮角仍是 FOC 原符号。
    //   [0] time
    //   [1] roll deg   [2] pitch deg   [3] yaw deg
    //   [4..6] gyro deg/s   [7..9] acc
    //   [10] 左轮角   [11] 右轮角   [12] 左轮速   [13] 右轮速
    //   [14] 左大腿计数 [15] 右大腿计数 [16] 左大腿速 [17] 右大腿速
    (void)dim;
    *frame = SimFrame();

    frame->roll_rad = deg2rad(sensors[1]);
    frame->pitch_rad = deg2rad(sensors[2]);
    frame->yaw_rad = deg2rad(sensors[3]);
    real_attitude.roll_rad = frame->roll_rad;
    real_attitude.pitch_rad = frame->pitch_rad;
    real_attitude.yaw_rad = frame->yaw_rad;
    frame->gyro[0] = deg2rad(sensors[4]);
    frame->gyro[1] = deg2rad(sensors[5]);
    frame->gyro[2] = deg2rad(sensors[6]);

    // 轮铰链：实机 FOC 正方向与仿真 XML 相反，必须取负才能和 qpos[9]/[12] 同号。
    frame->q_l = -sensors[10];
    frame->q_r = -sensors[11];
    frame->wheel_vel_l = -sensors[12];
    frame->wheel_vel_r = -sensors[13];

    mapping_thighs_position(sensors[14], sensors[15], frame->thigh_l, frame->thigh_r);
    mapping_thighs_velocity(sensors[16], sensors[17],
                            frame->thigh_vel_l, frame->thigh_vel_r);
}

}  // namespace real_io
