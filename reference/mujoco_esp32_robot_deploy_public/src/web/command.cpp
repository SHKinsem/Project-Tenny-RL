#include "web/command.h"
#include <cmath>

namespace web_cmd {

const float kCommandRange[4][2] = {
    {-0.2f, 0.2f},   // lin_vel_x
    {-1.5f, 1.5f},   // ang_vel_yaw
    {-0.5f, 0.f},    // height_value
    {-0.2f, 0.2f},   // roll_value
};

float clip(float x, float lo, float hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

float linear_map(float x, float x0, float x1, float y0, float y1) {
    return (x - x0) / (x1 - x0) * (y1 - y0) + y0;
}

void map_height(float height_bar, float command[4], const float range[4][2]) {
    command[2] = linear_map(height_bar, 100.f, 0.f, range[2][0], range[2][1]);
}

void map_roll(float oscillation_bar, float command[4], const float range[4][2]) {
    command[3] = linear_map(oscillation_bar, -50.f, 50.f, range[3][0], range[3][1]);
}

void map_joystick(float x, float y, float command[4], const float range[4][2],
                  bool *joystick_enable, bool *joystick_turn_enable) {
    bool enable = false;
    bool turn_enable = false;

    float magnitude = sqrtf(x * x + y * y);
    if (magnitude > 0.1f) {
        enable = true;
    } else {
        magnitude = 0.f;
    }

    // 用 |x|/|y| 的象限折叠，再按 x*y 符号翻回来。历史固件 π 是 3.1415926。
    float yaw_value = -atan2f(fabsf(x), fabsf(y));
    yaw_value = yaw_value * 2.f / 3.1415926f;
    if (fabsf(yaw_value) > 0.3f) {
        turn_enable = true;
    } else {
        yaw_value = 0.f;
    }
    if (!enable) {
        yaw_value = 0.f;
    }
    if (y * x < 0.f) {
        yaw_value = -yaw_value;
    }

    // 闭环实测 vx>=-0.15 仍前进、<=-0.18 才退。下拉过 0.2 行程就给满速。
    float vx_scale = fabsf(range[0][0]) > fabsf(range[0][1]) ? fabsf(range[0][0]) : fabsf(range[0][1]);
    const float dead = 0.1f;
    const float full_y = 0.2f;
    float x_vel = 0.f;
    if (enable && fabsf(y) > dead) {
        float t = clip((fabsf(y) - dead) / (full_y - dead), 0.f, 1.f);
        x_vel = (y > 0.f ? 1.f : -1.f) * t * vx_scale;
    }
    float yaw_vel = yaw_value * 1.f;

    command[0] = clip(x_vel, range[0][0], range[0][1]);
    command[1] = clip(yaw_vel, range[1][0], range[1][1]);

    if (joystick_enable) {
        *joystick_enable = enable;
    }
    if (joystick_turn_enable) {
        *joystick_turn_enable = turn_enable;
    }
}

void apply_control_page(float joy_x, float joy_y,
                        float height_bar, float oscillation_bar,
                        float command[4], const float range[4][2],
                        bool *joystick_enable, bool *joystick_turn_enable) {
    map_roll(oscillation_bar, command, range);
    map_height(height_bar, command, range);
    map_joystick(joy_x, joy_y, command, range, joystick_enable, joystick_turn_enable);
}

}  // namespace web_cmd
