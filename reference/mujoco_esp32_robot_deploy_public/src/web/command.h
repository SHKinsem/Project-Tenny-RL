#ifndef WEB_COMMAND_H
#define WEB_COMMAND_H

// =============================================================================
// Web 操作员通道的共享映射：摇杆 / 高度 / 横滚 → command[4]
//
// Web 不是与 sim/real 并列的第三条本体支路。
// sim / real 解决「状态从哪来」；Web 解决「人怎么给 command」。两条本体支路都能接。
// HTTP 传输分家：真机 lqrwebserver，主机 host/web。映射对着 host/web/command.py。
//
// compute() 顺序（两边必须相同）：
//   1) 先用 K 的 cmd_* 填 command
//   2) 若当前页是 control，再用摇杆/滑条覆盖
// =============================================================================

namespace web_cmd {

// 与 host/web/command.py COMMAND_RANGE 同一套。Control 不要再抄一份。
extern const float kCommandRange[4][2];

float clip(float x, float lo, float hi);
float linear_map(float x, float x0, float x1, float y0, float y1);

// height_bar: 0~100。100 → range[2][0]（下蹲 -0.5），0 → range[2][1]（站立 0）。
void map_height(float height_bar, float command[4], const float range[4][2]);

// oscillation_bar: -50~50 → range[3]。
void map_roll(float oscillation_bar, float command[4], const float range[4][2]);

// 摇杆 x/y 约 -1~1，Y 朝前为正。|y|>=0.2 即满速 ±0.2（策略在 vx>-0.18 时仍前进）。
// 固件 atan 用 3.1415926；|xy|<0.1 当没推；偏航 |yaw_norm|>0.3 才算转向。
void map_joystick(float x, float y, float command[4], const float range[4][2],
                  bool *joystick_enable, bool *joystick_turn_enable);

// page 为 "control" 时覆盖 command[0..3] 的摇杆/滑条通道。
void apply_control_page(float joy_x, float joy_y,
                        float height_bar, float oscillation_bar,
                        float command[4], const float range[4][2],
                        bool *joystick_enable, bool *joystick_turn_enable);

}  // namespace web_cmd

#endif
