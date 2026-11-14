#include <Arduino.h>

#include "app.h"
#include "pid_control.h"

// 1: PID 控制模式（速度 PID + 日志）
// 0: 纯电压控制模式（常值 Uq，无 PID、无日志）
#define USE_PID_MODE 0

// 0: 使用 driver1 引脚 (32,33,25,22)
// 1: 使用 driver2 引脚 (26,27,14,12)，与 wl_pro_robot.ino 保持一致
#define USE_DRIVER2 1

// 纯电压控制下使用的相电压大小（单位：V，约束在 [-7, 7]）
static constexpr float kPureVoltageUq = 0.0f;

void setup() {
  // 在启动控制任务之前选择使用哪个驱动引脚
  pid_set_use_driver2(USE_DRIVER2 != 0);

#if USE_PID_MODE
  app_start_pid();
#else
  app_start_voltage(kPureVoltageUq);
#endif
}

void loop() {
  // 所有控制运行在 FreeRTOS 任务中，这里保持空闲即可。
  delay(1000);
}
