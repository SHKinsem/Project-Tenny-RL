#include "app.h"

#include <Arduino.h>

#if defined(ARDUINO_ARCH_ESP32)
#include "pid_control.h"
#include "web_server.h"
#endif

void app_start_pid() {
#if defined(ARDUINO_ARCH_ESP32)
  Serial.begin(115200);
  delay(200);
  Serial.println("[left_wheel_voltage_pid_delta] boot");

  pid_control_start_task();
  web_server_start_task();
#endif
}

void app_start_voltage(float uq_voltage) {
#if defined(ARDUINO_ARCH_ESP32)
  Serial.begin(115200);
  delay(200);
  Serial.println("[left_wheel_voltage_pid_delta] boot (pure voltage mode), voltage=" + String(uq_voltage, 3) + " V");

  // Start pure-voltage control task and web server (for /set_voltage, /status, etc.).
  pid_voltage_start_task(uq_voltage);
  web_server_start_task();
#endif
}

void app_start() {
  // Default to PID behavior to preserve existing code semantics.
  app_start_pid();
}
