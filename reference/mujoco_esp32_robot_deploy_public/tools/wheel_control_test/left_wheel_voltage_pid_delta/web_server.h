#pragma once

#include <Arduino.h>

#if defined(ARDUINO_ARCH_ESP32)
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#endif

// Start web task pinned to core0
void web_server_start_task();
