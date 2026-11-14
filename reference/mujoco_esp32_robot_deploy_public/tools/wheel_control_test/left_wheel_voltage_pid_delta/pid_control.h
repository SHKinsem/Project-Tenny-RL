#pragma once

#include <Arduino.h>

#if defined(ARDUINO_ARCH_ESP32)
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#endif

// Fixed-point sample to reduce RAM usage (with real timestamp):
// - t_us: microseconds since run start (first sample is 0)
// - batt_mV: battery voltage in millivolts
// - uq_mV: control voltage (Uq) in millivolts
// - *_centi: velocity in centi(rad/s)
struct PidLogSample {
  uint32_t t_us;
  uint16_t batt_mV;
  int16_t uq_mV;
  int16_t target_centi;
  int16_t raw_centi;
  int16_t meas_centi;
};

struct PidLogStatus {
  bool running;
  bool ready;
  uint32_t samples;
  uint32_t capacity;
  uint32_t duration_s;
};

// Control/log loop period in microseconds (used for CSV time column).
uint32_t pidlog_period_us();

// Start logging + enable control run for duration_s seconds.
// Returns false if logging buffer is not available.
bool pidlog_start(uint32_t duration_s);
PidLogStatus pidlog_status();
bool pidlog_get_sample(uint32_t idx, PidLogSample* out);

// Start PID control task (velocity PID, logging) pinned to core1
void pid_control_start_task();

// Select which BLDC driver pin mapping to use:
// false: driver1 (32,33,25,22), true: driver2 (26,27,14,12).
void pid_set_use_driver2(bool use_driver2);

// Start pure voltage control task pinned to core1 (no PID, no logging).
// uq_voltage: desired phase voltage [V], typically within [-7, 7].
void pid_voltage_start_task(float uq_voltage);

// Update the pure voltage command Uq (in volts, [-7, 7]).
// Effective only when running in pure voltage mode.
void pid_set_voltage(float uq_voltage);

// Read current battery voltage in volts (based on ADC & divider).
float pid_get_battery_voltage();
// PID gains configuration (kept in RAM; default values are set in pid_control.cpp).
void pid_set_gains(float kp, float ki, float kd);
void pid_get_gains(float* kp, float* ki, float* kd);

// Velocity LowPassFilter time constant (seconds). Set via web /start (optional).
// If not provided, last/default value is used.
void pid_set_lpf_tf(float tf_s);
void pid_get_lpf_tf(float* tf_s);
