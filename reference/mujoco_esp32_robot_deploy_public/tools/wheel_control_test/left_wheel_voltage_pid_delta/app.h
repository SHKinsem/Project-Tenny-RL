#pragma once

// Start in PID (velocity control + logging) mode.
void app_start_pid();

// Start in pure voltage control mode (no PID, no logging).
// The voltage parameter is in volts, typically within [-7, 7].
void app_start_voltage(float uq_voltage);

// Backward-compatible alias: default to PID mode.
void app_start();
