#include <Arduino.h>
#include <SimpleFOC.h>

#if defined(ARDUINO_ARCH_ESP32)
#include "esp_adc_cal.h"
#endif

// Left wheel: motor1 + sensor1 + driver1 (copied from wl_pro_robot.ino)

// Motor pole-pairs
// Your log shows: "PP check: fail - estimated pp: 7.83" which is very close to 8.
// If you still see PP check fail or the motor feels weak/rough, try 7 or 8 here.
#ifndef MOTOR1_POLE_PAIRS
#define MOTOR1_POLE_PAIRS 7
#endif
BLDCMotor motor1 = BLDCMotor(MOTOR1_POLE_PAIRS);

// 3PWM driver pins: (UH, VH, WH, EN)
BLDCDriver3PWM driver1 = BLDCDriver3PWM(32, 33, 25, 22);

// I2C for AS5600
TwoWire I2Cone = TwoWire(0);
MagneticSensorI2C sensor1 = MagneticSensorI2C(AS5600_I2C);

// ---- Velocity target (rad/s) ----
static float g_left_target_velocity = 5.0f;

// ---- Actual bus/battery voltage measurement (via ADC + divider) ----
// NOTE: You must have a resistor divider wired to this ADC pin.
#ifndef BAT_PIN
#define BAT_PIN 35
#endif

// Divider ratio: Vbat = Vadc * BAT_DIVIDER_RATIO
// In your original project you used 3.97.
#ifndef BAT_DIVIDER_RATIO
#define BAT_DIVIDER_RATIO 3.97f
#endif

#if defined(ARDUINO_ARCH_ESP32)
static esp_adc_cal_characteristics_t g_adc_chars;

static void adc_init_for_battery() {
  // ESP32 Arduino core typically configures ADC internally; we just characterize.
  esp_adc_cal_characterize(ADC_UNIT_1, ADC_ATTEN_DB_11, ADC_WIDTH_BIT_12, 0, &g_adc_chars);
}

static float read_battery_voltage() {
  uint32_t raw = analogRead(BAT_PIN);
  uint32_t mv = esp_adc_cal_raw_to_voltage(raw, &g_adc_chars);
  return (mv * BAT_DIVIDER_RATIO) / 1000.0f;
}
#else
static void adc_init_for_battery() {}
static float read_battery_voltage() {
  // Fallback: no calibrated reading on non-ESP32.
  return NAN;
}
#endif

// 用 PID_velocity 控制左轮目标速度（单位：rad/s）
void set_left_wheel_target_velocity(float target_velocity_rad_s) {
  g_left_target_velocity = target_velocity_rad_s;
}

static void setup_left_wheel_velocity_pid() {
  // I2C (SDA=19, SCL=18)
  I2Cone.begin(19, 18, 400000UL);
  sensor1.init(&I2Cone);

  // Driver
  driver1.voltage_power_supply = 8;  // 与原工程一致
  driver1.init();

  // Link
  motor1.linkSensor(&sensor1);
  motor1.linkDriver(&driver1);

  // We are NOT doing torque/voltage command anymore.
  // Velocity loop will use PID_velocity internally.
  motor1.torque_controller = TorqueControlType::voltage;
  motor1.controller = MotionControlType::velocity;

  // Limits (建议根据电源与电机发热情况调整)
  motor1.voltage_limit = 7.0;    // 最大输出电压（给速度环用）<= voltage_power_supply
  motor1.velocity_limit = 50.0;  // rad/s

  // Velocity PID (copied from wl_pro_robot.ino defaults, tune as needed)
  motor1.PID_velocity.P = 0.05;
  motor1.PID_velocity.I = 1.0;
  motor1.PID_velocity.D = 0.0;
  motor1.PID_velocity.output_ramp = 0.0;
  // Make sure the velocity PID output (voltage) is allowed to reach voltage_limit
  motor1.PID_velocity.limit = motor1.voltage_limit;

  // Optional: smooth measurement
  motor1.LPF_velocity.Tf = 0.01f;

  // Init
  motor1.useMonitoring(Serial);
  motor1.monitor_downsample = 50;

  // IMPORTANT: sensor alignment voltage (copied from your original wl_pro_robot.ino)
  // If the wheel has large static friction, you may need a higher value (but <= supply).
  motor1.voltage_sensor_align = 7.0;

  motor1.init();
  int foc_result = motor1.initFOC();
  Serial.print("initFOC result=");
  Serial.println(foc_result);

  // Quick sanity prints (helps diagnose “no movement” cases)
  Serial.print("AS5600 angle(rad)=");
  Serial.println(sensor1.getAngle(), 6);
  Serial.print("Initial shaft_velocity(rad/s)=");
  Serial.println(motor1.shaft_velocity, 6);

  Serial.println("Left wheel velocity PID ready.");
  Serial.println("Send target velocity (rad/s) via Serial, e.g. 10 or -5");
}

static void handle_serial_setpoint() {
  // Simple serial setpoint: send a number like "12.5\n"
  if (!Serial.available()) return;

  float v = Serial.parseFloat();
  // Consume trailing newline/space
  while (Serial.available()) {
    char c = (char)Serial.peek();
    if (c == '\n' || c == '\r' || c == ' ' || c == '\t') {
      Serial.read();
    } else {
      break;
    }
  }
  set_left_wheel_target_velocity(v);
  Serial.print("target_velocity(rad/s)=");
  Serial.println(g_left_target_velocity, 4);
}

void setup() {
  Serial.begin(115200);
  delay(200);

  adc_init_for_battery();
  setup_left_wheel_velocity_pid();
}

void loop() {
  uint32_t t0_us = micros();
  handle_serial_setpoint();

  // FOC loop
  motor1.loopFOC();

  // velocity control (rad/s)
  motor1.move(g_left_target_velocity);

  // Optional debug
  static uint32_t last = 0;
  uint32_t now1 = millis();
  if (now1 - last > 2000) {
    last = now1;
    Serial.print(read_battery_voltage(), 3);
    Serial.print(", ");
    Serial.print(g_left_target_velocity, 4);
    Serial.print(", ");
    Serial.print(motor1.shaft_velocity, 4);
    Serial.print(", ");
    Serial.println((micros() - t0_us) / 1000.0f, 3);
  }
}
