#include "pid_control.h"

#include <SimpleFOC.h>

#if defined(ARDUINO_ARCH_ESP32)
#include "esp_adc_cal.h"
#include "esp_heap_caps.h"
#endif

// ---------------- Hardware ----------------
#ifndef MOTOR1_POLE_PAIRS
#define MOTOR1_POLE_PAIRS 7
#endif

static BLDCMotor motor1 = BLDCMotor(MOTOR1_POLE_PAIRS);
static BLDCDriver3PWM driver1 = BLDCDriver3PWM(32, 33, 25, 22);
static BLDCDriver3PWM driver2 = BLDCDriver3PWM(26, 27, 14, 12);

// Runtime flag selecting which driver pin mapping to use.
// false: driver1 pins (32,33,25,22)
// true : driver2 pins (26,27,14,12) as in wl_pro_robot.ino
static volatile bool g_use_driver2 = false;

static inline BLDCDriver3PWM& active_driver() {
  return g_use_driver2 ? driver2 : driver1;
}

static TwoWire I2Cone = TwoWire(0);
static TwoWire I2Ctwo = TwoWire(1);
static MagneticSensorI2C sensor1 = MagneticSensorI2C(AS5600_I2C);
static MagneticSensorI2C sensor2 = MagneticSensorI2C(AS5600_I2C);

// ---------------- Targets / state ----------------
static float g_target_velocity = 10.0f;
static float g_uq_cmd = 0.0f;
static float g_lpf_tf_s = 0.01f;
static LowPassFilter g_vel_lpf = LowPassFilter(0.01f);

// 0: velocity PID mode (default)
// 1: pure voltage mode (bypass PID, constant Uq)
static volatile uint8_t g_control_mode = 0;

#if defined(ARDUINO_ARCH_ESP32)
// Protect config updates coming from the web task (core0).
static portMUX_TYPE g_cfg_mux = portMUX_INITIALIZER_UNLOCKED;
#endif

static inline float clampf(float x, float lo, float hi) {
  if (x < lo) return lo;
  if (x > hi) return hi;
  return x;
}

// ---------------- Speed PID (delta-Uq) ----------------
class SpeedPidDeltaUq {
 public:
  void reset() {
    error_prev2_ = 0.0f;
    error_prev_ = 0.0f;
    d_output_prev_ = 0.0f;
    timestamp_prev_us_ = micros();
  }

  float update(float target_velocity, float meas_velocity) {
    float error = target_velocity - meas_velocity;

    uint32_t timestamp_now_us = micros();
    float Ts = (timestamp_now_us - timestamp_prev_us_) * 1e-6f;
    if (!(Ts > 0.0f) || Ts > 0.5f) Ts = 1e-3f;

    float de = error - error_prev_;
    float dde = error - 2.0f * error_prev_ + error_prev2_;

    float d_output = kp_ * de + ki_ * Ts * error + kd_ * (dde / Ts);
    d_output = clampf(d_output, -limit_, limit_);

    if (output_ramp_ > 0.0f) {
      float d_rate = (d_output - d_output_prev_) / Ts;
      if (d_rate > output_ramp_) d_output = d_output_prev_ + output_ramp_ * Ts;
      else if (d_rate < -output_ramp_) d_output = d_output_prev_ - output_ramp_ * Ts;
    }

    error_prev2_ = error_prev_;
    error_prev_ = error;
    d_output_prev_ = d_output;
    timestamp_prev_us_ = timestamp_now_us;
    return d_output;
  }

  void setGains(float kp, float ki, float kd) {
    kp_ = kp;
    ki_ = ki;
    kd_ = kd;
  }

  void getGains(float* kp, float* ki, float* kd) const {
    if (kp) *kp = kp_;
    if (ki) *ki = ki_;
    if (kd) *kd = kd_;
  }

 private:
  float kp_ = 0.05f;
  float ki_ = 2.0f;
  float kd_ = 0.0f;
  float output_ramp_ = 0.0f;
  float limit_ = 5.0f;

  float error_prev2_ = 0.0f;
  float error_prev_ = 0.0f;
  float d_output_prev_ = 0.0f;
  uint32_t timestamp_prev_us_ = 0;
};

static SpeedPidDeltaUq g_speed_pid;

// Keep last configured PID gains in RAM.
static float g_pid_kp = 0.05f;
static float g_pid_ki = 2.0f;
static float g_pid_kd = 0.0f;

void pid_set_gains(float kp, float ki, float kd) {
#if defined(ARDUINO_ARCH_ESP32)
  portENTER_CRITICAL(&g_cfg_mux);
#endif
  g_pid_kp = kp;
  g_pid_ki = ki;
  g_pid_kd = kd;
  g_speed_pid.setGains(kp, ki, kd);
#if defined(ARDUINO_ARCH_ESP32)
  portEXIT_CRITICAL(&g_cfg_mux);
#endif
}

void pid_get_gains(float* kp, float* ki, float* kd) {
#if defined(ARDUINO_ARCH_ESP32)
  portENTER_CRITICAL(&g_cfg_mux);
#endif
  if (kp) *kp = g_pid_kp;
  if (ki) *ki = g_pid_ki;
  if (kd) *kd = g_pid_kd;
#if defined(ARDUINO_ARCH_ESP32)
  portEXIT_CRITICAL(&g_cfg_mux);
#endif
}

// ---------------- Velocity LowPassFilter (Tf) ----------------
static volatile bool g_lpf_update_pending = false;
static volatile bool g_lpf_reset_pending = false;

void pid_set_lpf_tf(float tf_s) {
  if (!(tf_s >= 0.0f)) tf_s = 0.0f;
#if defined(ARDUINO_ARCH_ESP32)
  portENTER_CRITICAL(&g_cfg_mux);
#endif
  g_lpf_tf_s = tf_s;
  g_lpf_update_pending = true;
#if defined(ARDUINO_ARCH_ESP32)
  portEXIT_CRITICAL(&g_cfg_mux);
#endif
}

void pid_get_lpf_tf(float* tf_s) {
  if (!tf_s) return;
#if defined(ARDUINO_ARCH_ESP32)
  portENTER_CRITICAL(&g_cfg_mux);
#endif
  *tf_s = g_lpf_tf_s;
#if defined(ARDUINO_ARCH_ESP32)
  portEXIT_CRITICAL(&g_cfg_mux);
#endif
}

void pid_set_use_driver2(bool use_driver2) {
#if defined(ARDUINO_ARCH_ESP32)
  portENTER_CRITICAL(&g_cfg_mux);
#endif
  g_use_driver2 = use_driver2;
#if defined(ARDUINO_ARCH_ESP32)
  portEXIT_CRITICAL(&g_cfg_mux);
#endif
}

// ---------------- Battery voltage (ADC) ----------------
#ifndef BAT_PIN
#define BAT_PIN 35
#endif

#ifndef BAT_DIVIDER_RATIO
#define BAT_DIVIDER_RATIO 3.97f
#endif

#if defined(ARDUINO_ARCH_ESP32)
static esp_adc_cal_characteristics_t g_adc_chars;

static void adc_init_for_battery() {
  esp_adc_cal_characterize(ADC_UNIT_1, ADC_ATTEN_DB_11, ADC_WIDTH_BIT_12, 0, &g_adc_chars);
}

static float read_battery_voltage() {
  uint32_t raw = analogRead(BAT_PIN);
  uint32_t mv = esp_adc_cal_raw_to_voltage(raw, &g_adc_chars);
  return (mv * BAT_DIVIDER_RATIO) / 1000.0f;
}
#else
static void adc_init_for_battery() {}
static float read_battery_voltage() { return NAN; }
#endif

// ---------------- Motor setup ----------------
static void setup_motor_voltage_mode() {
  MagneticSensorI2C* sensor = nullptr;

  if (g_use_driver2) {
    // Match wl_pro_robot.ino: driver2 uses I2Ctwo on pins (23,5).
    I2Ctwo.begin(23, 5, 400000UL);
    sensor2.init(&I2Ctwo);
    sensor = &sensor2;
  } else {
    // Default: driver1 uses I2Cone on pins (19,18).
    I2Cone.begin(19, 18, 400000UL);
    sensor1.init(&I2Cone);
    sensor = &sensor1;
  }

  BLDCDriver3PWM& driver = active_driver();
  driver.voltage_power_supply = 8.0f;
  driver.init();

  motor1.linkSensor(sensor);
  motor1.linkDriver(&driver);

  motor1.torque_controller = TorqueControlType::voltage;
  motor1.controller = MotionControlType::torque;

  motor1.voltage_limit = 7.0f;
  motor1.velocity_limit = 50.0f;
  motor1.voltage_sensor_align = 7.0f;

  motor1.init();
  (void)motor1.initFOC();
}

// ---------------- Data logging ----------------
static constexpr uint32_t CONTROL_LOOP_PERIOD_US = 2000;
static constexpr uint32_t CONTROL_LOOP_HZ = 1000000UL / CONTROL_LOOP_PERIOD_US;
static constexpr uint32_t LOG_CAPACITY_DEFAULT = 12000;

uint32_t pidlog_period_us() {
  return CONTROL_LOOP_PERIOD_US;
}

static PidLogSample* g_log = nullptr;
static volatile uint32_t g_log_capacity = 0;
static volatile uint32_t g_log_count = 0;
static volatile bool g_log_running = false;
static volatile bool g_log_ready = false;
static volatile uint32_t g_log_duration_s = 0;
static volatile uint32_t g_log_stop_ms = 0;
static volatile uint32_t g_log_t0_us = 0;

#if defined(ARDUINO_ARCH_ESP32)
static portMUX_TYPE g_log_mux = portMUX_INITIALIZER_UNLOCKED;
#endif

bool pidlog_start(uint32_t duration_s) {
  const uint32_t cap = (uint32_t)g_log_capacity;
  if (cap == 0 || g_log == nullptr) {
    // logging not available
    return false;
  }

  const uint32_t max_s = (uint32_t)(cap / CONTROL_LOOP_HZ);
  if (duration_s > max_s) duration_s = max_s;
  if (duration_s < 1) duration_s = 1;

#if defined(ARDUINO_ARCH_ESP32)
  portENTER_CRITICAL(&g_log_mux);
#endif
  g_log_count = 0;
  g_log_ready = false;
  g_log_running = true;
  g_log_duration_s = duration_s;
  g_log_stop_ms = millis() + duration_s * 1000UL;
  g_log_t0_us = 0; // will be set by control task on first run tick
  g_lpf_reset_pending = true; // reset LPF at each run start
#if defined(ARDUINO_ARCH_ESP32)
  portEXIT_CRITICAL(&g_log_mux);
#endif

  return true;
}

static void pidlog_stop_if_running() {
#if defined(ARDUINO_ARCH_ESP32)
  portENTER_CRITICAL(&g_log_mux);
#endif
  if (g_log_running) {
    g_log_running = false;
    g_log_ready = true;
  }
#if defined(ARDUINO_ARCH_ESP32)
  portEXIT_CRITICAL(&g_log_mux);
#endif
}

static void force_motor_stop() {
  g_uq_cmd = 0.0f;
  g_speed_pid.reset();
  motor1.move(0.0f);
}

PidLogStatus pidlog_status() {
  PidLogStatus st;
#if defined(ARDUINO_ARCH_ESP32)
  portENTER_CRITICAL(&g_log_mux);
#endif
  st.running = g_log_running;
  st.ready = g_log_ready;
  st.samples = g_log_count;
  st.capacity = (uint32_t)g_log_capacity;
  st.duration_s = g_log_duration_s;
#if defined(ARDUINO_ARCH_ESP32)
  portEXIT_CRITICAL(&g_log_mux);
#endif
  return st;
}

bool pidlog_get_sample(uint32_t idx, PidLogSample* out) {
  if (!out) return false;
  bool ok = false;
#if defined(ARDUINO_ARCH_ESP32)
  portENTER_CRITICAL(&g_log_mux);
#endif
  if (idx < g_log_count) {
    if (g_log != nullptr) *out = g_log[idx];
    ok = true;
  }
#if defined(ARDUINO_ARCH_ESP32)
  portEXIT_CRITICAL(&g_log_mux);
#endif
  return ok;
}

// ---------------- Control task (core1, PID mode) ----------------
static void control_task(void* pv) {
  (void)pv;

  // Allocate log buffer at runtime to avoid large .bss usage.
  if (g_log == nullptr) {
    const uint32_t preferred_caps = MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT;
    const uint32_t fallback_caps = MALLOC_CAP_8BIT;

    uint32_t try_caps[] = {LOG_CAPACITY_DEFAULT, 10000, 8000, 6000};
    for (uint32_t i = 0; i < (sizeof(try_caps) / sizeof(try_caps[0])); i++) {
      const uint32_t cap = try_caps[i];
      const size_t bytes = (size_t)cap * sizeof(PidLogSample);
      void* p = heap_caps_malloc(bytes, preferred_caps);
      if (!p) p = heap_caps_malloc(bytes, fallback_caps);
      if (p) {
        g_log = (PidLogSample*)p;
        g_log_capacity = cap;
        break;
      }
    }

    if (g_log != nullptr && g_log_capacity > 0) {
      Serial.print("[log] buffer ok samples=");
      Serial.print((uint32_t)g_log_capacity);
      Serial.print(" bytes=");
      Serial.println((uint32_t)((size_t)g_log_capacity * sizeof(PidLogSample)));
    } else {
      Serial.println("[log] buffer alloc failed");
    }
  }

  adc_init_for_battery();
  setup_motor_voltage_mode();

  Serial.println("[control] ready (stopped). wait /start?seconds=N");

  g_speed_pid.reset();
  // Apply default/last gains.
  g_speed_pid.setGains(g_pid_kp, g_pid_ki, g_pid_kd);
  g_vel_lpf = LowPassFilter(g_lpf_tf_s);
  g_uq_cmd = 0.0f;
  force_motor_stop();

  uint32_t next_tick_us = micros() + CONTROL_LOOP_PERIOD_US;

  bool was_running = false;
  for (;;) {
    motor1.loopFOC();

    // armed?
    bool running;
    uint32_t stop_ms;
    uint32_t idx;
#if defined(ARDUINO_ARCH_ESP32)
    portENTER_CRITICAL(&g_log_mux);
#endif
    running = g_log_running;
    stop_ms = g_log_stop_ms;
    idx = g_log_count;
#if defined(ARDUINO_ARCH_ESP32)
    portEXIT_CRITICAL(&g_log_mux);
#endif

    if (!running) {
      if (was_running) {
        force_motor_stop();
        was_running = false;
      } else {
        motor1.move(0.0f);
      }
    } else {
      // Ensure time origin is set even if /start is called again while already running.
      if (!was_running || g_log_t0_us == 0) {
        const uint32_t t0 = micros();

        bool do_lpf_update = false;
        bool do_lpf_reset = false;
        float tf_s = 0.01f;

#if defined(ARDUINO_ARCH_ESP32)
        portENTER_CRITICAL(&g_log_mux);
#endif
        g_log_t0_us = t0;
        do_lpf_update = g_lpf_update_pending;
        do_lpf_reset = g_lpf_reset_pending;
        tf_s = g_lpf_tf_s;
        g_lpf_update_pending = false;
        g_lpf_reset_pending = false;
#if defined(ARDUINO_ARCH_ESP32)
        portEXIT_CRITICAL(&g_log_mux);
#endif

        if (do_lpf_update || do_lpf_reset) {
          g_vel_lpf = LowPassFilter(tf_s);
        }
      }

      was_running = true;

      const float raw_v = motor1.shaft_velocity;
      const float meas_v = g_vel_lpf(raw_v);
      float dUq = g_speed_pid.update(g_target_velocity, meas_v);
      g_uq_cmd = clampf(g_uq_cmd + dUq, -motor1.voltage_limit, motor1.voltage_limit);
      motor1.move(g_uq_cmd);

      if (g_log == nullptr || g_log_capacity == 0) {
        pidlog_stop_if_running();
      } else if ((uint32_t)millis() >= stop_ms || idx >= (uint32_t)g_log_capacity) {
        pidlog_stop_if_running();
        force_motor_stop();
        was_running = false;
      } else {
        // Fixed-point sample
        uint32_t t0_us;
      #if defined(ARDUINO_ARCH_ESP32)
        portENTER_CRITICAL(&g_log_mux);
      #endif
        t0_us = g_log_t0_us;
      #if defined(ARDUINO_ARCH_ESP32)
        portEXIT_CRITICAL(&g_log_mux);
      #endif

        const float batt_v = read_battery_voltage();
        int32_t batt_mV = (int32_t)(batt_v * 1000.0f + 0.5f);
        if (batt_mV < 0) batt_mV = 0;
        if (batt_mV > 65535) batt_mV = 65535;

        int32_t uq_mV = (int32_t)(g_uq_cmd * 1000.0f);
        uq_mV = clampf((float)uq_mV, -32768.0f, 32767.0f);

        int32_t target_centi = (int32_t)(g_target_velocity * 100.0f);
        target_centi = (int32_t)clampf((float)target_centi, -32768.0f, 32767.0f);

        int32_t raw_centi = (int32_t)(raw_v * 100.0f);
        raw_centi = (int32_t)clampf((float)raw_centi, -32768.0f, 32767.0f);

        int32_t meas_centi = (int32_t)(meas_v * 100.0f);
        meas_centi = (int32_t)clampf((float)meas_centi, -32768.0f, 32767.0f);

        PidLogSample s;
        // First sample should be exactly t=0.
        if (idx == 0 || t0_us == 0) s.t_us = 0;
        else s.t_us = (uint32_t)(micros() - t0_us);
        s.batt_mV = (uint16_t)batt_mV;
        s.uq_mV = (int16_t)uq_mV;
        s.target_centi = (int16_t)target_centi;
        s.raw_centi = (int16_t)raw_centi;
        s.meas_centi = (int16_t)meas_centi;

#if defined(ARDUINO_ARCH_ESP32)
        portENTER_CRITICAL(&g_log_mux);
#endif
        if (g_log_running && g_log != nullptr && g_log_count < (uint32_t)g_log_capacity) {
          g_log[g_log_count] = s;
          g_log_count++;
        }
#if defined(ARDUINO_ARCH_ESP32)
        portEXIT_CRITICAL(&g_log_mux);
#endif
      }
    }

    // fixed period pacing
    uint32_t now_us = micros();
    int32_t remaining_us = (int32_t)(next_tick_us - now_us);
    next_tick_us += CONTROL_LOOP_PERIOD_US;

    if (remaining_us > 0) {
      delayMicroseconds((uint32_t)remaining_us);
    } else {
      if (remaining_us < -(int32_t)CONTROL_LOOP_PERIOD_US) {
        next_tick_us = micros() + CONTROL_LOOP_PERIOD_US;
      }
    }
  }
}

void pid_control_start_task() {
#if defined(ARDUINO_ARCH_ESP32)
  xTaskCreatePinnedToCore(control_task, "control", 8192, nullptr, 2, nullptr, 1);
#endif
}

// ---------------- Pure voltage control task (core1) ----------------
static void voltage_task(void* pv) {
  (void)pv;

  adc_init_for_battery();
  setup_motor_voltage_mode();

  Serial.println("[control] pure voltage mode running");

  // Simple loop: keep FOC running and apply constant Uq (g_uq_cmd).
  for (;;) {
    motor1.loopFOC();
    motor1.move(g_uq_cmd);
    // Serial.print("[volt] Uq=");
    // Serial.print(g_uq_cmd, 3);
    // Serial.println(" V");
    // Small delay to yield CPU; control bandwidth is still high enough.
    delayMicroseconds(1000);
    
    vTaskDelay(1 / portTICK_PERIOD_MS);

  }
}

void pid_voltage_start_task(float uq_voltage) {
#if defined(ARDUINO_ARCH_ESP32)
  // Enter pure-voltage mode and set initial Uq.
  g_control_mode = 1;
  pid_set_voltage(uq_voltage);
  xTaskCreatePinnedToCore(voltage_task, "volt_ctrl", 4096, nullptr, 2, nullptr, 1);
#else
  (void)uq_voltage;
#endif
}

void pid_set_voltage(float uq_voltage) {
#if defined(ARDUINO_ARCH_ESP32)
  if (g_control_mode != 1) {
    // Ignore requests when not in pure-voltage mode.
    return;
  }

  float uq = clampf(uq_voltage, -7.0f, 7.0f);

  portENTER_CRITICAL(&g_cfg_mux);
  g_uq_cmd = uq;
  portEXIT_CRITICAL(&g_cfg_mux);

  Serial.print("[volt] set Uq=");
  Serial.print(uq, 3);
  Serial.println(" V");
#else
  (void)uq_voltage;
#endif
}

float pid_get_battery_voltage() {
#if defined(ARDUINO_ARCH_ESP32)
  return read_battery_voltage();
#else
  return NAN;
#endif
}
