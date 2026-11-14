#include "MPU6050Mahony.h"

#include <math.h>

namespace {
constexpr uint8_t MPU6050_ADDR = 0x68;
constexpr uint8_t REG_SMPLRT_DIV = 0x19;
constexpr uint8_t REG_CONFIG = 0x1a;
constexpr uint8_t REG_GYRO_CONFIG = 0x1b;
constexpr uint8_t REG_ACCEL_CONFIG = 0x1c;
constexpr uint8_t REG_DATA = 0x3b;
constexpr uint8_t REG_PWR_MGMT_1 = 0x6b;
constexpr float IMU_RAD_TO_DEG = 57.2957795f;
constexpr float IMU_DEG_TO_RAD = 0.0174532925f;
constexpr float TWO_KP = 4.0f;
constexpr float TWO_KI = 0.04f;

int16_t readInt16(TwoWire &wire) {
  return (int16_t)((uint16_t)wire.read() << 8 | (uint8_t)wire.read());
}
}

MPU6050Mahony::MPU6050Mahony(TwoWire &wire, bool flip_x_180)
    : wire_(wire), flip_x_180_(flip_x_180) {}

bool MPU6050Mahony::writeRegister(uint8_t reg, uint8_t value) {
  wire_.beginTransmission(MPU6050_ADDR);
  wire_.write(reg);
  wire_.write(value);
  return wire_.endTransmission() == 0;
}

bool MPU6050Mahony::readFrame(RawFrame &frame) {
  wire_.beginTransmission(MPU6050_ADDR);
  wire_.write(REG_DATA);
  if (wire_.endTransmission(false) != 0 ||
      wire_.requestFrom((int)MPU6050_ADDR, 14) != 14 ||
      wire_.available() < 14) {
    while (wire_.available()) wire_.read();
    ++read_error_count_;
    return false;
  }

  frame.acc_x = readInt16(wire_);
  frame.acc_y = readInt16(wire_);
  frame.acc_z = readInt16(wire_);
  readInt16(wire_);
  frame.gyro_x = readInt16(wire_);
  frame.gyro_y = readInt16(wire_);
  frame.gyro_z = readInt16(wire_);
  return true;
}

bool MPU6050Mahony::begin() {
  if (!writeRegister(REG_SMPLRT_DIV, 0x00) ||
      !writeRegister(REG_CONFIG, 0x00) ||
      !writeRegister(REG_GYRO_CONFIG, 0x08) ||
      !writeRegister(REG_ACCEL_CONFIG, 0x00) ||
      !writeRegister(REG_PWR_MGMT_1, 0x01)) {
    return false;
  }
  delay(100);
  RawFrame frame;
  if (!readFrame(frame)) return false;
  initializeAttitude(frame);
  last_update_us_ = micros();
  return true;
}

void MPU6050Mahony::initializeAttitude(const RawFrame &frame) {
  float ax = frame.acc_x / 16384.0f;
  float ay = frame.acc_y / 16384.0f;
  float az = frame.acc_z / 16384.0f;
  if (flip_x_180_) {
    ay = -ay;
    az = -az;
  }
  float roll = atan2f(ay, az);
  float pitch = atan2f(-ax, sqrtf(ay * ay + az * az));
  float cr = cosf(roll * 0.5f);
  float sr = sinf(roll * 0.5f);
  float cp = cosf(pitch * 0.5f);
  float sp = sinf(pitch * 0.5f);
  q0_ = cr * cp;
  q1_ = sr * cp;
  q2_ = cr * sp;
  q3_ = -sr * sp;
  roll_deg_ = roll * IMU_RAD_TO_DEG;
  pitch_deg_ = pitch * IMU_RAD_TO_DEG;
  yaw_deg_ = 0.0f;
  integral_x_ = integral_y_ = integral_z_ = 0.0f;
}

bool MPU6050Mahony::calcGyroOffsets(bool console, uint16_t delayBefore,
                                    uint16_t delayAfter) {
  delay(delayBefore);
  if (console) Serial.println("Calculating gyro offsets; keep MPU6050 still");

  float sum_x = 0.0f;
  float sum_y = 0.0f;
  float sum_z = 0.0f;
  uint16_t valid_samples = 0;
  RawFrame frame;
  for (uint16_t attempt = 0; attempt < 3300 && valid_samples < 3000; ++attempt) {
    if (!readFrame(frame)) continue;
    sum_x += frame.gyro_x / 65.5f;
    sum_y += frame.gyro_y / 65.5f;
    sum_z += frame.gyro_z / 65.5f;
    ++valid_samples;
  }
  if (valid_samples < 3000) return false;

  gyro_offset_x_dps_ = sum_x / valid_samples;
  gyro_offset_y_dps_ = sum_y / valid_samples;
  gyro_offset_z_dps_ = sum_z / valid_samples;
  initializeAttitude(frame);
  last_update_us_ = micros();
  total_update_cost_us_ = 0;
  update_count_ = 0;
  read_error_count_ = 0;

  if (console) {
    Serial.printf("Gyro offsets: %.6f, %.6f, %.6f deg/s\n",
                  gyro_offset_x_dps_, gyro_offset_y_dps_, gyro_offset_z_dps_);
  }
  delay(delayAfter);
  return true;
}

bool MPU6050Mahony::update() {
  uint32_t started_us = micros();
  RawFrame frame;
  if (!readFrame(frame)) {
    recordUpdateCost(started_us);
    return false;
  }

  acc_x_g_ = frame.acc_x / 16384.0f;
  acc_y_g_ = frame.acc_y / 16384.0f;
  acc_z_g_ = frame.acc_z / 16384.0f;
  gyro_x_dps_ = frame.gyro_x / 65.5f - gyro_offset_x_dps_;
  gyro_y_dps_ = frame.gyro_y / 65.5f - gyro_offset_y_dps_;
  gyro_z_dps_ = frame.gyro_z / 65.5f - gyro_offset_z_dps_;
  if (flip_x_180_) {
    acc_y_g_ = -acc_y_g_;
    acc_z_g_ = -acc_z_g_;
    gyro_y_dps_ = -gyro_y_dps_;
    gyro_z_dps_ = -gyro_z_dps_;
  }

  uint32_t now_us = micros();
  float dt = (now_us - last_update_us_) * 1.0e-6f;
  last_update_us_ = now_us;
  if (dt <= 0.0f || dt > 0.05f) {
    recordUpdateCost(started_us);
    return false;
  }

  float gx = gyro_x_dps_ * IMU_DEG_TO_RAD;
  float gy = gyro_y_dps_ * IMU_DEG_TO_RAD;
  float gz = gyro_z_dps_ * IMU_DEG_TO_RAD;
  float acc_norm = sqrtf(acc_x_g_ * acc_x_g_ + acc_y_g_ * acc_y_g_ +
                         acc_z_g_ * acc_z_g_);
  if (acc_norm > 0.5f && acc_norm < 1.5f) {
    float ax = acc_x_g_ / acc_norm;
    float ay = acc_y_g_ / acc_norm;
    float az = acc_z_g_ / acc_norm;
    float vx = 2.0f * (q1_ * q3_ - q0_ * q2_);
    float vy = 2.0f * (q0_ * q1_ + q2_ * q3_);
    float vz = q0_ * q0_ - q1_ * q1_ - q2_ * q2_ + q3_ * q3_;
    float ex = ay * vz - az * vy;
    float ey = az * vx - ax * vz;
    float ez = ax * vy - ay * vx;

    if (TWO_KI > 0.0f) {
      integral_x_ += 0.5f * TWO_KI * ex * dt;
      integral_y_ += 0.5f * TWO_KI * ey * dt;
      integral_z_ += 0.5f * TWO_KI * ez * dt;
      gx += integral_x_;
      gy += integral_y_;
      gz += integral_z_;
    }
    gx += 0.5f * TWO_KP * ex;
    gy += 0.5f * TWO_KP * ey;
    gz += 0.5f * TWO_KP * ez;
  }

  float half_dt = 0.5f * dt;
  float qa = q0_;
  float qb = q1_;
  float qc = q2_;
  float qd = q3_;
  q0_ += (-qb * gx - qc * gy - qd * gz) * half_dt;
  q1_ += (qa * gx + qc * gz - qd * gy) * half_dt;
  q2_ += (qa * gy - qb * gz + qd * gx) * half_dt;
  q3_ += (qa * gz + qb * gy - qc * gx) * half_dt;
  float q_norm = sqrtf(q0_ * q0_ + q1_ * q1_ + q2_ * q2_ + q3_ * q3_);
  q0_ /= q_norm;
  q1_ /= q_norm;
  q2_ /= q_norm;
  q3_ /= q_norm;

  roll_deg_ = atan2f(2.0f * (q0_ * q1_ + q2_ * q3_),
                     1.0f - 2.0f * (q1_ * q1_ + q2_ * q2_)) * IMU_RAD_TO_DEG;
  float sin_pitch = 2.0f * (q0_ * q2_ - q3_ * q1_);
  sin_pitch = constrain(sin_pitch, -1.0f, 1.0f);
  pitch_deg_ = asinf(sin_pitch) * IMU_RAD_TO_DEG;
  yaw_deg_ = atan2f(2.0f * (q0_ * q3_ + q1_ * q2_),
                    1.0f - 2.0f * (q2_ * q2_ + q3_ * q3_)) * IMU_RAD_TO_DEG;

  recordUpdateCost(started_us);
  return true;
}

void MPU6050Mahony::recordUpdateCost(uint32_t started_us) {
  last_update_cost_us_ = micros() - started_us;
  total_update_cost_us_ += last_update_cost_us_;
  ++update_count_;
}

uint32_t MPU6050Mahony::getAverageUpdateCostUs() const {
  return update_count_ ? total_update_cost_us_ / update_count_ : 0;
}
