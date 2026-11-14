#pragma once

#include <Arduino.h>
#include <Wire.h>

class MPU6050Mahony {
public:
  explicit MPU6050Mahony(TwoWire &wire, bool flip_x_180 = false);

  bool begin();
  bool calcGyroOffsets(bool console = false, uint16_t delayBefore = 1000,
                       uint16_t delayAfter = 3000);
  bool update();

  float getAngleX() const { return roll_deg_; }
  float getAngleY() const { return pitch_deg_; }
  float getAngleZ() const { return yaw_deg_; }
  float getGyroX() const { return gyro_x_dps_; }
  float getGyroY() const { return gyro_y_dps_; }
  float getGyroZ() const { return gyro_z_dps_; }
  float getAccX() const { return acc_x_g_; }
  float getAccY() const { return acc_y_g_; }
  float getAccZ() const { return acc_z_g_; }

  uint32_t getLastUpdateCostUs() const { return last_update_cost_us_; }
  uint32_t getAverageUpdateCostUs() const;
  uint32_t getReadErrorCount() const { return read_error_count_; }

private:
  struct RawFrame {
    int16_t acc_x;
    int16_t acc_y;
    int16_t acc_z;
    int16_t gyro_x;
    int16_t gyro_y;
    int16_t gyro_z;
  };

  bool writeRegister(uint8_t reg, uint8_t value);
  bool readFrame(RawFrame &frame);
  void initializeAttitude(const RawFrame &frame);
  void recordUpdateCost(uint32_t started_us);

  TwoWire &wire_;
  bool flip_x_180_;
  float gyro_offset_x_dps_ = 0.0f;
  float gyro_offset_y_dps_ = 0.0f;
  float gyro_offset_z_dps_ = 0.0f;
  float q0_ = 1.0f;
  float q1_ = 0.0f;
  float q2_ = 0.0f;
  float q3_ = 0.0f;
  float integral_x_ = 0.0f;
  float integral_y_ = 0.0f;
  float integral_z_ = 0.0f;
  float roll_deg_ = 0.0f;
  float pitch_deg_ = 0.0f;
  float yaw_deg_ = 0.0f;
  float gyro_x_dps_ = 0.0f;
  float gyro_y_dps_ = 0.0f;
  float gyro_z_dps_ = 0.0f;
  float acc_x_g_ = 0.0f;
  float acc_y_g_ = 0.0f;
  float acc_z_g_ = 0.0f;
  uint32_t last_update_us_ = 0;
  uint32_t last_update_cost_us_ = 0;
  uint64_t total_update_cost_us_ = 0;
  uint32_t update_count_ = 0;
  uint32_t read_error_count_ = 0;
};
