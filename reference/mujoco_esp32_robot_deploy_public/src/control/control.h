#ifndef CONTROL_H
#define CONTROL_H

#include <Arduino.h>
#include <ArduinoEigen.h>
#include <vector>
#include "assert.h"
#include "web/lqrwebserver.h"
#include "policy/rl_policy.h"
#include "controller/hold_obs.h"
#include "controller/obs.h"

// =============================================================================
// Control = 胶水：real I/O → HoldObs + ObsPack → 策略 / PID
//
// Arduino 只把 .ino 留在根目录。固件源码全在 src/（递归编译，src/ 在 include path）。
//
//   src/controller/  共享核 HoldObs + ObsPack + SimFrame  ↔ host/controller/
//   src/policy/      共享核 RL_Policy 权重/前向           ↔ host/policy/
//   src/real/        真机传感器 → SimFrame                ↔ host/real/
//   src/control/     本文件：接线。锚点 / 26 维拼装不要写回来
//   src/web/         操作员通道（摇杆映射 + HTTP）。sim / real 都能接
//   src/hal/         遥控协议、STS3032 舵机
//   host/sim/        仅主机：MJX → SimFrame；策略走 host/policy，不另 load ckpt
//   host/web/        同一套操作员映射；HTTP 仿真侧再接
// =============================================================================

using Eigen::MatrixXf;
using Eigen::VectorXf;

float deg2rad(float deg);
float clip(float x, float min, float max);

class ControlPID {
public:
    ControlPID(float kp = 0.003f, float ki = 0.0f, float kd = 0.0f,
        float out_min = -8.0f, float out_max = 8.0f)
        : kp_(kp), ki_(ki), kd_(kd),
          out_min_(out_min), out_max_(out_max),
                    setpoint_(0.0f),
                    integral_(0.0f), prev_error_(0.0f), has_prev_error_(false),
          last_time_us_(0), has_prev_time_(false) {}

    void setGains(float kp, float ki, float kd) {
        kp_ = kp;
        ki_ = ki;
        kd_ = kd;
    }

    void setSetpoint(float setpoint) {
        setpoint_ = setpoint;
    }

    void setOutputLimits(float out_min, float out_max) {
        out_min_ = out_min;
        out_max_ = out_max;
    }

    void reset() {
        setpoint_ = 0.0f;
        integral_ = 0.0f;
        prev_error_ = 0.0f;
        has_prev_error_ = false;
        last_time_us_ = 0;
        has_prev_time_ = false;
    }

    // dt 由内部通过 millis() 计算
    float compute(float setpoint, float measurement) {
        unsigned long now = micros();

        float dt = 0.0f;
        if (has_prev_time_) {
            unsigned long delta_us = now - last_time_us_;
            dt = delta_us * 0.000001f; // 转为秒
        }
        last_time_us_ = now;
        has_prev_time_ = true;

        if (dt <= 0.0f) {
            // 第一次调用或时间间隔过小，不更新积分微分，仅用比例
            float error_only = setpoint - measurement;
            prev_error_ = error_only;
            has_prev_error_ = true;
            float out_p = kp_ * error_only;
            if (out_p > out_max_) out_p = out_max_;
            else if (out_p < out_min_) out_p = out_min_;
            return out_p;
        }

        float error = setpoint - measurement;
        integral_ += error * dt;

        // anti-windup: limit integral term to avoid excessive accumulation
        if (ki_ != 0.0f) {
            float i_limit = out_max_;  // ensure |ki_ * integral_| <= out_max_
            if (integral_ > i_limit) {
                integral_ = i_limit;
            } else if (integral_ < -i_limit) {
                integral_ = -i_limit;
            }
        }

        float derivative = 0.0f;
        if (has_prev_error_) {
            derivative = (error - prev_error_) / dt;
        }

        prev_error_ = error;
        has_prev_error_ = true;

        float output = kp_ * error + ki_ * integral_ + kd_ * derivative;

        if (output > out_max_) {
            output = out_max_;
        } else if (output < out_min_) {
            output = out_min_;
        }

        return output;
    }

    // 使用内部保存的 setpoint，只传 measurement
    float compute(float measurement) {
        return compute(setpoint_, measurement);
    }

private:
    float kp_;
    float ki_;
    float kd_;

    float setpoint_;

    float out_min_;
    float out_max_;

    float integral_;
    float prev_error_;
    bool has_prev_error_;

    unsigned long last_time_us_;
    bool has_prev_time_;
};

class PRBSSignalGenerator {
private:
    unsigned int lfsr;  // PRBS寄存器
    unsigned int mask;  // 多项式掩码
    float amplitude;  // 信号幅度
    float bitDuration;  // 比特持续时间(毫秒)
    unsigned long lastUpdateTime;
    
public:
    PRBSSignalGenerator(float amp = 6.0, float duration = 100, std::array<int,2> poly = {7,6}) {
        amplitude = amp;
        bitDuration = duration;
        lfsr = 0x7F;  // 初始种子值
        
        // 根据多项式创建掩码
        mask = 0;
        int poly_size = sizeof(poly)/sizeof(poly[0]);
        for(int i=0; i<poly_size; i++) {
            mask |= (1 << (poly[i]-1));
        }
        
        lastUpdateTime = millis();
    }
    
    // 获取当前信号值，必要时更新PRBS状态
    float getSignal() {
        unsigned long currentTime = millis();
        if(currentTime - lastUpdateTime >= bitDuration) {
            lastUpdateTime = currentTime;
            
            // 计算下一个比特
            unsigned int lsb = lfsr & 1;
            lfsr >>= 1;
            if(lsb) {
                lfsr ^= mask;
            }
        }
        
        // 返回当前信号值
        return (lfsr & 1) ? amplitude : -amplitude;
    }
};



class Control {
public:
    Control(int run_interval_ = 1);
    ~Control();

    void compute(const float *sensors, int sensors_dim, float *control, const int control_dim);

    void set_battery_voltage(float voltage) { voltage_ = voltage; }

    static String cvtSensors2String(const float *dataArray, const int arraySize, const int deci = 6);

    bool is_off() { return !is_on; }

    bool is_turn_on_action() { return turn_on_action; }
    bool is_turn_off_action() { return turn_off_action; }

    bool do_shutdown() { return shutdown; }

    unsigned long last_rl_policy_us() const { return last_rl_policy_us_; }
    unsigned long benchmark_rl_policy();

    

private:
    PRBSSignalGenerator signalGen;  // 创建信号生成器

    void switch_action();

    bool joystick_enable = false;
    bool joystick_turn_enable = false;
    bool is_on = false;
    bool is_on_last = false;
    bool turn_on_action = false;
    bool turn_off_action = false;
    unsigned long is_off_start_time = 0;
    unsigned long is_off_duration = 0;
    bool shutdown = false;

    void align_sensors(const float *sensors, int sensors_dim);
    void fill_sensors_std_from_frame(const SimFrame &frame, const float *sensors, int sensors_dim);

    // 分段线性插值函数：使用类成员 x_breakpoints_ / y_values_ 作为节点。
    // 这里约定 x_breakpoints_ 已按升序排列，x 为 torque 数据，
    // 插值时对 |x| 进行运算，最后把原始符号作用在结果上，实现关于 0 对称。
    // 若 |x| 超出区间，则返回端点对应的 y 值（左/右端点外插为常值），再带上符号。
    float piecewise_linear(float x) {
        const int num_points = num_piecewise_points_;

        if (num_points <= 0) {
            return 0.0f;
        }

        // 保留符号，后续只对绝对值做插值
        float sign = (x >= 0.0f) ? 1.0f : -1.0f;
        float x_abs = x * sign;

        if (num_points == 1) {
            return sign * y_values_[0];
        }

        // 左端点外：返回第一个点
        if (x_abs <= x_breakpoints_[0]) {
            return sign * y_values_[0];
        }

        // 右端点外：返回最后一个点
        if (x_abs >= x_breakpoints_[num_points - 1]) {
            return sign * y_values_[num_points - 1];
        }

        // 区间内：找到所在的分段 [x_i, x_{i+1}] 并做线性插值
        for (int i = 0; i < num_points - 1; ++i) {
            float x0 = x_breakpoints_[i];
            float x1 = x_breakpoints_[i + 1];

            if (x_abs >= x0 && x_abs <= x1) {
                float y0 = y_values_[i];
                float y1 = y_values_[i + 1];

                float t = (x_abs - x0) / (x1 - x0);
                return sign * (y0 + t * (y1 - y0));
            }
        }

        // 理论上不会到这里，保险起见返回右端点
        return sign * y_values_[num_points - 1];
    }

    float clip(float x, float min, float max) {
        if (x < min) {
            return min;
        } else if (x > max) {
            return max;
        } else {
            return x;
        }
    }

    float ployfunc(float x) {
        float a = 4.5168f, b = -8.3565f, c = 6.3522f, d = -0.4637f;
        return a * x * x * x + b * x * x + c * x + d;
    }

    void set_output_data();

    RL_Policy *rl_policy;
    unsigned long last_rl_policy_us_ = 0;

    int history_length;
    int one_observation_length;

    float voltage2torque = 0.004345f;
    float voltage_ = 0.0f;

    // 分段线性拟合用到的节点：x 为 torque，y 为 voltage。
    static const int num_piecewise_points_ = 15;
    float x_breakpoints_[num_piecewise_points_];  // torque 节点，升序
    float y_values_[num_piecewise_points_];        // 对应的 voltage 值
    float wheel_ctrl_scale_ = 1.0f;

    // PID for wheel velocity control
    ControlPID wheelPIDLeft, wheelPIDRight;
    float torque_limit = 0.0295f;
    float voltage_limit = 7.0f;
    float delta_wheel_control_left = 0.0f;
    float delta_wheel_control_right = 0.0f;
    float pre_wheel_control_left = 0.0f;
    float pre_wheel_control_right = 0.0f;

    // PID for thigh position control
    ControlPID thighPIDLeft, thighPIDRight;
    float thigh_pid_smoothing = 0.5f; // lower is smoother
    std::vector<float> last_target_thigh_pos = std::vector<float>(2, 0.0f);
    float smoothed_thigh_left = 0.0f;
    float smoothed_thigh_right = 0.0f;
    float thigh_smoothing_alpha = 0.5f; // 0~1, higher is smoother

    float thigh_range[2] = {-0.4f, -0.06f};
    float pitch_squat_zero = 0.08;

    // 共享控制核。real I/O 填 SimFrame 后只走这里。
    HoldObs hold_obs_;
    ObsPack obs_pack_;
    SimFrame frame_;

    float transform_action[4] = {0.0, 0.0, 0.0, 0.0};
    float transform_action_bias[4] = {0.0, 0.0, 0.0, 0.0};
    float transform_action_scale[4] = {0.25, 0.25, 1.0, 1.0};

    float command[4] = {0.0, 0.0, 0.0, 0.0}; // lin_vel_x, ang_vel_yaw, height_value, roll_value
    float command_vx_filtered_ = 0.0f;
    float command_vx_rate_ = 0.2f;
    float last_act[4] = {0.0, 0.0, 0.0, 0.0}; // left_thigh, right_thigh, left_wheel, right_wheel
    float applied_act[4] = {0.0, 0.0, 0.0, 0.0}; // left_thigh, right_thigh, left_wheel, right_wheel
    float control_smooth[4] = {0.0, 0.0, 0.0, 0.0}; // left_thigh, right_thigh, left_wheel, right_wheel
    float control_smooth_alpha = 0.0;

    int run_interval = 1;
    int interval_counter = 0;

    // timing between consecutive compute() calls (microseconds)
    unsigned long last_compute_time_us_ = 0;
    unsigned long last_compute_interval_us_ = 0;
    bool has_last_compute_time_us_ = false;

    // statistics for average compute interval over 1 second
    unsigned long interval_sum_us_ = 0;
    unsigned long interval_count_ = 0;
    unsigned long last_interval_report_ms_ = 0;
    unsigned long last_debug_ms_ = 0;

    float *input;
    VectorXf sensors_std;
};

#endif // CONTROL_H