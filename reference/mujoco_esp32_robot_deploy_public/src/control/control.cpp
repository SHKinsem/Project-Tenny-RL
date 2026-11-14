#include "control.h"
#include "real/io.h"
#include "web/command.h"
#include <cmath>
#include <string>

// Control 只接线，不实现锚点或 26 维拼装。
// align_sensors：raw sensors → real_io::to_sim_frame → HoldObs::step → ObsPack::pack
// PID / 舵机逆映射仍读 sensors_std（已换成仿真系）。

#define WEB_DEBUG 1

float deg2rad(float deg) { return deg * 3.1415 / 180; }

float clip(float x, float min, float max) {
    return x < min ? min : (x > max ? max : x);
}

Control::Control(int run_interval_) {
    // sensors: [time:0, angX:1, angY:2, angZ:3, gyroX:4, gyroY:5, gyroZ:6, accX:7, accY:8, accZ:9, 
    //           motor1_shaft_angle:10, motor2_shaft_angle:11, motor1_shaft_velocity:12, motor2_shaft_velocity:13, 
    //           thigh_Left:14, thigh_Right:15, thigh_Left_velocity:16, thigh_Right_velocity:17]
    // control: [thigh_left_control:0, thigh_right_control:1, wheel_left_control:2, wheel_right_control:3]
    run_interval = run_interval_;
    history_length = 1;
    one_observation_length = kObsDim;
    rl_policy = new RL_Policy(history_length, one_observation_length);

    input = new float[one_observation_length];
    sensors_std = VectorXf::Zero(18);

#if WEB_DEBUG
    setup_lqrwebserver(true);
#endif

    wheelPIDLeft.setGains(0.00025f, 0.0f, 0.0f);
    wheelPIDRight.setGains(0.00025f, 0.0f, 0.0f);
    // wheelPIDLeft.setOutputLimits(-voltage_limit, voltage_limit);
    // wheelPIDRight.setOutputLimits(-voltage_limit, voltage_limit);

    wheelPIDLeft.setOutputLimits(-torque_limit, torque_limit);
    wheelPIDRight.setOutputLimits(-torque_limit, torque_limit);

    thighPIDLeft.setGains(0.5f, 0.0f, 0.00f);
    thighPIDRight.setGains(0.5f, 0.0f, 0.00f);

    // 使用与 voltage_torque.py 相同的数据进行初始化
    // x_breakpoints_：torque 升序
    const float torque_data[num_piecewise_points_] = {
        0.0f,
        0.000490f,
        0.00108f,
        0.00157f,
        0.00274f,
        0.00647f,
        0.00970f,
        0.0124f,
        0.0155f,
        0.0192f,
        0.0221f,
        0.0246f,
        0.0268f,
        0.0281f,
        0.0295f
    };

    // y_values_：对应的 voltage
    const float voltage_data[num_piecewise_points_] = {
        0.0f,
        0.1f,
        0.2f,
        0.3f,
        0.5f,
        1.0f,
        1.5f,
        2.0f,
        2.5f,
        3.0f,
        3.5f,
        4.0f,
        5.0f,
        6.0f,
        7.0f
    };

    for (int i = 0; i < num_piecewise_points_; ++i) {
        x_breakpoints_[i] = torque_data[i];
        y_values_[i] = voltage_data[i];
    }
}

Control::~Control() {
    delete rl_policy;
    delete[] input;
}

void Control::switch_action() {
    getSwitcher(&is_on);
    if (!is_on && is_on_last) {
        turn_off_action = true;
    } else {
        turn_off_action = false;
    }
    if (!is_on_last && is_on) {
        turn_on_action = true;
    } else {
        turn_on_action = false;
    }
    is_on_last = is_on;

    if (turn_off_action) {
        is_off_start_time = millis();
    }

    shutdown = false;
    if (!is_on) {
        is_off_duration = millis() - is_off_start_time;

        if (is_off_duration > 3000) {
            shutdown = true;
        }
    }

    // Serial.println("shutdown: " + String(shutdown));
}


void Control::fill_sensors_std_from_frame(const SimFrame &frame, const float *sensors, int sensors_dim) {
    // PID / 串口仍读 sensors_std。姿态、轮、大腿换成仿真系；time/acc 保持 raw。
    for (int i = 0; i < sensors_dim; ++i) {
        sensors_std(i) = sensors[i];
    }
    sensors_std(1) = frame.roll_rad;
    sensors_std(2) = frame.pitch_rad;
    sensors_std(3) = frame.yaw_rad;
    sensors_std(4) = frame.gyro[0];
    sensors_std(5) = frame.gyro[1];
    sensors_std(6) = frame.gyro[2];
    sensors_std(10) = frame.q_l;
    sensors_std(11) = frame.q_r;
    sensors_std(12) = frame.wheel_vel_l;
    sensors_std(13) = frame.wheel_vel_r;
    sensors_std(14) = frame.thigh_l;
    sensors_std(15) = frame.thigh_r;
    sensors_std(16) = frame.thigh_vel_l;
    sensors_std(17) = frame.thigh_vel_r;
}

void Control::align_sensors(const float *sensors, int sensors_dim) {
    // 1) 真机 raw → 仿真系。轮角取负、deg→rad、大腿映射都在 real_io，不在 HoldObs。
    real_io::to_sim_frame(sensors, sensors_dim, &frame_);
    fill_sensors_std_from_frame(frame_, sensors, sensors_dim);

    for (int i = 0; i < one_observation_length; ++i) {
        input[i] = 0.0f;
    }

    // 2) 双锚点。command 会被就地死区清零，所以下面 pack 用的已是清零后的值。
    HoldObsOut hold = hold_obs_.step(command, frame_, turn_on_action);

    // 3) 26 维与训练槽位对齐。拼装在 ObsPack，HoldObs 只负责 [7] 和 [11]/[13]。
    obs_pack_.pack(input, command, frame_, hold);
}

void Control::compute(const float *sensors, int sensors_dim, float *control, const int control_dim) {
    // init control to zero
    real_io::inv_mapping_thighs_position(thigh_range[1], thigh_range[1], control[0], control[1],
                                          thigh_range[0], thigh_range[1]);
    control[2] = 0.0f;
    control[3] = 0.0f;
    last_rl_policy_us_ = 0;
    // record interval between consecutive compute() calls in microseconds
    unsigned long now_us = micros();
    if (has_last_compute_time_us_) {
        last_compute_interval_us_ = now_us - last_compute_time_us_;
        // accumulate statistics for 1-second average
        interval_sum_us_ += last_compute_interval_us_;
        interval_count_++;
    } else {
        last_compute_interval_us_ = 0;
        has_last_compute_time_us_ = true;
    }
    last_compute_time_us_ = now_us;

    // every 2 second, print average compute interval
    // unsigned long now_ms_for_interval = millis();
    // if (last_interval_report_ms_ == 0) {
    //     last_interval_report_ms_ = now_ms_for_interval;
    // }
    // if (now_ms_for_interval - last_interval_report_ms_ >= 2000 && interval_count_ > 0) {
    //     float avg_interval_ms = (interval_sum_us_ / (float)interval_count_) / 1000.0f;
    //     Serial.println("avg_last_compute_interval_us_: " + String(avg_interval_ms, 3) + " ms");
    //     interval_sum_us_ = 0;
    //     interval_count_ = 0;
    //     last_interval_report_ms_ = now_ms_for_interval;
    // }

    wheel_ctrl_scale_ = getK("wheel_ctrl_scale", wheel_ctrl_scale_);
    float wheel_Kp = getK("wheel_Kp", 0.02f);
    float wheel_Ki = getK("wheel_Ki", 0.0f);
    float wheel_Kd = getK("wheel_Kd", 0.0001f);
    wheelPIDLeft.setGains(wheel_Kp, wheel_Ki, wheel_Kd);
    wheelPIDRight.setGains(wheel_Kp, wheel_Ki, wheel_Kd);

    int send_interval = static_cast<int>(getK("send_interval", 1000.0f));
    run_interval = static_cast<int>(getK("run_interval", static_cast<float>(run_interval)));

    obs_pack_.cfg.sinpitch_scale = getK("sinpitch_scale", obs_pack_.cfg.sinpitch_scale);
    // Kweb 调的是核上的字段，不要在 align_sensors 里再做一份 clip。
    hold_obs_.sinyaw_scale = getK("sinyaw_scale", 1.0f);
    hold_obs_.hold_error_clip_enable = getK("hold_error_clip_enable", 1.0f) > 0.5f;
    hold_obs_.hold_error_clip_ds = getK("hold_error_clip_ds", 3.0f);
    hold_obs_.hold_error_clip_dpsi = getK("hold_error_clip_dpsi", 1.0f);
    hold_obs_.pitch_max = getK("pitch_max", hold_obs_.pitch_max);
    
    thigh_smoothing_alpha = clip(getK("thigh_smoothing_alpha(0,1)", thigh_smoothing_alpha), 0.0f, 1.0f);
    obs_pack_.cfg.wheel_vel_smoothing_alpha = clip(getK("wheel_vel_smoothing_alpha(0,1)", obs_pack_.cfg.wheel_vel_smoothing_alpha), 0.0f, 1.0f);
    float thigh_Kp = getK("thigh_Kp", 0.3);
    float thigh_Kd = getK("thigh_Kd", 0.0);

    thighPIDLeft.setGains(thigh_Kp, 0.0f, thigh_Kd);
    thighPIDRight.setGains(thigh_Kp, 0.0f, thigh_Kd);

    control_smooth_alpha = clip(getK("control_smooth(0, 1)", control_smooth_alpha), 0.0f, 1.0f);

    // set commands from K
    command[0] = getK("cmd_lin_vel_x(-0.2,0.2)", command[0]);
    command[1] = getK("cmd_ang_vel_yaw(-1.5,1.5)", command[1]);
    command[2] = clip(getK("cmd_height_value(-0.5,0)", command[2]), thigh_range[0], thigh_range[1]);
    command[3] = getK("cmd_roll_value(-0.2,0.2)", command[3]);

    obs_pack_.cfg.pitch_zero_low = getK("pitch_zero_low", obs_pack_.cfg.pitch_zero_low);
    obs_pack_.cfg.pitch_zero_high = getK("pitch_zero_high", obs_pack_.cfg.pitch_zero_high);
    obs_pack_.cfg.sinroll_scale = getK("sinroll_scale", obs_pack_.cfg.sinroll_scale);
    obs_pack_.cfg.thigh_lo = thigh_range[0];
    obs_pack_.cfg.thigh_hi = thigh_range[1];

    switch_action();
    // Web 是操作员通道，不是本体支路。real 胶水在这里接同一套映射；sim 接 host/web。
    if (getCurrentPage() == "control") {
        float joy[2];
        getJoyStickSize2(joy);
        web_cmd::apply_control_page(
            joy[0], joy[1], get_height_bar(), get_oscillation_bar(),
            command, web_cmd::kCommandRange, &joystick_enable, &joystick_turn_enable);
    }
    command_vx_rate_ = clip(getK("command_vx_rate", command_vx_rate_), 0.0f, 100.0f);
    if (turn_on_action) {
        command_vx_filtered_ = 0.0f;
    }
    float max_delta_vx = command_vx_rate_ * last_compute_interval_us_ * 1e-6f;
    float delta_vx = clip(command[0] - command_vx_filtered_, -max_delta_vx, max_delta_vx);
    command_vx_filtered_ += delta_vx;
    command[0] = command_vx_filtered_;
    int current_time_start = millis();
    // // web client debug info end
    // if (millis() - lastCommandPrintTime >= 1000) {
    //     Serial.printf("pagetype: %s, command3: %.4f, %.4f, %.4f, %.4f\n", pagetype.c_str(), command[0], command[1], command[2], command[3]);
    //     lastCommandPrintTime = millis();
    // }
    // prepare sensors for thigh and wheel control
    align_sensors(sensors, sensors_dim);
    // print input
    // Serial.printf("input: ");
    // for (int i = 0; i < one_observation_length; ++i) {
    //     Serial.printf("%.3f, ", input[i]);
    // }
    // Serial.println();
    bool is_alive = !shutdown && fabs(sensors_std[2]) <= hold_obs_.pitch_max;
    // return;
    if (is_alive) {
        interval_counter++;
    // Serial.println("interval_counter: " + String(interval_counter) + ", " + String(run_interval));
        if (interval_counter == run_interval) {
            interval_counter = 0;
            unsigned long rl_start_us = micros();
            rl_policy->run(input, one_observation_length, last_act, 4);
            last_rl_policy_us_ = micros() - rl_start_us;
            // return;

            // transform last_act
            for (int i = 0; i < 4; ++i) {
                transform_action[i] = (last_act[i] + transform_action_bias[i]) * transform_action_scale[i];
            }
            // clip to (-0.5, 0)
            float target_thigh_L = clip(transform_action[0] + sensors_std(14), thigh_range[0], thigh_range[1]);
            float target_thigh_R = clip(transform_action[1] + sensors_std(15), thigh_range[0], thigh_range[1]);
            last_target_thigh_pos[0] = clip(last_target_thigh_pos[0], thigh_range[0], thigh_range[1]);
            last_target_thigh_pos[1] = clip(last_target_thigh_pos[1], thigh_range[0], thigh_range[1]);
            // smoothing target thigh position
            last_target_thigh_pos[0] = target_thigh_L * thigh_pid_smoothing + last_target_thigh_pos[0] * (1 - thigh_pid_smoothing);
            last_target_thigh_pos[1] = target_thigh_R * thigh_pid_smoothing + last_target_thigh_pos[1] * (1 - thigh_pid_smoothing);
            thighPIDLeft.setSetpoint(last_target_thigh_pos[0]);
            thighPIDRight.setSetpoint(last_target_thigh_pos[1]);
            
            
            for (int i = 0; i < 4; ++i) {
                obs_pack_.cfg.transform_action_bias[i] = transform_action_bias[i];
                obs_pack_.cfg.transform_action_scale[i] = transform_action_scale[i];
            }
            obs_pack_.note_policy_action(last_act, frame_);
            wheelPIDLeft.setSetpoint(obs_pack_.last_target_wheel_vel[0]);
            wheelPIDRight.setSetpoint(obs_pack_.last_target_wheel_vel[1]);
        }
        
        // thigh control by PID
        float delta_thigh_L = thighPIDLeft.compute(sensors_std[14]);
        float delta_thigh_R = thighPIDRight.compute(sensors_std[15]);
        applied_act[0] = clip(delta_thigh_L + sensors_std[14], thigh_range[0], thigh_range[1]);
        applied_act[1] = clip(delta_thigh_R + sensors_std[15], thigh_range[0], thigh_range[1]);

        // wheel control by PID
        delta_wheel_control_left = wheelPIDLeft.compute(sensors_std[12]);
        delta_wheel_control_right = wheelPIDRight.compute(sensors_std[13]);
        pre_wheel_control_left = applied_act[2];
        pre_wheel_control_right = applied_act[3];
        applied_act[2] = clip(pre_wheel_control_left + delta_wheel_control_left, -torque_limit, torque_limit);
        applied_act[3] = clip(pre_wheel_control_right + delta_wheel_control_right, -torque_limit, torque_limit);
        // applied_act[2] = clip(pre_wheel_control_left + delta_wheel_control_left, -voltage_limit, voltage_limit);
        // applied_act[3] = clip(pre_wheel_control_right + delta_wheel_control_right, -voltage_limit, voltage_limit);
    } else {
        obs_pack_.reset();
        rl_policy->cleanHistory();
        //PID reset
        wheelPIDLeft.reset();
        wheelPIDRight.reset();
        thighPIDLeft.reset();
        thighPIDRight.reset();
    }
    // return;

    if (!is_on) {
        last_act[0] = 0;
        last_act[1] = 0;
        applied_act[0] = 0;
        applied_act[1] = 0;
    }

    if (HoldObs::fallen(sensors_std(2), hold_obs_.pitch_max)) {
        last_act[0] = 0;
        last_act[1] = 0;
        last_act[2] = 0;
        last_act[3] = 0;

        applied_act[0] = 0;
        applied_act[1] = 0;
        applied_act[2] = 0;
        applied_act[3] = 0;
    }

    float control_thigh_inv[2];
    
    real_io::inv_mapping_thighs_position(applied_act[0], applied_act[1], control_thigh_inv[0], control_thigh_inv[1],
                                          thigh_range[0], thigh_range[1]);
    // combine thigh and wheel control
    float control_now[4];
    control_now[0] = control_thigh_inv[0];
    control_now[1] = control_thigh_inv[1];

    // control_now[2] = -piecewise_linear(applied_act[2]) * wheel_ctrl_scale_;
    // control_now[3] = -piecewise_linear(applied_act[3]) * wheel_ctrl_scale_;
    control_now[2] = -applied_act[2] * wheel_ctrl_scale_;
    control_now[3] = -applied_act[3] * wheel_ctrl_scale_;

    control_smooth[0] = control_smooth[0] * thigh_smoothing_alpha + control_now[0] * (1 - thigh_smoothing_alpha);
    control_smooth[1] = control_smooth[1] * thigh_smoothing_alpha + control_now[1] * (1 - thigh_smoothing_alpha);
    control_smooth[2] = control_smooth[2] * control_smooth_alpha + control_now[2] * (1 - control_smooth_alpha);
    control_smooth[3] = control_smooth[3] * control_smooth_alpha + control_now[3] * (1 - control_smooth_alpha);

    for (int i = 0; i < control_dim; ++i) {
        control[i] = control_smooth[i];
    }
    // return;

    // control[3] = signalGen.getSignal();
    // Serial.println("control[2]: " + String(control[2]));
#if WEB_DEBUG
    // debug info
    int current_time = millis();
    if (current_time - last_debug_ms_ >= static_cast<unsigned long>(send_interval))
    {
        last_debug_ms_ = current_time;
        int deci = 6;
        String debugInfo;
        debugInfo.reserve(256);
        float joystick[2];
        getJoyStickSize2(joystick);
        

        debugInfo += "time:" + String(current_time) +
                 ",  cost_ms:" + String(current_time-current_time_start) +
                 ",  dt_compute_ms:" + String(last_compute_interval_us_/1000.0f, 3) + "\n";
        debugInfo += "battery_voltage: " + String(voltage_, deci) + ", pagetype: " + getCurrentPage() + ", joystick:[" + String(joystick[0], deci) + ", " + String(joystick[1], deci) + "]\n";
        debugInfo += "hold ds_true:" + String(hold_obs_.ds_true, deci) +
                 " dpsi_true:" + String(hold_obs_.dpsi_true, deci) +
                 " ds_obs:" + String(input[11], deci) +
                 " dpsi_obs:" + String(input[7], deci) +
                 " s0:" + String(hold_obs_.position_start, deci) +
                 " yaw0:" + String(hold_obs_.yaw_sin_zero, deci) + "\n";
#if 0
        // add input
        debugInfo += "input:\n";
        for (int i = 0; i < one_observation_length; i++) {
            debugInfo += String(i) + ":" + String(input[i], deci) + ", ";
            if ((i+1) % 10 == 0) debugInfo += "\n";
        }
        debugInfo += "\n\n";
        // add rl_policy->debugInfo()
        // debugInfo += "rl_policy->debugInfo():\n";
        // debugInfo += rl_policy->debugInfo();
#endif
#endif
        setOtherString0_(debugInfo);
    }
    // return;

#if 0
        // setOtherData(sensors, sensors_dim, control, control_dim);
        float input_show[one_observation_length + 2] = {sensors[0], interval_counter};  // 初始化第一个元素为传感器时间
        std::copy(input, input + one_observation_length, input_show + 2);
        // return;
        // combine transform_action to control_smooth for debug
        const int temp_len = control_dim + 4 + 2 + 2;
        float temp_values[temp_len];
        int idx = 0;
        for (int i = 0; i < control_dim; ++i) {
            temp_values[idx++] = control_smooth[i];
        }
        temp_values[idx++] = pre_wheel_control_left;
        temp_values[idx++] = pre_wheel_control_right;
        temp_values[idx++] = delta_wheel_control_left;
        temp_values[idx++] = delta_wheel_control_right;
        for (int i = 0; i < 4; ++i) {
            temp_values[idx++] = applied_act[i];
        }
        assert(idx == temp_len);
        setOtherData(input_show, one_observation_length+2, temp_values, temp_len);
#endif
}

unsigned long Control::benchmark_rl_policy() {
    float action[4];
    unsigned long start_us = micros();
    rl_policy->run(input, one_observation_length, action, 4);
    return micros() - start_us;
}


String Control::cvtSensors2String(const float *dataArray, const int arraySize, const int deci) {
    String dataString = "";
    int deci_ = deci;
    for (int i = 0; i < arraySize; i++) {
        deci_ = deci;
        if (i == 0) deci_ = 0;
        dataString += String(dataArray[i], deci_);
        if (i < arraySize - 1) {
            dataString += ",";
        }
    }
    return dataString;
}