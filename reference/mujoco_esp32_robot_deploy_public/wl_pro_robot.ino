#include <WiFi.h>
#include <WiFiAP.h>
#include <WiFiClient.h>
#include <WiFiGeneric.h>
#include <WiFiMulti.h>
#include <WiFiSTA.h>
#include <WiFiScan.h>
#include <WiFiServer.h>
#include <WiFiType.h>
#include <WiFiUdp.h>

#include <WiFi.h>
#include <WiFiAP.h>
#include <WiFiClient.h>
#include <WiFiGeneric.h>
#include <WiFiMulti.h>
#include <WiFiSTA.h>
#include <WiFiScan.h>
#include <WiFiServer.h>
#include <WiFiType.h>
#include <WiFiUdp.h>

#include <freertos/FreeRTOS.h>
#include <freertos/task.h>


// -----------------------------------------------------------------------------
// Copyright (c) 2024 Mu Shibo
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
// -----------------------------------------------------------------------------

//机器人控制头文件
#include "hal/MPU6050Mahony.h"
#include "hal/Servo_STS3032.h"
#include <SimpleFOC.h>
#include <Arduino.h>

//wifi控制数据传输头文件
#include <WebSocketsServer.h>
#include <ArduinoJson.h>
#include <WebServer.h>
#include <WiFi.h>
#include <FS.h>
#include "hal/robot.h"

#include "control/control.h"

// 根目录只留本文件。固件在 src/，主机 Python 在 host/。详见 src/control/control.h。

#define BENCHMARK_POLICY 0
#define IMU_FLIP_X_180 1

void get_sensors_data(float *dataArray, const int arraySize);
void set_thighs_position(int thigh_left, int thigh_right, int acc, int speed);
void test_sts_response();
bool control_loop();
void printCPUInfo();
void scanWiFiNetworks();
void continuousWiFiScan(int intervalSeconds);
void setup();
void loop();
void jump_loop();
void blink(int led, int t);
void bat_check();

#define LOOP_SHOW 0

#define LEFT_SERVO_SQUAT_POS 2136
#define RIGHT_SERVO_SQUAT_POS 2019
#define SERVO_AVIALABE_LEN_POS 540

#define LEFT_SERVO_STAND_POS (LEFT_SERVO_SQUAT_POS+SERVO_AVIALABE_LEN_POS)
#define RIGHT_SERVO_STAND_POS (RIGHT_SERVO_SQUAT_POS-SERVO_AVIALABE_LEN_POS)

/************实例定义*************/

//电机实例
BLDCMotor motor1 = BLDCMotor(7);
BLDCMotor motor2 = BLDCMotor(7);
BLDCDriver3PWM driver1 = BLDCDriver3PWM(32,33,25,22);
BLDCDriver3PWM driver2  = BLDCDriver3PWM(26,27,14,12);

// configuration for AS5600 sensor
/*
MagneticSensorI2CConfig_s MyAS5600_I2C = {
  .chip_address = 0x36, 
  .bit_resolution = 12, 
  .angle_register=0x0E, 
  .data_start_bit=11
  };
*/
//编码器实例
TwoWire I2Cone = TwoWire(0);
TwoWire I2Ctwo = TwoWire(1);
MagneticSensorI2C sensor1 = MagneticSensorI2C(AS5600_I2C);
MagneticSensorI2C sensor2 = MagneticSensorI2C(AS5600_I2C);


//STS舵机实例
SMS_STS sms_sts;

//MPU6050实例
MPU6050Mahony mpu6050(I2Cone, IMU_FLIP_X_180 != 0);

/************参数定义*************/
#define pi 3.1415927

//LQR自平衡控制器参数
float LQR_angle = 0;
float LQR_gyro  = 0;
float LQR_speed = 0;
float LQR_distance = 0;
float angle_control   = 0;
float gyro_control    = 0;
float speed_control   = 0;
float distance_control = 0;
float LQR_u = 0;
float angle_zeropoint = 0; //-2.3; //squat:5.5, stand:-2.3
float roll_angle_zeropoint = 0.0;
float distance_zeropoint = -256.0;       //轮部位移零点偏置（-256为一个不可能的位移值，将其作为未刷新的标志）

//YAW轴控制数据
float YAW_gyro = 0;
float YAW_angle = 0;
float YAW_angle_last = 0;
float YAW_angle_total = 0;
float YAW_angle_zero_point = -10;
float YAW_output = 0;

//腿部舵机控制数据
byte ID[2];
s16 Position[2];
u16 Speed[2];
byte ACC[2];

//逻辑处理标志位
float robot_speed = 0;          //记录当前轮部转速
float robot_speed_last = 0;     //记录上一时刻的轮部转速
int wrobot_move_stop_flag = 0;  //记录摇杆停止的标识
int jump_flag = 0;              //跳跃时段标识
float leg_position_add = 0;     //roll轴平衡控制量
int uncontrolable = 0;          //机身倾角过大导致失控


//电压检测
uint16_t bat_check_num = 0;
int BAT_PIN = 35;    // select the input pin for the ADC

//电量显示LED
#define LED_BAT 13

#define S_RXD 16
#define S_TXD 17


const int arraySize = 18;
float dataArray[arraySize]; // raw HAL，还没进仿真系。映射在 src/real/io.cpp。
                    //0:time
                    //1~9: angX, angY, angZ, gyroX, gyroY, gyroZ, accX, accY, accZ  （角度是度）
                    //10~13: motor1_shaft_angle, motor2_shaft_angle, motor1_shaft_velocity, motor2_shaft_velocity
                    //      FOC 原符号；real_io 里取负后才与仿真 qpos 同号
                    //14~17: thigh_Left, thigh_Right, thigh_Left_velocity, thigh_Right_velocity  （舵机计数）
float dataArray_init[arraySize];
String sensors_str_gobal = "";


void get_sensors_data(float *dataArray, const int arraySize) {
  dataArray[0] = micros()/1000.f;

  float mpu_angleX = (float)mpu6050.getAngleX();
  float mpu_angleY = (float)mpu6050.getAngleY();
  float mpu_angleZ = (float)mpu6050.getAngleZ();

  float mpu_gyroX  = (float)mpu6050.getGyroX();
  float mpu_gyroY  = (float)mpu6050.getGyroY();
  float mpu_gyroZ  = (float)mpu6050.getGyroZ();

  float mpu_accX  = (float)mpu6050.getAccX();
  float mpu_accY  = (float)mpu6050.getAccY();
  float mpu_accZ  = (float)mpu6050.getAccZ();

  dataArray[1] = mpu_angleX;
  dataArray[2] = mpu_angleY;
  dataArray[3] = mpu_angleZ;
  dataArray[4] = mpu_gyroX;
  dataArray[5] = mpu_gyroY;
  dataArray[6] = mpu_gyroZ;
  dataArray[7] = mpu_accX;
  dataArray[8] = mpu_accY;
  dataArray[9] = mpu_accZ;
  // Serial.printf("motor1.shaft_angle:%.4f, motor1.shaftAngle():%.4f\n", motor1.shaft_angle, motor1.shaftAngle());
  float motor1_shaft_angle = motor1.shaft_angle; dataArray[10] = motor1_shaft_angle;
  float motor2_shaft_angle = motor2.shaft_angle; dataArray[11] = motor2_shaft_angle;
  float motor1_shaft_velocity = motor1.shaft_velocity; dataArray[12] = motor1_shaft_velocity;
  float motor2_shaft_velocity = motor2.shaft_velocity; dataArray[13] = motor2_shaft_velocity;

  // int t0 = millis();
  int thigh_Left = sms_sts.ReadPos(1);
  int thigh_Right = sms_sts.ReadPos(2);
  // int t1 = millis();
  dataArray[14] = thigh_Left;//thigh_Left;
  dataArray[15] = thigh_Right;//thigh_Right;
  dataArray[16] = 0.0; //sms_sts.ReadSpeed(1); //thigh_Left_velocity;
  dataArray[17] = 0.0; //sms_sts.ReadSpeed(2); //thigh_Right_velocity;
  // int t2 = millis();
  // if (t2 - t1 > 1) {
  //   Serial.printf("thigh_Left:%d, thigh_Right:%d, time:%dms\n", thigh_Left, thigh_Right, t1-t0);
  //   Serial.printf("thigh_Left_velocity:%d, thigh_Right_velocity:%d, time:%dms\n", dataArray[16], dataArray[17], t2-t1);
  // }
}

Control *control = nullptr;


void set_thighs_position(int thigh_left, int thigh_right, int acc = 30, int speed = 300) {
  ID[0] = 1;
  ID[1] = 2;
  ACC[0] = acc;
  ACC[1] = acc;
  Speed[0] = speed;
  Speed[1] = speed;
  Position[0] = thigh_left;
  Position[1] = thigh_right;
  sms_sts.SyncWritePosEx(ID, 2, Position, Speed, ACC);
}

void test_sts_response() {
  /*
  #define LEFT_SERVO_SQUAT_POS 2136
#define RIGHT_SERVO_SQUAT_POS 2019
#define SERVO_AVIALABE_LEN_POS 540

#define LEFT_SERVO_STAND_POS (LEFT_SERVO_SQUAT_POS+SERVO_AVIALABE_LEN_POS)
#define RIGHT_SERVO_STAND_POS (RIGHT_SERVO_SQUAT_POS-SERVO_AVIALABE_LEN_POS)
  */
  const int arraySize = 1000;
  const int period = 300;
  const int delay_t = 2;
  int times[arraySize];
  int thigh_left_ctrl[arraySize];
  int thigh_right_ctrl[arraySize];
  int thigh_left_pos[arraySize];
  int thigh_right_pos[arraySize];

  int left_ctrl = LEFT_SERVO_SQUAT_POS;
  int right_ctrl = RIGHT_SERVO_SQUAT_POS;

  for (int i = 1; i <= arraySize; i++) {
    times[i] = millis();
    
    if (i % period == 0) { // switch between squat and stand
      if (left_ctrl == LEFT_SERVO_SQUAT_POS) {
        left_ctrl = LEFT_SERVO_STAND_POS;
        right_ctrl = RIGHT_SERVO_STAND_POS;
      } else {
        left_ctrl = LEFT_SERVO_SQUAT_POS;
        right_ctrl = RIGHT_SERVO_SQUAT_POS;
      }
    }
    set_thighs_position(left_ctrl, right_ctrl, 8, 200);
    thigh_left_ctrl[i] = left_ctrl;
    thigh_right_ctrl[i] = right_ctrl;
    thigh_left_pos[i] = sms_sts.ReadPos(1);
    thigh_right_pos[i] = sms_sts.ReadPos(2);
    Serial.printf("%d, %d, %d, %d, %d, %d, %d, %d\n", i, delay_t, period, times[i], thigh_left_ctrl[i], thigh_right_ctrl[i], thigh_left_pos[i], thigh_right_pos[i]);
    delay(delay_t);
  }
  // for (int i = 0; i < arraySize; i++) {
  //   Serial.printf("%d, %d, %d, %d, %d\n", times[i], thigh_left_ctrl[i], thigh_right_ctrl[i], thigh_left_pos[i], thigh_right_pos[i]);
  // }
}

bool control_loop() {
  bool stop = false;
  unsigned long sensor_start_us = micros();
  get_sensors_data(dataArray, arraySize);
  unsigned int sensor_cost_us = micros() - sensor_start_us;
  // Serial.println(control->cvtSensors2String(dataArray, arraySize, 4));
  const int ctrl_num = 4;
  float controls[ctrl_num] = {0, 0, 0, 0};
  unsigned int timestamp = millis();
  static unsigned long last_attitude_print_ms = 0;
  if (timestamp - last_attitude_print_ms >= 1000) {
    last_attitude_print_ms = timestamp;
    Serial.printf("yaw=%.6f,pitch=%.6f,roll=%.6f imu_us=%lu avg_us=%lu i2c_errors=%lu\n",
                  deg2rad(dataArray[3]), deg2rad(dataArray[2]), deg2rad(dataArray[1]),
                  (unsigned long)mpu6050.getLastUpdateCostUs(),
                  (unsigned long)mpu6050.getAverageUpdateCostUs(),
                  (unsigned long)mpu6050.getReadErrorCount());
  }
  unsigned long compute_start_us = micros();
  control->compute(dataArray, arraySize, controls, ctrl_num);
  unsigned int compute_cost_us = micros() - compute_start_us;
  int cost = millis() - timestamp;
  static unsigned long last_compute_print_ms = 0;
  static uint64_t compute_sum_us = 0;
  static uint64_t sensor_sum_us = 0;
  static uint64_t model_sum_us = 0;
  static uint64_t other_sum_us = 0;
  static uint32_t compute_count = 0;
  static uint32_t compute_max_us = 0;
  static uint32_t compute_hist[7] = {0};
#if BENCHMARK_POLICY
  unsigned int model_cost_us = control->benchmark_rl_policy();
#else
  unsigned int model_cost_us = control->last_rl_policy_us();
#endif
  compute_sum_us += compute_cost_us;
  sensor_sum_us += sensor_cost_us;
  model_sum_us += model_cost_us;
  other_sum_us += compute_cost_us - model_cost_us;
  compute_count++;
  if (compute_cost_us > compute_max_us) {
    compute_max_us = compute_cost_us;
  }
  if (compute_cost_us < 100) compute_hist[0]++;
  else if (compute_cost_us < 200) compute_hist[1]++;
  else if (compute_cost_us < 400) compute_hist[2]++;
  else if (compute_cost_us < 800) compute_hist[3]++;
  else if (compute_cost_us < 1600) compute_hist[4]++;
  else if (compute_cost_us < 3200) compute_hist[5]++;
  else compute_hist[6]++;
  if (timestamp - last_compute_print_ms >= 1000) {
    last_compute_print_ms = timestamp;
    Serial.printf("compute_avg_us=%llu compute_max_us=%u compute_calls=%u\n",
                  (unsigned long long)(compute_sum_us / compute_count), compute_max_us, compute_count);
    Serial.printf("sensor_avg_us=%llu model_avg_us=%llu other_avg_us=%llu\n",
                  (unsigned long long)(sensor_sum_us / compute_count),
                  (unsigned long long)(model_sum_us / compute_count),
                  (unsigned long long)(other_sum_us / compute_count));
    static const char *hist_labels[7] = {"<100", "100-199", "200-399", "400-799", "800-1599", "1600-3199", ">=3200"};
    Serial.print("hist_us=");
    for (int i = 0; i < 7; ++i) {
      Serial.printf("%s:%u%s", hist_labels[i], compute_hist[i], i == 6 ? "\n" : ",");
      compute_hist[i] = 0;
    }
    compute_sum_us = 0;
    sensor_sum_us = 0;
    model_sum_us = 0;
    other_sum_us = 0;
    compute_count = 0;
    compute_max_us = 0;
  }

  if (control->do_shutdown()) {
    stop = true;
    return stop;
  }

  // Serial.printf("controls: %.4f, %.4f, %.4f, %.4f\n", controls[0], controls[1], controls[2], controls[3]);
  // Serial.printf("control->compute cost: %dms\n", cost);

  motor1.target = controls[2];
  motor2.target = controls[3];

  int thigh_left_pos = controls[0];
  int thigh_right_pos = controls[1];
  // int acc = ;
  // int speed = ;
  set_thighs_position(thigh_left_pos, thigh_right_pos, 30, 300);

  return stop;
}

void printCPUInfo() {
  // 获取CPU核心数量
  int coreCount = portNUM_PROCESSORS;
  
  // 获取当前运行的CPU核心
  int currentCore = xPortGetCoreID();
  
  // 获取每个核心的FreeRTOS任务信息
  for(int core = 0; core < coreCount; core++) {
    Serial.printf("核心 %d 信息:\n", core);
    Serial.printf("  当前运行任务: %s\n", pcTaskGetName(xTaskGetCurrentTaskHandleForCPU(core)));
    Serial.printf("  空闲内存: %d bytes\n", esp_get_free_heap_size());
    Serial.printf("  最小空闲内存: %d bytes\n", esp_get_minimum_free_heap_size());
    
    // 获取核心利用率(需要启用FreeRTOS任务跟踪)
    #ifdef CONFIG_FREERTOS_USE_TRACE_FACILITY
    TaskStatus_t *pxTaskStatusArray;
    UBaseType_t uxArraySize = uxTaskGetNumberOfTasks();
    pxTaskStatusArray = (TaskStatus_t *)pvPortMalloc(uxArraySize * sizeof(TaskStatus_t));
    
    if(pxTaskStatusArray != NULL) {
      uxArraySize = uxTaskGetSystemState(pxTaskStatusArray, uxArraySize, NULL);
      for(UBaseType_t x = 0; x < uxArraySize; x++) {
        if(pxTaskStatusArray[x].xCoreID == core) {
          Serial.printf("  任务: %s, 状态: %d, 优先级: %d\n", 
                       pxTaskStatusArray[x].pcTaskName,
                       pxTaskStatusArray[x].eCurrentState,
                       pxTaskStatusArray[x].uxCurrentPriority);
        }
      }
      vPortFree(pxTaskStatusArray);
    }
    #endif
    
    Serial.println();
  }
  
  Serial.printf("当前运行在核心: %d\n", currentCore);
}

// WiFi网络扫描函数
void scanWiFiNetworks() {
  Serial.println("开始扫描WiFi网络...");
  
  // 设置WiFi模式为STA（客户端模式）
  WiFi.mode(WIFI_STA);
  
  // 断开当前连接（如果有）
  WiFi.disconnect();
  
  // 等待一段时间让WiFi模块初始化
  delay(100);
  
  // 开始扫描WiFi网络
  int n = WiFi.scanNetworks();
  
  Serial.println("扫描完成！");
  
  if (n == 0) {
    Serial.println("未找到任何WiFi网络");
  } else {
    Serial.print("找到 ");
    Serial.print(n);
    Serial.println(" 个WiFi网络：");
    Serial.println("==================================");
    
    // 按信号强度排序
    int indices[n];
    for (int i = 0; i < n; i++) {
      indices[i] = i;
    }
    
    // 简单的冒泡排序，按信号强度排序
    for (int i = 0; i < n; i++) {
      for (int j = i + 1; j < n; j++) {
        if (WiFi.RSSI(indices[j]) > WiFi.RSSI(indices[i])) {
          std::swap(indices[i], indices[j]);
        }
      }
    }
    
    // 打印网络信息
    for (int i = 0; i < n; ++i) {
      Serial.print(i + 1);
      Serial.print(": ");
      Serial.print(WiFi.SSID(indices[i]));
      Serial.print(" (");
      Serial.print(WiFi.RSSI(indices[i]));
      Serial.print(" dBm) ");
      
      // 打印安全类型
      Serial.print("安全: ");
      switch (WiFi.encryptionType(indices[i])) {
        case WIFI_AUTH_OPEN:
          Serial.print("开放");
          break;
        case WIFI_AUTH_WEP:
          Serial.print("WEP");
          break;
        case WIFI_AUTH_WPA_PSK:
          Serial.print("WPA/WPA2");
          break;
        case WIFI_AUTH_WPA2_PSK:
          Serial.print("WPA2");
          break;
        case WIFI_AUTH_WPA_WPA2_PSK:
          Serial.print("WPA/WPA2混合");
          break;
        case WIFI_AUTH_WPA2_ENTERPRISE:
          Serial.print("WPA2企业");
          break;
        default:
          Serial.print("未知");
          break;
      }
      
      Serial.print(" | 信道: ");
      Serial.println(WiFi.channel(indices[i]));
    }
    
    Serial.println("==================================");
  }
  
  // 清理扫描结果
  WiFi.scanDelete();
}

// 持续扫描WiFi网络的函数
void continuousWiFiScan(int intervalSeconds = 10) {
  static unsigned long lastScanTime = 0;
  unsigned long currentTime = millis();
  
  if (currentTime - lastScanTime >= intervalSeconds * 1000) {
    scanWiFiNetworks();
    lastScanTime = currentTime;
  }
}

void setup() {
  Serial.begin(115200);//通讯串口
  Serial2.begin(1000000, SERIAL_8N1, S_RXD, S_TXD);//腿部sts舵机
  
  printCPUInfo();

  Serial.println("=== WiFi网络扫描 ===");
  scanWiFiNetworks();

  //舵机初始化
  //舵机有效行程SERVO_AVIALABE_LEN_POS
  //左侧舵机[LEFT_SERVO_SQUAT_POS+12+50,LEFT_SERVO_SQUAT_POS+12+SERVO_AVIALABE_LEN_POS]
  //左侧舵机[RIGHT_SERVO_SQUAT_POS-12-50,RIGHT_SERVO_SQUAT_POS-12-SERVO_AVIALABE_LEN_POS]
  sms_sts.pSerial = &Serial2;
  
  //舵机(ID1/ID2)以最高速度V=2400步/秒，加速度A=50(50*100步/秒^2)，运行至各自的Position位置
  int thigh_left_pos = LEFT_SERVO_SQUAT_POS+SERVO_AVIALABE_LEN_POS/10;
  int thigh_right_pos = RIGHT_SERVO_SQUAT_POS-SERVO_AVIALABE_LEN_POS/10;
  set_thighs_position(thigh_left_pos, thigh_right_pos);

  //电量显示LED
  pinMode(LED_BAT,OUTPUT);

  // 编码器设置
  I2Cone.begin(19,18, 400000UL); 
  I2Ctwo.begin(23,5, 400000UL);
  sensor1.init(&I2Cone);
  sensor2.init(&I2Ctwo);

  //mpu6050设置
  mpu6050.begin();
  mpu6050.calcGyroOffsets(true);

  float roll_angle  = (float)mpu6050.getAngleX();
  float pitch_angle  = (float)mpu6050.getAngleY();
  float yaw_angle  = (float)mpu6050.getAngleZ();
  Serial.printf("MPU6050 -> yaw_angle: %f, pitch_angle: %f, roll_angle:%f\n", yaw_angle, pitch_angle, roll_angle);
  //roll_angle_zeropoint = roll_angle;
  
  //连接motor对象与编码器对象
  motor1.linkSensor(&sensor1);
  motor2.linkSensor(&sensor2);

  //速度环PID参数
  motor1.PID_velocity.P = 0.05;
  motor1.PID_velocity.I = 1;
  motor1.PID_velocity.D = 0;

  motor2.PID_velocity.P = 0.05;
  motor2.PID_velocity.I = 1;
  motor2.PID_velocity.D = 0;

  // 驱动器设置
  motor1.voltage_sensor_align = 6;
  motor2.voltage_sensor_align = 6;
  driver1.voltage_power_supply = 8;
  driver2.voltage_power_supply = 8;
  driver1.init();
  driver2.init();

  //连接motor对象与驱动器对象
  motor1.linkDriver(&driver1);
  motor2.linkDriver(&driver2);
 
  motor1.torque_controller = TorqueControlType::voltage;
  motor2.torque_controller = TorqueControlType::voltage;   
  motor1.controller = MotionControlType::torque;
  motor2.controller = MotionControlType::torque;
  
  // monitor相关设置
  motor1.useMonitoring(Serial);
  motor2.useMonitoring(Serial);
  //电机初始化
  motor1.init();
  motor1.initFOC(); 
  motor2.init();
  motor2.initFOC();

  get_sensors_data(dataArray_init, arraySize);
  String datastr_init = Control::cvtSensors2String(dataArray_init, arraySize);
  Serial.println("init states:"+datastr_init);

  control = new Control();

  delay(2000);
  Serial.println("bye setup...");

  
}


void loop() {
  // test_sts_response();
  // return;
#if LOOP_SHOW
  unsigned long t0 = millis();
#endif
  bat_check();        //电压检测

  mpu6050.update();   //IMU数据更新

  bool stop = control_loop();     //控制器计算

  if (stop) {
    return;
  }

  //迭代计算FOC相电压
  motor1.loopFOC();
  motor2.loopFOC();
  
  //设置轮部电机输出
  motor1.move();
  motor2.move();
  
#if LOOP_SHOW
  unsigned long t1 = millis();
  Serial.printf("one loop cost: %d ms\n", (t1 - t0));
#endif
}


//跳跃控制
void jump_loop(){
  if( (wrobot.dir_last == 5) && (wrobot.dir == 4) && (jump_flag == 0) )
  {
      ACC[0] = 0;
      ACC[1] = 0;
      Speed[0] = 0;
      Speed[1] = 0;
      Position[0] = LEFT_SERVO_SQUAT_POS + 12 + 8.4*(80-32);
      Position[1] = RIGHT_SERVO_SQUAT_POS - 12 - 8.4*(80-32);
      sms_sts.SyncWritePosEx(ID, 2, Position, Speed, ACC);

      jump_flag = 1;
  }
  if( jump_flag > 0 )
  {
    jump_flag++;
    if( (jump_flag > 30) && (jump_flag < 35) )
    {
      ACC[0] = 0;
      ACC[1] = 0;
      Speed[0] = 0;
      Speed[1] = 0;
      Position[0] = LEFT_SERVO_SQUAT_POS + 12 + 8.4*(40-32);
      Position[1] = RIGHT_SERVO_SQUAT_POS - 12 - 8.4*(40-32);
      sms_sts.SyncWritePosEx(ID, 2, Position, Speed, ACC);

      jump_flag = 40;
    }
    if(jump_flag > 200)
    {
      jump_flag = 0;//已准备好再次跳跃
    }
  }
}

void blink(int led, int t) {
  digitalWrite(led, HIGH);//LED1引脚输出高电平，点亮
  delay(t);//延时200ms
  digitalWrite(led, LOW);//LED1引脚输出低电平，熄灭
  delay(t);//延时200ms
}

//电压检测
void bat_check()
{
  if(bat_check_num > 1000)
  {
    //电压读取
    uint32_t voltage = analogReadMilliVolts(BAT_PIN);
    double battery=(voltage*3.97)/1000.0;

    control->set_battery_voltage(battery);
    //电量显示
    if(battery>7.8)
      digitalWrite(LED_BAT, HIGH);
    else {
      digitalWrite(LED_BAT, LOW);
      Serial.println(battery);
    }

    bat_check_num = 0;
  }
  else
    bat_check_num++;
}