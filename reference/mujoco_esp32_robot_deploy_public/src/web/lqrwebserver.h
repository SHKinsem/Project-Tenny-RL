#ifndef LQR_WEB_SERVER_H
#define LQR_WEB_SERVER_H

// 真机 HTTP 传输（ESP32 WebServer + LittleFS）。摇杆/滑条/K 的值存在这里。
// 映射成 command[4] 在 web/command.h，不在本文件。仿真侧换 Python HTTP，仍用同一套映射。

#include <Arduino.h>
#include <ArduinoEigen.h>
#include <vector>

using Eigen::MatrixXf;
using Eigen::VectorXf;

struct KItem {
  String key;
  float value;
};

using KDict = std::vector<KItem>;

struct LabeledVectorXf {
  std::vector<std::string> labels;
  VectorXf values;
};

const String& getCurrentPage();
float get_height_bar();
float get_oscillation_bar();
void getJoyStickSize2(float *joyStick);
void getSwitcher(bool *switcher_);
void setVectorXf(int index, const LabeledVectorXf& newVector);
void getVectorXf(int index, LabeledVectorXf& vectorXfOut);
void setMatrixXf(const MatrixXf& newMatrix);
void getMatrixXf(MatrixXf& matrixXfOut);
void setup_lqrwebserver(bool useWiFi = true);
void setKValues(const KDict& newK);
void getKValues(KDict& K);
float getK(const String &key, float defaultValue);
void setOtherString(const String& newString);
void setOtherString0_(const String& newString);
void setOtherData(const float *data, const int dlen, const float *control, const int clen);

#endif // LQR_WEB_SERVER_H