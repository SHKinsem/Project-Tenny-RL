#include <WiFi.h>
#include <WebServer.h>
#include <WebSocketsServer.h>
#include <freertos/task.h>
#include <ArduinoJson.h>
#include <FS.h>
#include <LittleFS.h>
#include "lqrwebserver.h"
#include "real/io.h"
#include "wifi_config.h"

// STA WiFi configuration profiles (SSID + password + IP) for easy switching
struct WiFiProfile {
  const char* ssid;
  const char* password;
  IPAddress local_IP;
  IPAddress gateway;
};

const WiFiProfile WIFI_PROFILES[] = {
  { WIFI_STA_SSID, WIFI_STA_PASSWORD, IPAddress(192, 168, 1, 184), IPAddress(192, 168, 1, 1) },
};

constexpr int WIFI_PROFILE_COUNT = sizeof(WIFI_PROFILES) / sizeof(WIFI_PROFILES[0]);
int currentWifiProfileIndex = 0; // select which STA profile to use

// AP (hotspot) configuration profiles
struct WiFiApProfile {
  const char* ssid;
  const char* password;
  IPAddress local_IP;
  IPAddress gateway;
  uint8_t channel;
  bool hidden;      // 0: visible, 1: hidden
  uint8_t max_conn; // max connections
};

const WiFiApProfile WIFI_AP_PROFILES[] = {
  { WIFI_AP_SSID, WIFI_AP_PASSWORD, IPAddress(192, 168, 4, 1), IPAddress(192, 168, 4, 1), 1, false, 4 },
};

constexpr int WIFI_AP_PROFILE_COUNT = sizeof(WIFI_AP_PROFILES) / sizeof(WIFI_AP_PROFILES[0]);
int currentWifiApProfileIndex = 0; // select which AP profile to use

WebServer server(80);
WebSocketsServer controlSocket(81);

// Ordered K configuration stored directly in a vector
KDict Kweb = {
  {"wheel_ctrl_scale",          100.0f},
  {"wheel_Kp",                  0.0001f},
  {"wheel_Ki",                  0.000f},
  {"wheel_Kd",                  0.00f},
  {"sinwheel_scale",            1.0f},
  {"sinpitch_scale",            1.3f},
  {"sinyaw_scale",              0.2f},
  {"hold_error_clip_enable",    1.0f},
  {"hold_error_clip_ds",        3.0f},
  {"hold_error_clip_dpsi",      1.0f},
  {"pitch_max",                 0.6f},
  {"send_interval",          1000.0f},
  {"run_interval",              4.0f},
  {"thigh_Kp",                 0.40f},
  {"thigh_Kd",                 0.1f},
  {"thigh_smoothing_alpha(0,1)", 0.0f},
  {"wheel_vel_smoothing_alpha(0,1)", 0.0f},
  {"control_smooth(0, 1)",      0.0f},
  {"cmd_lin_vel_x(-0.2,0.2)",   0.0f},
  {"cmd_ang_vel_yaw(-1.5,1.5)", 0.0f},
  {"cmd_height_value(-0.5,0)", -0.1f},
  {"cmd_roll_value(-0.2,0.2)",  0.0f},
  {"command_vx_rate",           0.2f},
  {"pitch_zero_low",            0.22f},
  {"pitch_zero_high",           0.10f},
  {"sinroll_scale",             1.0f}
};
String otherString0_ = "Initial string";
String otherString = "";
int refreshInterval0_ = 1000; // 默认刷新间隔为1秒
std::vector<std::vector<float>> otherData;
const int otherDataLenMax = 200;
// const int onedatalen = 22; // 1 time + 17 sensors + 4 controls
SemaphoreHandle_t otherDataMutex;

MatrixXf matrixXf;

std::vector<LabeledVectorXf> vectors;

float joystick[2] = {0, 0};
unsigned long lastJoystickUpdate = 0;
const unsigned long JOYSTICK_TIMEOUT = 1000; // 1000ms timeout
bool switcher = false;

float height_bar = 0; // 初始高度值
float oscillation_bar = 0; // 初始振荡值

// 当前页面名称，根据实际加载的页面在各自处理函数中更新
// 可能的取值示例："main", "control" 等
String currentPage = "";


const String& getCurrentPage() {
  return currentPage;
}

float get_height_bar() {
  return height_bar;
}

float get_oscillation_bar() {
  return oscillation_bar;
}

void getJoyStickSize2(float *joyStick) {
  if (millis() - lastJoystickUpdate > JOYSTICK_TIMEOUT) {
    joystick[0] = 0;
    joystick[1] = 0;
    // Serial.println("Joystick reset to zero due to inactivity");
  }
  joyStick[0] = joystick[0];
  joyStick[1] = joystick[1];
}

void getSwitcher(bool *switcher_) {
  *switcher_ = switcher;
}

void setVectorXf(int index, const LabeledVectorXf& newVector) {
  if (index >= vectors.size()) {
    vectors.resize(index + 1);
  }
  vectors[index] = newVector;
}

void getVectorXf(int index, LabeledVectorXf& vectorXfOut) {
  if (index < vectors.size()) {
    vectorXfOut = vectors[index];
  }
}

void setMatrixXf(const MatrixXf& newMatrix) {
  matrixXf = newMatrix;
}

void getMatrixXf(MatrixXf& matrixXfOut) {
  matrixXfOut = matrixXf;
}

void setKValues(const KDict& newK) {
  // overwrite existing values in index order; assumes same layout
  Kweb = newK;
}

void getKValues(KDict& K) {
  K = Kweb;
}

float getK(const String &key, float defaultValue) {
  for (const auto &item : Kweb) {
    if (item.key == key) {
      return item.value;
    }
  }
  // output warning if key not found
  Serial.println("Warning: K key '" + key + "' not found. Using default value: " + String(defaultValue));
  return defaultValue;
}

void setOtherString(const String& newString) {
  otherString = newString;
}

void setOtherString0_(const String& newString) {
  otherString0_ = newString;
}


void setOtherData(const float *data, const int dlen, const float *control, const int clen) {
  if (xSemaphoreTake(otherDataMutex, portMAX_DELAY) == pdTRUE) {
    if (otherData.size() >= otherDataLenMax) {
      otherData.erase(otherData.begin());
    }
    // fill data into otherData
    std::vector<float> dataVec;
    for (int i = 0; i < dlen; i++) {
      dataVec.push_back(data[i]);
    }
    for (int i = 0; i < clen; i++) {
      dataVec.push_back(control[i]);
    }
    otherData.push_back(dataVec);
    xSemaphoreGive(otherDataMutex);
  }
}

void handleGetOtherData_json() {
  if (xSemaphoreTake(otherDataMutex, portMAX_DELAY) == pdTRUE) {
    // 确保 otherData 不为空
    if (otherData.empty()) {
      xSemaphoreGive(otherDataMutex);
      server.send(200, "application/json", "[]");
      return;
    }

    // 将 otherData 转换为JSON格式
    String jsonData = "[";
    for (size_t i = 0; i < otherData.size(); ++i) {
      jsonData += "[";
      for (size_t j = 0; j < otherData[i].size(); ++j) {
        jsonData += String(otherData[i][j]);
        if (j < otherData[i].size() - 1) {
          jsonData += ",";
        }
      }
      jsonData += "]";
      if (i < otherData.size() - 1) {
        jsonData += ",";
      }
    }
    jsonData += "]";

    // 清空 otherData
    otherData.clear();
    xSemaphoreGive(otherDataMutex);

    // 设置响应头
    server.sendHeader("Content-Type", "application/json");

    // 发送HTTP响应状态行和头部
    server.send(200, "application/json", jsonData);
  }
}

void handleGetOtherData() {
  if (xSemaphoreTake(otherDataMutex, portMAX_DELAY) == pdTRUE) {
    // 确保 otherData 不为空
    if (otherData.empty()) {
      xSemaphoreGive(otherDataMutex);
      server.send(200, "application/octet-stream", "");
      return;
    }

    // 将 otherData 转换为二进制格式
    String binaryData;
    float dataSize = static_cast<float>(otherData[0].size());
    char sizeBytes[sizeof(float)];
    memcpy(sizeBytes, &dataSize, sizeof(float));
    for (size_t k = 0; k < sizeof(float); ++k) {
      binaryData += sizeBytes[k];
    }

    for (size_t i = 0; i < otherData.size(); ++i) {
      for (size_t j = 0; j < otherData[i].size(); ++j) {
        float value = otherData[i][j];
        char bytes[sizeof(float)];
        memcpy(bytes, &value, sizeof(float));
        for (size_t k = 0; k < sizeof(float); ++k) {
          binaryData += bytes[k];
        }
      }
    }

    // 清空 otherData
    otherData.clear();
    xSemaphoreGive(otherDataMutex);

    // 设置响应头
    server.sendHeader("Content-Type", "application/octet-stream");

    // 发送HTTP响应状态行和头部
    server.send(200, "application/octet-stream", binaryData);
  }
}

void handleRoot() {
  if (!LittleFS.begin(true)) {
    Serial.println("An Error has occurred while mounting LittleFS");
    return;
  }

  File file = LittleFS.open("/main.html", "r");
  if (!file) {
    Serial.println("Failed to open file for reading");
    return;
  }

  String html = file.readString();
  file.close();

  currentPage = "main";

  // Insert K values into the HTML
  String kValuesHtml;
  int index = 0;
  for (const auto &item : Kweb) {
    String disabled = item.key == "sinwheel_scale" ? " disabled" : "";
    kValuesHtml += "<div class=\"k-item\" data-k-name=\"" + item.key + "\"><span class=\"k-index\">K" + String(index + 1) + "</span><span class=\"k-name\">" + item.key + "</span><span class=\"k-desc\"></span><span class=\"k-value\"><input type=\"text\" name=\"K" + String(index + 1) + "\" value=\"" + String(item.value, 5) + "\"" + disabled + "></span></div>";
    ++index;
  }
  html.replace("<!-- K values will be inserted here by the server -->", kValuesHtml);

  server.send(200, "text/html", html);
}

void handleRootJS() {
  if (!LittleFS.begin(true)) {
    Serial.println("An Error has occurred while mounting LittleFS");
    return;
  }

  File file = LittleFS.open("/main.js", "r");
  if (!file) {
    Serial.println("Failed to open file for reading");
    return;
  }

  String js = file.readString();
  file.close();

  server.send(200, "application/javascript", js);
}

void handleControl() {
  if (!LittleFS.begin(true)) {
    Serial.println("An Error has occurred while mounting LittleFS");
    return;
  }

  File file = LittleFS.open("/control.html", "r");
  if (!file) {
    Serial.println("Failed to open file for reading");
    return;
  }

  String html = file.readString();
  file.close();

  currentPage = "control";

  server.send(200, "text/html", html);
}

void handleControlJS() {
  if (!LittleFS.begin(true)) {
    Serial.println("An Error has occurred while mounting LittleFS");
    return;
  }

  File file = LittleFS.open("/control.js", "r");
  if (!file) {
    Serial.println("Failed to open file for reading");
    return;
  }

  String js = file.readString();
  file.close();

  server.send(200, "application/javascript", js);
}

void handleSetK() {
  int index = 0;
  for (auto &item : Kweb) {
    String argName = "K" + String(index + 1);
    if (server.hasArg(argName)) {
      item.value = server.arg(argName).toFloat();
    }
    ++index;
  }
  String message = "K Values Updated Successfully";

  // 使用 ArduinoJson 生成 JSON 响应
  StaticJsonDocument<200> doc;
  doc["message"] = message;
  doc["otherString"] = otherString;
  String jsonResponse;
  serializeJson(doc, jsonResponse);

  server.send(200, "application/json", jsonResponse);
}

void handleGetOtherString0_() {
  // 使用 ArduinoJson 生成 JSON 响应
  StaticJsonDocument<1000> doc; //size is 100KB
  doc["otherString0_"] = otherString0_;
  String jsonResponse;
  serializeJson(doc, jsonResponse);

  server.send(200, "application/json", jsonResponse);
  
  otherString0_ = ""; // 清空 otherString0_
}

void handleGetOtherString() {
  // 使用 ArduinoJson 生成 JSON 响应
  StaticJsonDocument<100000> doc; //size is 100KB
  doc["otherString"] = otherString;
  String jsonResponse;
  serializeJson(doc, jsonResponse);

  server.send(200, "application/json", jsonResponse);
  
  otherString = ""; // 清空 otherString
}


void handleGetVector(int index) {
  StaticJsonDocument<1000> doc;
  JsonArray vector = doc.createNestedArray("vector");
  JsonArray labels = doc.createNestedArray("labels");
  if (index < vectors.size()) {
    for (int i = 0; i < vectors[index].values.size(); i++) {
      vector.add(vectors[index].values(i));
      labels.add(vectors[index].labels[i]);
    }
  }
  String jsonResponse;
  serializeJson(doc, jsonResponse);
  server.send(200, "application/json", jsonResponse);
}

void handleSetVector(int index) {
  bool success = true;
  if (index >= vectors.size()) {
    vectors.resize(index + 1);
  }
  for (int i = 0; i < vectors[index].values.size(); i++) {
    String argName = "V" + String(index) + "_" + String(i);
    if (server.hasArg(argName)) {
      vectors[index].values(i) = server.arg(argName).toFloat();
    } else {
      success = false;
    }
  }
  String message = success ? "Vector" + String(index) + " Updated Successfully" : "Error: Failed to Update Vector" + String(index);
  StaticJsonDocument<200> doc;
  doc["message"] = message;
  String jsonResponse;
  serializeJson(doc, jsonResponse);
  server.send(200, "application/json", jsonResponse);
}

void handleGetVectorSize() {
  StaticJsonDocument<200> doc;
  doc["size"] = vectors.size();
  String jsonResponse;
  serializeJson(doc, jsonResponse);
  server.send(200, "application/json", jsonResponse);
}

void handleGetMatrix() {
  StaticJsonDocument<1000> doc;
  JsonArray matrix = doc.createNestedArray("matrix");
  for (int i = 0; i < matrixXf.rows(); i++) {
    JsonArray row = matrix.createNestedArray();
    for (int j = 0; j < matrixXf.cols(); j++) {
      row.add(matrixXf(i, j));
    }
  }
  String jsonResponse;
  serializeJson(doc, jsonResponse);
  server.send(200, "application/json", jsonResponse);
}

void handleSetMatrix() {
  bool success = true;
  for (int i = 0; i < matrixXf.rows(); i++) {
    for (int j = 0; j < matrixXf.cols(); j++) {
      String argName = "M" + String(i) + "_" + String(j);
      if (server.hasArg(argName)) {
        matrixXf(i, j) = server.arg(argName).toFloat();
      } else {
        success = false;
      }
    }
  }
  String message = success ? "Matrix Updated Successfully" : "Error: Failed to Update Matrix";
  StaticJsonDocument<200> doc;
  doc["message"] = message;
  String jsonResponse;
  serializeJson(doc, jsonResponse);
  server.send(200, "application/json", jsonResponse);
}

void setJoystickCommand(float x, float y) {
  joystick[0] = x;
  joystick[1] = y;
  lastJoystickUpdate = millis();
}

void handleControlSocketEvent(uint8_t num, WStype_t type, uint8_t *payload, size_t length) {
  if (type == WStype_DISCONNECTED) {
    setJoystickCommand(0.0f, 0.0f);
    return;
  }
  if (type != WStype_TEXT || length == 0) {
    return;
  }
  if (length == 1 && payload[0] == 'P') {
    controlSocket.sendTXT(num, "P");
    return;
  }
  if (length < 3) {
    return;
  }

  char message[64];
  size_t copy_len = length < sizeof(message) - 1 ? length : sizeof(message) - 1;
  memcpy(message, payload, copy_len);
  message[copy_len] = '\0';
  char *value = message + 2;

  switch (message[0]) {
    case 'J': {
      char *end;
      float x = strtof(value, &end);
      if (end == value || *end != ',') {
        return;
      }
      char *y_text = end + 1;
      float y = strtof(y_text, &end);
      if (end == y_text) {
        return;
      }
      setJoystickCommand(x, y);
      break;
    }
    case 'H':
      height_bar = atof(value);
      break;
    case 'O':
      oscillation_bar = atof(value);
      break;
    case 'S':
      switcher = strcmp(value, "true") == 0;
      break;
  }
}

void handleJoystickMove() {
  if (server.hasArg("x") && server.hasArg("y")) {
    setJoystickCommand(server.arg("x").toFloat(), server.arg("y").toFloat());
    server.send(200, "text/plain", "Joystick move received");
  } else {
    server.send(400, "text/plain", "Invalid request");
  }
}

void handleSwitchToggle() {
  if (server.hasArg("state")) {
    switcher = server.arg("state") == "true";
    server.send(200, "text/plain", "Switch toggle received");
  } else {
    server.send(400, "text/plain", "Invalid request");
  }
}

void handleHeightChange() {
  if (server.hasArg("value")) {
    height_bar = server.arg("value").toFloat();
    server.send(200, "text/plain", "Height updated");
  } else {
    server.send(400, "text/plain", "Missing value parameter");
  }
}

void handleOscillationChange() {
  if (server.hasArg("value")) {
    oscillation_bar = server.arg("value").toFloat();
    server.send(200, "text/plain", "Oscillation updated");
  } else {
    server.send(400, "text/plain", "Missing value parameter");
  }
}

// void handleSavePageState() {
//   // if (server.hasArg("currentPage")) {
//   //   currentPage = server.arg("currentPage");
//   //   server.send(200, "text/plain", "Page state saved");
//   // } else {
//   //   server.send(400, "text/plain", "Invalid request");
//   // }
// }

void handleGetControlPageData() {
  StaticJsonDocument<200> doc;
  doc["height_bar"] = height_bar;
  doc["oscillation_bar"] = oscillation_bar;
  doc["switcher"] = switcher;
  String jsonResponse;
  serializeJson(doc, jsonResponse);
  server.send(200, "application/json", jsonResponse);
}

void handleGetImuData() {
  StaticJsonDocument<200> doc;
  doc["yaw_rad"] = real_io::real_attitude.yaw_rad;
  doc["pitch_rad"] = real_io::real_attitude.pitch_rad;
  doc["roll_rad"] = real_io::real_attitude.roll_rad;
  String jsonResponse;
  serializeJson(doc, jsonResponse);
  server.send(200, "application/json", jsonResponse);
}

void HttpTask(void *pvParameters) {
  (void)pvParameters;
  while (true) {
    server.handleClient();
    vTaskDelay(10 / portTICK_PERIOD_MS);
  }
}

void WiFiTask(void *pvParameters) {
  bool useWiFi = *((bool*)pvParameters);
  Serial.println("useWiFi: " + String(useWiFi));
  if (useWiFi) {
    // STA 模式：连接路由器
    IPAddress subnet(255, 255, 255, 0);
    IPAddress primaryDNS(8, 8, 8, 8);   // 可选
    IPAddress secondaryDNS(8, 8, 4, 4); // 可选

    // choose active WiFi STA profile
    if (currentWifiProfileIndex < 0 || currentWifiProfileIndex >= WIFI_PROFILE_COUNT) {
      currentWifiProfileIndex = 0;
    }
    WiFiProfile profile = WIFI_PROFILES[currentWifiProfileIndex];

    if (!WiFi.config(profile.local_IP, profile.gateway, subnet, primaryDNS, secondaryDNS)) {
      Serial.println("STA Failed to configure");
    }

    WiFi.begin(profile.ssid, profile.password);
    while (WiFi.status() != WL_CONNECTED) {
      delay(1000);
      Serial.print("connecting...");
    }
    Serial.println("Connected to WiFi");
  } else {
    // AP 模式：开启热点
    if (currentWifiApProfileIndex < 0 || currentWifiApProfileIndex >= WIFI_AP_PROFILE_COUNT) {
      currentWifiApProfileIndex = 0;
    }
    WiFiApProfile ap = WIFI_AP_PROFILES[currentWifiApProfileIndex];

    IPAddress subnet(255, 255, 255, 0);  // 子网掩码
    // 先配置IP再启动AP
    WiFi.softAPConfig(ap.local_IP, ap.gateway, subnet);

    WiFi.softAP(ap.ssid, ap.password, ap.channel, ap.hidden ? 1 : 0, ap.max_conn);
    Serial.println("Access Point started");
    Serial.print("AP IP address: ");
    Serial.println(WiFi.softAPIP());
  }

  server.on("/", handleControl);
  server.on("/main.html", handleRoot);
  server.on("/main.js", HTTP_GET, handleRootJS);
  server.on("/control.html", handleControl);
  server.on("/control.js", HTTP_GET, handleControlJS);
  server.on("/setK", HTTP_POST, handleSetK);
  server.on("/getOtherString", HTTP_GET, handleGetOtherString);
  server.on("/getOtherString0_", HTTP_GET, handleGetOtherString0_);
  server.on("/getOtherData", HTTP_GET, handleGetOtherData);
  server.on("/getMatrix", HTTP_GET, handleGetMatrix); // 注册新的处理函数
  server.on("/setMatrix", HTTP_POST, handleSetMatrix); // 注册新的处理函数
  server.on("/getVectorSize", HTTP_GET, handleGetVectorSize); // 注册获取 vectors 大小的处理函数
  for (int i = 0; i < vectors.size(); i++) {
    server.on("/getVector" + String(i), HTTP_GET, [i]() { handleGetVector(i); });
    server.on("/setVector" + String(i), HTTP_POST, [i]() { handleSetVector(i); });
  }
  server.on("/joystickMove", HTTP_POST, handleJoystickMove); // 注册摇杆移动处理函数
  server.on("/switchToggle", HTTP_POST, handleSwitchToggle); // 注册开关切换处理函数
  server.on("/setHeight", handleHeightChange);
  server.on("/setOscillation", handleOscillationChange);
  // server.on("/savePageState", HTTP_POST, handleSavePageState);
  server.on("/getControlPageData", HTTP_GET, handleGetControlPageData); // 注册新的处理函数
  server.on("/getImuData", HTTP_GET, handleGetImuData);
  controlSocket.begin();
  controlSocket.onEvent(handleControlSocketEvent);
  server.begin();
  Serial.println("Web server started");
  xTaskCreatePinnedToCore(HttpTask, "HttpTask", 4096, NULL, 1, NULL, 0);

  while (true) {
    controlSocket.loop();
    vTaskDelay(1 / portTICK_PERIOD_MS);
  }
}

void setup_lqrwebserver(bool useWiFi) {
  Serial.begin(115200);

  // 初始化 otherDataMutex
  otherDataMutex = xSemaphoreCreateMutex();
  if (otherDataMutex == NULL) {
    Serial.println("Failed to create mutex");
    while (1); // 停止程序
  }

  // 创建 FreeRTOS 任务
  Serial.println("Creating WiFiTask:" + String(useWiFi));
  static bool usewifi = useWiFi; // 防止局部变量被销毁
  // 将 Web/WiFi 相关任务固定在 core 0 上运行
  xTaskCreatePinnedToCore(WiFiTask, "WiFiTask", 4096, &usewifi, 2, NULL, 0);
}
