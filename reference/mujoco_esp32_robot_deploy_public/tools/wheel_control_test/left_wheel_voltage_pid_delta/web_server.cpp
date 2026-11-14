#include "web_server.h"

#if defined(ARDUINO_ARCH_ESP32)
#include <WiFi.h>
#include <WebServer.h>

#include "pid_control.h"

static WebServer g_server(80);

static volatile bool g_wifi_ok = false;
static volatile bool g_wifi_is_sta = false;
static volatile uint32_t g_wifi_ip_u32 = 0;

// Last run parameters for naming downloads.
static portMUX_TYPE g_param_mux = portMUX_INITIALIZER_UNLOCKED;
static String g_last_csv_filename = "pid_log.csv";

static String format_float_compact(float v, int decimals) {
  String s(v, decimals);
  // Trim trailing zeros for nicer filenames.
  while (s.length() > 1 && s.endsWith("0") && s.indexOf('.') >= 0) s.remove(s.length() - 1);
  if (s.endsWith(".")) s.remove(s.length() - 1);
  return s;
}

static bool is_windows_filename_char(char c) {
  // Windows forbids: < > : " / \ | ? *
  if (c == '<' || c == '>' || c == ':' || c == '"' || c == '/' || c == '\\' || c == '|' || c == '?' || c == '*') {
    return false;
  }
  // Control chars also forbidden.
  if ((unsigned char)c < 32) return false;
  return true;
}

static String sanitize_windows_filename(const String& in) {
  String out;
  out.reserve(in.length());

  for (size_t i = 0; i < in.length(); i++) {
    char c = in[i];
    if (!is_windows_filename_char(c)) {
      out += '_';
      continue;
    }
    // Keep it shell/URL friendly too: avoid spaces.
    if (c == ' ') c = '_';
    out += c;
  }

  // Windows disallows trailing dot/space.
  while (out.length() > 0 && (out.endsWith(" ") || out.endsWith("."))) {
    out.remove(out.length() - 1);
  }

  // Reasonable length cap for typical filesystems.
  const size_t kMaxLen = 180;
  if (out.length() > kMaxLen) {
    out = out.substring(0, kMaxLen);
    while (out.length() > 0 && (out.endsWith(" ") || out.endsWith("."))) {
      out.remove(out.length() - 1);
    }
  }

  if (out.length() == 0) return String("pid_log.csv");
  return out;
}

static void update_last_filename_from_params(float kp, float ki, float kd, float lpf_tf) {
  String name;
  name.reserve(96);
  name += "pid_log_";
  // Use '_' separators to keep filenames portable.
  name += "kp=" + format_float_compact(kp, 6);
  name += "_ki=" + format_float_compact(ki, 6);
  name += "_kd=" + format_float_compact(kd, 6);
  name += "_lpf_tf=" + format_float_compact(lpf_tf, 6);
  name += ".csv";

  name = sanitize_windows_filename(name);

  portENTER_CRITICAL(&g_param_mux);
  g_last_csv_filename = name;
  portEXIT_CRITICAL(&g_param_mux);
}

// Prefer STA (router) first; fallback to AP if connection fails.
static constexpr bool WIFI_PREFER_STA = true;
static constexpr uint32_t WIFI_STA_TIMEOUT_MS = 8000;

#ifndef WIFI_STA_SSID
#define WIFI_STA_SSID "YOUR_WIFI_SSID"
#endif
#ifndef WIFI_STA_PASSWORD
#define WIFI_STA_PASSWORD "YOUR_WIFI_PASSWORD"
#endif
#ifndef WIFI_AP_SSID
#define WIFI_AP_SSID "ESP32-Access-Point"
#endif
#ifndef WIFI_AP_PASSWORD
#define WIFI_AP_PASSWORD "CHANGE_ME_123"
#endif

// STA (router)
static const char* WIFI_SSID_STA = WIFI_STA_SSID;
static const char* WIFI_PASS_STA = WIFI_STA_PASSWORD;
static const IPAddress WIFI_STA_LOCAL_IP(192, 168, 1, 111);
static const IPAddress WIFI_STA_GATEWAY(192, 168, 1, 1);
static const IPAddress WIFI_STA_SUBNET(255, 255, 255, 0);
static const IPAddress WIFI_STA_DNS1(8, 8, 8, 8);
static const IPAddress WIFI_STA_DNS2(8, 8, 4, 4);

static const char* WIFI_SSID_AP = WIFI_AP_SSID;
static const char* WIFI_PASS_AP = WIFI_AP_PASSWORD;

static bool wifi_try_sta() {
  WiFi.mode(WIFI_STA);

  // Static IP. If it fails, still try DHCP.
  (void)WiFi.config(WIFI_STA_LOCAL_IP, WIFI_STA_GATEWAY, WIFI_STA_SUBNET, WIFI_STA_DNS1, WIFI_STA_DNS2);

  WiFi.begin(WIFI_SSID_STA, WIFI_PASS_STA);
  Serial.print("[wifi] sta connect ssid=");
  Serial.println(WIFI_SSID_STA);
  const uint32_t t0 = millis();
  while (WiFi.status() != WL_CONNECTED) {
    if ((uint32_t)(millis() - t0) >= WIFI_STA_TIMEOUT_MS) return false;
    vTaskDelay(200 / portTICK_PERIOD_MS);
  }
  return true;
}

static void wifi_start_ap() {
  WiFi.mode(WIFI_AP);
  IPAddress local_IP(192, 168, 4, 1);
  IPAddress gateway(192, 168, 4, 1);
  IPAddress subnet(255, 255, 255, 0);
  WiFi.softAPConfig(local_IP, gateway, subnet);
  WiFi.softAP(WIFI_SSID_AP, WIFI_PASS_AP, 1, 0, 4);
}

static void handle_status() {
  PidLogStatus st = pidlog_status();
  float kp, ki, kd;
  pid_get_gains(&kp, &ki, &kd);
  float lpf_tf;
  pid_get_lpf_tf(&lpf_tf);
  float batt_v = pid_get_battery_voltage();

  String json = "{";
  json += "\"running\":" + String(st.running ? "true" : "false");
  json += ",\"ready\":" + String(st.ready ? "true" : "false");
  json += ",\"samples\":" + String(st.samples);
  json += ",\"capacity\":" + String(st.capacity);
  json += ",\"duration_s\":" + String(st.duration_s);
  json += ",\"wifi_ok\":" + String(g_wifi_ok ? "true" : "false");
  json += ",\"wifi_mode\":\"" + String(g_wifi_is_sta ? "sta" : "ap") + "\"";
  json += ",\"ip\":\"" + IPAddress(g_wifi_ip_u32).toString() + "\"";
  json += ",\"kp\":" + String(kp, 6);
  json += ",\"ki\":" + String(ki, 6);
  json += ",\"kd\":" + String(kd, 6);
  json += ",\"lpf_tf\":" + String(lpf_tf, 6);
  json += ",\"battery_v\":" + String(batt_v, 3);
  json += "}";

  g_server.send(200, "application/json", json);
}

// Set pure-voltage control command: GET /set_voltage?uq=V
// Only effective when firmware is running in pure-voltage mode.
static void handle_set_voltage() {
  if (!g_server.hasArg("uq")) {
    g_server.send(400, "text/plain", "missing uq parameter");
    return;
  }

  const float uq = g_server.arg("uq").toFloat();
  pid_set_voltage(uq);

  String msg;
  msg.reserve(96);
  msg += "set Uq=";
  msg += String(uq, 3);
  msg += " V";

  const float batt_v = pid_get_battery_voltage();
  msg += ", battery=";
  msg += String(batt_v, 3);
  msg += " V";

  g_server.send(200, "text/plain", msg);
}

static void handle_start() {
  uint32_t seconds = 5;
  if (g_server.hasArg("seconds")) {
    seconds = (uint32_t)g_server.arg("seconds").toInt();
  }

  // Optional PID gains: kp, ki, kd (set at start; if absent, keep last/default).
  bool has_pid = false;
  float kp, ki, kd;
  pid_get_gains(&kp, &ki, &kd);
  if (g_server.hasArg("kp")) {
    kp = g_server.arg("kp").toFloat();
    has_pid = true;
  }
  if (g_server.hasArg("ki")) {
    ki = g_server.arg("ki").toFloat();
    has_pid = true;
  }
  if (g_server.hasArg("kd")) {
    kd = g_server.arg("kd").toFloat();
    has_pid = true;
  }
  if (has_pid) {
    pid_set_gains(kp, ki, kd);
    Serial.print("[pid] gains kp=");
    Serial.print(kp, 6);
    Serial.print(" ki=");
    Serial.print(ki, 6);
    Serial.print(" kd=");
    Serial.println(kd, 6);
  }

  // Optional velocity LPF time constant: lpf_tf (seconds).
  if (g_server.hasArg("lpf_tf")) {
    const float tf = g_server.arg("lpf_tf").toFloat();
    pid_set_lpf_tf(tf);
    Serial.print("[pid] lpf_tf=");
    Serial.println(tf, 6);
  }

  // Store effective params for download filename (works with any combination).
  float eff_kp, eff_ki, eff_kd;
  float eff_lpf_tf;
  pid_get_gains(&eff_kp, &eff_ki, &eff_kd);
  pid_get_lpf_tf(&eff_lpf_tf);
  update_last_filename_from_params(eff_kp, eff_ki, eff_kd, eff_lpf_tf);

  if (!pidlog_start(seconds)) {
    g_server.send(500, "text/plain", "log buffer not available");
    return;
  }

  String html;
  html.reserve(384);
  html += "<!doctype html><meta charset=utf-8>";
  html += "<title>pid log</title>";
  html += "<script>setTimeout(()=>{location.href='/data.csv'},";
  html += String(seconds);
  html += "*1000);</script>";
  html += "running ";
  html += String(seconds);
  html += "s...";

  g_server.send(200, "text/html", html);
}

static void handle_data_csv() {
  PidLogStatus st = pidlog_status();
  if (!st.ready) {
    g_server.send(409, "text/plain", "not ready");
    return;
  }

  String filename;
  portENTER_CRITICAL(&g_param_mux);
  filename = g_last_csv_filename;
  portEXIT_CRITICAL(&g_param_mux);

  g_server.sendHeader("Content-Type", "text/csv");
  g_server.sendHeader("Content-Disposition", "attachment; filename=\"" + filename + "\"");
  g_server.setContentLength(CONTENT_LENGTH_UNKNOWN);
  g_server.send(200, "text/csv", "");
  g_server.sendContent("t_s,battery_v,control_v,target_vel,current_vel,raw_vel\n");

  String line;
  line.reserve(96);

  for (uint32_t i = 0; i < st.samples; i++) {
    PidLogSample s;
    if (!pidlog_get_sample(i, &s)) break;

    line = "";
    line += String(((float)s.t_us) * 1e-6f, 6);
    line += ",";
    line += String(((float)s.batt_mV) / 1000.0f, 3);
    line += ",";
    line += String(((float)s.uq_mV) / 1000.0f, 3);
    line += ",";
    line += String(((float)s.target_centi) / 100.0f, 4);
    line += ",";
    line += String(((float)s.meas_centi) / 100.0f, 4);
    line += ",";
    line += String(((float)s.raw_centi) / 100.0f, 4);
    line += "\n";

    g_server.sendContent(line);
  }
}

static void web_task(void* pv) {
  (void)pv;

  WiFi.setSleep(false);

  if (WIFI_PREFER_STA && wifi_try_sta()) {
    g_wifi_ok = true;
    g_wifi_is_sta = true;
    g_wifi_ip_u32 = (uint32_t)WiFi.localIP();
    Serial.print("[wifi] sta ok ip=");
    Serial.println(WiFi.localIP());
  } else {
    wifi_start_ap();
    g_wifi_ok = true;
    g_wifi_is_sta = false;
    g_wifi_ip_u32 = (uint32_t)WiFi.softAPIP();
    Serial.print("[wifi] ap fallback ssid=");
    Serial.print(WIFI_SSID_AP);
    Serial.print(" ip=");
    Serial.println(WiFi.softAPIP());
  }

  g_server.on("/status", HTTP_GET, handle_status);
  g_server.on("/start", HTTP_GET, handle_start);
  g_server.on("/data.csv", HTTP_GET, handle_data_csv);
  g_server.on("/set_voltage", HTTP_GET, handle_set_voltage);
  g_server.on("/", HTTP_GET, []() {
    String html;
    html.reserve(1024);
    html += F("<!doctype html><html><head><meta charset='utf-8'>");
    html += F("<title>left_wheel_voltage_pid_delta - help</title>");
    html += F("<style>body{font-family:sans-serif;margin:12px;}code{background:#f4f4f4;padding:2px 4px;border-radius:3px;}h1{font-size:20px;}h2{font-size:16px;margin-top:18px;}ul{margin-top:4px;}li{margin:2px 0;}</style>");
    html += F("</head><body>");
    html += F("<h1>left_wheel_voltage_pid_delta Web Help</h1>");

    html += F("<p>电池电压：<span id='batt_v'>--</span> V</p>");

    html += F("<h2>PID 模式 (USE_PID_MODE=1)</h2>");
    html += F("<ul>");
    html += F("<li><code>GET /start?seconds=N[&kp=..&ki=..&kd=..&lpf_tf=..]</code> ");
    html += F("&ndash; 运行 N 秒并采集日志，结束后自动跳转 <code>/data.csv</code></li>");
    html += F("<li><code>GET /data.csv</code> &ndash; 下载最近一次运行的 CSV 日志</li>");
    html += F("<li><code>GET /status</code> &ndash; 查看日志缓冲区、WiFi 状态、PID 参数等</li>");
    html += F("</ul>");

    html += F("<h2>纯电压模式 (USE_PID_MODE=0)</h2>");
    html += F("<ul>");
    html += F("<li><code>GET /set_voltage?uq=V</code> &ndash; 在线设置相电压 Uq，单位 V，推荐范围 [-7,7]</li>");
    html += F("<li><code>GET /status</code> &ndash; 仍可查看 WiFi 状态</li>");
    html += F("</ul>");

    html += F("<h2>当前设备</h2>");
    html += F("<p>可直接在浏览器地址栏输入上述 URL，或通过脚本/工具发起 HTTP 请求。</p>");

    html += F("<script>\n");
    html += F("async function updateStatus(){\n");
    html += F("  try {\n");
    html += F("    const r = await fetch('/status');\n");
    html += F("    if(!r.ok) return;\n");
    html += F("    const s = await r.json();\n");
    html += F("    if(typeof s.battery_v === 'number'){\n");
    html += F("      document.getElementById('batt_v').textContent = s.battery_v.toFixed(3);\n");
    html += F("    }\n");
    html += F("  } catch(e) {}\n");
    html += F("}\n");
    html += F("updateStatus();\n");
    html += F("setInterval(updateStatus, 2000);\n");
    html += F("</script>\n");

    html += F("</body></html>");

    g_server.send(200, "text/html", html);
  });

  g_server.begin();
  Serial.println("[web] server started");
  Serial.println("[web] GET /status");
  Serial.println("[web] GET /start?seconds=N");
  Serial.println("[web] GET /data.csv");
  Serial.println("[web] GET /set_voltage?uq=V");

  for (;;) {
    g_server.handleClient();
    vTaskDelay(5 / portTICK_PERIOD_MS);
  }
}

void web_server_start_task() {
  xTaskCreatePinnedToCore(web_task, "web", 4096, nullptr, 1, nullptr, 0);
}

#endif
