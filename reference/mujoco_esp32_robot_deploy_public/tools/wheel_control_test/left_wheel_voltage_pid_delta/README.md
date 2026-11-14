# left_wheel_voltage_pid_delta

ESP32 双核示例，支持两种模式：
- Core1：
  - PID 模式：SimpleFOC + 速度 PID（delta-Uq）+ 每个控制循环采集数据到 RAM
  - 纯电压模式：SimpleFOC + 固定 Uq 电压输出（无速度环、无日志）
- Core0：WebServer（PC 端启动采集/下载 CSV，或在纯电压模式下在线调节电压）

## 代码结构
- `left_wheel_voltage_pid_delta.ino`：
  - 通过宏切换 PID / 纯电压模式、driver1 / driver2：
    - `USE_PID_MODE`：`1`=PID 模式，`0`=纯电压模式
    - `USE_DRIVER2`：`0`=driver1 引脚 `(32,33,25,22)`，`1`=driver2 引脚 `(26,27,14,12)`
    - `kPureVoltageUq`：纯电压模式下的初始 Uq（单位 V，范围约 [-7,7]）
  - `setup()` 中先调用 `pid_set_use_driver2()` 选择驱动，再调用 `app_start_pid()` 或 `app_start_voltage()`
- `app.cpp/.h`：
  - `app_start_pid()`：启动 PID 控制任务 + WebServer（CSV 采集/下载、参数设置）
  - `app_start_voltage(float uq)`：启动纯电压控制任务 + WebServer（在线调节电压、查看状态）
- `pid_control.cpp/.h`：
  - Core1 控制环 + 数据采集（PID 模式）
  - 纯电压模式下保持 FOC 运行，持续输出 `g_uq_cmd`
  - 支持运行时切换使用 `driver1` 或 `driver2`，对应不同的 AS5600 I2C 总线
- `web_server.cpp/.h`：Core0 Web API + CSV 下载 + 纯电压在线调节

## 采集数据（每个 loop）
CSV 列：
- `t_s`：时间（秒），真实时间戳（从收到 `/start` 后第一次采样起算为 0）
- `battery_v`：电池电压（ADC 读取后换算）
- `control_v`：控制电压（q-axis 电压命令 `Uq`）
- `target_vel`：目标速度（rad/s）
- `current_vel`：当前速度（rad/s，低通后，用于 PID）
- `raw_vel`：原始速度（rad/s，低通前）

控制周期：`2ms (500Hz)`
缓存容量：默认尝试 `12000`（约 24 秒），如果内存不足会自动降到 `10000/8000/6000`（满了自动截断）

## Web 使用方法
启动时优先尝试 STA 连接到本地路由器，连接失败会自动回退到 AP。

### PID 模式（USE_PID_MODE=1）
默认上电后电机保持停止（0V），只有收到 `/start?seconds=N` 才会运行 N 秒并采集数据；N 秒到后自动停止并等待下一次 `/start`。

编译前通过本机私有构建配置设置 `WIFI_STA_SSID`、`WIFI_STA_PASSWORD`、`WIFI_AP_SSID` 和 `WIFI_AP_PASSWORD`，不要把真实凭据提交到仓库。未设置时使用源码中的占位符。

### STA（连接路由器）
- 默认静态 IP：`192.168.1.111`，网关：`192.168.1.1`，可按实际网络修改。
- 如果 STA 连接成功，请用：`http://<设备IP>/start?seconds=5`

### AP（回退热点）
- 默认 IP：`192.168.4.1`
- SSID 和密码使用上述构建参数配置。

接口：
- 启动并自动下载：`GET /start?seconds=N`
  - 返回一个极简页面：等待 N 秒后自动跳转 `GET /data.csv`
  - 可选参数（都可不传；任意组合；不传则沿用上次或默认）：
    - PID：`kp`、`ki`、`kd`
    - 速度低通滤波：`lpf_tf`（秒，LowPassFilter 的 time constant）
    - 组合示例：`/start?seconds=5&kp=0.05&ki=2.0&kd=0&lpf_tf=0.01`
- 下载 CSV：`GET /data.csv`
  - 下载文件名会包含本次运行的参数，便于区分（任意组合都适用）：
    - `pid_log_kp=..._ki=..._kd=..._lpf_tf=....csv`（会做 Windows 兼容字符清理）
  - 未采集完成时返回 `409 not ready`
- 状态：`GET /status`
  - 额外包含 WiFi 信号：`wifi_ok`（true/false）、`wifi_mode`（"sta"/"ap"）、`ip`（当前访问 IP）
  - 额外包含 PID 参数：`kp`、`ki`、`kd`、`lpf_tf`

示例：
- `http://192.168.4.1/start?seconds=5`

### 纯电压模式（USE_PID_MODE=0）

在 `left_wheel_voltage_pid_delta.ino` 中设置：

```cpp
#define USE_PID_MODE 0
static constexpr float kPureVoltageUq = -1.0f;  // 例如 -1V
```

上电后 Core1 以固定 Uq 进行扭矩控制（无速度反馈、无 PID/日志），Core0 仍然运行 WebServer，可通过以下接口在线调节电压：

- 设置电压：`GET /set_voltage?uq=V`
  - `V` 为目标相电压（单位 V），范围约在 `[-7,7]` 内
  - 仅在固件处于纯电压模式时生效
  - 设置成功后串口会打印当前 Uq，例如：`[volt] set Uq=-1.000 V`
- 状态：`GET /status`
  - 仍然返回 WiFi 状态和最近一次 PID 参数（PID 模式下使用）。

## 关键参数
在 `pid_control.cpp` 内：
- PID 参数：`kp_ / ki_ / kd_`
- 目标速度：`g_target_velocity`
- 供电/限幅：`driver1.voltage_power_supply`、`motor1.voltage_limit`
- 电池分压：`BAT_DIVIDER_RATIO`、ADC 引脚：`BAT_PIN`

## 画控制响应曲线
用 `plot_response.py` 从 CSV 画曲线（速度跟踪 + 控制电压 + 电池电压）：

```bash
python3 plot_response.py data/pid_log\ \(3\).csv
```

不指定 `--save` 时，默认输出到 `image/` 目录：`image/pid_log (3).png`。

保存图片：

```bash
python3 plot_response.py data/pid_log\ \(3\).csv --save response.png
```

只显示某个时间段（例如 0.5s~3.0s）：

```bash
python3 plot_response.py data/pid_log\ \(3\).csv --tmin 0.5 --tmax 3.0
```

## 多个 CSV 对比绘图
用 `plot_compare.py` 把多个 CSV 叠加到同一张图里做对比（current/target、Uq、电池电压）：

```bash
python3 plot_compare.py data/pid_log\ \(3\).csv data/pid_log\ \(4\).csv
```

不指定 `--save` 时，默认输出到 `image/` 目录：`image/compare_<a>_vs_<b>....png`。

指定保存路径：

```bash
python3 plot_compare.py data/pid_log\ \(3\).csv data/pid_log\ \(4\).csv --save image/compare.png
```

只显示某个时间段：

```bash
python3 plot_compare.py data/pid_log\ \(3\).csv data/pid_log\ \(4\).csv --tmin 0.5 --tmax 3.0
```

自定义曲线标签（逗号分隔）。如果 labels 数量少于 CSV 个数，会自动用文件名补齐；多于则截断：

```bash
python3 plot_compare.py data/pid_log\ \(3\).csv data/pid_log\ \(4\).csv --labels one,two
```

绘图脚本会把 raw_vel 单独画在最上面的子图，同时也画 target 线；current_vel 保持在它下面，Uq 和 battery 紧随其后。

### plot.sh（可选）
本目录的 `plot.sh` 提供了一个可改的示例：用 bash 数组传递 CSV 路径（文件名带空格也安全），并可选传 `--labels`。
