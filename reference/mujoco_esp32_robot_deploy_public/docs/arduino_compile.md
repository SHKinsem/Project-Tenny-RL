# Arduino 固件编译

## 直接使用

先安装 Arduino CLI、ESP32 Arduino Core 和所需库，然后在仓库根目录执行：

```bash
./arduino_compile.sh
```

脚本默认使用 `ESP32 Dev Module`（`esp32:esp32:esp32`）。它会在临时标准目录中创建 sketch 副本，因此仓库克隆目录可以保留任意名称；临时目录在编译结束后自动删除。

可通过环境变量覆盖板型和库目录，也可以向脚本追加 `arduino-cli compile` 参数：

```bash
FQBN=esp32:esp32:esp32 ./arduino_compile.sh
ARDUINO_LIB_ROOT=/path/to/libraries ./arduino_compile.sh
./arduino_compile.sh --verbose
```

脚本只编译，不上传真机。

## 依赖与库目录

需要以下库：

| 库 | 已验证版本 | 用途或原因 |
|---|---:|---|
| ArduinoJson | 6.21.0 | Web JSON |
| SCServo | 使用现有兼容版本 | 舵机通信 |
| Ethernet | 2.0.2 | WebSockets 的 `Ethernet.h` 依赖 |
| Eigen | 0.3.2 | 提供 `ArduinoEigen.h` |
| ArxTypeTraits | 0.3.2 | Eigen 自动安装的依赖 |
| NeuralNetwork | 4.1.9 | 提供 `NeuralNetwork.h` |
| Simple FOC | 2.4.0 | 电机控制 |
| WebSockets | 2.7.2 | WebSocket 通信 |

可通过 Arduino IDE 库管理器安装，或使用 Arduino CLI：

```bash
arduino-cli lib install ArduinoJson Ethernet Eigen NeuralNetwork "Simple FOC" WebSockets
```

如果 SCServo 未出现在库管理器中，请使用已有的兼容库。脚本默认复用 `../.arduino/user/libraries`；其他环境应把 `ARDUINO_LIB_ROOT` 指向同时包含 `ArduinoJson` 和 `SCServo` 的目录。其余库可由 Arduino CLI 自动解析。

IMU 使用项目内的 `src/hal/MPU6050Mahony.cpp`，不再依赖 MPU6050_tockn。

## Wi-Fi 配置

`src/web/wifi_config.h` 只包含占位符。使用 CLI 编译时，可通过环境变量在本机传入凭据；脚本只在临时 sketch 副本中生成配置，不会改写仓库文件：

```bash
WIFI_STA_SSID='YOUR_WIFI_SSID' \
WIFI_STA_PASSWORD='YOUR_WIFI_PASSWORD' \
WIFI_AP_SSID='ESP32-Access-Point' \
WIFI_AP_PASSWORD='CHANGE_ME_123' \
./arduino_compile.sh
```

请替换占位值，并根据自己的网络调整静态 IP 和网关；不要提交真实 SSID、密码或网络配置。

## Arduino IDE

使用 Arduino IDE 2.3.10 打开 `wl_pro_robot.ino`，选择 `ESP32 Dev Module`（`esp32:esp32:esp32`）后执行“验证”：

- 在 IDE 的 `directories.user` 中设置自己的库目录，并安装上述依赖。
- `sketch.yaml` 声明板型和主要库版本。
- `build_opt.h` 加入项目的 `src/` 头文件搜索路径。
- Wi-Fi 凭据通过本机私有构建配置提供，不要写入待提交文件。

## 本次验证环境

- macOS
- Arduino CLI 1.5.1
- ESP32 Arduino Core 3.3.12
- 目标板：ESP32 Dev Module

## 本次解决的问题

### 1. 已有库不在 CLI 默认库目录

`arduino-cli lib list` 最初看不到 `../.arduino/user/libraries` 中已有的库。脚本通过多个 `--library` 参数显式复用这些库，避免重复安装。

### 2. 找不到 `hal/Servo_STS3032.h`

报错：

```text
fatal error: hal/Servo_STS3032.h: No such file or directory
```

Arduino 构建器没有按项目注释所述自动把 `src/` 加入头文件搜索路径。脚本加入：

```text
-I<仓库绝对路径>/src
```

### 3. 缺少 Ethernet、Eigen 和 NeuralNetwork

分别表现为找不到 `Ethernet.h` 或对应库头文件。使用 Arduino CLI 安装 `Ethernet`、`Eigen` 和 `NeuralNetwork`；`ArxTypeTraits` 作为 Eigen 依赖自动安装。

### 4. SimpleFOC 2.2.1 与 ESP32 Core 3.3.12 不兼容

报错：

```text
Simple_FOC/src/current_sense/hardware_specific/esp32_adc_driver.cpp:
fatal error: esp_intr.h: No such file or directory
```

旧 SimpleFOC 使用了新 ESP32 Core 已移除的接口。升级到 SimpleFOC 2.4.0，并确保编译不再优先选择旧目录中的 2.2.1。

### 5. 旧 `esp_adc_cal` 电量校准接口失效

固件原来使用 `esp_adc_cal_characteristics_t`、`esp_adc_cal_characterize()` 和 `esp_adc_cal_raw_to_voltage()`，与新 ESP32 Core 不兼容。

`wl_pro_robot.ino` 已改为：

```cpp
uint32_t voltage = analogReadMilliVolts(BAT_PIN);
double battery = (voltage * 3.97) / 1000.0;
```

Arduino 的 `analogReadMilliVolts()` 负责 ADC 校准，原来的分压换算保持不变。旧校准类型、初始化代码和头文件引用已删除。

### 6. Arduino CLI 自动函数声明丢失返回类型

预处理输出中出现了没有返回类型的声明，例如：

```cpp
get_sensors_data(float *dataArray, const int arraySize);
control_loop();
bat_check();
```

这会产生大量 `expected constructor, destructor, or type conversion` 级联错误。`wl_pro_robot.ino` 已为相关函数添加显式声明，不再依赖这部分自动生成结果。

### 7. WebSockets 与 ESP32 Core 的 Base64 符号重复

链接时报错：

```text
multiple definition of `base64_init_decodestate'
multiple definition of `base64_decode_value'
multiple definition of `base64_decode_block'
```

WebSockets 的 `cdecode.c` 只有在定义 `ESP32` 时才使用核心自带的 Base64 实现。脚本统一添加：

```text
-DESP32
```

避免 WebSockets 再编译一份相同符号。

## 编译结果

ESP32 Core 3.3.12、SimpleFOC 2.4.0、WebSockets 2.7.2 下已完整编译并链接成功：

当前配置使用 Arduino CLI 的参考编译结果：

```text
Sketch uses 1169363 bytes (89%) of program storage space.
Maximum is 1310720 bytes.
Global variables use 79440 bytes (24%) of dynamic memory,
leaving 248240 bytes for local variables.
Maximum is 327680 bytes.
```

## 上传

默认 921600 波特率曾出现 `Unable to verify flash chip connection`。改用 115200 后，`flash-id` 能正确读出 4MB、3.3V 闪存，IDE 的 Upload Speed 也已固定为 115200。

已通过 USB 串口实际上传固件，bootloader、分区表和主程序均写入成功并通过哈希校验，随后正常触发硬复位。

上传成功不代表已验证真机控制行为；ADC 读数、传感器和执行器仍需单独检查。
