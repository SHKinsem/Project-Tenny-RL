# 手机摇杆控制仿真：外部资料核验

核验日期：2026-11-14。范围仅限平台能力与传输架构；未验证本项目源码、手机 App、蓝牙适配器或实际网络。未安装软件、修改系统/防火墙配置或烧录硬件。

## 结论

**蓝牙可以实现，但手机上的“蓝牙摇杆 App”与电脑识别的 Bluetooth HID 游戏手柄不是同一件事。** 对 Windows + WSL2 Ubuntu + Python 仿真，建议先确定手机 App 输出 BLE GATT、经典蓝牙 RFCOMM/SPP，还是 HID。若只需要快速获得触摸摇杆，Wi-Fi 网页/客户端发送控制量是值得优先评估的工程方案；这是架构判断，并非已完成延迟或兼容性测试。

## BLE：先匹配连接角色与数据协议

- Android 官方区分两组角色：central 扫描并连接、peripheral 广播；GATT client 发请求、server 提供服务。两组角色概念不同，不能仅凭“可以蓝牙连接”断定互通。通常手机遥控 App 为 central/GATT client，接收端应广播它所需的服务 UUID、特征 UUID，并解码其写入的数据。[Android BLE 概述](https://developer.android.com/develop/connectivity/bluetooth/ble/ble-overview)
- Android 也提供 `BluetoothManager.openGattServer` 和 BLE 广播 API，可构建手机 peripheral/GATT server；但广播支持取决于设备，`getBluetoothLeAdvertiser()` 可能返回 null。因此“手机做 peripheral → Windows Python 做 central”技术上有路线，现成 App 是否提供此模式需核实。[Android BluetoothManager](https://developer.android.com/reference/android/bluetooth/BluetoothManager#openGattServer(android.content.Context,%20android.bluetooth.BluetoothGattServerCallback))、[BluetoothAdapter 广播能力](https://developer.android.com/reference/android/bluetooth/BluetoothAdapter#getBluetoothLeAdvertiser())
- iOS Core Bluetooth 提供 `CBCentralManager` 与 `CBPeripheralManager`；后者管理并广播本 App 的 peripheral 服务。因此 iPhone 可开发 BLE 控制端，也可开发 BLE peripheral。不能据此推断任何现成 iPhone App 与电脑协议兼容。[Apple Core Bluetooth](https://developer.apple.com/documentation/corebluetooth)、[CBPeripheralManager](https://developer.apple.com/documentation/corebluetooth/cbperipheralmanager)
- Windows 有 GATT server API（`GattServiceProvider`、`GattLocalCharacteristic`），可接手机 central 写入；peripheral 能力依赖适配器/驱动，须查询 `BluetoothAdapter.IsPeripheralRoleSupported`。官方 API 可用于 Win32 桌面应用，常见实现可用 Windows 端桥接程序，再经 TCP 将控制量送入 WSL；该桥接尚未实现。[Windows GATT Server](https://learn.microsoft.com/en-us/windows/apps/develop/devices-sensors/gatt-server)、[Windows Bluetooth FAQ](https://learn.microsoft.com/en-us/windows/apps/develop/devices-sensors/bluetooth-dev-faq)
- Python Bleak 官方定位是 **GATT client**，支持扫描、连接、读写、订阅通知；其 Windows backend 使用 WinRT。它适合“电脑 central → 手机/设备 GATT server”，不能把 `BleakClient` 当成 Windows peripheral/GATT server。若现成手机 App 只作为 central，须另写 server 或采用其他经过验证的 server 实现。[Bleak 文档](https://bleak.readthedocs.io/en/latest/)、[Bleak Windows backend 官方代码文档](https://github.com/hbldh/bleak/blob/develop/docs/backends/windows.rst)

## 经典蓝牙串口与 HID 的边界

- Android 支持 RFCOMM socket，Windows 也有 `Windows.Devices.Bluetooth.Rfcomm` 及双向串口式数据流 API。若手机 App 明确支持 RFCOMM/SPP，Windows 原生接收程序是一条可行路线；服务 UUID、客户端/服务端方向及报文格式仍须一致。**仅配对并不保证自动生成 COM 口**；若要用 Python 串口方案，必须先确认 Windows 实际存在可用 COM 服务。[Android 连接蓝牙设备](https://developer.android.com/develop/connectivity/bluetooth/connect-bluetooth-devices)、[Windows RFCOMM](https://learn.microsoft.com/en-us/windows/apps/develop/devices-sensors/send-or-receive-files-with-rfcomm)
- iOS 不应简单概括成“完全没有 Classic”：当前 Core Bluetooth 文档明确提及 BR/EDR，并提供 Classic 示例；这不等于任意 Android 式 SPP/RFCOMM App 能直接移植到 iOS。Apple 的 External Accessory/Classic 配件路线还涉及 MFi。对跨 Android/iOS 的新控制协议，BLE 是更清晰的候选，具体 Classic/SPP 需求应逐项核实。[Apple Core Bluetooth Classic 示例](https://developer.apple.com/documentation/corebluetooth/using-core-bluetooth-classic)、[Apple Bluetooth / MFi](https://developer.apple.com/bluetooth/)
- Android API 28 起有 `BluetoothHidDevice`，包括 gamepad/joystick subclass；App 必须注册 HID 服务并发送 HID report，非前台时会自动注销。**普通 BLE UART/SPP 发送摇杆坐标，不会自动成为系统游戏手柄。** 若选择 HID 路线，还需验证手机固件/应用支持、Windows 是否识别游戏控制器，以及 WSL 仿真如何读取或桥接 Windows 输入。[Android BluetoothHidDevice](https://developer.android.com/reference/android/bluetooth/BluetoothHidDevice)

## WSL2：优先在 Windows 接蓝牙，再通过网络桥接

不能假定 WSL 内的 BlueZ/Bleak 直接使用 Windows 已配对的蓝牙适配器。微软说明 USB 设备访问需要 USB/IP 路线，附加到 WSL 后设备不能同时由 Windows 使用；蓝牙 USB 适配器是否可转交、Linux 驱动与 BlueZ 是否可用必须单独检查。**也不能断言 WSL 永远不支持蓝牙或必需自行编译内核**：微软当前 6.18 分支 `config-wsl` 已列出 `CONFIG_BT=m`、`CONFIG_BT_HCIBTUSB=m` 等；这不证明本机运行的内核及其模块具有相同配置。基于这些约束，“Windows 接蓝牙 → TCP/HTTP → WSL 仿真”是减少硬件直通依赖的工程建议。[微软 WSL USB 文档](https://learn.microsoft.com/en-us/windows/wsl/connect-usb)、[微软 WSL 内核配置](https://github.com/microsoft/WSL2-Linux-Kernel/blob/linux-msft-wsl-6.18.y/arch/x86/configs/config-wsl)

## Wi-Fi HTTP/WebSocket 网络路径

微软文档支持下列判断；这里没有执行任何网络配置。

1. 手机与电脑须网络可达；同一 SSID 仍可能受到访客网络/AP 隔离影响。手机应连接电脑 LAN 地址，手机上的 `localhost` 是手机自身。
2. WSL2 默认 NAT：Windows 通常可用 `localhost:端口` 访问 WSL 服务，但这不等于手机能访问该端口。可由 Windows 本地 HTTP/TCP 桥接转发；或评估文档中的 TCP `portproxy`。WSL 服务须绑定可达地址，防火墙也须允许对应连接。
3. Windows 11 22H2+ 的 mirrored networking 支持从 LAN 直接访问 WSL；仍需核实当前模式、端口监听与 Windows/Hyper-V 防火墙规则，不能假设改为 mirrored 即自动可用。
4. HTTP/WebSocket 都可作为此网络路径上的应用层方案；若使用 WebSocket，它使用 TCP，不能把 TCP `portproxy` 当成任意 UDP 转发。

来源：[微软 WSL 网络文档](https://learn.microsoft.com/en-us/windows/wsl/networking)、[netsh interface portproxy 官方说明](https://learn.microsoft.com/en-us/windows-server/networking/technologies/netsh/netsh-interface-portproxy)

## 下一步需要实测的信息

- 手机系统/版本、现成 App 名称与协议文档；是否可配置 UUID 和连续发送摇杆数值，是否只支持 HC-05 一类经典串口目标。
- Windows 蓝牙适配器是否支持 peripheral，Windows/WSL 的实际版本与网络模式。
- 既有仿真的输入接口：读取 HID、网络指令还是其他数据源，由本地源码调查确定。
- 蓝牙/Wi-Fi 的实际更新率、延迟、丢包/断线表现。建议控制量采用限幅、死区、序号/时间戳与超时归零；这是待实现的控制设计建议，非平台文档给出的性能保证。
- 未验证任何特定第三方摇杆 App，因此不列出“装好就能用”的 App 推荐。
