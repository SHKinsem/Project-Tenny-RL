# Android 手机摇杆控制 MuJoCo 仿真：可行性调查

日期：2026-11-14。用户条件：Android；参考 SHKinsem/watch-link；连接方式不限，优先尽快可用；本次不初始化 CodeGraph。

## 结论

可行。项目已有手机触摸事件、摇杆输入、HTTP 服务和策略控制链路，接入不需要重新训练策略，也不需要 ESP32。最快验证路线是手机浏览器使用现有控制页，通过 Wi-Fi 访问电脑转发到 WSL 的服务；若需要独立 Android APK，可参考 watch-link 的 Flutter 分层，新增一个通过 HTTP 发送指令的控制页。蓝牙适合作为后续传输适配器，当前没有开箱即用的手机到电脑蓝牙通道。

本次完成源码和接口层验证，没有进行手机触摸实测、局域网端到端演示、蓝牙配对或视频传输。Windows reference 是调查依据；没有比对当前 WSL 工作副本的文件哈希，不据此宣称其运行版本完全一致。

## 当前项目已经具备什么

以下路径均相对 reference/mujoco_esp32_robot_deploy_public：

| 层 | 已有实现 | 依据 |
|---|---|---|
| 手机输入 | Canvas 摇杆，pointerdown/move/up/cancel，触摸时阻止页面滚动，松手发零 | data/control.js:142、348、383 |
| 发送 | 移动事件限频间隔20ms，拖动期间100ms续发；这是发送逻辑，不能当作实测无线频率/延迟 | data/control.js:211、313 |
| 仿真接口 | POST /joystickMove，表单 x/y；另有开关、高度、横滚及 /status | host/web/server.py:194 |
| 命令映射 | Operator → [前后速度、偏航角速度、高度参数、横滚参数] | host/web/state.py:79；host/web/command.py:42 |
| 仿真执行 | KeyCmd.merge → HoldObs → ObsPack → RL_Policy.run → env.step | host/sim/viewer.py:179、374–389 |
| 失联 | 超过1000ms没有摇杆更新，将摇杆视为回中 | host/web/state.py:67 |

前后速度限制为 ±0.2 m/s；偏航命令的允许范围为 ±1.5 rad/s，但当前摇杆公式满幅大约给出 ±1 rad/s。高度参数范围 [-0.5,0]、横滚参数范围 [-0.2,0.2]，保持原项目语义，不把高度参数当成绝对离地高度。

控制的是策略目标，底层仍由现有平衡策略生成动作；手机不直接设置关节或修改模型位置。仿真画面继续显示在电脑的 MuJoCo 独立窗口，网页当前只有控制与姿态数据。手机同时看实时模型画面需要另做视频串流，属于额外工作。

## 方案比较

| 方案 | 改动 | 判断 |
|---|---|---|
| Android 浏览器 + Wi-Fi + 现有网页 | 处理 Windows→WSL 转发；改善手机 viewport/布局；处理已发现的转向边界 | 最快验证输入体验 |
| 独立 Flutter APK + Wi-Fi HTTP | 新增摇杆页、连接状态、HTTP transport、生命周期归零；复用仿真接口 | 适合正式保留一个 App |
| Android App + BLE + Windows 桥接 + HTTP | 增加 GATT 服务、连接/重连/协议，以及主机蓝牙能力检查 | 可行，但首版工作更多 |
| App + Classic SPP 或 HID | 分别需要匹配 RFCOMM 服务或实现 HID，并增加仿真输入适配 | 需明确协议后再选；无现成项目实现证据 |

推荐首版链路：

```text
Android 触摸摇杆
    → Wi-Fi / HTTP
    → Windows 局域网入口（代理或端口转发）
    → WSL :8765 /joystickMove
    → Operator → 平衡策略 → MuJoCo 模型运动
```

现有 tasks/start-sim-wsl.sh 绑定 hostname -I 的第一个 WSL 地址。复现记录中的 172.* 地址是过去的 WSL 网卡地址，可能变化，不是可直接交给手机使用的稳定局域网入口。手机应访问 Windows 的实际局域网地址和被正确转发的端口；还要确认 Wi-Fi 无客户端隔离、监听正确、防火墙允许。微软说明 NAT 与 mirrored 的 LAN 访问配置不同：[WSL 网络官方说明](https://learn.microsoft.com/en-us/windows/wsl/networking)。本轮没有修改网络模式、转发或防火墙。

现有 JS 尝试 ws://hostname:81，但 Python 仿真服务只提供 HTTP :8765，没有实现对应 WebSocket 服务。当前仿真应使用 HTTP fallback；不能把前端已有 WebSocket 代码当成仿真 WebSocket 已可用。首版直接 HTTP 即可，后续有测量依据再增加同端口 WebSocket。

## watch-link 可以怎样参考

已通过 GitHub connector 检查私有仓库 [SHKinsem/watch-link](https://github.com/SHKinsem/watch-link)，固定 main 快照 c9149c145a1a5692870052322b19c238c47010ff。

- [合并 App 说明](https://github.com/SHKinsem/watch-link/blob/c9149c145a1a5692870052322b19c238c47010ff/link_app/README.md) 和 [main.dart](https://github.com/SHKinsem/watch-link/blob/c9149c145a1a5692870052322b19c238c47010ff/link_app/lib/main.dart)：现有页签为同时看/XIAO/手表；使用 IndexedStack 保持设备页和连接。可以参考这个结构增加“仿真控制”页，也可先建独立小 App。
- [XiaoBleTransport](https://github.com/SHKinsem/watch-link/blob/c9149c145a1a5692870052322b19c238c47010ff/packages/xiao_link/lib/xiao_link_ble.dart)：手机扫描、连接 XIAO，发现服务、写 characteristic、订阅通知。它是面向板子协议的客户端，不能直接连接只开放 HTTP 的 MuJoCo 仿真。
- [PlatformWatchTransport](https://github.com/SHKinsem/watch-link/blob/c9149c145a1a5692870052322b19c238c47010ff/packages/watch_link/lib/src/platform_transport.dart)：通过平台通道使用厂商手表通信；不是通用手机到电脑控制链路。
- [link_core](https://github.com/SHKinsem/watch-link/blob/c9149c145a1a5692870052322b19c238c47010ff/packages/link_core/README.md)：主要是传感器样本、序号、单位及命令队列。可以参考 UI/会话/transport 分离，但不应把运动命令塞进 MotionSample。连续摇杆指令应保留最新值，避免排队重放旧运动；开关等离散操作可以串行确认。

以上是复用建议，不表示 watch-link 已有摇杆、已支持 PC，或已经修改该仓库。本次只读访问它。

## 若后续必须使用蓝牙

沿用 watch-link 的手机 BLE client 思路时，电脑需要广播并提供可写的 GATT service，Windows 收到摇杆数据后再调用 WSL 的现有 HTTP 接口。无需把 Windows 蓝牙设备直接交给 WSL。

Windows 官方支持 GATT server，但具体适配器需查询 peripheral 能力：[Windows GATT server](https://learn.microsoft.com/en-us/windows/apps/develop/devices-sensors/gatt-server)。[Bleak](https://bleak.readthedocs.io/en/latest/) 本身是 GATT client，不能直接作为此方案的电脑 GATT server。另一种角色分配是 Android 做 peripheral、Windows 做 client，但需要实现新的手机端服务。

普通 BLE 遥控 App、Classic SPP 和 HID gamepad 是不同协议；配对成功并不代表仿真能收到摇杆。不需要额外板子的无线方案在架构上成立，但本机蓝牙能力、具体手机兼容性、丢包和延迟均未验证。更多官方依据见 phone-control-external-research.md。

## 接入前需要处理的具体问题

1. **纯横向转向边界存在问题。** 当前 map_joystick 在 y=0 时，x=-1 和 x=+1 都输出约 -1.0 rad/s。这是直接调用实际 Python 映射确认的结果；现有12项映射测试没有左右转向用例。实现控制器时先修正符号语义，并验证左/右、前后组合和精确零点。尚未修改原作者映射。
2. 当前纵向映射在 |y|≥0.2 时就到 ±0.2 m/s 满幅；并非整段摇杆行程线性调速。源码说明这是为原策略倒车偏弱做的处理。首版保留时应明确手感，是否改成线性须在实际仿真中验证。
3. 现有页面缺少移动端 viewport 声明，且没有手机布局验收。按系统缩放能显示不代表摇杆触摸尺寸合适。
4. 超时只归零摇杆速度目标，不关闭策略、不重置高度和横滚；模型仍有速度平滑、惯性和平衡响应。不要把“1秒命令超时”描述为“1秒内物理位置完全停住”。
5. App 首版建议前台以20Hz发送最新值（设计目标，非实测）；松手、取消触摸、切后台和断开时尽力发零，接收端继续保留超时保护。恢复连接默认回中并要求用户重新开启，避免恢复旧运动指令。蓝牙或 HTTP 都应做值域/非有限数校验和队列限流。

## 本次验证与边界

- Windows Python 3.13，以 -X utf8 -B 运行原有 test_web_command.py：12/12通过。
- 原有 test_web_http.py：12/12通过；临时回环服务，未开启仿真、未发指令给运行中的模型。默认沙箱禁止 localhost 连接，沙箱外同命令通过。
- 额外只读 Operator 探针：前推 +0.2、后推 -0.2、回中0；1000ms仍有效、1001ms归零；左右纯横向均约-1.0，记录为待修问题。探针不构成机器人动力学验证。
- 首次映射测试因 Windows 默认 GBK 读取 UTF-8 源码失败，改用 Python -X utf8 后通过，未改源码编码。
- 未安装 App、未构建 APK、未配对蓝牙、未触发固件编译/烧录、未重训策略、未进行手机到仿真的端到端测量。

## 建议的最小实施顺序

先核对 Windows 与 WSL 工作副本的实际文件，修正纯左右转向并覆盖相关用例；再让手机访问现有控制页，依次验收前进/后退/左右/松手/切后台/断网，记录命令到达和实际运动响应。之后若需要独立 APK，按 watch-link 风格实现小控制页和 HTTP transport；确定体验后再加 BLE transport。每一步保持原模型、观测、策略和训练环境。
