# 正式策略预览性能：官方资料核对

日期：2026-11-15。范围仅为预览，未运行训练、仿真、基准或服务。通过 Context7 定位官方资料后，直接核对了 MuJoCo 3.12.0 文档；该版本与仓库 `experiments/torque200/requirements.txt` 固定版本一致。下列性能判断均不代表本机实测。

## 官方事实与本项目适用性

| 问题 | 官方事实 | 本项目推断与限制 |
|---|---|---|
| MJX 单场景 CPU 是否合适 | MJX-JAX 面向大批量相同场景；官方提醒单个场景可能比 CPU 优化的 native MuJoCo 慢约 10 倍，最适合数千至数万场景并行。[S1] | 单机器人 CPU 预览值得优先研究 native MuJoCo。10 倍不是本项目收益承诺，也不能从批量训练吞吐反推单机器人速度。此警告明确针对 MJX-JAX，不能泛化为所有 MJX 实现。 |
| 物理与显示能否解耦 | MuJoCo 将抽象场景更新与 OpenGL 绘制分开；可接入其他渲染器。官方示例在每个显示帧前执行多个 `mj_step`。[S2] | 保持策略与物理固定 dt，单独降低显示频率；显示线程只消费一致的最新快照。慢显示帧不应决定控制 dt，也不应无限补积过期显示帧。 |
| EGL 是否等于 CPU 隔离 | MuJoCo 的 Linux 文档将 EGL 列为硬件加速离屏渲染；Khronos 将其定义为图形上下文、表面和同步接口。Mesa 的 EGL 实现也存在软件回退。[S2][S3][S4] | `MUJOCO_GL=egl` 不能证明 CPU-only，也不能证明图形工作已与训练 GPU 隔离。独立进程只隔离进程状态，不自动隔离共享 GPU、CPU 或内存带宽。实际后端和设备仍需单独确认。 |
| OSMesa 的意义 | MuJoCo 明确将 Linux OSMesa 列为无显示器的软件渲染路径。[S2] | 可作为减少服务器 GPU 绘图竞争的候选，但会增加 CPU 光栅化负担；是否更顺畅取决于 CPU 余量、画面大小与帧率，不能认定更快。此处官方支持说明针对 Linux。 |
| 阴影、反射是否有额外成本 | 经典渲染器的 `mjvScene.flags` 可关闭阴影、反射等；官方说明阴影与反射会增加额外绘制 pass。[S2] | 先研究降低画面分辨率、显示帧率，关闭阴影与反射。它们影响显示成本，不能降低必须执行的策略或物理工作；具体收益待测。 |
| 浏览器只收姿态是否可行 | WebGL 是浏览器内的 3D 图形 API；Three.js 的 `Object3D` 支持位置、四元数和变换矩阵。[S5][S6] | 服务器保留策略与权威物理，浏览器加载一次静态几何，再更新刚体/geom 姿态即可绘图。该方案不要求把 MuJoCo 物理搬到浏览器。 |

## 浏览器姿态方案的取舍（工程推断，未实现）

优点：服务器可省去逐帧 OpenGL 绘制、像素读回和 JPEG 编码；小型刚体场景的姿态流有望小于图像流，带宽收益必须按实际编码测量。相机交互可由浏览器本地处理。发送状态频率与浏览器绘制频率可不同，客户端插值只用于显示。

成本与边界：需要维护几何、材质、坐标系、四元数顺序、层级和模型版本映射。只发送 `qpos` 还需要客户端正运动学；发送服务端已算好的刚体/geom 世界姿态可减少这部分实现风险。接触力、调试标记及其他动态装饰必须另传；可变形几何不能只用刚体姿态表达。网络延迟、丢帧和插值会产生显示滞后，界面应携带仿真时间和状态序号，并丢弃过期快照。

浏览器仍需要图形资源。如果浏览器运行在训练同一台机器、同一块显卡上，WebGL 仍可能与训练争用该 GPU；在另一台设备上查看，才更有机会把绘图负担移出训练机器。姿态方案减轻服务器图像生成，并不自动提高服务端物理步进速度。

## 针对当前代码的研究顺序

只读观察：`experiments/torque200/web_preview.py` 使用 JIT 的 `env.step`，每次推进约 0.05 秒仿真后，将状态复制给 `MjData`，再调用 `mj_forward`、更新场景、渲染及 JPEG 编码，最后等待约 0.05 秒周期。这是约 20 fps 的调度目标，不是实际帧率测量。

1. 先研究显示降采样与一致快照消费，保留当前策略与物理语义；关闭非必要视觉效果。计时应分开策略、物理、状态搬运/前向更新、绘制、读回、编码和发送，避免只看总 fps。
2. CPU 单场景 native 路线作为独立候选。`env.step` 可能含观测/历史、动作缩放、滤波、重置和终止处理；这些需逐项保留。官方还记录了 MJX-JAX 与 native 的碰撞算法差异，[S1] 因此切换后必须做行为一致性验证，不能把 `mj_step` 当作环境调用的直接等价替换。
3. 如果主要成本是服务端图像生成，再研究浏览器姿态渲染；若主要成本是单场景 MJX 物理，WebGL 本身不能解决它。OSMesa 是资源取舍备选，不宜仅凭“CPU 渲染”选择。

本轮没有证据给出 native 收益倍数、EGL/OSMesa 实际设备、WebGL 带宽收益或对在训任务的影响；没有启动任何验证负载。

## 官方来源

- [S1 — MuJoCo 3.12.0 MJX-JAX: Sharp Bits](https://mujoco.readthedocs.io/en/3.12.0/mjx.html#mjx-jax-the-sharp-bits)：单场景适配性、批量适配性及碰撞算法差异。
- [S2 — MuJoCo 3.12.0 Visualization](https://mujoco.readthedocs.io/en/3.12.0/programming/visualization.html)：两阶段可视化、步进/绘图示例、Linux GLX/OSMesa/EGL、阴影与反射 pass。
- [S3 — Khronos EGL](https://www.khronos.org/egl/)：EGL 的上下文、表面与同步职责。
- [S4 — Mesa EGL: Bootstrapping](https://docs.mesa3d.org/egl.html#bootstrapping)：Mesa EGL 初始化的软件回退；这是 Mesa 实现事实，不能据此断言当前 NVIDIA 路径正在软件回退。
- [S5 — Khronos WebGL](https://www.khronos.org/webgl/)：浏览器内 3D API。
- [S6 — Three.js Object3D](https://threejs.org/docs/pages/Object3D.html)：位置、四元数与矩阵变换 API。
