# 训练侧 vs 固件部署代码 逐项核对（2026-08-25）

对比双方：

- **训练侧**：本仓库 `improvements/deploy-consistent` 分支（对齐部署快照 82ba252 后，含大腿速度观测改喂噪声的真机适配）
- **固件侧**：`wl_pro_robot` 仓库（ESP32/Arduino，`RLPID` 分支，HEAD d98145d），关键文件 `control.cpp` / `control.h` / `rl_policy.cpp` / `lqrwebserver.cpp`

总体结论：**观测布局与动作链路的结构、顺序、时序已对齐**；剩余差异中 "实机 getK 可调增益"属调参范畴（不算本质区别），真正需要处理的是 观测编码/输入分布类的结构性差异。

---

## 一、观测链路逐项核对（训练 `_get_obs` 26 维 vs 固件 `align_sensors` input[0..25]）

| idx | 内容 | 结论 | 说明 |
|---|---|---|---|
| 0–3 | command [lin_vel_x, ang_vel_yaw, height, roll] | ✅ 一致 | 两侧均原始指令不归一化（env.py:508；control.cpp:282–285）。固件 cmd2 clip 到 [-0.4,-0.06]（control.cpp:430），训练采样范围 [-0.5,0] |
| 4–6 | gyro [x,y,z] | ✅ 一致 | 训练 rad/s；固件 deg→rad（control.cpp:247–249,293–295）。轴符号依赖 IMU 安装，需实机日志确认 |
| 7 | sinyaw | ❌ 不一致 | 训练 `sin(yaw×0.5)/0.5 − 起始值`（sinyaw_scale=0.5）；固件 `sin(yaw×1.0)`、yaw_sin_scale=1.0 硬编码（control.h:340） |
| 8 | sinpitch | ⚠️ 基公式一致，参考不同 | 基公式恒等（训练 −R[2,0] ≡ sin(pitch)）；参考：训练质心几何 `sin_balance_zero` vs 固件参数化曲线 low=0.22/high=0.05/power=1.0（lqrwebserver.cpp:72–73） |
| 9 | sinroll | ❌ 缩放不一致 | 公式经核实数学恒等（ZYX 下 R[2,1]/cos(pitch) ≡ sin(roll)）；但固件外层乘 `sinroll_scale`，Kweb 运行时默认 **0.2**（lqrwebserver.cpp:74），比训练小 5 倍 |
| 10, 12 | 大腿位置 [L,R] | ✅ 一致 | 固件舵机值线性映射到 sim rad，squat=0/stand=−0.5 对齐 |
| 11, 13 | 轮子 sin 位置 | ❌ 不一致 | 训练 `sin(p×0.1)/0.1 − start`；固件 0.25 硬编码（control.cpp:304–305；control.h:339） |
| 14, 16 | 大腿速度 | ⚠️ 训练侧已适配 | 训练改喂零均值纯噪声（本分支 93ddd81，std≈0.58），策略学会忽略该维；固件喂真实速度。**注意已部署旧 ckpt（20260111-2109）std=1e-6 仍存爆炸问题** |
| 15, 17 | 轮角速度 | ✅ 一致 | 固件 wheel_vel_smoothing_alpha=0.0，平滑退化为原始值 |
| 18, 19 | 轮线速度 | ✅ 一致（微小） | 训练 ×0.0195 vs 固件 ×0.0194，差 0.5% |
| 20–23 | previous_wheels_policyout | ✅ 一致 | 两侧均记残差 a×30，flatten [旧L, 旧R, 新L, 新R]，时序一致 |
| 24–25 | 轮速误差 | ✅ 一致 | 两侧均为"平滑后目标 − 当前原始轮速" |

其他核对点：

- 观测维度/历史：两侧均 26 维、history=1、不含 last_act（固件 last_act 喂入已注释 control.cpp:364–368）。✅
- zero margin：两侧数值完全相同（0.000195 / 0.01）；触发逻辑细节略异、语义等价。✅
- 归一化参数：`deploy/params.cpp.txt` 的 state_mean/std 与固件 `rl_policy.cpp:15,20` **逐数值相同**（同源 ckpt 20260111-2109）；新训练策略经 `scripts/export_params.py` 导出后可直接替换。唯一 epsilon 差异：训练 `(x−mean)/(std+1e-6)` vs 固件裸除。✅
- 噪声：仅训练侧有观测噪声，部署无噪声，正常。

## 二、控制/动作链路逐项核对

| 项目 | 训练侧 | 固件侧 | 结论 |
|---|---|---|---|
| 大腿增量公式 | `qpos_now + a×0.25`（env.py:271-273） | `sensors_std + a×0.25`（control.cpp:482-486） | ✅ |
| 大腿 clip | [-0.5, 0] | [-0.4, -0.06]（control.h:330） | ❌ 硬件限位，动作空间收缩 |
| 轮残差目标 | `clip(step起始轮速 + a×30, ±30)` | 同（control.cpp:497-506） | ✅ |
| 目标轮速低通 | 0.5（notsmoothing） | 0.5（vel_pid_smoothing，control.h:310） | ✅ |
| wheels_commands_history | 记残差 (2,2) | 同 | ✅ |
| 扭矩生成 | delta-torque 累加 `clip(last + kp×err, ±0.0295)` | ControlPID（Ki=Kd=0 时同构）±0.0295 | ✅ 结构一致 |
| 轮 Kp | 0.0002 | getK 运行时覆盖（见下） | ⚠️ 调参范畴但需固化 |
| action_delay | 0（关闭） | 无延迟 | ✅ |
| 轮速误差观测 | `last_target − 当前轮速` | 同（control.cpp:362-363） | ✅ |

### 轮 PID 增益的运行时值（重点）

固件 `compute()` 每周期执行 `wheelPID.setGains(getK("wheel_Kp", 0.02f), getK("wheel_Ki", 0.0f), getK("wheel_Kd", 0.0001f))`（control.cpp:404-408）。若实机 K 表从未写入 `wheel_Kp`，运行时 Kp=**0.02**（训练的 100 倍），且 Kd 默认 0.0001 会引入训练不存在的微分项。属实机调参范畴，但**建议在实机 K 表固化 wheel_Kp=0.0002、wheel_Ki=0、wheel_Kd=0 作为训练对齐的起点**，之后实机再按需调。

### 其他结构差异

- 大腿通道：固件增量目标后多两级处理——目标 0.5 低通（thigh_pid_smoothing）+ 大腿 P 环（getK 默认 0.3）；训练直接把位置目标交给 MuJoCo 位置执行器。实机大腿响应比训练假设更慢更柔（执行器建模差距，靠域随机化/调参吸收）。
- 轮内环频率：训练固定 4×2.5ms 每 env step；固件 free-running（loop 无固定周期），delta-torque 累加速度取决于实际 loop 周期。
- 轮输出映射：固件 `control_now = −applied_act × wheel_ctrl_scale_`（符号为硬件方向约定）；分段线性 torque→voltage 表已注释，当前直通；`control_smooth_alpha` 默认 0.0 不平滑。
- 大腿/轮状态清零：固件关机/摔倒时清零，训练 reset 时清零，语义一致。

## 三、差异分类（按处置方式）

**实机调参范畴（非本质区别，保留实机可调）**

- 轮 PID 增益（getK 实时调）、大腿目标低通/伺服环增益、大腿 clip（硬件限位）、wheel_radius 0.0194/0.0195（0.5%）

**需对齐的观测编码/输入分布差异（调参调不掉）**

1. ~~input[14]/[16] 大腿速度~~ —— 已在训练侧适配（93ddd81），新训练 ckpt 的 std 正常；已部署旧 ckpt 仍需固件置零该两维
2. `wheel_sin_scale` 0.25 vs 0.1、`yaw_sin_scale` 1.0 vs 0.5（固件硬编码常量，改 control.h 即可）
3. `sinroll_scale=0.2`（Kweb 运行时默认，建议固化为 1.0）
4. sinpitch 参考：固件参数化曲线 vs 训练质心几何（需二选一对齐）

**代码层面无法验证、需实机日志确认**

- IMU 陀螺/角度与 MJX 机体坐标系的符号对应关系
- 固件 `compute()` 实际调用频率（run_interval 对应的策略频率）
