# 修改说明：`improvements/deploy-consistent` 分支

日期：2026-08-25 提交：`51b394f`

## 涉及项目

本工作横跨两个仓库：

- **训练项目（本仓库）**：`legwheel_rl` 仓库（检查点 `ckpts/20260111-2109/000801792000` 实体文件仅在训练机上）
- **固件项目（部署侧）**：`wl_pro_robot` 仓库（ESP32/Arduino 固件，`RLPID` 分支；跨项目对比详见该仓库根目录的 `deploy_consistency.md` 与 `branch_comparison.md`）

## 目的

当前部署在机器人固件（wl_pro_robot 仓库 `RLPID` 分支，`rl_policy.cpp`）上的策略是 **`ckpts/20260111-2109/000801792000`**，其训练代码对应本仓库提交 `82ba252` （"update policy 20260111-2109"，旧单文件 `legwheel_env.py` 时代）。

`improvements/bugfix-cleanup` 在重构为 `legwheel/` 包的过程中，环境语义已经偏离该快照 （30 维观测、绝对目标动作、direct-torque PI、动作延迟等），直接用其训练的新策略 **无法部署到现有固件**。

本分支在保留 bugfix-cleanup 重构成果（包结构、测试、脚本、训练改进）的前提下，将**训练逻辑与参数对齐到与部署一致的 82ba252 训练快照**。

## 改动内容（3 个文件）

### `legwheel/config.py` — 参数对齐

| 参数 | bugfix-cleanup | 本分支（=82ba252） |
|------|----------------|---------------------|
| `last_act_obs` | True（30 维观测） | **False**（26 维，与固件一致） |
| `delta_thigh_action` | False（绝对目标） | **True**（增量 `pos + a×0.25`，clip [-0.5, 0]） |
| `vel_pid.kp` | 0.005 | **0.0002** |
| `vel_pid.torque_limit` | 0.04 | **0.0295** N·m |
| `vel_pid.target_scale` | 40.0 | **30.0** |
| `vel_pid.notsmoothing` | 1.0（不平滑） | **0.5** |
| `vel_pid.direct_torque` | True（PI 直接生成扭矩） | **False**（delta-torque 累加） |
| `action_delay_steps` | 1 | **0**（快照时代无动作延迟） |
| `action_scale.wheel` | 0.04 | **0.0295**（仅 vel_pid 禁用时生效） |

新增开关 `vel_pid.residual_target=True`（见下）。

### `legwheel/env.py` — 残差目标模式

`step()` 中新增 `vel_pid.residual_target` 模式（配置开关控制，head 的绝对目标模式保留）：

- 策略输出 → 速度残差 `a × target_scale`（即 `a×30`）
- 目标轮速 = **step 起始时的当前轮速** + 残差，clip ±30，再经 0.5 低通平滑
- `wheels_commands_history` 记录**残差**（head 记录的是限幅后的绝对目标）——该历史对应观测 `previous_wheels_policyout`，与固件 `control.cpp` 中 `wheels_commands_history` 的语义一致

delta-torque 累加（`direct_torque=False`）路径为重构中原有代码，无需改动：`new_torque = clip(last_torque + kp×err, ±torque_limit)`，与固件 `ControlPID`（Ki=0、Kd=0 时）的增量 PID 结构等价。

### `tests/test_env.py` — 维度断言更新

- `state`：30 → **26** 维；`privileged_state`：63 → **55** 维

## 验证

- `python tests/test_env.py`:3 项测试全部通过
- 数值验证（reset 后 step 固定动作 `[0.2, -0.1, 0.5, -0.5]`）：
  - `wheels_commands_history[-1] == a×30 = [15, -15]` ✓
  - `last_target_wheel_vel == 0.5×(当前轮速 + 残差) = [7.5, -7.5]` ✓
  - 观测 26 维且有限 ✓

## 与固件侧已知差异（不属于本分支改动范围）

以下差异存在于**固件** `control.cpp`（wl_pro_robot 仓库），训练侧无此问题，详见该仓库 `deploy_consistency.md`：

1. 固件在观测 input[14]/input[16] 喂入真实大腿速度，而训练侧硬置零 （部署检查点 `state_std` 相应位为 1e-6），归一化后远超训练分布 ——**已在训练侧适配（2026-08-25）**：因真机无法稳定测量大腿速度，训练侧该两维由恒零改为喂零均值纯噪声（`env.py:_get_obs`，`noise_config.scales.joint_vel` 尺度），策略学会忽略该维，且新训练导出的 `state_std` 退化为噪声尺度（≈0.58）而非 1e-6，固件喂入的实测速度经归一化后保持有界。注意：仅对**本分支新训练**的 策略生效，已部署的 20260111-2109 检查点仍是 std=1e-6
2. 固件 `wheel_sin_scale=0.25`（训练 0.1）、`yaw_sin_scale=1.0`（训练 0.5），均为硬编码
3. 固件 sinroll 用 `sin(roll)`，训练用 `R[2,1]/sqrt(1-sinpitch²)` ——经核实两式在 ZYX 欧拉角下数学恒等，公式本身无差异；但固件外层 另有 `sinroll_scale`（Kweb 运行时默认 0.2），使该维比训练小 5 倍，建议实机 Kweb 固化为 1.0
4. 固件俯仰参考为参数化曲线，训练为质心几何 `sin_balance_zero`
5. 小参数：固件大腿 clip [-0.4, -0.06] vs 训练 [-0.5, 0]；轮 Kp 实机经 getK 实时调（属调参范畴，非本质差异，但建议实机 K 表固化 wheel_Ki=0、wheel_Kd=0 以避免引入训练不存在的微分项）

## 有意保留、未对齐到快照的项

- **奖励函数**：head 启用了 `action_rate_scale=-0.5`、`pitch_velocity_penalty=-0.5`，并移除了快照的 `pid_vel` 轮速跟踪奖励（82ba252 中 `pid_vel_scale=5.0`、`sigma_pid_vel=5.0`，两项惩罚为 0）。奖励不影响部署接口一致性，故保留 head 版本；如需完全复现快照训练可再对齐
- **训练超参**：如折扣率改为按时间半衰期定义，当前 `half_life_s=0.5` → γ≈0.9862（bugfix-cleanup 的折扣率改进，评估与取值折中见 `docs/compare_target_direct_33238d9.md` 的 discounting 小节），与部署接口无关，保留
- **重构基础设施**：动作延迟缓冲（`action_delay_steps`，本分支设 0 关闭）、termination penalty（默认关闭）等保留，便于后续实验

## 与 82ba252 的复核（2026-08-25，commit 93ddd81 时点）

对当前工作区与 82ba252 做了系统复核：config 数值、step 主流程、26 维观测 构成与顺序、终止条件、轮子 PID/动作链路、奖励公式（除上述有意项）、域随机化的范围数值均逐行确认一致。复核新发现以下**此前未记录的实质偏差**：

1. **指令分布不同（有意偏离快照）**：82ba252 的 `sample_command` 死区判定 无 abs，所有负向 lin_vel_x / ang_vel_yaw 指令被清零——已部署策略实际在 "只前进/只正偏航 + 约一半零指令"的分布上训练。本分支已修复为对称双向 （`env.py:633,636`），新训策略的后退/反转能力是经修复后新训出来的，与已部署策略训练分布不同，部署前需实机验证。
2. **域随机化差异比此前记载更大（有意偏离快照）**：82ba252 的 `domain_randomize.py` 开关全关，连地面摩擦都不随机，实际只随机初始位姿；本分支全开且新增执行器随机化（thigh kp/kv ×U(0.9,1.1)、轮 gear ×U(0.8,1.2)）、质量/质心范围非零（[-0.003,0.003]/[-0.01,0.01]）、初始 pitch ±π/6 → ±π/8。保留全开是为了 sim2real 鲁棒性，但与快照训练分布差异显著。
3. **首次指令重采样时机**：快照硬编码 step==100（1.0s），本分支 `cmd_start_delay_s=0.2` → step 20，每个 3s 指令窗口的有效跟踪时长不同。
4. **XML 轮电机限幅**：快照 XML `ctrlrange/forcerange=±0.0295`，当前 `assets/legwheel_robot2_stl_nolinkage6_terrain.xml:149-153` 为 ±0.04。无实际影响（config `torque_limit=0.0295` 在进仿真前已截断），仅文件不一致。

结论：本分支与 82ba252 的"部署接口"（观测布局、动作链路、轮子控制律）已一致；但训练分布（指令对称性、域随机化、γ、奖励）有意偏离快照，**"对齐 82ba252"指的是部署接口而非训练分布完全复刻**。

## 与固件的逐项核对明细

观测 26 维逐位、控制/动作链路逐项的核对表见 `docs/deploy_firmware_comparison.md`（2026-08-25）。要点：

- 观测布局、残差历史语义、归一化管线（state_mean/std 逐数值相同）已对齐
- 属"实机 getK 调参范畴"的差异不算本质区别（轮 PID 增益、大腿伺服环、大腿 clip 等），但建议实机 K 表固化 wheel_Kp=0.0002、wheel_Ki=0、wheel_Kd=0 作为与训练对齐的起点
- 仍需处理的结构性差异：`wheel_sin_scale`/`yaw_sin_scale` 硬编码常量、`sinroll_scale` Kweb 默认 0.2（建议固化 1.0）、sinpitch 参考曲线定义
- 代码层面无法验证、需实机日志确认：IMU 坐标系符号对应、策略实际调用频率

## 使用注意

- 用本分支训练出的 26 维观测策略，可通过 `scripts/export_params.py` 导出后直接 部署到现有固件（观测布局、动作链路、轮子控制律均一致）
- 若需训练 30 维（含 `last_act`）或绝对目标等新语义策略，请使用 `improvements/bugfix-cleanup`，但需同步修改固件 `control.cpp` 才能部署
