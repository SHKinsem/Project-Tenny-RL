# 固件与训练仓库算法一致性对比

> 历史记录：本文对比的是 RLPID 阶段的旧固件和 `20260111-2109` 策略，不代表当前 `main` 使用的 `ckpts/newobs-norecovery/000403660800`。下文“当前 head”均指当时训练仓库的状态。

涉及项目（本工作横跨两个仓库）：

- **固件项目**：本仓库
- **训练项目**：独立的 `legwheel_rl` 仓库
  （训练机上使用对应的本地检出目录；
  该仓库 `docs/deploy_consistent_branch.md` 记录了为对齐部署快照新建的
  `improvements/deploy-consistent` 分支）

对比对象：

- **固件**：本仓库 `RLPID` 分支（`control.cpp` / `rl_policy.cpp`，部署策略 `ckpts/20260111-2109`，26 维观测）
- **训练仓库**：上述 legwheel_rl 的 `improvements/bugfix-cleanup` 分支
  - 当前 head(`legwheel/env.py` + `legwheel/config.py`)
  - 训练出部署策略时的旧代码：提交 `82ba252`(`legwheel_env.py`)

## 结论

- 固件与 `bugfix-cleanup` **当前 head 不一致，且不兼容**（观测维度、动作语义、轮子控制律都已改变）。
- 固件与训练出部署策略的旧代码（`82ba252`）**框架一致**，但存在若干实质差异，其中最严重的是大腿速度观测未置零。

## 与 82ba252（部署策略对应代码）对比

### 一致的部分

- **观测结构**:26 维、`history_length=1`，排列顺序完全相同：

  | 索引 | 内容 | 维度 |
  |------|------|------|
  | 0~3 | command（原始指令，不归一化） | 4 |
  | 4~6 | gyro | 3 |
  | 7~9 | sinyaw / sinpitch / sinroll（减参考值） | 3 |
  | 10~13 | 关节位置 [大腿L, 轮L sin, 大腿R, 轮R sin] | 4 |
  | 14~17 | 关节速度 [大腿L, 轮L, 大腿R, 轮R] | 4 |
  | 18~19 | 轮线速度（角速度 × 轮半径） | 2 |
  | 20~23 | previous_wheels_policyout（残差 a×30，旧的在先） | 4 |
  | 24~25 | 轮速误差（平滑目标 − 当前轮速） | 2 |

- **网络**:26→64→64→8,SiLU 隐藏层 + 输出 tanh；归一化用检查点烘焙的 `state_mean/state_std`(`rl_policy.cpp:11` 头部确认检查点为 `20260111-2109`)
- **大腿动作**:delta 模式，`target = 当前位置 + a×0.25`，两侧一致
- **轮子动作链**：残差速度目标 `a×30` → 加当前轮速、clip ±30 → 0.5 低通平滑（固件 `vel_pid_smoothing=0.5` 与训练 `smoothing=0.5` 公式相同）→ P 控制以**增量累加**形式输出，限幅 ±0.0295 N·m。固件 `ControlPID` 在 Ki=0、Kd=0 时 `output=Kp×err`，`applied = clip(prev + delta)` 与训练的 `new_torque = clip(last + kp×err, ±0.0295)` 结构等价
- 轮速误差两侧均为"平滑目标 − 当前轮速"

### 不一致的部分（按严重程度排序）

1. **大腿速度观测（最严重）**：训练侧把 input[14]、input[16]（左右大腿速度）**硬置零**——部署检查点的 `state_std[14]=state_std[16]=0.000001` 是直接证据。而固件 `control.cpp:347,352` 喂的是真实大腿速度。归一化除以 1e-6 后输入达 10⁵ 量级，远超训练分布。
2. **轮子角度编码尺度**：训练 `wheel_sin_scale=0.1`，固件硬编码 `0.25`(`control.h:339`，Kweb 不可覆盖）
3. **偏航编码尺度**：训练 `sin(yaw×0.5)/0.5`，固件内层 `yaw_sin_scale=1.0`（硬编码，`control.h:340`);Kweb 的 `sinyaw_scale` 只能改外层乘子
4. **sinroll 公式**：训练用 `R[2,1]/sqrt(1-sinpitch²)`（含俯仰耦合修正），固件直接 `sin(roll)`
5. **俯仰参考**：训练用质心几何计算的 `sin_balance_zero`，固件用参数化曲线 `pitch_zero_low/high/power` 近似
6. **小参数差异**:
   - 大腿 clip 范围：固件 `[-0.4, -0.06]` vs 训练 `[-0.5, 0]`
   - 轮 Kp：固件 0.00025（实机调）vs 训练 0.0002
   - `wheel_radius`：固件 0.0194 vs 训练 0.0195（可忽略）
   - 固件多了大腿目标平滑 `thigh_pid_smoothing=0.5` 和可能的 Kd 项（训练无）

> 注：第 2、3 条涉及训练时的实际配置（可能被命令行覆盖），需与训练机上的检查点配置最终确认。

## 与 bugfix-cleanup 当前 head 的差异

当前 head 已大幅演进，与固件**不兼容**:

| 项目 | 固件 / 82ba252（已部署） | bugfix-cleanup head |
|------|--------------------------|---------------------|
| 观测维度 | 26 | 30(`last_act_obs=True`，追加 `last_act[4]`) |
| 大腿动作 | delta 目标 `pos + a×0.25` | 绝对目标 `(a-1)×0.25` |
| 轮子目标 | 残差 `a×30` + 当前轮速，0.5 平滑 | 绝对 `a×40`，无平滑，`action_delay_steps=1` |
| 轮子控制律 | 增量累加 P(kp=0.0002，±0.0295 N·m) | direct-torque PI(kp=0.005，±0.04 N·m) |

**含义**：用当前 head 新训的策略不能直接部署到当前固件，需同步修改 `control.cpp` 的观测组装与动作链路（或用 82ba252 时代的配置训练）。

## 附注

- `branch_comparison.md` 中"策略 20260111-1432"来自提交信息，实际 `rl_policy.cpp` 内为更新的 `20260111-2109`（随 `aa4f5d1` 引入），该文件此条已过时。
- 建议优先核实"大腿速度置零"一条：若实机运行正常，可能是网络对该维度不敏感，但从训练分布看这属于 bug 级差异。
