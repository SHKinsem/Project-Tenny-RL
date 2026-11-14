# 分支对比：RLPID / RLPID_newOBS / deepRL

> 历史记录：本文反映 2026 年 1 月的旧分支和旧策略，不代表当前 `main`。仓库公开时已重建历史并只保留 `main`，下文提交哈希和分支名仅供理解演进过程。

## 分支关系

```
origin/deepRL (6efe115, 2025-12-16)   ← 共同祖先
        │
        ▼  (几十个提交：轮子PID、数据分析、web调参等)
   eaf67f3 (2026-01-05)               ← 分叉点
        ├── RLPID_newOBS: +2 提交 (6a31530, 8a9dafa)
        └── RLPID:        +4 提交 (ecb742b → aa4f5d1)
```

> 注意：本地 `deepRL`（370ccb6）落后于 `origin/deepRL` 一个提交。

各分支最新提交：

| 分支 | 最新提交 | 日期 | 说明 |
|------|----------|------|------|
| `origin/RLPID` | aa4f5d1 | 2026-01-24 | set wheel_Kp=0.00025 |
| `origin/RLPID_newOBS` | 8a9dafa | 2026-01-07 | last_act |
| `origin/deepRL` | 6efe115 | 2025-12-16 | add extract2.py for ubuntu |

## 核心差异（强化学习观测与轮子控制）

### `deepRL` — 纯策略控制轮子的旧方案

- 观测 20 维，`history_length=5`
- 没有 wheelPID，轮子完全由神经网络输出控制

### `RLPID_newOBS`（分叉后 2 个提交）— "新观测"实验

- 观测 **30 维**：在输入里加入了 `sin_balance_zero`、`sinyaw_start`、`wheels_sinposition_start` 等起始姿态特征，以及 `wheel_ctrl_vel_err` 和 `last_act`（上一次动作，见提交 `8a9dafa`）
- 策略更新到 `20260107-1908`

### `RLPID`（分叉后 4 个提交）— 另一种观测设计

- 观测 **26 维**：不用起始姿态特征，改用缩放差值——如 `(yaw_sin - yaw_sin_zero)*sinyaw_scale`、`(wheel_sin - wheel_sin_zero)*sinwheel_scale`；轮速做了低通平滑，并加入 `previous_wheels_policyout`（上 4 步轮子策略输出）和轮速误差
- 不含 `last_act`（代码里被注释掉）
- 策略更新到 `20260111-1432`，并调参：`wheel_Ki=0`、`wheel_Kp=0.00025`

两个 RLPID 分支都引入了轮子电压 PID（`wheelPIDLeft/Right`，默认 Kp=0.00025，可在 web 端调），这是相对 `deepRL` 最大的架构变化。

## 文件层面差异

- `RLPID` vs `RLPID_newOBS`：只差 3 个文件
  - `control.cpp`：观测组装逻辑不同（26 维 vs 30 维）
  - `rl_policy.cpp`：网络权重不同
  - `lqrwebserver.cpp`：4 行小改
- 两者 vs `deepRL`：差异很大（5000+ 行），主要包括：
  - 新增 `wheel_control_test/left_wheel_voltage_pid_delta/`（轮子 PID 独立测试工程，含 web server、电压-力矩拟合脚本）
  - 新增 `data_analysis/` 绘图工具
  - `params.cpp.txt` 大改
  - `control.cpp` / `control.h` / `lqrwebserver.cpp` 大量改动

## 总结

`RLPID` 和 `RLPID_newOBS` 是同一轮子 PID 架构下的两种观测方案实验；`RLPID` 更新（策略日期 01-11 vs 01-07），也是最新提交所在的分支。
