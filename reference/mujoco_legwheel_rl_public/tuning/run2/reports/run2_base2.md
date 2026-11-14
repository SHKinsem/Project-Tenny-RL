# R0:base2 建立与验证(gate-R0 审查材料)

base2 = cand010 的 4 个增益(lr/updates/kp/notsmoothing)移植回 baseline 网络 + 固件对齐映射(计划 6.R0)。

- RMSE 均值: **0.6238**(逐 seed: s0=0.5999, s1=0.6478)

| seed | RMSE | 存活(步) | action_rate 部署口径/步 | 报表口径/步 |
|---|---|---|---|---|
| 0 | 0.5999 | 566.0 | 0.3660 | 1.2852 |
| 1 | 0.6478 | 541.4 | 0.3814 | 1.2882 |

## 达成率(尾部 3 eval 均值,seeds 平均)

| 分项 | 达成率 |
|---|---|---|
| tracking_lin_vel | 76.3% |
| tracking_ang_vel | 63.0% |
| tracking_height_jnt | 53.9% |
| tracking_height_action | 55.8% |
| pitch_cost | 73.5% |
| roll_cost | 75.6% |
| tracking_sinyaw | 82.8% |
| tracking_wheels_sinposition | 54.9% |

→ gate-R0:与 run1 baseline 1e8 对照(RMSE 0.7084±0.0472),预期 0.60~0.70;若反而更差,增益移植不成立,回退纯 baseline 起点。
