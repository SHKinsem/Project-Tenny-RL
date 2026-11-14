# 阶段 1:Plackett-Burman 主效应筛选(gate-1 审查材料)

- run 数 N = 32,因子数 k = 29,y = 尾部 3 eval 归一化 RMSE(死亡/发散 run 记 WORST=10.0)
- σ_noise = 0.0472,SE(E_i) = 2σ/√N = 0.0167,分档阈值 2·SE = 0.0334(同 seed=0 CRN,判据偏保守)

## run 一览

| run | y (RMSE) | 状态 |
|---|---|---|
| 0 | 1.5116 | 完成 |
| 1 | 0.7311 | 完成 |
| 2 | 0.7753 | 完成 |
| 3 | 1.1144 | 完成 |
| 4 | 1.2572 | 完成 |
| 5 | 0.9465 | 完成 |
| 6 | 1.2409 | 完成 |
| 7 | 0.8427 | 完成 |
| 8 | 4.3053 | 完成 |
| 9 | 0.9522 | 完成 |
| 10 | 10.6799 | 完成 |
| 11 | 1.0492 | 完成 |
| 12 | 0.7658 | 完成 |
| 13 | 1.2281 | 完成 |
| 14 | 1.2732 | 完成 |
| 15 | 0.8121 | 完成 |
| 16 | 1.0474 | 完成 |
| 17 | 0.9029 | 完成 |
| 18 | 1.0840 | 完成 |
| 19 | 0.5847 | 完成 |
| 20 | 3.0366 | 完成 |
| 21 | 0.5324 | 完成 |
| 22 | 0.9042 | 完成 |
| 23 | 0.9248 | 完成 |
| 24 | 1.0629 | 完成 |
| 25 | 0.8412 | 完成 |
| 26 | 0.8693 | 完成 |
| 27 | 1.0431 | 完成 |
| 28 | 9.9165 | 完成 |
| 29 | 1.1323 | 完成 |
| 30 | 0.8558 | 完成 |
| 31 | 2.4027 | 完成 |

## 主效应排序(|E_i| 降序)

| 排名 | 因子 | 层 | E_i | \|E_i\| vs 2·SE | 建议 |
|---|---|---|---|---|---|
| 1 | vel_pid.target_scale | B | +1.5639 | 显著 | 进阶段二 |
| 2 | learning_rate | A | -1.5341 | 显著 | 进阶段二 |
| 3 | action_scale.wheel | B | +1.3609 | 显著 | 进阶段二 |
| 4 | policy_net | A | +1.3596 | 显著 | 进阶段二 |
| 5 | vel_pid.notsmoothing | B | -1.3592 | 显著 | 进阶段二 |
| 6 | vel_pid.torque_limit | B | +1.1916 | 显著 | 进阶段二 |
| 7 | action_range_thigh_lo | B | -1.1626 | 显著 | 进阶段二 |
| 8 | num_updates_per_batch | A | +1.0601 | 显著 | 进阶段二 |
| 9 | action_scale.thigh | B | +1.0493 | 显著 | 进阶段二 |
| 10 | value_net | A | -0.9994 | 显著 | 进阶段二 |
| 11 | vel_pid.kp | B | -0.9933 | 显著 | 冻结 |
| 12 | num_minibatches | A | -0.9628 | 显著 | 冻结 |
| 13 | reward_config.pitch_velocity_penalty_scale | B | +0.8511 | 显著 | 冻结 |
| 14 | reward_config.action_rate_scale | B | -0.8096 | 显著 | 冻结 |
| 15 | cmd_roll_value_range | B | -0.7998 | 显著 | 冻结 |
| 16 | cmd_lin_vel_x_range | B | +0.5418 | 显著 | 冻结 |
| 17 | reward_config.energy_scale | B | +0.4940 | 显著 | 冻结 |
| 18 | cmd_height_value_range | B | +0.4797 | 显著 | 冻结 |
| 19 | half_life_s | A | +0.4205 | 显著 | 冻结 |
| 20 | reward_config.zero_keep_scale | B | -0.2719 | 显著 | 冻结 |
| 21 | reward_config.sigma_wheel | B | +0.2325 | 显著 | 冻结 |
| 22 | entropy_cost | A | -0.2321 | 显著 | 冻结 |
| 23 | unroll_length | A | +0.2305 | 显著 | 冻结 |
| 24 | cmd_ang_vel_yaw_range | B | -0.2212 | 显著 | 冻结 |
| 25 | cmd_config.cmd_prob | B | -0.1944 | 显著 | 冻结 |
| 26 | reward_config.sigma | B | -0.1465 | 显著 | 冻结 |
| 27 | reward_config.height_scale | B | +0.0797 | 显著 | 冻结 |
| 28 | reward_scaling | A | -0.0302 | 低于噪声底 | 冻结 |
| 29 | reward_config.roll_scale | B | +0.0202 | 低于噪声底 | 冻结 |

## 建议冻结/搜索清单

- 进阶段二(top 10): action_range_thigh_lo, action_scale.thigh, action_scale.wheel, learning_rate, num_updates_per_batch, policy_net, value_net, vel_pid.notsmoothing, vel_pid.target_scale, vel_pid.torque_limit
- 建议冻结: cmd_ang_vel_yaw_range, cmd_config.cmd_prob, cmd_height_value_range, cmd_lin_vel_x_range, cmd_roll_value_range, entropy_cost, half_life_s, num_minibatches, reward_config.action_rate_scale, reward_config.energy_scale, reward_config.height_scale, reward_config.pitch_velocity_penalty_scale, reward_config.roll_scale, reward_config.sigma, reward_config.sigma_wheel, reward_config.zero_keep_scale, reward_scaling, unroll_length, vel_pid.kp
- 低于噪声底(gate-1 人工判断是否放回): reward_config.roll_scale, reward_scaling

→ 人工核准:编辑 `stage1_space.yaml`(放回误冻参数、收窄离谱区间)后放行阶段二。
