# R1:reward shaping 搜索结果(gate-R1 审查材料)

- 完成 trial 数: 19,满足全部硬约束(硬门槛口径): **0**
- base2 RMSE: 0.6238;搜索目标 = 真实 RMSE + 软约束罚(gate-R1 决议),死亡/发散记 WORST=10.0

## Top 5(按搜索目标 objective)

| 排名 | trial | objective | 真实 RMSE | 罚 | 存活 | 能耗/步 | ar 部署/步 | sinyaw 达成 | 违反约束 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | search2-000 | 1.2129 | 0.6798 | 0.5330 | 555.8 | 0.0381 | 0.3392 | 80.9% | sinyaw_gain |
| 2 | search2-007 | 1.8998 | 0.7059 | 1.1939 | 520.6 | 0.0385 | 0.3644 | 81.8% | alive; action_rate_deploy; sinyaw_gain; no_regress[tracking_lin_vel] |
| 3 | search2-003 | 2.1234 | 0.6363 | 1.4871 | 513.7 | 0.0224 | 0.3493 | 80.5% | alive; sinyaw_gain |
| 4 | search2-026 | 2.5213 | 0.7052 | 1.8162 | 563.3 | 0.0366 | 0.2596 | 84.7% | no_regress[tracking_lin_vel]; no_regress[tracking_ang_vel] |
| 5 | search2-047 | 2.6164 | 0.6233 | 1.9932 | 551.9 | 0.0246 | 0.3869 | 84.3% | action_rate_deploy; no_regress[roll_cost] |

## 约束违反统计

| 约束 | 违反 trial 数 |
|---|---|---|
| sinyaw_gain | 14 |
| no_regress[roll_cost] | 14 |
| action_rate_deploy | 13 |
| alive | 6 |
| no_regress[tracking_lin_vel] | 3 |
| no_regress[tracking_ang_vel] | 3 |
| energy | 1 |
| no_regress[pitch_cost] | 1 |

## Top trial 参数

### #1 trial search2-000

```json
{
  "reward_config.vel_scale": 2.1230721545439306,
  "reward_config.height_scale": 3.114157810617098,
  "reward_config.roll_scale": 4.807780394357274,
  "reward_config.zero_keep_scale": 10.51972623055997,
  "reward_config.action_rate_scale": 0.32005396430807687,
  "reward_config.energy_scale": 1.0833308661063878,
  "reward_config.pitch_velocity_penalty_scale": 0.25119498706364835,
  "reward_config.sigma": 0.8606783008129293,
  "reward_config.sigma_wheel": 0.950873626413166,
  "vel_pid.kp": 0.00015111382568935223,
  "vel_pid.notsmoothing": 0.2866250702947006
}
```

| 分项 | 达成率 | vs base2 |
|---|---|---|
| tracking_lin_vel | 84.3% | +8.0pp |
| tracking_ang_vel | 84.9% | +21.8pp |
| tracking_height_jnt | 67.9% | +14.0pp |
| tracking_height_action | 80.7% | +24.9pp |
| pitch_cost | 86.1% | +12.6pp |
| roll_cost | 83.8% | +8.2pp |
| tracking_sinyaw | 80.9% | -1.8pp |
| tracking_wheels_sinposition | 55.8% | +0.9pp |

### #2 trial search2-007

```json
{
  "reward_config.vel_scale": 0.7913121091030084,
  "reward_config.height_scale": 1.1865647651984197,
  "reward_config.roll_scale": 1.5772806120609535,
  "reward_config.zero_keep_scale": 6.2388484205628005,
  "reward_config.action_rate_scale": 0.3071494635756644,
  "reward_config.energy_scale": 0.12669752683695518,
  "reward_config.pitch_velocity_penalty_scale": 0.6432086583369704,
  "reward_config.sigma": 0.5483629135888869,
  "reward_config.sigma_wheel": 0.36117726357628405,
  "vel_pid.kp": 0.0001647747009251751,
  "vel_pid.notsmoothing": 0.18608787436966148
}
```

| 分项 | 达成率 | vs base2 |
|---|---|---|
| tracking_lin_vel | 81.0% | +4.7pp |
| tracking_ang_vel | 87.4% | +24.4pp |
| tracking_height_jnt | 70.8% | +16.9pp |
| tracking_height_action | 82.4% | +26.7pp |
| pitch_cost | 86.8% | +13.2pp |
| roll_cost | 81.3% | +5.7pp |
| tracking_sinyaw | 81.8% | -1.0pp |
| tracking_wheels_sinposition | 50.6% | -4.2pp |

### #3 trial search2-003

```json
{
  "reward_config.vel_scale": 2.2211873899843853,
  "reward_config.height_scale": 0.6265288166963007,
  "reward_config.roll_scale": 4.975271018786807,
  "reward_config.zero_keep_scale": 12.280526240898489,
  "reward_config.action_rate_scale": 0.746536276742779,
  "reward_config.energy_scale": 3.25043165257102,
  "reward_config.pitch_velocity_penalty_scale": 0.6184249844308961,
  "reward_config.sigma": 0.41151467973779804,
  "reward_config.sigma_wheel": 0.45820461914851945,
  "vel_pid.kp": 0.00018355770640072548,
  "vel_pid.notsmoothing": 0.18224429620593693
}
```

| 分项 | 达成率 | vs base2 |
|---|---|---|
| tracking_lin_vel | 85.8% | +9.5pp |
| tracking_ang_vel | 87.8% | +24.7pp |
| tracking_height_jnt | 43.3% | -10.6pp |
| tracking_height_action | 45.9% | -9.8pp |
| pitch_cost | 87.0% | +13.5pp |
| roll_cost | 88.1% | +12.5pp |
| tracking_sinyaw | 80.5% | -2.3pp |
| tracking_wheels_sinposition | 54.9% | +0.1pp |

### #4 trial search2-026

```json
{
  "reward_config.vel_scale": 1.1873456214508187,
  "reward_config.height_scale": 1.7690102975648725,
  "reward_config.roll_scale": 1.7485453973763958,
  "reward_config.zero_keep_scale": 7.572043456865868,
  "reward_config.action_rate_scale": 0.5534776853501364,
  "reward_config.energy_scale": 2.517534549342281,
  "reward_config.pitch_velocity_penalty_scale": 0.23861174244737682,
  "reward_config.sigma": 0.8682504778554553,
  "reward_config.sigma_wheel": 0.6263458020088386,
  "vel_pid.kp": 0.0001759197389951316,
  "vel_pid.notsmoothing": 0.30972500391006313
}
```

| 分项 | 达成率 | vs base2 |
|---|---|---|
| tracking_lin_vel | 81.8% | +5.5pp |
| tracking_ang_vel | 79.9% | +16.8pp |
| tracking_height_jnt | 64.5% | +10.6pp |
| tracking_height_action | 76.4% | +20.6pp |
| pitch_cost | 84.4% | +10.9pp |
| roll_cost | 81.4% | +5.8pp |
| tracking_sinyaw | 84.7% | +1.9pp |
| tracking_wheels_sinposition | 58.8% | +3.9pp |

### #5 trial search2-047

```json
{
  "reward_config.vel_scale": 3.4727809134751624,
  "reward_config.height_scale": 0.9940400426976008,
  "reward_config.roll_scale": 2.6397863499682526,
  "reward_config.zero_keep_scale": 14.82539946476885,
  "reward_config.action_rate_scale": 0.1422638932328053,
  "reward_config.energy_scale": 0.4676516326248976,
  "reward_config.pitch_velocity_penalty_scale": 0.38048979755455864,
  "reward_config.sigma": 0.9084309066407097,
  "reward_config.sigma_wheel": 0.7090889508744631,
  "vel_pid.kp": 0.00019789419659174923,
  "vel_pid.notsmoothing": 0.25473002351962587
}
```

| 分项 | 达成率 | vs base2 |
|---|---|---|
| tracking_lin_vel | 84.8% | +8.5pp |
| tracking_ang_vel | 85.7% | +22.6pp |
| tracking_height_jnt | 41.1% | -12.8pp |
| tracking_height_action | 45.0% | -10.7pp |
| pitch_cost | 85.2% | +11.7pp |
| roll_cost | 76.8% | +1.2pp |
| tracking_sinyaw | 84.3% | +1.5pp |
| tracking_wheels_sinposition | 57.7% | +2.8pp |

