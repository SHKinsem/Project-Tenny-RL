# run2 阶段 R1:reward shaping 搜索(约束满足 + 帕累托判定,gate-R1 审查材料)

- 完成 trial 数: 8,主目标:归一化跟踪 RMSE(越小越好,WORST=10.0)
- base2 RMSE: 0.6591,显著改善线 base2 − 2σ = 0.5647(σ 复用 run1 噪声底 0.0472)
- base2 关键分项达成率: {"tracking_lin_vel": 0.7658, "tracking_ang_vel": 0.6384, "tracking_height_jnt": 0.5936, "tracking_height_action": 0.6008, "pitch_cost": 0.7306, "roll_cost": 0.8037, "tracking_sinyaw": 0.8402, "tracking_wheels_sinposition": 0.6311, "action_rate_per_step": 1.2846, "energy_cost_per_step": 0.0366, "pitch_velocity_penalty_per_step": 0.6162}

## 约束满足情况(按 RMSE 升序,全量完成 trial)

| trial | RMSE | 存活 | 能耗/步 | action_rate 部署 | sinyaw 达成率 | 约束 | 违反项 |
|---|---|---|---|---|---|---|---|
| search2-015 | 0.5561 | 560.0 | 0.0298 | 0.3348 | 0.8615 | 违反 | action_rate_deploy<=0.15 |
| search2-006 | 0.5953 | 522.8 | 0.0273 | 0.3183 | 0.8384 | 违反 | survival>=550, action_rate_deploy<=0.15, sinyaw_rate>=base2+2pp |
| search2-004 | 0.6483 | 518.9 | 0.0354 | 0.2818 | 0.8021 | 违反 | survival>=550, action_rate_deploy<=0.15, sinyaw_rate>=base2+2pp, pitch_cost退化<=2pp |
| search2-001 | 0.6743 | 569.0 | 0.0310 | 0.3535 | 0.8498 | 违反 | action_rate_deploy<=0.15, sinyaw_rate>=base2+2pp, tracking_lin_vel退化<=2pp |
| search2-003 | 0.6957 | 524.0 | 0.0429 | 0.3507 | 0.8320 | 违反 | survival>=550, action_rate_deploy<=0.15, sinyaw_rate>=base2+2pp, tracking_lin_vel退化<=2pp, tracking_ang_vel退化<=2pp, pitch_cost退化<=2pp, roll_cost退化<=2pp |
| search2-000 | 0.7017 | 552.0 | 0.0385 | 0.2828 | 0.8952 | 违反 | survival>=550, action_rate_deploy<=0.15 |
| search2-005 | 0.7316 | 529.1 | 0.0285 | 0.3313 | 0.8457 | 违反 | survival>=550, action_rate_deploy<=0.15, sinyaw_rate>=base2+2pp, pitch_cost退化<=2pp, roll_cost退化<=2pp |
| search2-002 | 0.9980 | 547.6 | 0.0430 | 0.2801 | 0.8189 | 违反 | survival>=550, action_rate_deploy<=0.15, sinyaw_rate>=base2+2pp, tracking_lin_vel退化<=2pp, tracking_ang_vel退化<=2pp, pitch_cost退化<=2pp, roll_cost退化<=2pp |

## 帕累托判定(top 5,RMSE 显著 + 分项硬约束全过才算赢)

| 排名 | trial | RMSE | RMSE 显著 | 约束全过 | 判定 |
|---|---|---|---|---|---|
| 1 | search2-015 | 0.5561 | 是 | 否 | 否 |
| 2 | search2-006 | 0.5953 | 否 | 否 | 否 |
| 3 | search2-004 | 0.6483 | 否 | 否 | 否 |
| 4 | search2-001 | 0.6743 | 否 | 否 | 否 |
| 5 | search2-003 | 0.6957 | 否 | 否 | 否 |

- 约束全过的 trial 数: **0** / 8(gate-R1:不足 3 个说明约束过紧或空间不够,按计划 6.gate-R1 两档处置,不要让搜索空跑)

## Top trial 参数

### #1 trial search2-015

```json
{
  "reward_config.action_rate_scale": 1.153785630512394,
  "reward_config.energy_scale": 1.985357657185739,
  "reward_config.pitch_velocity_penalty_scale": 0.5456196340471977,
  "reward_config.zero_keep_scale": 4.808752441110493,
  "reward_config.vel_scale": 3.42139093600775,
  "reward_config.height_scale": 1.9026812529471335,
  "reward_config.roll_scale": 11.753932833782558,
  "reward_config.sigma": 0.8353789057077303,
  "reward_config.sigma_wheel": 0.648808293395497,
  "vel_pid.kp": 0.0001688999075002288,
  "vel_pid.notsmoothing": 0.27688876415902886
}
```

### #2 trial search2-006

```json
{
  "reward_config.action_rate_scale": 1.8255511227718364,
  "reward_config.energy_scale": 0.14307444471758823,
  "reward_config.pitch_velocity_penalty_scale": 1.100038830238109,
  "reward_config.zero_keep_scale": 3.742998582237653,
  "reward_config.vel_scale": 5.683432617631446,
  "reward_config.height_scale": 1.765234683958553,
  "reward_config.roll_scale": 11.374762413318672,
  "reward_config.sigma": 0.5782202498733064,
  "reward_config.sigma_wheel": 0.6966602525368217,
  "vel_pid.kp": 0.00012192209473398765,
  "vel_pid.notsmoothing": 0.20848238894265336
}
```

### #3 trial search2-004

```json
{
  "reward_config.action_rate_scale": 0.9287248456977505,
  "reward_config.energy_scale": 1.1868671225073562,
  "reward_config.pitch_velocity_penalty_scale": 0.10864643696586337,
  "reward_config.zero_keep_scale": 4.036895918892134,
  "reward_config.vel_scale": 1.240450965033791,
  "reward_config.height_scale": 1.3863153159348665,
  "reward_config.roll_scale": 4.4604432318445335,
  "reward_config.sigma": 0.4592026990699765,
  "reward_config.sigma_wheel": 0.9840119056749078,
  "vel_pid.kp": 0.00012675970809186005,
  "vel_pid.notsmoothing": 0.19915608732054882
}
```

### #4 trial search2-001

```json
{
  "reward_config.action_rate_scale": 0.5075801506336866,
  "reward_config.energy_scale": 0.8129081255257445,
  "reward_config.pitch_velocity_penalty_scale": 1.5199571569452797,
  "reward_config.zero_keep_scale": 3.533111252256514,
  "reward_config.vel_scale": 0.7332980837225367,
  "reward_config.height_scale": 0.6285931554651909,
  "reward_config.roll_scale": 8.162084593855202,
  "reward_config.sigma": 0.7352534235799667,
  "reward_config.sigma_wheel": 0.8351019833009119,
  "vel_pid.kp": 0.00021809409912512234,
  "vel_pid.notsmoothing": 0.2870040878124922
}
```

### #5 trial search2-003

```json
{
  "reward_config.action_rate_scale": 0.6036024038597264,
  "reward_config.energy_scale": 0.10717719521090709,
  "reward_config.pitch_velocity_penalty_scale": 0.4880441634946439,
  "reward_config.zero_keep_scale": 12.280526240898489,
  "reward_config.vel_scale": 2.4836205650145264,
  "reward_config.height_scale": 5.271076617127191,
  "reward_config.roll_scale": 5.76768515439796,
  "reward_config.sigma": 0.41151467973779804,
  "reward_config.sigma_wheel": 0.45820461914851945,
  "vel_pid.kp": 0.00018327441339048014,
  "vel_pid.notsmoothing": 0.18164748299140274
}
```

