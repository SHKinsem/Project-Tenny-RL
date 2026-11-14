# 阶段 2:TPE + ASHA 搜索结果(gate-2 审查材料)

- 完成 trial 数: 7,主目标:归一化跟踪 RMSE(越小越好)
- baseline RMSE: 0.7084, S(θ)=固定权重重算分(sanity check,越大越好)

## Top 5(按主指标)

| 排名 | trial | RMSE | 存活 | 能耗/步 | return | S(θ) | 异常标记 |
|---|---|---|---|---|---|---|---|
| 1 | search-010 | 0.5162 | 584.5 | 0.0279 | 5042.65 | 5164.85 | - |
| 2 | search-015 | 0.5544 | 577.1 | 0.0172 | 4420.26 | 4528.56 | - |
| 3 | search-027 | 0.5587 | 564.6 | 0.0268 | 4648.78 | 4651.43 | - |
| 4 | search-000 | 0.5698 | 581.4 | 0.0264 | 4611.48 | 4769.64 | - |
| 5 | search-012 | 0.6498 | 416.2 | 0.0350 | 3613.61 | 3640.29 | - |

## Top trial 参数

### #1 trial search-010

```json
{
  "vel_pid.target_scale": 44.36961409945713,
  "learning_rate": 0.0002994581357971835,
  "action_scale.wheel": 0.023222504090077256,
  "policy_net": "128x64",
  "vel_pid.notsmoothing": 0.25082138932605313,
  "vel_pid.torque_limit": 0.025022828722066966,
  "action_range_thigh_lo": -0.2830184280073902,
  "num_updates_per_batch": 8,
  "action_scale.thigh": 0.36746194952269073,
  "value_net": "512x512",
  "vel_pid.kp": 0.00017026277433303862
}
```

- 正项: {"pitch_cost": 430.4607238769531, "roll_cost": 480.77203369140625, "tracking_ang_vel": 399.68560791015625, "tracking_height_action": 298.9226989746094, "tracking_height_jnt": 302.24578857421875, "tracking_lin_vel": 448.58929443359375, "tracking_sinyaw": 55.10888671875, "tracking_wheels_sinposition": 36.3588752746582, "tracking_ang_vel_std": 69.40584564208984, "tracking_height_action_std": 97.18872833251953, "tracking_height_jnt_std": 125.15971374511719, "tracking_lin_vel_std": 78.5958023071289, "tracking_sinyaw_std": 124.68002319335938, "tracking_wheels_sinposition_std": 88.19596099853516}
- 负项: {"action_rate": 742.2473754882812, "energy_cost": 16.052627563476562, "pitch_velocity_penalty": 369.248779296875, "termination_penalty": 0.0, "action_rate_std": 121.24903869628906, "energy_cost_std": 3.7188546657562256, "pitch_cost_std": 73.7718276977539, "pitch_velocity_penalty_std": 95.341796875, "roll_cost_std": 81.11680603027344, "termination_penalty_std": 0.0}

### #2 trial search-015

```json
{
  "vel_pid.target_scale": 31.045984035679098,
  "learning_rate": 0.00012199351216949624,
  "action_scale.wheel": 0.021044228356500348,
  "policy_net": "128x64",
  "vel_pid.notsmoothing": 0.4314885094947459,
  "vel_pid.torque_limit": 0.039418882700450256,
  "action_range_thigh_lo": -0.31774112603376026,
  "num_updates_per_batch": 8,
  "action_scale.thigh": 0.1296938987419988,
  "value_net": "512x512",
  "vel_pid.kp": 0.000100217541996427
}
```

- 正项: {"pitch_cost": 404.60791015625, "roll_cost": 438.54620361328125, "tracking_ang_vel": 382.82476806640625, "tracking_height_action": 240.93447875976562, "tracking_height_jnt": 233.94451904296875, "tracking_lin_vel": 424.4542236328125, "tracking_sinyaw": 52.80232238769531, "tracking_wheels_sinposition": 29.14114761352539, "tracking_ang_vel_std": 70.0219497680664, "tracking_height_action_std": 126.8410873413086, "tracking_height_jnt_std": 134.72039794921875, "tracking_lin_vel_std": 82.36241149902344, "tracking_sinyaw_std": 123.10028076171875, "tracking_wheels_sinposition_std": 74.74592590332031}
- 负项: {"action_rate": 813.6553955078125, "energy_cost": 9.661079406738281, "pitch_velocity_penalty": 385.76806640625, "termination_penalty": 0.0, "action_rate_std": 141.74822998046875, "energy_cost_std": 2.941230297088623, "pitch_cost_std": 74.36817932128906, "pitch_velocity_penalty_std": 85.62464141845703, "roll_cost_std": 91.68492889404297, "termination_penalty_std": 0.0}

### #3 trial search-027

```json
{
  "vel_pid.target_scale": 28.368127095665386,
  "learning_rate": 0.0002506996807342352,
  "action_scale.wheel": 0.026484073123022842,
  "policy_net": "128x64",
  "vel_pid.notsmoothing": 0.585570479885961,
  "vel_pid.torque_limit": 0.03065396916258502,
  "action_range_thigh_lo": -0.29245537582081443,
  "num_updates_per_batch": 4,
  "action_scale.thigh": 0.27529122762852193,
  "value_net": "512x512",
  "vel_pid.kp": 0.00010582463207425557
}
```

- 正项: {"pitch_cost": 404.6287841796875, "roll_cost": 449.359130859375, "tracking_ang_vel": 364.81488037109375, "tracking_height_action": 260.0447998046875, "tracking_height_jnt": 259.0502014160156, "tracking_lin_vel": 417.318603515625, "tracking_sinyaw": 51.053287506103516, "tracking_wheels_sinposition": 31.007545471191406, "tracking_ang_vel_std": 94.21305084228516, "tracking_height_action_std": 100.06906127929688, "tracking_height_jnt_std": 124.23570251464844, "tracking_lin_vel_std": 115.19017028808594, "tracking_sinyaw_std": 118.64186096191406, "tracking_wheels_sinposition_std": 77.64417266845703}
- 负项: {"action_rate": 762.8385009765625, "energy_cost": 14.672979354858398, "pitch_velocity_penalty": 345.956298828125, "termination_penalty": 0.0, "action_rate_std": 191.90126037597656, "energy_cost_std": 3.9970409870147705, "pitch_cost_std": 107.28627014160156, "pitch_velocity_penalty_std": 91.51182556152344, "roll_cost_std": 111.56791687011719, "termination_penalty_std": 0.0}

### #4 trial search-000

```json
{
  "vel_pid.target_scale": 31.464405117819744,
  "learning_rate": 0.00015570789053085475,
  "action_scale.wheel": 0.03253151959411349,
  "policy_net": "128x64",
  "vel_pid.notsmoothing": 0.46879360563134626,
  "vel_pid.torque_limit": 0.04105730352307135,
  "action_range_thigh_lo": -0.26816861974948536,
  "num_updates_per_batch": 4,
  "action_scale.thigh": 0.26701114027348305,
  "value_net": "512x512",
  "vel_pid.kp": 0.00011004523464883122
}
```

- 正项: {"pitch_cost": 422.7634582519531, "roll_cost": 474.811767578125, "tracking_ang_vel": 367.98193359375, "tracking_height_action": 258.31103515625, "tracking_height_jnt": 254.85934448242188, "tracking_lin_vel": 439.835693359375, "tracking_sinyaw": 51.485267639160156, "tracking_wheels_sinposition": 32.72117233276367, "tracking_ang_vel_std": 63.95557403564453, "tracking_height_action_std": 87.16868591308594, "tracking_height_jnt_std": 108.8835678100586, "tracking_lin_vel_std": 73.07737731933594, "tracking_sinyaw_std": 119.71540832519531, "tracking_wheels_sinposition_std": 78.65838623046875}
- 负项: {"action_rate": 826.874755859375, "energy_cost": 14.935349464416504, "pitch_velocity_penalty": 370.55181884765625, "termination_penalty": 0.0, "action_rate_std": 121.47782135009766, "energy_cost_std": 2.4707136154174805, "pitch_cost_std": 67.53318786621094, "pitch_velocity_penalty_std": 80.21078491210938, "roll_cost_std": 73.69609832763672, "termination_penalty_std": 0.0}

### #5 trial search-012

```json
{
  "vel_pid.target_scale": 17.22281001516542,
  "learning_rate": 0.00021568639529362697,
  "action_scale.wheel": 0.0431127105352739,
  "policy_net": "128x64",
  "vel_pid.notsmoothing": 0.4695145158649908,
  "vel_pid.torque_limit": 0.03764561907326964,
  "action_range_thigh_lo": -0.7385537317076618,
  "num_updates_per_batch": 8,
  "action_scale.thigh": 0.25498496013268235,
  "value_net": "512x512",
  "vel_pid.kp": 0.00021537779089498304
}
```

- 正项: {"pitch_cost": 304.13079833984375, "roll_cost": 318.39129638671875, "tracking_ang_vel": 267.02398681640625, "tracking_height_action": 222.96351623535156, "tracking_height_jnt": 220.13845825195312, "tracking_lin_vel": 302.8580627441406, "tracking_sinyaw": 44.59777069091797, "tracking_wheels_sinposition": 28.510623931884766, "tracking_ang_vel_std": 160.5977020263672, "tracking_height_action_std": 143.7022705078125, "tracking_height_jnt_std": 157.44375610351562, "tracking_lin_vel_std": 194.6500701904297, "tracking_sinyaw_std": 114.78988647460938, "tracking_wheels_sinposition_std": 77.4642333984375}
- 负项: {"action_rate": 530.8570556640625, "energy_cost": 13.624833106994629, "pitch_velocity_penalty": 222.54635620117188, "termination_penalty": 0.0, "action_rate_std": 316.1166076660156, "energy_cost_std": 9.762293815612793, "pitch_cost_std": 197.22592163085938, "pitch_velocity_penalty_std": 98.90686798095703, "roll_cost_std": 195.97341918945312, "termination_penalty_std": 0.0}

## Top-1 与 baseline 的参数差异

```json
{
  "env": {
    "vel_pid.target_scale": 44.36961409945713,
    "action_scale.wheel": 0.023222504090077256,
    "vel_pid.notsmoothing": 0.25082138932605313,
    "vel_pid.torque_limit": 0.025022828722066966,
    "action_range.thigh": [
      -0.2830184280073902,
      0.0
    ],
    "action_scale.thigh": 0.36746194952269073,
    "vel_pid.kp": 0.00017026277433303862
  },
  "ppo": {
    "learning_rate": 0.0002994581357971835,
    "policy_hidden_layer_sizes": [
      128,
      64
    ],
    "num_updates_per_batch": 8,
    "value_hidden_layer_sizes": [
      512,
      512
    ]
  },
  "randomization": {}
}
```

→ 人工核准:排除可疑组合,确认进入全量确认的名单(默认 top-2)。
