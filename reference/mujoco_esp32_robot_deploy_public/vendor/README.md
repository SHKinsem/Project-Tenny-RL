# 仿真本体快照

跑 `sim_web.sh` / `tests/sim_verify_hold_obs.py` 需要的 MuJoCo 资源和环境代码，拷自训练仓 `legwheel_rl`，不包含 PPO、checkpoint、调参。

| | |
|---|---|
| 来源 | `legwheel_rl` @ `680cd3a`（`improvements/hold-position-heading`） |
| XML / mesh / hfield | `vendor/assets/` |
| MJX 环境 | `vendor/legwheel/`（`LegWheelRobot`、`default_config`、奖励 mixin） |

训练仍在 `legwheel_rl`。换本体（XML、网格、环境动力学）时，从那边同步到这里，并确认 `src/policy/rl_policy.cpp` 仍对得上。

权重不在这里：策略数组在 `src/policy/rl_policy.cpp`，由 `tools/sync_rl_policy.py` 从训练 checkpoint 导出。
