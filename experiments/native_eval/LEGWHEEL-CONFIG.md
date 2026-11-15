# Legwheel task configuration

`type: "legwheel"` is the deterministic, single-frame 24-observation / 4-action
adapter for production `HybridLegWheelRobot` and `StudentEnv`. The generic engine
does not depend on its dimensions or indices. Config is a flat dictionary; the
binding groups values by their native types. Unknown keys and wrong types fail.

All names below refer to the loaded MuJoCo model. Left precedes right. Required
lists are `hip_joints[2]`, `knee_joints[2]`, `wheel_joints[2]`, `wheel_bodies[2]`,
and `actuators[4]` (hip L/R, wheel L/R). Required strings are `root_body`,
`gyro_sensor`, `local_linvel_sensor`, and `action_mode` (`virtual_leg_force`,
`direct_torque`, or `matched_envelope`). Motors must be stateless, fixed unit gain,
zero bias, and unit gear, and must transmit to the named joints. The task never
substitutes ctrl for the actual `qfrc_actuator` values.

Required numeric vectors:

| Key | Length | Meaning |
| --- | --- | --- |
| `leg_geometry_left`, `leg_geometry_right` | 28 each | Production `LegGeometry`, flattened as described below |
| `jmax` | 2 | `scan_geometry`'s `jmax_bound` |
| `force_scales` | 2 | `scan_geometry`'s `force_scale_n` |
| `torque_scales` | 4 | Hip L/R and wheel L/R limits / normalization |
| `hip_range` | 2 | Common calibrated experimental hip envelope |
| `command_scales` | 4 | Actual production `_cmd_scales`, including float32 rounding if applicable |
| `thigh_range` | 2 | Production `action_range.thigh` for height projection |

Each geometry vector is ordered:
`[hip_ref, knee_ref, *poly(5), *knee_offset(3), *knee_rotation(row-major 9),
*knee_anchor(3), *knee_axis(3), *wheel_center(3)]`.
This is a direct flattening of `dataclasses.asdict(leg)` except the `side` string.
The knee axis is normalized by the production XML reader. The task evaluates the
actual knee angle, including the soft equality residual, at every physics substep.

Optional scalar numbers (defaults are production standing_v2):

| Key | Default |
| --- | ---: |
| `reference_dt` | 0.01 |
| `wheel_radius` | 0.0195 |
| `sinyaw_scale` | 0.5 |
| `cmd_lin_vel_x_zero_margin` | 0.000195 |
| `cmd_ang_vel_zero_margin` | 0.01 |
| `sigma` | 0.5 |
| `sigma_wheel` | 0.5 |
| `height_scale`, `vel_scale` | 2.0 each |
| `roll_scale` | 4.0 |
| `pitch_scale` | 0.0 |
| `zero_keep_scale` | 10.0 |
| `energy_scale` | -1.0 |
| `action_rate_scale` | -0.5 |
| `pitch_velocity_penalty_scale` | -0.5 |
| `roll_thigh_coupling` | 0.433 |
| `equality_position_limit` | 0.01 |
| `length_velocity_limit` | 0.05 |
| `minimum_jacobian_ratio` | 0.05 |
| `pitch_min`, `pitch_max` | -0.6, 0.85 |
| `roll_limit` | 0.6 |

Optional bool flags `sinpitch_zero`, `sinyaw`, `wheel_sinpos_start`,
`enable_lin_vel_x`, `enable_ang_vel_yaw`, `enable_height_value`,
`enable_roll_value`, and `height_feasible_projection` default to true.
Optional strings `geometry_schema` and `observation_version`, if supplied, must
equal `wheel-center-axial-v1` and `torque24-mean-sent-v1`, respectively.

The following explicit compatibility declarations are accepted only at the
listed values: numbers `noise_level=0`, `command_probability=0`,
`action_delay_steps=0`, `action_repeat=1`, `obs_history_len=1`; flags
`push_enabled=false`, `domain_randomization=false`, `last_act_obs=false`,
`hold_error_clip=false`, `obs_normalization=false`, `termination_penalty=false`,
`do_deconflict=false`. The Python exporter must check the source configuration
before emitting this supported subset. Noise, push, random dynamics, sampled or
time-varying commands, observation normalization/history, delay, and termination
penalties are unsupported and must never be silently dropped.

An initial state's command is four fixed raw values `[vx, yaw_rate, height,
roll]`; empty means zero. Nonzero commands use the production position/yaw
dead-zone gates and refresh the corresponding hold anchor after each step.
Supported `extra` fields are `position_start`, `sinyaw_start`, and
`sin_balance_zero`, each containing one finite value. Omitted anchors are computed
from the reset physical state. Reset clears every action and actual-torque
history. Random reset states should be supplied explicitly by the caller.

The engine supplies physics dt and an integer substep count. Reward is scaled by
`substeps * model.opt.timestep / reference_dt`; actual-torque first and second
differences use `reference_dt / model.opt.timestep` and carry history across
control periods. Invalid geometry latches within a period, sends zero torque for
the remaining substeps, completes every requested physics step, and terminates.
