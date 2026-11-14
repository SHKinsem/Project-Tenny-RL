"""Torque200 environment regressions; run: python tests/test_env.py.

These check the action/observation contract and inherited hold behaviour, not
whether a randomly initialized policy can balance. No firmware is exercised.
"""

import functools
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jp
import mujoco
import numpy as np

from legwheel import LegWheelRobot, default_config
from legwheel.randomization import domain_randomize


@functools.lru_cache(maxsize=None)
def _env(delay=0, hold_clip=False, ctrl_dt=0.005):
    cfg = default_config()
    cfg.action_delay_steps = delay
    cfg.hold_error_clip.enable = hold_clip
    cfg.noise_config.level = 0.0
    cfg.push_config.enable = False
    cfg.cmd_config.cmd_prob = 0.0
    cfg.ctrl_dt = ctrl_dt
    cfg.episode_length = int(round(cfg.episode_time / ctrl_dt))
    cfg.cmd_config.command_change_interval = int(round(3.0 / ctrl_dt))
    return LegWheelRobot(cfg)


@functools.lru_cache(maxsize=None)
def _step_fn(env):
    return jax.jit(env.step)


def _reset(env):
    return env.reset(jax.random.PRNGKey(0))


def _step(env, state, action):
    return _step_fn(env)(state, jp.asarray(action, dtype=jp.float32))


def _assert_close(actual, expected, atol=1e-6):
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=atol, rtol=0)


def _scales(env):
    return jp.asarray(env._config.torque_control.scales)


def test_default_timing_and_shapes():
    cfg = default_config()
    assert cfg.ctrl_dt == 0.005
    assert cfg.sim_dt == 0.0025
    assert cfg.episode_length == 1200
    assert cfg.cmd_config.command_change_interval == 600
    env = _env()
    assert env.n_substeps == 2
    state = _reset(env)
    assert state.obs["state"].shape == (24,)
    assert state.obs["privileged_state"].shape == (51,)
    assert env.action_size == 4
    after = _step(env, state, jp.zeros(4))
    _assert_close(after.data.time - state.data.time, 0.005)
    assert np.isfinite(np.asarray(after.obs["state"])).all()
    assert np.isfinite(np.asarray(after.obs["privileged_state"])).all()
    assert np.isfinite(float(after.reward))
    _assert_close(state.obs["state"][-4:], np.zeros(4))


def test_absolute_torque_sign_scale_and_saturation():
    env = _env()
    # The out-of-range channels must saturate, not wrap or accumulate.
    action = jp.array([2.0, -2.0, 0.5, -0.25])
    expected_action = jp.clip(action, -1.0, 1.0)
    expected_torque = expected_action * _scales(env)
    state = _step(env, _reset(env), action)
    _assert_close(state.data.ctrl, expected_torque)
    _assert_close(state.data.actuator_force, expected_torque)
    _assert_close(state.info["applied_torque"], expected_torque)
    _assert_close(state.info["last_torque_action"], expected_action)
    _assert_close(state.obs["state"][-4:], expected_action)
    # Returning to zero must immediately clear actuator force: no old PI state.
    state = _step(env, state, jp.zeros(4))
    _assert_close(state.data.ctrl, np.zeros(4))
    _assert_close(state.data.actuator_force, np.zeros(4))
    _assert_close(state.obs["state"][-4:], np.zeros(4))


def test_zero_action_has_no_hidden_position_servo():
    model = _env().mj_model
    _assert_close(model.actuator_biasprm, np.zeros_like(model.actuator_biasprm))
    _assert_close(model.actuator_gear[:, 0], np.ones(4))
    _assert_close(model.actuator_gear[:, 1:], np.zeros((4, 5)))
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    # Change position and velocity while leaving all control inputs zero.
    data.qpos[[7, 10]] = [-0.1, -0.4]
    data.qvel[[6, 9]] = [1.0, -1.0]
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)
    _assert_close(data.actuator_force, np.zeros(4))
    # Passive forces may remain: preserving the original robot is intentional.


def test_delay_observation_describes_applied_not_requested_action():
    env = _env(delay=2)
    action = jp.array([0.2, -0.4, 0.6, -0.8])
    state = _reset(env)
    for _ in range(2):
        state = _step(env, state, action)
        _assert_close(state.data.ctrl, np.zeros(4))
        _assert_close(state.obs["state"][-4:], np.zeros(4))
    state = _step(env, state, action)
    _assert_close(state.data.ctrl, action * _scales(env))
    _assert_close(state.obs["state"][-4:], action)
    _assert_close(state.info["act_delay_buffer"], np.tile(np.asarray(action), (2, 1)))


def test_thigh_velocity_is_measured_not_replaced_with_noise():
    env = _env()
    state = _reset(env)
    qvel = state.data.qvel.at[6].set(1.25).at[9].set(-0.75)
    obs = env._get_obs(state.data.replace(qvel=qvel), dict(state.info))["state"]
    _assert_close(obs[jp.array([14, 16])], [1.25, -0.75])


def test_no_obsolete_pid_state():
    state = _reset(_env())
    obsolete = {"pid_integral", "pid_prev_error", "desired_wheel_angvel",
                "wheels_commands_history", "last_target_wheel_vel", "last_wheel_torque"}
    assert not obsolete.intersection(state.info), obsolete.intersection(state.info)


def test_height_reward_does_not_compare_torque_to_joint_target():
    env = _env()
    state = _reset(env)
    info = dict(state.info)
    info["command"] = jp.array([0.0, 0.0, -0.25, 0.0])
    r_zero = env._get_reward(state.data, jp.zeros(4), dict(info), dict(state.metrics))
    r_force = env._get_reward(state.data, _scales(env), dict(info), dict(state.metrics))
    _assert_close(r_zero["height_reward"], r_force["height_reward"])
    assert "reward/tracking_height_action" not in state.metrics


def test_effort_penalty_uses_all_four_normalized_torques():
    env = _env()
    state = _reset(env)
    normalized = jp.array([0.25, -0.5, 0.75, -1.0])
    rewards = env._get_reward(state.data, normalized * _scales(env),
                              dict(state.info), dict(state.metrics))
    expected = jp.mean(normalized ** 2) * env._config.reward_config.energy_scale
    _assert_close(rewards["energy_cost"], expected)


def test_reward_total_has_reference_time_scaling():
    env = _env()
    before = _reset(env)
    after = _step(env, before, jp.zeros(4))
    # Zero action keeps action history unchanged, and cmd_prob=0 prevents resampling.
    components = env._get_reward(after.data, jp.zeros(4),
                                 dict(before.info), dict(before.metrics))
    expected = sum(components.values()) * env.dt / 0.010
    _assert_close(after.reward, expected, atol=2e-5)


def test_action_rate_matches_same_physical_linear_ramp():
    # A ramp of 10 normalized units/s has the same cost at both sampling rates.
    costs = []
    for dt in (0.005, 0.010):
        env = _env(ctrl_dt=dt)
        current = jp.full(4, 2 * dt * 10)
        previous = jp.full(4, dt * 10)
        costs.append(env._cost_action_rate(current, previous, jp.zeros(4)))
    _assert_close(costs[0], costs[1])
    assert float(costs[0]) > 0


def test_randomization_preserves_torque_actuators():
    env = _env()
    keys = jax.random.split(jax.random.PRNGKey(12), 4)
    model, _ = domain_randomize(env.mjx_model, keys)
    bias = np.asarray(model.actuator_biasprm)
    gains = np.asarray(model.actuator_gainprm)[..., 0]
    assert np.isfinite(gains).all()
    assert (gains > 0).all()
    assert gains.shape == (4, 4), gains.shape
    assert (gains[:, :2] >= 0.9 - 1e-6).all()
    assert (gains[:, :2] <= 1.1 + 1e-6).all()
    assert (gains[:, 2:] >= 0.8 - 1e-6).all()
    assert (gains[:, 2:] <= 1.2 + 1e-6).all()
    assert np.ptp(gains[:, :2]) > 0
    assert np.ptp(gains[:, 2:]) > 0
    _assert_close(bias, np.zeros_like(bias))
    # Randomization must not reintroduce a position or velocity controller.
    assert np.isfinite(np.asarray(model.body_mass)).all()


def test_sample_command_covers_negative_cmds():
    cfg = default_config()
    env = LegWheelRobot(cfg)
    keys = jax.random.split(jax.random.PRNGKey(0), 5000)
    commands = jax.vmap(env.sample_command)(keys)
    for column, margin in ((0, cfg.cmd_lin_vel_x_zero_margin),
                           (1, cfg.cmd_ang_vel_zero_margin)):
        values = commands[:, column]
        assert float((values < -margin).mean()) > 0.2
        assert float((values > margin).mean()) > 0.2
    assert np.max(np.abs(np.asarray(commands[:, 0]))) <= 0.2 + 1e-6
    assert np.max(np.abs(np.asarray(commands[:, 1]))) <= 1.5 + 1e-6


def test_hold_obs_clip_preserves_anchor():
    env = _env(hold_clip=True)
    state = _reset(env)
    info = dict(state.info)
    info["command"] = jp.zeros(4)
    info["position_start"] = env.get_hold_position(state.data) - 8.0
    expected_anchor = info["position_start"]
    state = env.prepare_inference(state.replace(info=info))
    _assert_close(state.info["position_start"], expected_anchor)
    _assert_close(state.obs["state"][jp.array([11, 13])], [5.0, 5.0])
    state = _step(env, state, jp.zeros(4))
    _assert_close(state.info["position_start"], expected_anchor)
    _assert_close(state.obs["state"][11], state.obs["state"][13])


def test_training_hold_observation_is_not_clipped():
    env = _env()
    state = _reset(env)
    info = dict(state.info)
    info["position_start"] = env.get_hold_position(state.data) - 8.0
    obs = env._get_obs(state.data, info)["state"]
    _assert_close(obs[jp.array([11, 13])], [8.0, 8.0])


def test_yaw_anchor_frozen_on_forward_command():
    env = _env()
    state = _reset(env)
    info = dict(state.info)
    info["command"] = jp.array([0.2, 0.0, -0.25, 0.0])
    info["sinyaw_start"] = env._getSinYaw_newest(state.data) + 1.0
    expected = info["sinyaw_start"]
    state = _step(env, state.replace(info=info), jp.zeros(4))
    _assert_close(state.info["sinyaw_start"], expected)


def test_position_anchor_frozen_on_yaw_command():
    env = _env()
    state = _reset(env)
    info = dict(state.info)
    info["command"] = jp.array([0.0, 0.5, -0.25, 0.0])
    info["position_start"] = env.get_hold_position(state.data) + 3.0
    expected = info["position_start"]
    state = _step(env, state.replace(info=info), jp.zeros(4))
    _assert_close(state.info["position_start"], expected)


def test_hold_position_wheel_signs_and_reward():
    env = _env()
    state = _reset(env)
    qpos = state.data.qpos
    s0 = env.get_hold_position(state.data)
    forward = state.data.replace(qpos=qpos.at[9].add(1).at[12].add(1))
    turning = state.data.replace(qpos=qpos.at[9].add(1).at[12].add(-1))
    _assert_close(env.get_hold_position(forward) - s0, 1.0)
    _assert_close(env.get_hold_position(turning) - s0, 0.0)
    _assert_close(env._reward_tracking_wheels_sinposition(jp.array(0.), jp.array(0.)), 1.0)
    assert env._reward_tracking_wheels_sinposition(jp.array(1.), jp.array(0.)) < 0.2


def test_reset_clears_action_history_and_preserves_original_pose():
    env = _env()
    state = _step(env, _reset(env), jp.ones(4) * 0.2)
    assert np.any(np.asarray(state.obs["state"][-4:]) != 0)
    state = _reset(env)
    _assert_close(state.data.qpos, env.mjx_model.qpos0)
    _assert_close(state.info["last_torque_action"], np.zeros(4))
    _assert_close(state.info["last_act"], np.zeros(4))
    _assert_close(state.info["last_last_act"], np.zeros(4))
    _assert_close(state.info["applied_torque"], np.zeros(4))


if __name__ == "__main__":
    failures = 0
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}", flush=True)
        except Exception:
            failures += 1
            print(f"FAIL {name}", flush=True)
            traceback.print_exc()
    print(f"TORQUE200_ENV_TESTS {len(tests) - failures}/{len(tests)} passed", flush=True)
    sys.exit(1 if failures else 0)
