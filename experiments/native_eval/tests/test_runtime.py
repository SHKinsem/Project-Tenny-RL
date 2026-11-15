"""Contract tests against real MuJoCo; no training or production runtime required."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import math
import sys
import threading

import mujoco
import numpy as np
import pytest

from native_eval import _native


XML = Path(__file__).parent / "fixtures" / "pendulum.xml"
ERRORS = (ValueError, TypeError, RuntimeError, OverflowError)


def constant_policy(value=0.0, observation_size=3, action_size=1):
    return {
        "type": "mlp",
        "mean": [0.0] * observation_size,
        "std": [1.0] * observation_size,
        "layers": [{"weights": [[0.0] * observation_size] * action_size,
                    "bias": [value] * action_size, "activation": "identity"}],
        "action_size": action_size,
        "output_activation": "identity",
    }


def evaluator(config=None, policy=None, xml=XML, physics_dt=0.0):
    return _native.Evaluator(str(xml),
                             {"type": "pendulum", "config": config or {}},
                             policy or constant_policy(), physics_dt)


def options(num_envs=1, rollout_steps=7, episode_steps=100, substeps=1, **extra):
    return dict(num_envs=num_envs, rollout_steps=rollout_steps,
                episode_steps=episode_steps, control_dt=0.002 * substeps,
                trace_worlds=list(range(num_envs)), **extra)


def observations(angle, velocity, target=0.0):
    return np.array([math.cos(angle - target), math.sin(angle - target), velocity])


def reference_trajectory(initial, steps, episode_steps, substeps, action, config, physics_dt=0.002):
    model = mujoco.MjModel.from_xml_path(str(XML))
    model.opt.timestep = physics_dt
    data = mujoco.MjData(model)
    result = {key: [] for key in ("observation", "action", "qpos", "qvel", "ctrl", "xmat", "reward", "done")}
    elapsed = 0

    def reset():
        mujoco.mj_resetData(model, data)
        data.qpos[:] = initial["qpos"]
        data.qvel[:] = initial["qvel"]
        mujoco.mj_forward(model, data)

    reset()
    for _ in range(steps):
        result["observation"].append(observations(data.qpos[0], data.qvel[0], config.get("target", 0.0)))
        result["action"].append([action])
        data.ctrl[0] = action * config.get("torque_scale", 1.0)
        for _ in range(substeps):
            mujoco.mj_step(model, data)
        elapsed += 1
        angle = data.qpos[0] - config.get("target", 0.0)
        threshold = config.get("terminate_angle", False)
        terminated = bool(threshold and abs(math.remainder(angle, 2.0 * math.pi)) > threshold)
        reward = (math.cos(angle) - config.get("velocity_cost", 0.1) * data.qvel[0] ** 2
                  - config.get("action_cost", 0.001) * action ** 2)
        done = terminated or elapsed >= episode_steps
        result["qpos"].append(data.qpos.copy())
        result["qvel"].append(data.qvel.copy())
        result["ctrl"].append(data.ctrl.copy())
        # Keep MuJoCo's raw mj_step stage; no post-step forward recomputes geometry.
        result["xmat"].append(data.xmat.copy().reshape(model.nbody, 3, 3))
        result["reward"].append(reward)
        result["done"].append(done)
        if done:
            reset()
            elapsed = 0
    return {key: np.asarray(value) for key, value in result.items()}


def test_capabilities_are_explicit_and_bounded():
    capabilities = _native.capabilities()
    assert capabilities["schema"] == 1
    assert {"pendulum", "legwheel"}.issubset(capabilities["tasks"])
    assert capabilities["stateless_policy"] is True
    assert capabilities["deterministic_policy"] is True
    assert capabilities["native_control_threads"] == 1


@pytest.mark.parametrize("num_envs,horizon,episode_steps,substeps", [
    (1, 7, 3, 1), (3, 11, 4, 3), (5, 9, 12, 4),
])
def test_dynamic_worlds_substeps_and_raw_traces(num_envs, horizon, episode_steps, substeps):
    config = dict(torque_scale=2.3, velocity_cost=0.17, action_cost=0.08, target=0.13)
    action = float(np.float32(0.125))
    engine = evaluator(config, constant_policy(action))
    assert (engine.nq, engine.nv, engine.nu, engine.nbody) == (1, 1, 1, 2)
    assert (engine.observation_size, engine.action_size, engine.task_name) == (3, 1, "pendulum")
    assert engine.physics_dt == pytest.approx(0.002)
    initial = [{"qpos": [0.3 + 0.11 * i], "qvel": [-0.2 + 0.09 * i]} for i in range(num_envs)]
    result = engine.evaluate(initial, options(num_envs, horizon, episode_steps, substeps))
    assert result["control_steps"] == num_envs * horizon
    assert result["physics_steps"] == num_envs * horizon * substeps
    assert not result["cancelled"]
    assert result["wall_seconds"] > 0
    assert len(result["traces"]) == num_envs
    # A task with no custom metrics still supplies the reserved reward series.
    assert set(result["episode_metrics"]) == {"reward"}
    assert result["episode_metrics"]["reward"].shape == (num_envs,)
    np.testing.assert_array_equal(result["episode_metrics"]["reward"], result["returns"])
    np.testing.assert_array_equal(result["lengths"], [min(horizon, episode_steps)] * num_envs)
    np.testing.assert_array_equal(result["terminated"], [False] * num_envs)
    np.testing.assert_array_equal(result["truncated"], [horizon >= episode_steps] * num_envs)
    np.testing.assert_array_equal(result["resets"], [horizon // episode_steps] * num_envs)
    assert list(result["reasons"]) == ["time_limit" if horizon >= episode_steps else ""] * num_envs
    for world, trace in enumerate(result["traces"]):
        expected = reference_trajectory(initial[world], horizon, episode_steps, substeps, action, config)
        assert trace["world"] == world
        assert trace["steps"] == horizon
        for key, dimension in (("observation", 3), ("action", 1), ("qpos", 1), ("qvel", 1), ("ctrl", 1)):
            assert trace[key].shape == (horizon, dimension)
        for key in ("reward", "done"):
            assert trace[key].shape == (horizon,)
        assert trace["xmat"].shape == (horizon, engine.nbody, 3, 3)
        np.testing.assert_array_equal(trace["xmat"], expected["xmat"])
        for key in expected.keys() - {"xmat"}:
            np.testing.assert_allclose(trace[key], expected[key], rtol=2e-12, atol=2e-12, err_msg=key)
        np.testing.assert_allclose(result["returns"][world], expected["reward"][:min(horizon, episode_steps)].sum(),
                                   rtol=2e-12, atol=2e-12)
        if horizon > episode_steps:
            # The reset observation returns to the initial state, but the prior
            # post-step qpos in the trace remains the actual terminal state.
            np.testing.assert_allclose(trace["observation"][episode_steps], observations(*[initial[world][key][0] for key in ("qpos", "qvel")], target=config["target"]))
            assert not np.isclose(trace["qpos"][episode_steps - 1, 0], initial[world]["qpos"][0], atol=1e-8)



def test_xmat_preserves_raw_step_stage_before_reset():
    initial = {"qpos": [0.4], "qvel": [-0.7]}
    action = 0.125
    config = {"torque_scale": 2.3}
    engine = evaluator(config, constant_policy(action))
    trace = engine.evaluate([initial], options(1, 3, 2, 3))["traces"][0]
    expected = reference_trajectory(initial, 3, 2, 3, action, config)
    np.testing.assert_array_equal(trace["xmat"], expected["xmat"])
    # With the fixture's Euler integrator, mj_step integrates qpos after the
    # kinematic stage. This motion makes recomputing geometry from final qpos
    # observably different, so a hidden mj_forward cannot satisfy the oracle.
    angle = trace["qpos"][:, 0]
    final_qpos_rotation = np.zeros((3, 3, 3))
    final_qpos_rotation[:, 0, 0] = np.cos(angle)
    final_qpos_rotation[:, 0, 2] = np.sin(angle)
    final_qpos_rotation[:, 1, 1] = 1.0
    final_qpos_rotation[:, 2, 0] = -np.sin(angle)
    final_qpos_rotation[:, 2, 2] = np.cos(angle)
    assert np.max(np.abs(trace["xmat"][:, 1] - final_qpos_rotation)) > 1e-6


def test_first_episode_mask_early_done_and_reset_state():
    config = dict(terminate_angle=0.2)
    initial = [{"qpos": [0.7], "qvel": [0.0]}, {"qpos": [0.1], "qvel": [0.0]}]
    result = evaluator(config).evaluate(initial, options(2, 7, 3, 3))
    np.testing.assert_array_equal(result["lengths"], [1, 3])
    np.testing.assert_array_equal(result["terminated"], [True, False])
    np.testing.assert_array_equal(result["truncated"], [False, True])
    assert list(result["reasons"]) == ["angle_limit", "time_limit"]
    np.testing.assert_array_equal(result["resets"], [7, 2])
    assert result["control_steps"] == 14
    assert result["physics_steps"] == 42
    np.testing.assert_array_equal(result["episode_metrics"]["reward"], result["returns"])
    for world, length in enumerate((1, 3)):
        expected = reference_trajectory(initial[world], 7, 3, 3, 0.0, config)
        trace = result["traces"][world]
        np.testing.assert_array_equal(trace["xmat"], expected["xmat"])
        for key in expected.keys() - {"xmat"}:
            np.testing.assert_allclose(trace[key], expected[key], rtol=2e-12, atol=2e-12)
        assert result["returns"][world] == pytest.approx(expected["reward"][:length].sum(), rel=2e-12)
    # World zero resets every control period, including velocity and observation.
    trace = result["traces"][0]
    np.testing.assert_allclose(trace["observation"], np.tile(observations(0.7, 0.0), (7, 1)))


def test_termination_wins_simultaneous_time_limit():
    result = evaluator({"terminate_angle": 0.2}).evaluate(
        [{"qpos": [0.7], "qvel": [0.0]}], options(1, 3, 1))
    assert result["terminated"].tolist() == [True]
    assert result["truncated"].tolist() == [False]
    assert result["reasons"] == ["angle_limit"]
    assert result["lengths"].tolist() == [1]
    assert result["returns"][0] == pytest.approx(result["traces"][0]["reward"][0])
    assert result["resets"].tolist() == [3]


@pytest.mark.parametrize("angle,terminated", [(2 * math.pi + 0.1, False), (2 * math.pi + 0.5, True)])
def test_termination_uses_wrapped_target_error(angle, terminated):
    result = evaluator({"terminate_angle": 0.2, "target": 0.1}).evaluate(
        [{"qpos": [angle + 0.1], "qvel": [0.0]}], options(1, 1, 10))
    assert bool(result["terminated"][0]) is terminated


def test_seed_reproducibility_world_episode_independence_and_new_evaluation():
    engine = evaluator({"initial_angle_range": 0.7, "initial_velocity_range": 0.4})
    opts = options(5, 5, 2, seed=913)
    first = engine.evaluate([{}], opts)
    repeated = engine.evaluate([{}], opts)
    different = engine.evaluate([{}], dict(opts, seed=914))
    for world in range(5):
        for key in ("observation", "qpos", "qvel", "reward", "done"):
            np.testing.assert_array_equal(first["traces"][world][key], repeated["traces"][world][key])
        initial_obs = first["traces"][world]["observation"][0]
        assert abs(math.atan2(initial_obs[1], initial_obs[0])) <= 0.7
        assert abs(initial_obs[2]) <= 0.4
        assert not np.array_equal(initial_obs, different["traces"][world]["observation"][0])
        assert not np.array_equal(initial_obs, first["traces"][world]["observation"][2])
    initial_observations = np.stack([trace["observation"][0] for trace in first["traces"]])
    assert np.unique(initial_observations, axis=0).shape[0] == 5
    np.testing.assert_array_equal(first["returns"], repeated["returns"])
    np.testing.assert_array_equal(first["lengths"], repeated["lengths"])
    assert first["resets"].tolist() == [2] * 5


def test_explicit_initial_state_overrides_only_supplied_coordinates():
    config = {"initial_angle_range": 0.2, "initial_velocity_range": 0.1}
    engine = evaluator(config)
    result = engine.evaluate([{"qpos": [0.85], "qvel": [0.9]}], options(1, 4, 1, seed=42))
    np.testing.assert_allclose(result["traces"][0]["observation"], np.tile(observations(0.85, 0.9), (4, 1)))
    supplied_angle = engine.evaluate([{"qpos": [0.85]}], options(1, 1, 2, seed=42))["traces"][0]["observation"][0]
    np.testing.assert_allclose(supplied_angle[:2], observations(0.85, 0.0)[:2])
    assert abs(supplied_angle[2]) <= 0.1
    supplied_velocity = engine.evaluate([{"qvel": [0.9]}], options(1, 1, 2, seed=42))["traces"][0]["observation"][0]
    assert abs(math.atan2(supplied_velocity[1], supplied_velocity[0])) <= 0.2
    assert supplied_velocity[2] == 0.9


def test_named_joint_actuator_xml_default_and_physics_override(tmp_path):
    xml = tmp_path / "renamed.xml"
    source = XML.read_text().replace('name="hinge"', 'name="pivot" ref="0.25"')
    source = source.replace('joint="hinge"', 'joint="pivot"').replace('name="motor"', 'name="torque"')
    xml.write_text(source)
    engine = evaluator({"joint": "pivot", "actuator": "torque"}, xml=xml, physics_dt=0.001)
    assert engine.physics_dt == 0.001
    result = engine.evaluate([{}], dict(options(1, 1, 2), control_dt=0.004))
    np.testing.assert_allclose(result["traces"][0]["observation"][0], observations(0.25, 0.0))
    assert result["physics_steps"] == 4


def test_broadcast_selected_traces_and_no_trace():
    engine = evaluator()
    initial = [{"qpos": [0.3], "qvel": [0.1]}]
    result = engine.evaluate(initial, dict(options(3, 4, 9), trace_worlds=[2, 0]))
    assert [trace["world"] for trace in result["traces"]] == [2, 0]
    np.testing.assert_array_equal(result["returns"], [result["returns"][0]] * 3)
    np.testing.assert_array_equal(result["traces"][0]["qpos"], result["traces"][1]["qpos"])
    without_trace = engine.evaluate(initial, dict(options(3, 4, 9), trace_worlds=[]))
    assert without_trace["traces"] == []
    np.testing.assert_array_equal(without_trace["returns"], result["returns"])


@pytest.mark.parametrize("config", [
    {"unknown": 1}, {"torque_scale": 0}, {"torque_scale": -1},
    {"velocity_cost": -1}, {"action_cost": -1}, {"initial_angle_range": -1},
    {"initial_velocity_range": -1}, {"terminate_angle": 0},
    {"terminate_angle": math.pi + 0.1}, {"terminate_angle": True},
    {"joint": "absent"}, {"actuator": "absent"}, {"joint": ""},
    {"target": "wrong"}, {"target": [1.0]}, {"target": float("nan")},
    {"torque_scale": float("inf")},
])
def test_invalid_task_configuration_rejected(config):
    with pytest.raises(ERRORS):
        evaluator(config)


def test_scaled_actuator_command_overflow_is_rejected():
    with pytest.raises(ERRORS, match="(?i)(finite|overflow|command)"):
        evaluator({"torque_scale": 1e308}, constant_policy(2.0)).evaluate([{}], options(1, 1, 5))


def test_explicit_termination_disabled():
    result = evaluator({"terminate_angle": False}).evaluate([{"qpos": [2.7]}], options(1, 1, 5))
    assert not result["terminated"][0]


@pytest.mark.parametrize("initial,changes", [
    ([], {}), ([{}, {}, {}], {"num_envs": 2}),
    ([{"qpos": [0.0, 1.0]}], {}), ([{"qvel": []}, {"qvel": []}], {}),
    ([{"qvel": [0.0, 1.0]}], {}), ([{"qpos": [float("nan")]}], {}),
    ([{"qvel": [float("inf")]}], {}), ([{"command": [float("nan")]}], {}),
    ([{"extra": {"bad": [float("nan")]}}], {}),
    ([{"command": [1.0]}], {}), ([{"extra": {"unsupported": [1.0]}}], {}),
    ([{}], {"num_envs": 0}), ([{}], {"rollout_steps": 0}),
    ([{}], {"episode_steps": 0}), ([{}], {"control_dt": 0.0}),
    ([{}], {"control_dt": -0.01}), ([{}], {"control_dt": 0.003}),
    ([{}], {"control_dt": float("nan")}), ([{}], {"trace_worlds": [1]}),
])
def test_invalid_evaluation_inputs(initial, changes):
    engine = evaluator()
    with pytest.raises(ERRORS):
        engine.evaluate(initial, dict(options(), **changes))
    # A validation failure must release the evaluator's in-use guard.
    assert engine.evaluate([{}], options(1, 1, 2))["control_steps"] == 1


@pytest.mark.parametrize("edit", [
    lambda s: s.replace('type="hinge"', 'type="slide"'),
    lambda s: s.replace('<motor name="motor" joint="hinge" gear="1"/>',
                        '<position name="motor" joint="hinge" kp="1"/>'),
    lambda s: s.replace('<joint name="hinge"', '<joint name="extra" type="slide" axis="1 0 0"/><joint name="hinge"'),
])
def test_pendulum_model_assumptions_are_task_validation(tmp_path, edit):
    xml = tmp_path / "unsupported.xml"
    xml.write_text(edit(XML.read_text()))
    with pytest.raises(ERRORS):
        evaluator(xml=xml)


def test_missing_xml_is_reported(tmp_path):
    with pytest.raises(ERRORS, match="(?i)(missing|open|exist|file|xml)"):
        evaluator(xml=tmp_path / "missing.xml")


def test_cancel_releases_gil_preserves_partial_work_and_rejects_concurrent_evaluate():
    engine = evaluator()
    entered = threading.Event()
    requested = 2_000_000
    opts = options(1, requested, 97, 1)

    def run():
        entered.set()
        return engine.evaluate([{}], opts)

    # Keep the short Event.set -> native call sequence on the worker thread.
    # The next hand-off occurs when the native binding releases the GIL.
    old_interval = sys.getswitchinterval()
    with ThreadPoolExecutor(max_workers=1) as pool:
        try:
            sys.setswitchinterval(1.0)
            future = pool.submit(run)
            assert entered.wait(timeout=5)
            assert not future.done(), "native evaluate completed before another Python thread could run"
            with pytest.raises(RuntimeError, match="(?i)(running|concurrent|evaluat|use)"):
                engine.evaluate([{}], options(1, 1, 2))
            engine.cancel()
            result = future.result(timeout=5)
        finally:
            sys.setswitchinterval(old_interval)
            engine.cancel()
    assert result["cancelled"]
    assert 0 <= result["control_steps"] < requested
    assert result["physics_steps"] == result["control_steps"]
    trace = result["traces"][0]
    assert trace["steps"] == result["control_steps"]
    assert trace["qpos"].shape == (result["control_steps"], 1)
    # Keep model axes even if cancellation wins before the first transition.
    assert trace["xmat"].shape == (result["control_steps"], engine.nbody, 3, 3)
    assert result["lengths"][0] == min(result["control_steps"], 97)
    assert result["returns"][0] == pytest.approx(trace["reward"][:97].sum())
    assert set(result["episode_metrics"]) == {"reward"}
    assert result["episode_metrics"]["reward"].shape == (1,)
    np.testing.assert_array_equal(result["episode_metrics"]["reward"], result["returns"])
    assert result["resets"][0] == result["control_steps"] // 97
    fresh = engine.evaluate([{}], options(1, 3, 10))
    assert not fresh["cancelled"]
    assert fresh["control_steps"] == 3
    assert fresh["lengths"].tolist() == [3]
