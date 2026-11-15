"""Float32 mathematical oracles for the generic native dense policy."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from native_eval import _native


XML = Path(__file__).parent / "fixtures" / "pendulum.xml"
ERRORS = (ValueError, TypeError, RuntimeError, OverflowError)


def activate(values, kind):
    values = np.asarray(values, dtype=np.float32)
    if kind == "identity":
        return values
    if kind == "silu":
        return (values / (np.float32(1.0) + np.exp(-values))).astype(np.float32)
    if kind == "tanh":
        return np.tanh(values).astype(np.float32)
    if kind == "relu":
        return np.maximum(values, np.float32(0.0))
    raise AssertionError(kind)


def oracle(spec, observations):
    raw = np.asarray(observations, dtype=np.float32)
    value = ((raw - np.asarray(spec["mean"], dtype=np.float32)) /
             np.asarray(spec["std"], dtype=np.float32)).astype(np.float32)
    for layer in spec["layers"]:
        weights = np.asarray(layer["weights"], dtype=np.float32)
        bias = np.asarray(layer["bias"], dtype=np.float32)
        value = activate((value @ weights.T + bias).astype(np.float32), layer.get("activation", "identity"))
    count = spec["action_size"]
    network_scales = np.asarray(spec.get("network_scales", [1.0] * count), dtype=np.float32)
    value = (value * network_scales).astype(np.float32)
    skip_scales = np.asarray(spec.get("skip_scales", [1.0] * count), dtype=np.float32)
    epsilon = np.float32(spec.get("skip_clip_epsilon", 1e-5))
    for index, source in enumerate(spec.get("skip_indices", [-1] * count)):
        if source < 0:
            continue
        skip = raw[:, source]
        if spec.get("skip_transform", "identity") == "atanh":
            skip = np.arctanh(np.clip(skip, np.float32(-1.0) + epsilon,
                                     np.float32(1.0) - epsilon)).astype(np.float32)
        value[:, index] = (value[:, index] + skip_scales[index] * skip).astype(np.float32)
    return activate(value, spec.get("output_activation", "tanh"))


def make_spec(inputs=3, outputs=1, hidden=(5, 4), output_activation="tanh"):
    rng = np.random.default_rng(70 + inputs + outputs + len(hidden))
    dimensions = (inputs, *hidden, outputs)
    activations = ("silu", "tanh", "relu", "identity")
    layers = []
    for index, (width, height) in enumerate(zip(dimensions, dimensions[1:])):
        layers.append({"weights": rng.normal(0, 0.35, (height, width)).astype(np.float32).tolist(),
                       "bias": rng.normal(0, 0.15, height).astype(np.float32).tolist(),
                       "activation": activations[index % len(activations)] if index < len(hidden) else "identity"})
    return {"type": "mlp", "mean": np.linspace(-0.2, 0.3, inputs).astype(np.float32).tolist(),
            "std": np.linspace(0.4, 1.4, inputs).astype(np.float32).tolist(),
            "layers": layers, "action_size": outputs, "output_activation": output_activation}


@pytest.mark.parametrize("inputs,outputs,hidden,head", [
    (1, 1, (), "identity"), (3, 2, (5,), "tanh"),
    (7, 5, (4, 9), "identity"), (2, 4, (8, 3, 6, 7), "tanh"),
])
def test_generic_dimensions_depth_and_dense_math(inputs, outputs, hidden, head):
    spec = make_spec(inputs, outputs, hidden, head)
    observations = np.random.default_rng(981).normal(0, 0.8, (11, inputs)).astype(np.float32)
    actual = _native.policy_actions(spec, observations)
    assert actual.shape == (11, outputs)
    np.testing.assert_allclose(actual, oracle(spec, observations), rtol=3e-6, atol=3e-7)
    np.testing.assert_array_equal(actual, _native.policy_actions(spec, observations))


@pytest.mark.parametrize("transform", ["identity", "atanh"])
@pytest.mark.parametrize("head", ["identity", "tanh"])
def test_generic_input_skip_uses_configured_raw_coordinates(transform, head):
    spec = make_spec(7, 5, (4, 9), head)
    spec.update(skip_indices=[6, -1, 2, 0, 4], network_scales=[0.3, 1.0, 0.8, 2.0, -0.2],
                skip_scales=[2.0, 0.0, -0.7, 0.3, 1.5],
                skip_transform=transform, skip_clip_epsilon=0.02)
    observations = np.array([
        [-1.4, 0.1, 0.2, 0.3, 1.7, -0.4, 0.8],
        [0.15, -0.3, -1.0, 0.2, 0.5, 0.9, 1.3],
        [0.4, 0.2, 0.6, 0.1, -0.7, 0.4, -1.2],
    ], dtype=np.float32)
    actual = _native.policy_actions(spec, observations)
    np.testing.assert_allclose(actual, oracle(spec, observations), rtol=3e-6, atol=2e-6)
    # Disabling a skip must leave this output entirely controlled by the MLP.
    plain = deepcopy(spec)
    plain["skip_indices"] = [-1] * 5
    np.testing.assert_allclose(actual[:, 1], _native.policy_actions(plain, observations)[:, 1], atol=1e-7)


@pytest.mark.parametrize("activation", ["silu", "relu", "tanh", "identity"])
def test_supported_layer_activations(activation):
    spec = make_spec(3, 1, (4,))
    spec["layers"][0]["activation"] = activation
    observations = np.array([[-3, 0, 2], [0.3, -0.2, -1.3]], dtype=np.float32)
    np.testing.assert_allclose(_native.policy_actions(spec, observations), oracle(spec, observations),
                               rtol=3e-6, atol=3e-7)


def test_evaluator_policy_actions_matches_standalone_without_mutating_physics():
    spec = make_spec()
    engine = _native.Evaluator(str(XML), {"type": "pendulum", "config": {}}, spec)
    observations = [[0.3, -0.2, 0.1], [-0.5, 0.7, 0.2]]
    np.testing.assert_array_equal(engine.policy_actions(observations), _native.policy_actions(spec, observations))
    opts = dict(num_envs=1, rollout_steps=3, episode_steps=10, control_dt=0.002, trace_worlds=[0], seed=12)
    before = engine.evaluate([{}], opts)
    engine.policy_actions(observations)
    after = engine.evaluate([{}], opts)
    np.testing.assert_array_equal(before["traces"][0]["qpos"], after["traces"][0]["qpos"])


@pytest.mark.parametrize("mutation", [
    lambda p: p.update(type="unsupported"),
    lambda p: p.update(mean=[]),
    lambda p: p.update(std=[1.0, 1.0]),
    lambda p: p.update(std=[1.0, 0.0, 1.0]),
    lambda p: p.update(std=[1.0, -1.0, 1.0]),
    lambda p: p.update(std=[1.0, float("inf"), 1.0]),
    lambda p: p.update(mean=[1.0, float("nan"), 1.0]),
    lambda p: p.update(layers=[]),
    lambda p: p["layers"][0].update(weights=[[0.0, 1.0], [2.0]]),
    lambda p: p["layers"][0].update(bias=[0.0]),
    lambda p: p["layers"][0].update(activation="gelu"),
    lambda p: p["layers"][1].update(weights=[[0.0, 1.0]]),
    lambda p: p["layers"][-1].update(bias=[float("nan")]),
    lambda p: p["layers"][-1].update(weights=[[float("inf")] * 4]),
    lambda p: p.update(action_size=0),
    lambda p: p.update(action_size=2),
    lambda p: p.update(output_activation="relu"),
    lambda p: p.update(skip_indices=[]),
    lambda p: p.update(network_scales=[]),
    lambda p: p.update(skip_scales=[]),
    lambda p: p.update(skip_indices=[3]),
    lambda p: p.update(skip_indices=[-2]),
    lambda p: p.update(skip_indices=[0, 1]),
    lambda p: p.update(network_scales=[1.0, 2.0]),
    lambda p: p.update(skip_scales=[float("nan")]),
    lambda p: p.update(skip_transform="sigmoid"),
    lambda p: p.update(skip_clip_epsilon=0.0),
    lambda p: p.update(skip_clip_epsilon=1.0),
])
def test_invalid_policy_descriptor_rejected(mutation):
    spec = make_spec()
    mutation(spec)
    with pytest.raises(ERRORS):
        _native.policy_actions(spec, [[0.0, 0.0, 0.0]])


@pytest.mark.parametrize("observations", [
    [[0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0]],
    [[0.0, float("nan"), 0.0]], [[0.0, 0.0, float("inf")]],
])
def test_policy_observation_shape_and_finiteness(observations):
    with pytest.raises(ERRORS):
        _native.policy_actions(make_spec(), observations)


@pytest.mark.parametrize("inputs,outputs", [(2, 1), (3, 2)])
def test_policy_task_dimension_mismatch_rejected_at_construction(inputs, outputs):
    with pytest.raises(ERRORS, match="(?i)(observ|action|dimension|size|match)"):
        _native.Evaluator(str(XML), {"type": "pendulum", "config": {}}, make_spec(inputs, outputs))
