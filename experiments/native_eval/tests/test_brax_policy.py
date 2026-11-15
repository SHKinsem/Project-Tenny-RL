"""Live host-parameter conversion checked against independent Flax-layout math."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from native_eval import policy_actions
from native_eval.brax_policy import policy_from_brax


def fixture(inputs=3, outputs=2, hidden=(5,), distribution=True, prefix="hidden"):
    rng = np.random.default_rng(352 + inputs + outputs + len(hidden))
    dims = [inputs, *hidden, outputs]
    template = dict(type="mlp", mean=[19.] * inputs, std=[17.] * inputs,
                    layers=[], action_size=outputs, output_activation="tanh")
    actor = {}
    activations = ("silu", "tanh", "relu", "identity")
    for index, (width, height) in enumerate(zip(dims, dims[1:])):
        last = index == len(dims) - 2
        template["layers"].append(dict(weights=np.zeros((height, width)).tolist(),
                                       bias=np.full(height, 11.).tolist(),
                                       activation="identity" if last else activations[index % 4]))
        live_height = 2 * height if last and distribution else height
        actor[f"{prefix}_{index}"] = dict(kernel=rng.normal(0, .3, (width, live_height)).astype(np.float32),
                                         bias=rng.normal(0, .2, live_height).astype(np.float32))
    stats = SimpleNamespace(mean={"state": rng.normal(0, .2, inputs).astype(np.float32)},
                            std={"state": rng.uniform(.3, 1.5, inputs).astype(np.float32)})
    return (stats, {"params": dict(reversed(tuple(actor.items())))}, object()), template


def activate(values, name):
    if name == "silu":
        return values / (np.float32(1.) + np.exp(-values))
    if name == "tanh":
        return np.tanh(values)
    if name == "relu":
        return np.maximum(values, np.float32(0.))
    assert name == "identity"
    return values


def reference(params, template, observations):
    stats, actor = params[:2]
    raw = np.asarray(observations, dtype=np.float32)
    value = (raw - stats.mean["state"]) / stats.std["state"]
    order = sorted(actor["params"], key=lambda name: int(name.rsplit("_", 1)[1]))
    for index, name in enumerate(order):
        layer = actor["params"][name]
        value = activate(value @ layer["kernel"] + layer["bias"], template["layers"][index]["activation"])
    count = template["action_size"]
    value = value[:, :count] * np.asarray(template.get("network_scales", [1.] * count), dtype=np.float32)
    for index, source in enumerate(template.get("skip_indices", [-1] * count)):
        if source < 0:
            continue
        skip = raw[:, source]
        if template.get("skip_transform", "identity") == "atanh":
            eps = np.float32(template.get("skip_clip_epsilon", 1e-5))
            skip = np.arctanh(np.clip(skip, np.float32(-1.) + eps, np.float32(1.) - eps))
        value[:, index] += np.float32(template.get("skip_scales", [1.] * count)[index]) * skip
    return activate(value, template["output_activation"])


@pytest.mark.parametrize("inputs,outputs,hidden,distribution", [
    (1, 1, (), False), (3, 2, (5,), True), (7, 5, (4, 9), True),
    (2, 4, (8, 3, 6), False), (24, 4, (64, 64), True), (2, 1, (2,) * 11, True),
])
@pytest.mark.parametrize("prefix", ["hidden", "Dense"])
def test_live_parameters_match_flax_layout_math(inputs, outputs, hidden, distribution, prefix):
    params, template = fixture(inputs, outputs, hidden, distribution, prefix)
    result = policy_from_brax(params, template)
    observations = np.random.default_rng(21).normal(0, .4, (7, inputs)).astype(np.float32)
    np.testing.assert_allclose(policy_actions(result, observations), reference(params, template, observations),
                               rtol=4e-6, atol=2e-6)
    for index, layer in enumerate(result["layers"]):
        source = params[1]["params"][f"{prefix}_{index}"]
        outputs = len(layer["bias"])
        np.testing.assert_array_equal(layer["weights"], source["kernel"][:, :outputs].T)


def test_current_actor_and_normalizer_replace_saved_template_on_every_call():
    params, template = fixture()
    observation = np.array([[.4, -.1, .2]], dtype=np.float32)
    before = policy_actions(policy_from_brax(params, template), observation)
    params[1]["params"]["hidden_1"]["bias"][:2] += .7
    after_actor = policy_actions(policy_from_brax(params, template), observation)
    assert np.max(np.abs(before - after_actor)) > .01
    params[0].mean["state"] += .5
    params[0].std["state"] *= 1.7
    result = policy_from_brax(params, template)
    after_normalizer = policy_actions(result, observation)
    assert np.max(np.abs(after_actor - after_normalizer)) > 1e-4
    np.testing.assert_allclose(after_normalizer, reference(params, template, observation), atol=1e-6)


@pytest.mark.parametrize("transform", ["identity", "atanh"])
def test_task_independent_skip_and_activation_contract_preserved(transform):
    params, template = fixture(7, 3, (6,))
    template.update(skip_indices=[6, -1, 2], network_scales=[.4, 1., .7],
                    skip_scales=[.9, 0., -.3], skip_transform=transform, skip_clip_epsilon=.01,
                    output_activation="identity")
    observations = np.random.default_rng(17).normal(0, .6, (8, 7)).astype(np.float32)
    result = policy_from_brax(params, template)
    for name in ("skip_indices", "network_scales", "skip_scales", "skip_transform", "skip_clip_epsilon"):
        assert result[name] == template[name]
    np.testing.assert_allclose(policy_actions(result, observations), reference(params, template, observations),
                               rtol=4e-6, atol=2e-6)


def test_output_owns_data_and_does_not_mutate_inputs():
    params, template = fixture()
    template.update(skip_indices=[-1, 2], skip_scales=[0., .5])
    saved = deepcopy(template)
    expected_kernel = params[1]["params"]["hidden_0"]["kernel"].copy()
    result = policy_from_brax(params, template)
    assert template == saved
    result["layers"][0]["weights"][0][0] = -100
    result["mean"][0] = -100
    result["skip_scales"][1] = -100
    assert template == saved
    np.testing.assert_array_equal(params[1]["params"]["hidden_0"]["kernel"], expected_kernel)
    assert params[0].mean["state"][0] != -100


def test_mapping_normalizer_and_alternative_observation_key():
    params, template = fixture()
    stats = {"mean": {"actor": params[0].mean["state"]}, "std": {"actor": params[0].std["state"]}}
    result = policy_from_brax((stats, params[1]), template, observation_key="actor")
    np.testing.assert_array_equal(result["mean"], params[0].mean["state"])
    np.testing.assert_array_equal(result["std"], params[0].std["state"])


@pytest.mark.parametrize("mutation", [
    "stats_shape", "stats_missing", "std_zero", "std_negative", "stats_nan", "stats_inf",
    "kernel_shape", "bias_shape", "kernel_nan", "bias_inf", "distribution_tail_nan", "overflow",
    "extra_layer", "missing_layer", "unknown_name", "mixed_prefix", "noncontiguous", "leading_zero",
    "extra_layer_field", "extra_actor_field", "wrong_kernel_dtype", "wrong_stats_dtype",
])
def test_incompatible_live_parameters_are_rejected(mutation):
    params, template = fixture()
    stats, actor = params[:2]
    layer = actor["params"]["hidden_0"]
    if mutation == "stats_shape":
        stats.mean["state"] = [0., 0.]
    elif mutation == "stats_missing":
        del stats.std["state"]
    elif mutation == "std_zero":
        stats.std["state"][0] = 0
    elif mutation == "std_negative":
        stats.std["state"][0] = -1
    elif mutation == "stats_nan":
        stats.mean["state"][0] = np.nan
    elif mutation == "stats_inf":
        stats.std["state"][0] = np.inf
    elif mutation == "kernel_shape":
        layer["kernel"] = np.zeros((4, 5))
    elif mutation == "bias_shape":
        layer["bias"] = np.zeros(4)
    elif mutation == "kernel_nan":
        layer["kernel"][0, 0] = np.nan
    elif mutation == "bias_inf":
        layer["bias"][0] = np.inf
    elif mutation == "distribution_tail_nan":
        actor["params"]["hidden_1"]["bias"][-1] = np.nan
    elif mutation == "overflow":
        layer["kernel"] = np.full((3, 5), 1e300)
    elif mutation == "extra_layer":
        actor["params"]["hidden_2"] = deepcopy(layer)
    elif mutation == "missing_layer":
        del actor["params"]["hidden_1"]
    elif mutation in ("unknown_name", "mixed_prefix", "noncontiguous", "leading_zero"):
        name = {"unknown_name": "other", "mixed_prefix": "Dense_0",
                "noncontiguous": "hidden_2", "leading_zero": "hidden_00"}[mutation]
        actor["params"][name] = actor["params"].pop("hidden_0")
    elif mutation == "extra_layer_field":
        layer["scale"] = np.ones(5)
    elif mutation == "extra_actor_field":
        actor["batch_stats"] = {}
    elif mutation == "wrong_kernel_dtype":
        layer["kernel"] = np.full((3, 5), "1")
    elif mutation == "wrong_stats_dtype":
        stats.mean["state"] = np.array([True, False, True])
    with pytest.raises(ValueError):
        policy_from_brax(params, template)


@pytest.mark.parametrize("mutation", ["not_mlp", "layer_count", "broken_chain", "action_width", "bias_width"])
def test_incompatible_template_architecture_is_rejected(mutation):
    params, template = fixture()
    if mutation == "not_mlp":
        template["type"] = "rnn"
    elif mutation == "layer_count":
        template["layers"].pop()
    elif mutation == "broken_chain":
        template["layers"][1]["weights"] = np.zeros((2, 6)).tolist()
    elif mutation == "action_width":
        template["action_size"] = 3
    else:
        template["layers"][0]["bias"] = [0.]
    with pytest.raises(ValueError):
        policy_from_brax(params, template)


@pytest.mark.parametrize("params", [None, {}, (), (None,), (None, None, None, None)])
def test_parameter_container_must_be_supported_brax_tuple(params):
    _, template = fixture()
    with pytest.raises(ValueError):
        policy_from_brax(params, template)
