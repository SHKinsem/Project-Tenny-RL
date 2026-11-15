"""Translate a host Brax dense actor into the existing native policy contract.

The caller owns device synchronization and selecting the policy/head contract.
Only the normalizer and actor are read; no critic, checkpoint or ML runtime is
needed here. Each call returns independent arrays from the current parameters.
"""
from collections.abc import Mapping
from copy import deepcopy
import re

import numpy as np


def _array(value, shape, label):
    value = np.asarray(value)
    if value.dtype.kind not in "fiu" or value.shape != shape:
        raise ValueError(f"Invalid {label}: expected numeric array with shape {shape}")
    with np.errstate(over="ignore", invalid="ignore"):
        value = value.astype(np.float32)
    if not np.isfinite(value).all():
        raise ValueError(f"Nonfinite {label}")
    return value


def _statistics(normalizer, field, observation_key):
    value = (normalizer.get(field) if isinstance(normalizer, Mapping)
             else getattr(normalizer, field, None))
    if not isinstance(value, Mapping) or observation_key not in value:
        raise ValueError(f"Normalizer needs {field}[{observation_key!r}]")
    return value[observation_key]


def _ordered_layers(actor):
    if not isinstance(actor, Mapping) or set(actor) != {"params"}:
        raise ValueError("Actor must contain exactly one Flax params mapping")
    values = actor["params"]
    if not isinstance(values, Mapping) or not values:
        raise ValueError("Actor params must contain dense layers")
    ordered, prefix = {}, None
    for name, layer in values.items():
        match = re.fullmatch(r"(hidden|Dense)_(0|[1-9][0-9]*)", name) if isinstance(name, str) else None
        if match is None:
            raise ValueError("Unsupported actor layer name; expected hidden_N or Dense_N")
        if prefix is None:
            prefix = match[1]
        if match[1] != prefix:
            raise ValueError("Actor layer names must use one consistent prefix")
        index = int(match[2])
        if not isinstance(layer, Mapping) or set(layer) != {"kernel", "bias"}:
            raise ValueError("Each actor layer must contain only kernel and bias")
        ordered[index] = layer
    if set(ordered) != set(range(len(ordered))):
        raise ValueError("Actor dense layer indices must be contiguous from zero")
    return [ordered[index] for index in range(len(ordered))]


def policy_from_brax(policy_params, template, *, observation_key="state"):
    """Return a native MLP using the current host normalizer and Flax actor.

    ``policy_params`` is ``(normalizer, actor[, critic])`` after device_get.
    ``template`` is a validated native MLP descriptor providing architecture,
    elementwise activations and generic skip/output transformations. Its saved
    weights and statistics are never used as live policy values. Flax kernels
    are [input, output]; native weights are [output, input]. A final layer of
    twice the action width is interpreted as deterministic mean/scale logits.
    """
    if not isinstance(policy_params, (tuple, list)) or len(policy_params) not in (2, 3):
        raise ValueError("Expected host Brax (normalizer, actor[, critic]) parameters")
    if not isinstance(template, Mapping) or template.get("type") != "mlp":
        raise ValueError("A validated native MLP policy template is required")
    if not isinstance(observation_key, str) or not observation_key:
        raise ValueError("observation_key must be a nonempty string")
    normalizer, actor = policy_params[:2]
    layers = _ordered_layers(actor)
    declared = template.get("layers")
    if not isinstance(declared, (tuple, list)) or not declared or len(layers) != len(declared):
        raise ValueError("Live actor layer count differs from the native policy template")
    inputs = len(template.get("mean", ()))
    action_size = template.get("action_size")
    if inputs <= 0 or type(action_size) is not int or action_size <= 0:
        raise ValueError("Invalid native policy template dimensions")
    result = deepcopy(dict(template))
    mean = _array(_statistics(normalizer, "mean", observation_key), (inputs,), "normalizer mean")
    std = _array(_statistics(normalizer, "std", observation_key), (inputs,), "normalizer std")
    if np.any(std <= 0):
        raise ValueError("Normalizer std must be positive")
    result["mean"], result["std"] = mean.tolist(), std.tolist()
    converted = []
    for index, (live, expected) in enumerate(zip(layers, declared, strict=True)):
        if not isinstance(expected, Mapping):
            raise ValueError("Invalid native policy template layer")
        shape = np.shape(expected.get("weights"))
        if len(shape) != 2 or shape[0] <= 0 or shape[1] != inputs:
            raise ValueError("Native policy template layers have incompatible dimensions")
        outputs = shape[0]
        final = index == len(declared) - 1
        if final and outputs != action_size:
            raise ValueError("Native policy template final width differs from action_size")
        if np.shape(expected.get("bias")) != (outputs,):
            raise ValueError("Native policy template bias has incompatible dimensions")
        actual_shape = np.shape(live["kernel"])
        allowed_outputs = (outputs, 2 * outputs) if final else (outputs,)
        if len(actual_shape) != 2 or actual_shape[0] != inputs or actual_shape[1] not in allowed_outputs:
            raise ValueError("Live actor architecture differs from the native policy template")
        kernel = _array(live["kernel"], actual_shape, f"actor layer {index} kernel")
        bias = _array(live["bias"], (actual_shape[1],), f"actor layer {index} bias")
        layer = deepcopy(dict(expected))
        layer.update(weights=kernel[:, :outputs].T.tolist(), bias=bias[:outputs].tolist())
        converted.append(layer)
        inputs = outputs
    result["layers"] = converted
    return result
