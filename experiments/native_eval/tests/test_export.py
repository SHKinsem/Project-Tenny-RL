"""Artifact decoding: row order, deterministic mean head and strict metadata."""
import hashlib

import numpy as np
import pytest

from native_eval import policy_actions
from native_eval.legwheel import policy_descriptor


def artifact(tmp_path, *, observation_size=3, action_size=2, hidden=5, residual=False):
    rng = np.random.default_rng(317)
    dimensions = [observation_size, hidden, 2 * action_size]
    weights = [rng.normal(0, .2, (b, a)).astype(np.float32) for a, b in zip(dimensions, dimensions[1:])]
    biases = [rng.normal(0, .1, b).astype(np.float32) for b in dimensions[1:]]
    mean = rng.normal(0, .1, observation_size).astype(np.float32)
    std = rng.uniform(.5, 1.5, observation_size).astype(np.float32)
    arrays = dict(state_mean=mean, state_std=std, weights=np.concatenate([w.ravel() for w in weights]),
                  biases=np.concatenate(biases))
    file = tmp_path / "policy.cpp.txt"
    file.write_text("\n".join("float " + key + "[] = {" + ",".join(f"{float(x):.10g}f" for x in value) + "};"
                              for key, value in arrays.items()))
    metadata = dict(actor_dim=observation_size, action_order=list(map(str, range(action_size))),
                    export_sha256=hashlib.sha256(file.read_bytes()).hexdigest())
    if residual:
        metadata.update(policy_head="wheel_logit_residual_v1", observation_version="torque24-mean-sent-v1",
                        wheel_residual_scale=.05, previous_torque_clip=1e-5)
    ppo = {"network_factory": {"policy_hidden_layer_sizes": [hidden]}}
    return file, metadata, ppo, (mean, std, weights, biases)


@pytest.mark.parametrize("dimensions", [(3, 2, 5), (7, 1, 9), (2, 5, 3), (24, 4, 11)])
def test_dynamic_export_uses_mean_rows_and_row_major_weights(tmp_path, dimensions):
    obs_size, action_size, hidden = dimensions
    file, metadata, ppo, arrays = artifact(tmp_path, observation_size=obs_size, action_size=action_size, hidden=hidden)
    mean, std, weights, biases = arrays
    descriptor = policy_descriptor(file, metadata, ppo)
    assert np.asarray(descriptor["layers"][-1]["weights"]).shape == (action_size, hidden)
    obs = np.random.default_rng(9).normal(0, .2, (13, obs_size)).astype(np.float32)
    layer = ((obs - mean) / std) @ weights[0].T + biases[0]
    layer = layer / (1 + np.exp(-layer))
    expected = np.tanh((layer @ weights[1].T + biases[1])[:, :action_size])
    np.testing.assert_allclose(policy_actions(descriptor, obs), expected, atol=2e-6, rtol=2e-6)


def test_residual_head_exports_generic_skip_parameters(tmp_path):
    file, metadata, ppo, _ = artifact(tmp_path, observation_size=24, action_size=4, residual=True)
    descriptor = policy_descriptor(file, metadata, ppo)
    assert descriptor["skip_indices"] == [-1, -1, 22, 23]
    assert descriptor["network_scales"] == [1, 1, .05, .05]
    assert descriptor["skip_transform"] == "atanh"


@pytest.mark.parametrize("mutation", ["hash", "dimension", "head", "layout", "nan", "zero_std"])
def test_invalid_or_unsupported_export_is_rejected(tmp_path, mutation):
    file, metadata, ppo, _ = artifact(tmp_path)
    if mutation == "hash":
        metadata["export_sha256"] = "0" * 64
    elif mutation == "dimension":
        metadata["actor_dim"] += 1
    elif mutation == "head":
        metadata["policy_head"] = "rnn"
    elif mutation == "layout":
        metadata.update(policy_head="wheel_logit_residual_v1", observation_version="unknown")
    else:
        source = file.read_text()
        if mutation == "nan":
            source = source.replace("float state_mean[] = {", "float state_mean[] = {nan,", 1)
        else:
            start = source.index("float state_std[] = {") + len("float state_std[] = {")
            end = source.index(",", start)
            source = source[:start] + "0f" + source[end:]
        file.write_text(source)
        metadata["export_sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        policy_descriptor(file, metadata, ppo)
