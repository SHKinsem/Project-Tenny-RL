"""Public Python API contracts, using the real native pendulum evaluator."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from native_eval import Evaluator, capabilities, load_descriptor, policy_actions
from native_eval import _native


XML = Path(__file__).parent / "fixtures" / "pendulum.xml"
ERRORS = (ValueError, TypeError, RuntimeError, OverflowError)


def descriptor():
    return {
        "schema_version": 1,
        "xml_path": str(XML.resolve()),
        "task": {"type": "pendulum", "config": {}},
        "policy": {
            "type": "mlp", "mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0],
            "layers": [{"weights": [[0.0, 0.0, 0.0]], "bias": [0.0], "activation": "identity"}],
            "action_size": 1, "output_activation": "identity",
        },
        "eval_defaults": {"num_envs": 3, "rollout_steps": 7, "episode_steps": 3,
                          "control_dt": 0.006, "seed": 41, "trace_worlds": [2, 0]},
        "metadata": {"purpose": "单摆公共接口验收"},
    }


def test_load_mapping_accepts_mapping_protocol_and_does_not_mutate_input():
    source = descriptor()
    snapshot = deepcopy(source)
    loaded = load_descriptor(MappingProxyType(source))
    assert loaded == source
    assert loaded is not source
    assert source == snapshot


@pytest.mark.parametrize("as_string", [False, True])
def test_json_path_resolves_xml_relative_to_descriptor_and_runs_native(tmp_path, monkeypatch, as_string):
    assets = tmp_path / "assets"
    configs = tmp_path / "configs"
    elsewhere = tmp_path / "elsewhere"
    for directory in (assets, configs, elsewhere):
        directory.mkdir()
    model = assets / "pendulum.xml"
    model.write_text(XML.read_text(encoding="utf8"), encoding="utf8")
    source = descriptor()
    source["xml_path"] = "../assets/pendulum.xml"
    config_file = configs / "摆任务.json"
    config_file.write_text(json.dumps(source, ensure_ascii=False), encoding="utf8")
    monkeypatch.chdir(elsewhere)
    selected_source = str(config_file) if as_string else config_file
    loaded = load_descriptor(selected_source)
    assert loaded["xml_path"] == str(model.resolve())
    assert loaded["metadata"] == source["metadata"]
    result = Evaluator(selected_source).evaluate(steps=2, episode_steps=5)
    assert result["control_steps"] == 6
    assert result["physics_steps"] == 18
    np.testing.assert_array_equal(result["returns"], [2.0] * 3)
    np.testing.assert_array_equal(result["lengths"], [2] * 3)


def test_json_absolute_xml_path_is_preserved(tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(descriptor()), encoding="utf8")
    assert load_descriptor(config_file)["xml_path"] == str(XML.resolve())


def test_relative_descriptor_source_path(tmp_path, monkeypatch):
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(descriptor()), encoding="utf8")
    monkeypatch.chdir(tmp_path)
    assert load_descriptor(Path("config.json"))["xml_path"] == str(XML.resolve())


def test_public_evaluate_defaults_overrides_and_model_info():
    source = descriptor()
    snapshot = deepcopy(source)
    engine = Evaluator(source)
    assert engine.model_info == {"nq": 1, "nv": 1, "nu": 1, "physics_dt": 0.002,
                                 "observation_size": 3, "action_size": 1, "task_name": "pendulum"}
    result = engine.evaluate()
    assert result["control_steps"] == 21
    assert result["physics_steps"] == 63
    assert not result["cancelled"]
    np.testing.assert_array_equal(result["returns"], [3.0] * 3)
    np.testing.assert_array_equal(result["lengths"], [3] * 3)
    np.testing.assert_array_equal(result["terminated"], [False] * 3)
    np.testing.assert_array_equal(result["truncated"], [True] * 3)
    np.testing.assert_array_equal(result["resets"], [2] * 3)
    assert result["reasons"] == ["time_limit"] * 3
    assert [trace["world"] for trace in result["traces"]] == [2, 0]
    for trace in result["traces"]:
        np.testing.assert_array_equal(trace["qpos"], np.zeros((7, 1)))
        np.testing.assert_array_equal(trace["action"], np.zeros((7, 1)))
    changed = engine.evaluate(num_envs=5, steps=4, episode_steps=9,
                              control_dt=0.008, seed=99, trace_worlds=[4])
    assert changed["control_steps"] == 20
    assert changed["physics_steps"] == 80
    np.testing.assert_array_equal(changed["returns"], [4.0] * 5)
    np.testing.assert_array_equal(changed["lengths"], [4] * 5)
    np.testing.assert_array_equal(changed["truncated"], [False] * 5)
    assert [trace["world"] for trace in changed["traces"]] == [4]
    assert changed["traces"][0]["observation"].shape == (4, 3)
    # Overrides belong to this call, not to later calls or the descriptor.
    assert source == snapshot
    assert engine.descriptor == snapshot
    assert engine.evaluate()["control_steps"] == 21
    info = engine.model_info
    info["nq"] = 999
    assert engine.model_info["nq"] == 1


def test_public_evaluate_preserves_explicit_initial_state_and_empty_traces():
    engine = Evaluator(descriptor())
    result = engine.evaluate([{"qpos": [0.3], "qvel": [-0.2]}],
                             num_envs=1, steps=2, episode_steps=8,
                             control_dt=0.002, trace_worlds=[0])
    np.testing.assert_allclose(result["traces"][0]["observation"][0],
                               [np.cos(0.3), np.sin(0.3), -0.2])
    assert result["lengths"].tolist() == [2]
    assert result["physics_steps"] == 2
    without_trace = engine.evaluate(num_envs=1, steps=1, trace_worlds=[])
    assert without_trace["traces"] == []


def test_public_physics_override_and_minimal_defaults():
    source = descriptor()
    source.pop("eval_defaults")
    source["physics_dt"] = 0.001
    engine = Evaluator(source)
    assert engine.model_info["physics_dt"] == 0.001
    result = engine.evaluate()
    # Native defaults are one world, one transition, time limit one, dt=0.01.
    assert result["control_steps"] == 1
    assert result["physics_steps"] == 10
    assert result["lengths"].tolist() == [1]
    assert result["truncated"].tolist() == [True]
    assert result["traces"] == []


def test_public_policy_and_capabilities_match_native():
    source = descriptor()
    source["policy"]["layers"][0]["bias"] = [0.125]
    obs = np.array([[0.2, -0.3, 0.1], [0.5, 0.4, -0.2]])
    expected = _native.policy_actions(source["policy"], obs)
    np.testing.assert_array_equal(policy_actions(source["policy"], obs), expected)
    np.testing.assert_array_equal(Evaluator(source).policy_actions(obs), expected)
    np.testing.assert_array_equal(expected, [[0.125], [0.125]])
    assert capabilities() == _native.capabilities()


@pytest.mark.parametrize("mutation", [
    lambda d: d.update(unknown=True),
    lambda d: d.pop("schema_version"),
    lambda d: d.update(schema_version=0),
    lambda d: d.update(schema_version=2),
    lambda d: d.update(schema_version=1.0),
    lambda d: d.update(schema_version=True),
    lambda d: d.update(schema_version="1"),
    lambda d: d.pop("task"),
    lambda d: d.update(task=[]),
    lambda d: d.update(task=None),
    lambda d: d.pop("policy"),
    lambda d: d.update(policy="mlp"),
    lambda d: d.update(policy=None),
    lambda d: d.pop("xml_path"),
    lambda d: d.update(xml_path=""),
    lambda d: d.update(xml_path=None),
    lambda d: d.update(xml_path=17),
    lambda d: d.update(eval_defaults=[]),
    lambda d: d.update(eval_defaults=None),
    lambda d: d.update(eval_defaults={"threads": 8}),
    lambda d: d.update(eval_defaults={"steps": 3}),
])
def test_descriptor_fields_version_and_default_keys_are_strict(mutation):
    source = descriptor()
    mutation(source)
    with pytest.raises((ValueError, TypeError)):
        load_descriptor(source)


def test_empty_json_xml_path_is_rejected_before_path_resolution(tmp_path):
    source = descriptor()
    source["xml_path"] = ""
    config_file = tmp_path / "invalid.json"
    config_file.write_text(json.dumps(source), encoding="utf8")
    with pytest.raises(ValueError, match="xml_path"):
        load_descriptor(config_file)


@pytest.mark.parametrize("source", [None, [], 42, True, object()])
def test_unsupported_descriptor_source_type(source):
    with pytest.raises(TypeError):
        load_descriptor(source)


@pytest.mark.parametrize("value", [None, [], True, "descriptor"])
def test_json_root_must_be_mapping(tmp_path, value):
    config_file = tmp_path / "invalid.json"
    config_file.write_text(json.dumps(value), encoding="utf8")
    with pytest.raises(ValueError):
        load_descriptor(config_file)


def test_descriptor_read_errors_are_not_hidden(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_descriptor(tmp_path / "missing.json")
    config_file = tmp_path / "bad.json"
    config_file.write_text("{invalid json", encoding="utf8")
    with pytest.raises(json.JSONDecodeError):
        load_descriptor(config_file)


@pytest.mark.parametrize("key,value", [
    ("num_envs", 0), ("num_envs", -1), ("num_envs", True), ("num_envs", 1.5),
    ("rollout_steps", 0), ("rollout_steps", -1), ("rollout_steps", 2.5),
    ("episode_steps", 0), ("episode_steps", -1),
    ("control_dt", 0.0), ("control_dt", -0.002), ("control_dt", 0.003),
    ("control_dt", float("nan")), ("control_dt", float("inf")),
    ("seed", -1), ("seed", True), ("seed", 1.5), ("seed", 2 ** 64),
    ("trace_worlds", [3]), ("trace_worlds", [-1]), ("trace_worlds", [1, 1]),
    ("trace_worlds", ["0"]),
])
def test_invalid_default_values_fail_through_public_api(key, value):
    source = descriptor()
    source["eval_defaults"][key] = value
    with pytest.raises(ERRORS):
        Evaluator(source).evaluate()


@pytest.mark.parametrize("mutation", [
    lambda d: d["task"].update(unrecognized=True),
    lambda d: d["task"].update(type="unsupported"),
    lambda d: d["task"].update(config={"unknown": 1.0}),
    lambda d: d["policy"].update(unrecognized=True),
    lambda d: d["policy"].update(type="unsupported"),
    lambda d: d.update(physics_dt=-0.001),
    lambda d: d.update(physics_dt=float("nan")),
])
def test_native_descriptor_validation_is_preserved_by_wrapper(mutation):
    source = descriptor()
    mutation(source)
    with pytest.raises(ERRORS):
        Evaluator(source)


def test_unknown_public_evaluation_argument_is_rejected():
    with pytest.raises(TypeError):
        Evaluator(descriptor()).evaluate(threads=2)
