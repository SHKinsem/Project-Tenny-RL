"""Brax seam contracts without importing JAX or running a training loop."""
from contextlib import contextmanager
from copy import deepcopy
from functools import partial
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from native_eval import brax


class StopEvaluation(Exception):
    pass


class PoisonCritic:
    def __array__(self, *args, **kwargs):
        raise AssertionError("The evaluation must not transfer or inspect the critic")


def parameters():
    stats = SimpleNamespace(mean={"state": np.array([.1, -.2, .3], dtype=np.float32)},
                            std={"state": np.array([.5, 1., 2.], dtype=np.float32)})
    actor = {"params": {"hidden_0": {
        "kernel": np.array([[.2, .1, 1., 2.], [.3, -.4, 3., 4.], [.5, .6, 5., 6.]], dtype=np.float32),
        "bias": np.array([.05, -.15, 10., 20.], dtype=np.float32)}}}
    return stats, actor, PoisonCritic()


def policy_template():
    return dict(type="mlp", mean=[99.] * 3, std=[88.] * 3,
                layers=[dict(weights=[[0.] * 3] * 2, bias=[77.] * 2, activation="identity")],
                action_size=2, output_activation="tanh")


def reset_state(count):
    # Distinct worlds and float32 anchors expose reordering/rounding mistakes.
    return SimpleNamespace(
        data=SimpleNamespace(qpos=np.arange(count * 4, dtype=np.float64).reshape(count, 4) + .125,
                             qvel=np.arange(count * 3, dtype=np.float64).reshape(count, 3) - .375),
        info={"command": np.arange(count * 4, dtype=np.float32).reshape(count, 4) / np.float32(7),
              "position_start": np.arange(count, dtype=np.float32) + np.float32(.1234567),
              "sinyaw_start": np.arange(count, dtype=np.float32) + np.float32(-.2345678),
              "sin_balance_zero": np.arange(count, dtype=np.float32) + np.float32(.3456789)},
        metrics={"score": np.zeros(count), "cost_per_step": np.zeros(count)})


def result_value():
    return {"returns": np.array([10., 20., 30.]), "lengths": np.array([2, 3, 4]),
            "episode_metrics": {"reward": np.array([10., 20., 30.]),
                                "score": np.array([3., 6., 9.]),
                                "cost_per_step": np.array([4., 9., 8.])},
            "cancelled": False, "control_steps": 15, "physics_steps": 60, "wall_seconds": .25}


class Harness:
    def __init__(self, monkeypatch):
        self.gets, self.splits, self.jits, self.reset_calls = [], [], [], []
        self.states, self.conversions, self.engines, self.reports = [], [], [], []
        self.spans, self.stop_calls = [], 0
        self.stop_at = None
        self.during_rollout = None
        self.rollout_error = None
        self.result = result_value()
        self.ticks = iter((10., 12., 20., 23., 30., 34., 40., 45., 50., 56.))
        fake_jax = ModuleType("jax")
        fake_jax.random = SimpleNamespace(split=self.split)
        fake_jax.jit = self.jit
        fake_jax.device_get = self.device_get
        self.jax = fake_jax
        monkeypatch.setitem(sys.modules, "jax", fake_jax)
        monkeypatch.setattr(brax, "time", SimpleNamespace(monotonic=lambda: next(self.ticks)))
        harness = self

        class FakeEngine:
            def __init__(self, descriptor):
                self.descriptor = deepcopy(descriptor)
                self.calls, self.cancel_calls = [], 0
                harness.engines.append(self)

            def evaluate(self, states, **options):
                self.calls.append((states, options))
                if harness.during_rollout is not None:
                    harness.during_rollout(self)
                if harness.rollout_error is not None:
                    raise harness.rollout_error
                result = deepcopy(harness.result)
                if self.cancel_calls:
                    result["cancelled"] = True
                return result

            def cancel(self):
                self.cancel_calls += 1

        monkeypatch.setattr(brax, "Evaluator", FakeEngine)

    def split(self, key, count=2):
        self.splits.append((key, count))
        return tuple(key * 10 + index + 1 for index in range(count))

    def jit(self, function):
        self.jits.append(function)
        return function

    def device_get(self, value):
        self.gets.append(value)
        # Checking the tree ensures a regression passing the critic would fail.
        def check(item):
            if isinstance(item, PoisonCritic):
                raise AssertionError("Critic reached device_get")
            if isinstance(item, (tuple, list)):
                for child in item:
                    check(child)
            elif isinstance(item, dict):
                for child in item.values():
                    check(child)
        check(value)
        return value

    def reset(self, keys):
        self.reset_calls.append(keys)
        state = reset_state(len(keys))
        self.states.append(state)
        return state

    def convert(self, state, device_get):
        self.conversions.append((state, device_get))
        return brax.legwheel_initial_states(state, device_get)

    def stop(self):
        self.stop_calls += 1
        if self.stop_calls == self.stop_at:
            raise StopEvaluation("cooperative stop")

    @contextmanager
    def span(self, name):
        self.spans.append(("enter", name))
        try:
            yield
        finally:
            self.spans.append(("exit", name))

    def make(self, **changes):
        def forbidden_policy(*args, **kwargs):
            raise AssertionError("Native rollout must not call the Python policy")
        values = dict(eval_env=SimpleNamespace(reset=self.reset),
                      eval_policy_fn=partial(forbidden_policy, deterministic=True),
                      num_eval_envs=3, episode_length=5, action_repeat=1, key=7,
                      descriptor={"schema_version": 1, "xml_path": "unused.xml", "physics_dt": .0025,
                                  "task": {"type": "unused", "config": {}},
                                  "eval_defaults": {"control_dt": .01}},
                      policy_template=policy_template(), initial_states=self.convert,
                      check_stop=self.stop, span=self.span, report=self.reports.append)
        values.update(changes)
        return brax.BraxEvaluator(**values)


@pytest.fixture
def harness(monkeypatch):
    return Harness(monkeypatch)


def test_uses_current_actor_and_normalizer_and_never_transfers_critic(harness):
    adapter = harness.make()
    params = parameters()
    adapter.run_evaluation(params, {})
    first = harness.engines[0].descriptor["policy"]
    assert len(harness.gets[0]) == 2
    assert harness.gets[0][0] is params[0]
    assert harness.gets[0][1] is params[1]
    np.testing.assert_array_equal(first["layers"][0]["weights"], params[1]["params"]["hidden_0"]["kernel"][:, :2].T)
    params[1]["params"]["hidden_0"]["bias"][:2] += .7
    params[0].mean["state"] += .4
    params[0].std["state"] *= 1.5
    adapter.run_evaluation(params, {})
    second = harness.engines[1].descriptor["policy"]
    assert second["layers"][0]["bias"] != first["layers"][0]["bias"]
    np.testing.assert_array_equal(second["mean"], params[0].mean["state"])
    np.testing.assert_array_equal(second["std"], params[0].std["state"])
    assert harness.reports[0]["evaluation_actor_sha256"] != harness.reports[1]["evaluation_actor_sha256"]
    assert harness.reports[1]["native_evaluations"] == 2
    assert adapter.last_result is not None


def test_reset_key_flow_exact_transfer_and_native_workload_options(harness):
    adapter = harness.make()
    assert len(harness.jits) == 1
    assert not harness.reset_calls
    adapter.run_evaluation(parameters(), {})
    adapter.run_evaluation(parameters(), {})
    assert harness.splits == [(7, 2), (72, 3), (71, 2), (712, 3)]
    assert harness.reset_calls == [(721, 722, 723), (7121, 7122, 7123)]
    assert len(harness.conversions) == 2
    assert harness.conversions[0][0] is harness.states[0]
    assert harness.conversions[0][1] == harness.device_get
    engine = harness.engines[0]
    states, options = engine.calls[0]
    assert options == dict(num_envs=3, steps=5, episode_steps=5, trace_worlds=[])
    original = harness.states[0]
    for world, state in enumerate(states):
        np.testing.assert_array_equal(state["qpos"], original.data.qpos[world])
        np.testing.assert_array_equal(state["qvel"], original.data.qvel[world])
        np.testing.assert_array_equal(state["command"], original.info["command"][world])
        assert state["extra"] == {name: [float(original.info[name][world])]
                                   for name in ("position_start", "sinyaw_start", "sin_balance_zero")}
    assert engine.descriptor["eval_defaults"]["control_dt"] == .01
    phases = [name for kind, name in harness.spans if kind == "enter"]
    assert phases == ["evaluation.params_to_host", "evaluation.reset_to_host",
                      "evaluation.native_prepare", "evaluation.native_rollout"] * 2
    assert [name for kind, name in harness.spans if kind == "exit"] == phases


def test_metric_names_aggregation_normalization_and_training_metric_merge(harness):
    adapter = harness.make()
    metrics = adapter.run_evaluation(parameters(), {"training/loss": .75, "eval/episode_reward": -999.})
    expected_values = harness.result["episode_metrics"]
    lengths = harness.result["lengths"]
    for name, values in expected_values.items():
        normalized = values / lengths if name.endswith("per_step") else values
        assert metrics[f"eval/episode_{name}"] == pytest.approx(np.mean(normalized))
        assert metrics[f"eval/episode_{name}_std"] == pytest.approx(np.std(normalized))
    assert metrics["training/loss"] == .75
    assert metrics["eval/avg_episode_length"] == 3
    assert metrics["eval/std_episode_length"] == pytest.approx(np.std(lengths))
    assert metrics["eval/epoch_eval_time"] == 2
    assert metrics["eval/sps"] == 7.5
    assert metrics["eval/walltime"] == 2
    assert harness.reports[0] == dict(evaluation_backend="native_mujoco_cpp", native_evaluations=1,
                                     evaluation_actor_sha256=harness.reports[0]["evaluation_actor_sha256"],
                                     native_eval_control_steps=15, native_eval_physics_steps=60,
                                     native_eval_seconds=.25, native_eval_complete_call_seconds=2.)
    again = adapter.run_evaluation(parameters(), {})
    assert again["eval/epoch_eval_time"] == 3
    assert again["eval/walltime"] == 5


def test_aggregate_false_keeps_per_world_raw_values_for_both_suffixes(harness):
    adapter = harness.make()
    metrics = adapter.run_evaluation(parameters(), {}, aggregate_episodes=False)
    for name, values in harness.result["episode_metrics"].items():
        np.testing.assert_array_equal(metrics[f"eval/episode_{name}"], values)
        np.testing.assert_array_equal(metrics[f"eval/episode_{name}_std"], values)
    # Per-step normalization also stays disabled, matching Brax's _agg_fn.
    np.testing.assert_array_equal(metrics["eval/episode_cost_per_step"], [4., 9., 8.])


@pytest.mark.parametrize("mutation", ["extra", "missing", "different_reset"])
def test_task_metric_schema_mismatch_is_rejected_before_publication(harness, mutation):
    adapter = harness.make()
    if mutation == "extra":
        harness.result["episode_metrics"]["unexpected"] = np.zeros(3)
    elif mutation == "missing":
        del harness.result["episode_metrics"]["score"]
    else:
        original_reset = adapter._reset
        def changed_reset(keys):
            state = original_reset(keys)
            state.metrics["reset_only"] = np.zeros(3)
            return state
        adapter._reset = changed_reset
    with pytest.raises(ValueError, match="metric names differ"):
        adapter.run_evaluation(parameters(), {})
    assert not harness.reports
    assert adapter.last_result is None
    assert adapter._evaluations == 0
    assert adapter._eval_walltime == 0
    assert adapter._active is None


@pytest.mark.parametrize("bad", [
    np.array(1.), np.array([1., 2.]), np.ones((3, 1)),
    np.array([1., np.nan, 2.]), np.array([1., 2., np.inf]),
])
def test_invalid_metric_vector_is_rejected_without_publication(harness, bad):
    harness.result["episode_metrics"]["score"] = bad
    adapter = harness.make()
    with pytest.raises(ValueError, match="Invalid native episode metric: score"):
        adapter.run_evaluation(parameters(), {})
    assert not harness.reports
    assert adapter.last_result is None
    assert adapter._eval_walltime == 0
    assert adapter._evaluations == 0


@pytest.mark.parametrize("bad", [
    np.array(2), np.array([2, 3]), np.ones((3, 1)),
    np.array([0., 3., 4.]), np.array([-1., 3., 4.]), np.array([2., 3., 6.]),
    np.array([2., np.nan, 4.]), np.array([2., 3., np.inf]),
])
def test_invalid_first_episode_lengths_are_rejected_without_publication(harness, bad):
    harness.result["lengths"] = bad
    adapter = harness.make()
    with pytest.raises(ValueError, match="Invalid native first-episode lengths"):
        adapter.run_evaluation(parameters(), {})
    assert not harness.reports
    assert adapter.last_result is None
    assert adapter._eval_walltime == 0
    assert adapter._evaluations == 0


def test_cancelled_partial_episodes_do_not_publish_metrics(harness):
    adapter = harness.make()
    adapter.cancel()  # Before activation this does not touch any engine.
    assert not harness.engines
    harness.during_rollout = lambda engine: adapter.cancel()
    with pytest.raises(RuntimeError, match="cancelled; partial episodes"):
        adapter.run_evaluation(parameters(), {})
    assert harness.engines[0].cancel_calls == 1
    assert adapter._active is None
    assert adapter.last_result is None
    assert adapter._evaluations == 0
    assert not harness.reports
    adapter.cancel()
    assert harness.engines[0].cancel_calls == 1


@pytest.mark.parametrize("boundary", [1, 2, 3])
def test_cooperative_stop_at_each_safe_boundary_does_not_publish(harness, boundary):
    harness.stop_at = boundary
    adapter = harness.make()
    with pytest.raises(StopEvaluation, match="cooperative stop"):
        adapter.run_evaluation(parameters(), {})
    assert harness.stop_calls == boundary
    assert adapter._active is None
    assert adapter.last_result is None
    assert not harness.reports
    assert adapter._eval_walltime == 0
    assert adapter._evaluations == 0
    assert len(harness.reset_calls) == (0 if boundary == 1 else 1)
    assert len(harness.engines) == (1 if boundary == 3 else 0)


def test_engine_failure_clears_active_and_does_not_publish(harness):
    adapter = harness.make()
    harness.rollout_error = ValueError("physics failure")
    with pytest.raises(ValueError, match="physics failure"):
        adapter.run_evaluation(parameters(), {})
    assert adapter._active is None
    assert adapter._evaluations == 0
    assert adapter.last_result is None
    assert not harness.reports


def test_instances_keep_separate_counts_clocks_and_last_result(harness):
    first, second = harness.make(), harness.make()
    first.run_evaluation(parameters(), {})
    first.run_evaluation(parameters(), {})
    assert first._evaluations == 2 and first._eval_walltime == 5
    assert second._evaluations == 0 and second._eval_walltime == 0
    assert second.last_result is None
    metrics = second.run_evaluation(parameters(), {})
    assert metrics["eval/walltime"] == 4
    assert harness.reports[-1]["native_evaluations"] == 1
    assert first.last_result is not second.last_result


@pytest.mark.parametrize("arguments", [
    {"num_eval_envs": 0}, {"num_eval_envs": True}, {"num_eval_envs": 2.5},
    {"episode_length": 0}, {"episode_length": True}, {"episode_length": 3.5},
    {"action_repeat": 2}, {"action_repeat": 0},
    {"eval_policy_fn": partial(lambda: None, deterministic=False)},
    {"eval_policy_fn": lambda: None},
])
def test_incompatible_constructor_options_rejected(harness, arguments):
    with pytest.raises(ValueError):
        harness.make(**arguments)
    assert not harness.engines
