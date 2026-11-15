"""Brax evaluator seam; reset/parameter transfer stay outside the native rollout."""
from contextlib import nullcontext
import hashlib
import time

import numpy as np

from .api import Evaluator
from .brax_policy import policy_from_brax


def _span_factory(name):
    return nullcontext()


class BraxEvaluator:
    """Keep Brax's metric contract while evaluating the current actor natively.

    The task adapter supplies ``initial_states`` to convert one batched reset.
    Neither that conversion nor Python inference runs in the physics loop.
    """

    def __init__(self, eval_env, eval_policy_fn, num_eval_envs, episode_length,
                 action_repeat, key, *, descriptor, policy_template, initial_states,
                 check_stop=None, span=None, report=None):
        import jax
        if type(num_eval_envs) is not int or num_eval_envs < 1:
            raise ValueError("num_eval_envs must be positive")
        if type(episode_length) is not int or episode_length < 1 or action_repeat != 1:
            raise ValueError("Native Brax evaluation requires positive length and action_repeat=1")
        if getattr(eval_policy_fn, "keywords", {}).get("deterministic") is not True:
            raise ValueError("Native Brax evaluation requires an explicitly deterministic policy")
        self._jax = jax
        self._key = key
        self._reset = jax.jit(eval_env.reset)
        self._count = num_eval_envs
        self._length = episode_length
        self._descriptor = descriptor
        self._template = policy_template
        self._initial_states = initial_states
        self._check_stop = check_stop or (lambda: None)
        self._span = span or _span_factory
        self._report = report or (lambda value: None)
        self._eval_walltime = 0.0
        self._evaluations = 0
        self._active = None
        self.last_result = None

    def cancel(self):
        if self._active is not None:
            self._active.cancel()

    def run_evaluation(self, policy_params, training_metrics, aggregate_episodes=True):
        self._check_stop()
        self._key, unroll_key = self._jax.random.split(self._key)
        start = time.monotonic()
        with self._span("evaluation.params_to_host"):
            # The critic is unused by deterministic evaluation and remains on GPU.
            host = self._jax.device_get((policy_params[0], policy_params[1]))
            policy = policy_from_brax(host, self._template)
            digest = hashlib.sha256()
            for layer in policy["layers"]:
                for field in ("weights", "bias"):
                    digest.update(np.asarray(layer[field], dtype=np.float32).tobytes())
            actor_sha256 = digest.hexdigest()
        with self._span("evaluation.reset_to_host"):
            state = self._reset(self._jax.random.split(unroll_key, self._count))
            states = self._initial_states(state, self._jax.device_get)
            expected_metrics = set(state.metrics) | {"reward"}
        self._check_stop()
        with self._span("evaluation.native_prepare"):
            descriptor = dict(self._descriptor, policy=policy)
            engine = Evaluator(descriptor)
        self._active = engine
        try:
            with self._span("evaluation.native_rollout"):
                result = engine.evaluate(states, num_envs=self._count, steps=self._length,
                                         episode_steps=self._length, trace_worlds=[])
        finally:
            self._active = None
        self._check_stop()
        if result["cancelled"]:
            raise RuntimeError("Native evaluation cancelled; partial episodes are not evaluation metrics")
        values = result["episode_metrics"]
        if set(values) != expected_metrics:
            raise ValueError("Native and Brax task metric names differ: " +
                             repr(sorted(set(values) ^ expected_metrics)))
        for name, value in values.items():
            if np.shape(value) != (self._count,) or not np.isfinite(value).all():
                raise ValueError("Invalid native episode metric: " + name)
        elapsed = time.monotonic() - start
        lengths = np.asarray(result["lengths"], dtype=float)
        if (lengths.shape != (self._count,) or not np.isfinite(lengths).all() or
                np.any(lengths < 1) or np.any(lengths > self._length)):
            raise ValueError("Invalid native first-episode lengths")
        metrics = {}
        for function, suffix in ((np.mean, ""), (np.std, "_std")):
            for name, value in values.items():
                value = np.asarray(value)
                if aggregate_episodes:
                    metric = value / np.maximum(lengths, 1.) if name.endswith("per_step") else value
                    metric = function(metric)
                else:
                    metric = value
                metrics[f"eval/episode_{name}{suffix}"] = metric
        self._eval_walltime += elapsed
        self._evaluations += 1
        metrics.update({"eval/avg_episode_length": np.mean(lengths),
                        "eval/std_episode_length": np.std(lengths),
                        "eval/epoch_eval_time": elapsed,
                        "eval/sps": self._count * self._length / elapsed})
        self.last_result = result
        self._report({"evaluation_backend": "native_mujoco_cpp", "native_evaluations": self._evaluations,
                      "evaluation_actor_sha256": actor_sha256,
                      "native_eval_control_steps": result["control_steps"],
                      "native_eval_physics_steps": result["physics_steps"],
                      "native_eval_seconds": result["wall_seconds"],
                      "native_eval_complete_call_seconds": elapsed})
        return {"eval/walltime": self._eval_walltime, **training_metrics, **metrics}


def legwheel_initial_states(state, device_get):
    """Preserve the exact production reset, including its float32 reference anchors."""
    keys = ("command", "position_start", "sinyaw_start", "sin_balance_zero")
    qpos, qvel, info = device_get((state.data.qpos, state.data.qvel,
                                 {name: state.info[name] for name in keys}))
    return [{"qpos": qpos[i], "qvel": qvel[i], "command": info["command"][i],
             "extra": {name: [float(info[name][i])] for name in keys[1:]}}
            for i in range(len(qpos))]
