"""Small Python interface; configuration and results cross the native seam once."""
from collections.abc import Mapping
import importlib
import json
from pathlib import Path


def _extension():
    try:
        return importlib.import_module("native_eval._native")
    except ImportError as exc:
        raise RuntimeError(
            "Native evaluator is unavailable. Build with experiments/native_eval/build-wsl.sh "
            "and use the printed NATIVE_EVAL_PYTHONPATH. Original import error: " + str(exc)
        ) from exc


def capabilities():
    return _extension().capabilities()


def policy_actions(policy, observations):
    """Validate and execute a supported policy without loading a physics model."""
    return _extension().policy_actions(policy, observations)


def load_descriptor(source):
    if isinstance(source, (str, Path)):
        source = Path(source).expanduser().resolve()
        value = json.loads(source.read_text(encoding="utf8"))
        if isinstance(value, dict) and isinstance(value.get("xml_path"), str) and value["xml_path"]:
            value["xml_path"] = str((source.parent / value["xml_path"]).resolve())
    elif isinstance(source, Mapping):
        value = dict(source)
    else:
        raise TypeError("Descriptor must be a JSON path or mapping")
    allowed = {"schema_version", "xml_path", "task", "policy", "physics_dt", "eval_defaults", "metadata"}
    if not isinstance(value, dict) or set(value) - allowed:
        raise ValueError("Unknown descriptor fields or invalid descriptor")
    if type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise ValueError("Expected native evaluator descriptor schema_version=1")
    for name in ("task", "policy"):
        if not isinstance(value.get(name), dict):
            raise ValueError(f"Descriptor needs a {name} mapping")
    if not isinstance(value.get("xml_path"), str) or not value["xml_path"]:
        raise ValueError("Descriptor needs xml_path")
    if not isinstance(value.get("eval_defaults", {}), dict):
        raise ValueError("eval_defaults must be a mapping")
    unknown = set(value.get("eval_defaults", {})) - {
        "num_envs", "rollout_steps", "episode_steps", "control_dt", "seed", "trace_worlds"}
    if unknown:
        raise ValueError("Unknown eval_defaults: " + ", ".join(sorted(unknown)))
    return value


class Evaluator:
    """Own a reusable native model/task/policy; evaluate never calls back to Python."""

    def __init__(self, descriptor):
        self.descriptor = load_descriptor(descriptor)
        value = self.descriptor
        self._engine = _extension().Evaluator(value["xml_path"], value["task"], value["policy"],
                                              value.get("physics_dt", 0.0))

    def evaluate(self, initial_states=None, *, num_envs=None, steps=None, episode_steps=None,
                 control_dt=None, seed=None, trace_worlds=None):
        options = dict(self.descriptor.get("eval_defaults", {}))
        changes = {"num_envs": num_envs, "rollout_steps": steps, "episode_steps": episode_steps,
                   "control_dt": control_dt, "seed": seed, "trace_worlds": trace_worlds}
        options.update({key: value for key, value in changes.items() if value is not None})
        if initial_states is None:
            initial_states = [{}]
        return self._engine.evaluate(initial_states, options)

    def cancel(self):
        self._engine.cancel()

    def policy_actions(self, observations):
        return self._engine.policy_actions(observations)

    @property
    def model_info(self):
        return {name: getattr(self._engine, name) for name in
                ("nq", "nv", "nu", "physics_dt", "observation_size", "action_size", "task_name")}
