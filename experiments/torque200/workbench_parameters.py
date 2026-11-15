"""Small, simulator-free contract for user-editable training options.

Runtime presets remain owned by runtime_defaults. Empty user overrides retain
the existing defaults and five-field checkpoint inheritance unchanged.
"""
from collections.abc import Mapping
import json
import math
from pathlib import Path

if __package__:
    from .control_modes import mode_spec
    from .runtime_defaults import resolve_profile, apply_profile, RUNTIME_FIELDS
else:
    from control_modes import mode_spec
    from runtime_defaults import resolve_profile, apply_profile, RUNTIME_FIELDS


# Bounds protect the local interactive workflow from accidental unbounded jobs;
# they do not claim that every accepted configuration fits the available GPU.
FIELDS = (
    dict(key="num_envs", label="并行环境", type="integer", min=1, max=8192, step=1, group="并行"),
    dict(key="batch_size", label="批量大小", type="integer", min=1, max=8192, step=1, group="并行"),
    dict(key="num_minibatches", label="小批次数", type="integer", min=1, max=64, step=1, group="并行"),
    dict(key="unroll_length", label="展开步数", type="integer", min=1, max=1024, step=1, group="并行"),
    dict(key="num_updates_per_batch", label="每批更新", type="integer", min=1, max=64, step=1, group="PPO"),
    dict(key="learning_rate", label="学习率", type="number", min=1e-8, max=0.1, step="any", group="PPO"),
    dict(key="entropy_cost", label="熵系数", type="number", min=0, max=1, step="any", group="PPO"),
    dict(key="reward_scaling", label="奖励缩放", type="number", min=1e-6, max=100, step="any", group="PPO"),
    dict(key="clipping_epsilon", label="裁剪范围", type="number", min=0.001, max=1, step="any", group="PPO"),
    dict(key="max_grad_norm", label="梯度上限", type="number", min=1e-6, max=1000, step="any", group="PPO"),
    dict(key="num_eval_envs", label="评估环境", type="integer", min=1, max=128, step=1, group="评估"),
)
_BY_KEY = {field["key"]: field for field in FIELDS}
MAX_JSON_BYTES = 64 * 1024


def validate_parameters(overrides, base=None):
    """Validate actual JSON types and cross-field batch shape; return a copy."""
    if not isinstance(overrides, Mapping) or len(overrides) > len(FIELDS):
        raise ValueError("parameters must be an object of supported training fields")
    result = dict(overrides)
    unknown = set(result) - _BY_KEY.keys()
    if unknown:
        raise ValueError("Unsupported training parameter: " + ", ".join(sorted(map(str, unknown))))
    for key, value in result.items():
        field = _BY_KEY[key]
        if field["type"] == "integer":
            valid = type(value) is int
        else:
            valid = type(value) in (int, float)
        try:
            valid = valid and math.isfinite(value) and field["min"] <= value <= field["max"]
        except (OverflowError, TypeError):
            valid = False
        if not valid:
            raise ValueError(f"{key} must be a finite {field['type']} between {field['min']} and {field['max']}")
    combined = {**(dict(base) if base is not None else {}), **result}
    if all(key in combined for key in ("num_envs", "batch_size", "num_minibatches")):
        envs, batch, minibatches = (combined[key] for key in ("num_envs", "batch_size", "num_minibatches"))
        if any(type(value) is not int or value <= 0 for value in (envs, batch, minibatches)):
            raise ValueError("PPO batch shape must use positive integers")
        if (batch * minibatches) % envs:
            raise ValueError("并行环境数必须整除 批量大小 × 小批次数")
        # A batch is materialized before SGD; bound the editable product too.
        unroll = combined.get("unroll_length", 1)
        if type(unroll) is int and batch * minibatches * unroll > 16_777_216:
            raise ValueError("PPO rollout batch exceeds the interactive configuration limit")
    return result


def apply_parameters(config, overrides):
    if not isinstance(config, Mapping):
        raise ValueError("PPO configuration must be an object")
    values = validate_parameters(overrides, base=config)
    return {**dict(config), **values}


def default_parameters(mode):
    """Mirror the deployed entry's editable values without importing JAX."""
    spec = mode_spec(mode)
    position = mode == "original" or (spec["teacher"] and not spec["hybrid"])
    config = dict(num_envs=128, batch_size=8 if position else 128,
                  num_minibatches=16 if position else 1,
                  unroll_length=75 if position else 16, num_updates_per_batch=4 if position else 1,
                  learning_rate=3e-5 if mode == "leg_force_bc" else 1e-4,
                  entropy_cost=.001 if mode == "leg_force_bc" else .01,
                  reward_scaling=.1, clipping_epsilon=.3, max_grad_norm=1.,
                  num_eval_envs=8 if mode == "leg_force_bc" else 2)
    return apply_profile(resolve_profile(mode), config)


def inherited_parameters(checkpoint):
    """Only a previous explicit Workbench edit adds non-preset inheritance."""
    if checkpoint is None:
        return {}
    path = Path(checkpoint).parent.parent / "request.json"
    if not path.exists():
        return {}
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("Checkpoint training request is not a bounded regular file")
    try:
        request = json.loads(path.read_text(encoding="utf-8"))
        values = request.get("parameters", {}) if isinstance(request, dict) else None
        return validate_parameters(values)
    except (ValueError, UnicodeError, OSError) as error:
        raise ValueError("Invalid inherited training parameters") from error


def resolve_parameters(mode, checkpoint=None, overrides=None):
    profile = resolve_profile(mode, restore=checkpoint)
    base = apply_profile(profile, default_parameters(mode))
    edits = {**inherited_parameters(checkpoint), **validate_parameters({} if overrides is None else overrides)}
    effective = apply_parameters(base, edits)
    return effective, edits


def parameter_schema(mode, *, checkpoint=None, overrides=None, locked=False):
    effective, edits = resolve_parameters(mode, checkpoint, overrides)
    return dict(schemaVersion=1, mode=mode, checkpoint=str(checkpoint) if checkpoint else None,
                effective=effective, overrides=edits, locked=bool(locked),
                fields=[{**field, "value": effective[field["key"]], "editable": not locked} for field in FIELDS],
                resumeSemantics={"optimizerRestored": False})
