"""Resolve auditable PPO execution settings outside model-source contracts.

Only the performance fields below are selected here. Task semantics, networks,
learning options and persistence remain owned by the existing training entry.
"""
from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType


RUNTIME_FIELDS = ("num_envs", "batch_size", "num_minibatches", "unroll_length",
                  "num_updates_per_batch")
PROFILE_CHOICES = ("auto", "throughput512", "legacy", "checkpoint")
_REQUESTS = frozenset(PROFILE_CHOICES)
_THROUGHPUT512 = MappingProxyType(dict(num_envs=512, batch_size=512, num_minibatches=1,
                                     unroll_length=16, num_updates_per_batch=1))


def _validated_settings(options):
    missing = [field for field in RUNTIME_FIELDS if field not in options]
    if missing:
        raise ValueError("Incomplete PPO performance settings: " + ", ".join(missing))
    values = {field: options[field] for field in RUNTIME_FIELDS}
    for field, value in values.items():
        if type(value) is not int or value <= 0:
            raise ValueError(f"PPO {field} must be a positive integer")
    if (values["batch_size"] * values["num_minibatches"]) % values["num_envs"]:
        raise ValueError("PPO num_envs must divide batch_size * num_minibatches")
    return values


@dataclass(frozen=True, slots=True)
class RuntimeProfile:
    """An immutable selection; as_dict returns an independent JSON-safe record."""

    requested: str
    effective: str
    source: str
    settings: Mapping
    checkpoint_config_path: str | None = None
    overrides_checkpoint: bool = False

    def __post_init__(self):
        if not isinstance(self.settings, Mapping):
            raise TypeError("Profile settings must be a mapping")
        settings = dict(self.settings)
        if settings:
            if set(settings) != set(RUNTIME_FIELDS):
                raise ValueError("Profile settings must contain exactly the PPO performance fields")
            settings = _validated_settings(settings)
        elif self.effective not in ("legacy", "legacy_missing_config"):
            raise ValueError("An active runtime profile needs complete performance settings")
        object.__setattr__(self, "settings", MappingProxyType(settings))

    def as_dict(self):
        record = {"requested": self.requested, "effective": self.effective,
                  "source": self.source, "settings": dict(self.settings),
                  "overrides_checkpoint": self.overrides_checkpoint}
        if self.checkpoint_config_path is not None:
            record["checkpoint_config_path"] = self.checkpoint_config_path
        return record


def _checkpoint_config_path(restore):
    if isinstance(restore, str) and not restore.strip():
        raise ValueError("restore must identify a checkpoint path")
    checkpoint = Path(restore).expanduser().resolve()
    if len(checkpoint.parents) < 2:
        raise ValueError("restore must identify a checkpoint below its run directory")
    return checkpoint.parent.parent / "ppo_config.json"


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate field in checkpoint PPO configuration: " + key)
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError("Nonfinite JSON constant in checkpoint PPO configuration: " + value)


def resolve_profile(mode, restore=None, requested="auto"):
    """Choose a preset, inherit a checkpoint, or explicitly retain entry defaults.

    A missing legacy config is distinct from malformed existing metadata. An
    explicit throughput preset intentionally bypasses old performance settings.
    This function reads metadata only and never rewrites a run or checkpoint.
    """
    if not isinstance(mode, str) or not mode:
        raise ValueError("mode must be a nonempty string")
    if not isinstance(requested, str) or requested not in _REQUESTS:
        raise ValueError("requested must be auto, throughput512, legacy or checkpoint")
    if requested == "throughput512":
        if mode != "leg_force_bc":
            raise ValueError("throughput512 is validated for leg_force_bc only")
        return RuntimeProfile(requested, "throughput512", "builtin", _THROUGHPUT512,
                              overrides_checkpoint=restore is not None)
    if requested == "legacy":
        return RuntimeProfile(requested, "legacy", "entrypoint", {})
    if restore is None:
        if requested == "checkpoint":
            raise ValueError("The checkpoint runtime profile requires restore")
        if mode == "leg_force_bc":
            return RuntimeProfile(requested, "throughput512", "builtin", _THROUGHPUT512)
        return RuntimeProfile(requested, "legacy", "entrypoint", {})
    config_path = _checkpoint_config_path(restore)
    try:
        source = config_path.read_text(encoding="utf8")
    except FileNotFoundError:
        if requested == "checkpoint":
            raise ValueError("Checkpoint PPO configuration is missing: " + str(config_path)) from None
        return RuntimeProfile(requested, "legacy_missing_config", "entrypoint", {}, str(config_path))
    except (OSError, UnicodeError) as error:
        raise ValueError("Cannot read checkpoint PPO configuration: " + str(config_path)) from error
    try:
        config = json.loads(source, object_pairs_hook=_json_object, parse_constant=_invalid_constant)
        if not isinstance(config, dict):
            raise ValueError("Checkpoint PPO configuration must be an object")
        settings = _validated_settings(config)
    except (ValueError, UnicodeError) as error:
        raise ValueError("Invalid checkpoint PPO configuration: " + str(config_path)) from error
    return RuntimeProfile(requested, "checkpoint", "checkpoint", settings, str(config_path))


def apply_profile(profile, options):
    """Copy and merge once, then validate the exact settings used and persisted."""
    if not isinstance(profile, RuntimeProfile):
        raise TypeError("profile must be a RuntimeProfile")
    if not isinstance(options, Mapping):
        raise TypeError("PPO options must be a mapping")
    result = dict(options)
    result.update(profile.settings)
    _validated_settings(result)
    return result
