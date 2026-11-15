"""Force PPO with calibrated, fixed observation scales and deterministic evaluation."""


def configure_ppo(ppo_cfg):
    """Record the real Brax options before the console persists its config."""
    ppo_cfg.normalize_observations = True
    ppo_cfg.normalize_observations_mode = "ema"
    ppo_cfg.normalize_observations_std_eps = 0.0
    ppo_cfg.normalize_until_count = 0
    ppo_cfg.deterministic_eval = True
    ppo_cfg.entropy_cost = .001
    ppo_cfg.learning_rate = 3e-5
    return ppo_cfg


def fixed_normalizer(state):
    """Convert fitted Welford statistics to EMA variance without changing means."""
    import jax
    import jax.numpy as jp
    from brax.training.acme import running_statistics as rs
    variance = jax.tree.map(lambda std: jp.square(jp.asarray(std)), state.std)
    # Canonicalize before BC fitting so EMA's sqrt/clip is bitwise unchanged.
    std = jax.tree.map(lambda var: jp.clip(jp.sqrt(var), 1e-6, 1e6), variance)
    return state.replace(summed_variance=variance, std=std, std_eps=0.0,
                         mode=rs.NormalizationMode.EMA)


def normalizer_max_abs_change(reference, current):
    """Compare calibrated scales, excluding only the growing sample counter."""
    import jax
    import numpy as np
    maximum = 0.0
    for name in ("mean", "std", "summed_variance"):
        before, after = getattr(reference, name), getattr(current, name)
        if jax.tree_util.tree_structure(before) != jax.tree_util.tree_structure(after):
            raise ValueError("Normalizer structure changed: " + name)
        for old, new in zip(jax.tree.leaves(before), jax.tree.leaves(after), strict=True):
            old, new = np.asarray(old), np.asarray(new)
            if old.shape != new.shape or not np.isfinite(old).all() or not np.isfinite(new).all():
                raise ValueError("Invalid normalizer arrays: " + name)
            maximum = max(maximum, float(np.max(np.abs(new-old))))
    return maximum


def _restore_fixed_mode(state):
    """Orbax can omit Flax's static mode; validate EMA variance before restoring it."""
    import jax
    import jax.numpy as jp
    import numpy as np
    from brax.training.acme import running_statistics as rs
    from brax.training import types
    if float(state.std_eps) != 0.0:
        raise ValueError("Force checkpoint has incompatible normalization epsilon")
    for var, std in zip(jax.tree.leaves(state.summed_variance), jax.tree.leaves(state.std), strict=True):
        expected = np.asarray(jp.clip(jp.sqrt(jp.maximum(jp.asarray(var), 0)), 1e-6, 1e6))
        if not np.array_equal(expected, np.asarray(std)) or not np.isfinite(expected).all():
            raise ValueError("Force checkpoint is not calibrated fixed-EMA statistics")
    count = state.count
    if isinstance(count, dict) and set(count) == {"hi", "lo"}:
        count = types.UInt64(**count)
    return state.replace(mode=rs.NormalizationMode.EMA, count=count)


restore_fixed_normalizer = _restore_fixed_mode


def train_student(*, env, eval_env, env_cfg, ppo_params, progress_fn, params=None):
    from functools import partial
    import inspect
    from brax.training import checkpoint
    from brax.training.agents.ppo import train
    from force_teacher.networks import make_ppo_networks
    from mujoco_playground import wrapper

    options = dict(ppo_params)
    required = {"restore_params", "eval_env", "policy_params_fn", "deterministic_eval",
                "normalize_until_count", "normalize_observations_mode"}
    supported = inspect.signature(train.train).parameters
    if not required.issubset(supported):
        raise RuntimeError("Installed Brax lacks supported fixed-normalization PPO")
    for name, value in dict(normalize_observations=True, normalize_observations_mode="ema",
                            normalize_observations_std_eps=0.0, normalize_until_count=0,
                            deterministic_eval=True, entropy_cost=.001, learning_rate=3e-5).items():
        options[name] = value
    factory = partial(make_ppo_networks, **dict(options.pop("network_factory")))
    unknown = set(options) - set(supported)
    if unknown:
        raise ValueError("Unsupported force PPO options: " + ", ".join(sorted(unknown)))
    restore_path = options.pop("restore_checkpoint_path", None)
    if params is not None and restore_path:
        raise ValueError("Choose BC initialization or checkpoint resume, not both")
    if restore_path:
        params = checkpoint.load(restore_path)
    if params is None:
        raise ValueError("Force PPO requires a passed BC gate or a verified standing checkpoint")
    normalizer = _restore_fixed_mode(params[0])
    options["restore_params"] = (normalizer, params[1], params[2])
    return train.train(environment=env, eval_env=eval_env,
        wrap_env_fn=wrapper.wrap_for_brax_training, network_factory=factory,
        randomization_fn=None, progress_fn=progress_fn, **options)
