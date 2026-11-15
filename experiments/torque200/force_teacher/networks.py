"""Shared absolute-torque skip head. ML dependencies are loaded only by the factory."""

POLICY_HEAD_VERSION = "wheel_logit_residual_v1"
WHEEL_RESIDUAL_SCALE = 0.05
PREVIOUS_TORQUE_CLIP = 1e-5


def transform_logits(raw_logits, raw_obs, xp=None, *, wheel_head="residual", wheel_residual_scale=WHEEL_RESIDUAL_SCALE):
    """Preserve legs/scale logits and add the physical previous-wheel-torque skip."""
    if xp is None:
        import numpy as xp
    if raw_obs.shape[-1] != 24 or raw_logits.shape[-1] not in (4, 8):
        raise ValueError("Force skip head requires 24D raw observations and 4/8 logits")
    if wheel_head not in ("absolute", "residual"):
        raise ValueError("Unknown force policy head")
    if not 0 < wheel_residual_scale <= 1:
        raise ValueError("Invalid wheel residual scale")
    if wheel_head == "absolute":
        return raw_logits
    previous = xp.clip(raw_obs[..., 22:24], -1 + PREVIOUS_TORQUE_CLIP, 1 - PREVIOUS_TORQUE_CLIP)
    wheel = xp.arctanh(previous) + wheel_residual_scale * raw_logits[..., 2:4]
    return xp.concatenate((raw_logits[..., :2], wheel, raw_logits[..., 4:]), axis=-1)


def make_ppo_networks(observation_size, action_size, preprocess_observations_fn=None,
                      policy_hidden_layer_sizes=(64, 64), value_hidden_layer_sizes=(512, 512, 512, 512),
                      policy_obs_key="state", value_obs_key="privileged_state",
                      wheel_head="absolute", wheel_residual_scale=WHEEL_RESIDUAL_SCALE):
    """Brax-compatible factory; inference, BC and PPO share the identical head."""
    from collections.abc import Mapping
    from dataclasses import replace
    import jax.numpy as jp
    from brax.training.agents.ppo import networks
    if action_size != 4:
        raise ValueError("Force policy requires four physical action channels")
    kwargs = dict(policy_hidden_layer_sizes=policy_hidden_layer_sizes,
                  value_hidden_layer_sizes=value_hidden_layer_sizes,
                  policy_obs_key=policy_obs_key, value_obs_key=value_obs_key)
    if preprocess_observations_fn is not None:
        kwargs["preprocess_observations_fn"] = preprocess_observations_fn
    base = networks.make_ppo_networks(observation_size, action_size, **kwargs)
    obs_key = policy_obs_key
    original_apply = base.policy_network.apply
    def apply(normalizer, params, observation):
        raw = observation[obs_key] if isinstance(observation, Mapping) else observation
        logits = original_apply(normalizer, params, observation)
        return transform_logits(logits, raw, xp=jp, wheel_head=wheel_head, wheel_residual_scale=wheel_residual_scale)
    return base.replace(policy_network=replace(base.policy_network, apply=apply))
