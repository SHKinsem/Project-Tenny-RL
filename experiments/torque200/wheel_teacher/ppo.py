"""Explicit student train/eval wiring without modifying the reference package."""


def train_student(*, env, eval_env, env_cfg, ppo_params, progress_fn, params=None):
    from functools import partial
    import inspect
    from brax.training.agents.ppo import networks, train
    from mujoco_playground import wrapper
    from legwheel.randomization import domain_randomize

    options = dict(ppo_params)
    factory = partial(networks.make_ppo_networks, **dict(options.pop("network_factory")))
    supported = inspect.signature(train.train).parameters
    required = {"restore_params", "eval_env", "policy_params_fn"}
    if not required.issubset(supported):
        raise RuntimeError("Installed Brax lacks explicit student evaluation/BC initialization")
    unknown = set(options) - set(supported)
    if unknown:
        raise ValueError("Unsupported student PPO options: " + ", ".join(sorted(unknown)))
    if params is not None:
        if options.get("restore_checkpoint_path"):
            raise ValueError("Choose BC initialization or checkpoint resume, not both")
        options["restore_params"] = params
    return train.train(environment=env, eval_env=eval_env,
                       wrap_env_fn=wrapper.wrap_for_brax_training,
                       network_factory=factory, randomization_fn=domain_randomize,
                       progress_fn=progress_fn, **options)
