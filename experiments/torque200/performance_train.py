#!/usr/bin/env python3
"""Bounded torque200 PPO performance round, with checkpoint/export verification.

Uses the original training/export pipeline with the independent 24D NumPy kernel.
No balancing/convergence claim. Run on WSL ext4 in a fresh output directory.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MPLBACKEND", "Agg")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--restore", type=Path)
    parser.add_argument("--steps", type=int, default=32768)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-envs", type=int, default=2)
    parser.add_argument("--expected-platform", choices=("cpu", "gpu"))
    parser.add_argument("--export-atol", type=float, default=1e-3)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.repo.resolve()))

    import jax
    import numpy as np
    if args.expected_platform and any(d.platform != args.expected_platform for d in jax.devices()):
        raise RuntimeError(f"Unexpected devices: {jax.devices()}")
    from legwheel import LegWheelRobot, default_config
    from legwheel.deployment.extract import export_params, load
    from legwheel.training.ppo import compute_discounting, get_ppo_config, train_ppo
    from legwheel.utils import save_cpu_checkpoint
    from legwheel.deployment.numpy_policy import RL_Policy

    import legwheel
    if not Path(legwheel.__file__).resolve().is_relative_to(args.repo.resolve()):
        raise RuntimeError("Wrong legwheel source imported")
    cfg = default_config()
    params_cfg = get_ppo_config(
        cfg, num_evals=5, maxstep_candidate=args.steps,
        save_checkpoint_path=str(output / "checkpoints"),
        ppo_overrides={"batch_size": args.batch_size, "num_minibatches": 1,
                       "unroll_length": 16, "num_updates_per_batch": 1},
    )
    # Exactly the defaults applied by curriculum_learning, plus small eval batch.
    params_cfg.learning_rate = 1e-4
    params_cfg.reward_scaling = 0.1
    params_cfg.entropy_cost = 1e-2
    params_cfg.discounting = compute_discounting(cfg.ctrl_dt, 0.5)
    params_cfg.network_factory.policy_hidden_layer_sizes = [64, 64]
    params_cfg.network_factory.value_hidden_layer_sizes = [512, 512, 512, 512]
    params_cfg.num_eval_envs = args.eval_envs
    if args.restore:
        params_cfg.restore_checkpoint_path = str(args.restore.resolve())
    (output / "env_config.json").write_text(cfg.to_json_best_effort(indent=2))
    (output / "ppo_config.json").write_text(params_cfg.to_json_best_effort(indent=2))
    events = []

    def progress(steps, metrics):
        event = {"steps": int(steps), "metrics": {
            key: float(value) for key, value in metrics.items()
            if np.asarray(value).size == 1
        }}
        events.append(event)
        (output / "progress.json").write_text(json.dumps(events, indent=2))
        print(json.dumps(event), flush=True)

    started = time.monotonic()
    env = LegWheelRobot(config=cfg)
    make_inference, params, metrics = train_ppo(
        env=env, env_cfg=cfg, ppo_params=params_cfg, progress_fn=progress,
    )
    parameter_platforms = sorted({d.platform for leaf in jax.tree.leaves(params)
                                 if isinstance(leaf, jax.Array) for d in leaf.devices()})
    if args.expected_platform and parameter_platforms != [args.expected_platform]:
        raise RuntimeError(f"Unexpected parameter devices: {parameter_platforms}")
    save_cpu_checkpoint(params, str(output))
    ckpts = [path for path in (output / "checkpoints").iterdir()
             if path.is_dir() and path.name.isdigit()]
    if not ckpts:
        raise RuntimeError("No numeric Orbax checkpoint directory was saved")
    checkpoint = max(ckpts, key=lambda path: int(path.name))
    restored = load(str(checkpoint))
    from verify_artifacts import check_parameter_trees, check_losses
    parameter_checks = check_parameter_trees(params, restored)
    parameter_checks.update(check_losses(events))
    before = jax.jit(make_inference(params, deterministic=True))
    after = jax.jit(make_inference(restored, deterministic=True))
    export_path = output / "smoke_params.cpp.txt"
    export_params(str(checkpoint), export_path)
    metadata = {
        "variant": "original-robot-torque200", "action_semantics": "absolute_joint_torque",
        "action_order": ["thigh_left", "thigh_right", "wheel_left", "wheel_right"],
        "layer_sizes": [24, 64, 64, 8], "ctrl_dt": cfg.ctrl_dt, "sim_dt": cfg.sim_dt,
        "torque_scales_nm": list(cfg.torque_control.scales),
        "normalization": "(state-state_mean)/state_std", "checkpoint": str(checkpoint),
    }
    (output / "policy_metadata.json").write_text(json.dumps(metadata, indent=2))
    deployed = RL_Policy(path=export_path)
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    restore_errors, export_errors = [], []
    action_rows = []
    for seed in range(8):
        key = jax.random.PRNGKey(seed)
        sample = reset(key)
        # Include dynamic states, not only identical initial robot poses.
        if seed >= 4:
            for _ in range(seed - 3):
                sample = step(sample, before(sample.obs, key)[0])
        obs = sample.obs
        original = np.asarray(before(obs, key)[0])
        recovered = np.asarray(after(obs, key)[0])
        deployed.clean_history()
        exported = deployed.run(np.asarray(obs["state"]))
        if not all(np.isfinite(x).all() for x in (original, recovered, exported)):
            raise RuntimeError("Non-finite policy action")
        restore_errors.append(float(np.max(np.abs(original - recovered))))
        export_errors.append(float(np.max(np.abs(original - exported))))
        action_rows.append({"seed": seed, "jax": original.tolist(),
                            "restored": recovered.tolist(),
                            "exported_numpy": exported.tolist()})
    report = {
        "purpose": "bounded throughput comparison; not a converged balancing policy",
        "source": str(Path(legwheel.__file__).resolve()),
        "control_hz": 1.0 / cfg.ctrl_dt,
        "sim_dt": cfg.sim_dt,
        "episode_length": cfg.episode_length,
        "observation_sizes": {k: list(v.shape) for k, v in obs.items()},
        "torque_scales_nm": list(cfg.torque_control.scales),
        "devices": [str(d) for d in jax.devices()],
        "parameter_platforms": parameter_platforms,
        "checkpoint": str(checkpoint),
        "restored_training_from": str(args.restore) if args.restore else None,
        "elapsed_seconds": time.monotonic() - started,
        "actual_timesteps": events[-1]["steps"],
        "num_envs": int(params_cfg.num_envs),
        "num_eval_envs": int(params_cfg.num_eval_envs),
        "requested_timesteps": int(params_cfg.num_timesteps),
        "observed_progress_steps": [event["steps"] for event in events],
        "restore_max_abs_error": max(restore_errors),
        "export_max_abs_error": max(export_errors),
        "export_atol": args.export_atol,
        "actions": action_rows,
        **parameter_checks,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    np.testing.assert_allclose(restore_errors, 0, atol=1e-6, rtol=0)
    if max(export_errors) > args.export_atol:
        raise AssertionError("Export quantization/action mismatch; see report.json")
    if report["observed_progress_steps"][-1] < params_cfg.num_timesteps:
        raise AssertionError("Training did not reach requested progress")
    print("TORQUE200_PERFORMANCE_ROUND_OK", flush=True)


if __name__ == "__main__":
    main()
