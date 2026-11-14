"""Local PPO entry for the hybrid family; never silently loads legacy weights.

Use --steps 16 --num-envs 2 --unroll-length 4 for plumbing, not performance.
The supported deployment adapter is hybrid.deployment.HybridPolicy, not the old
raw-torque policy loader. The exported C arrays alone are NOT a force controller.
"""
import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MPLBACKEND", "Agg")

from hybrid.core import MODES, metadata, read_geometry, scan_geometry, source_fingerprint, require_metadata, write_json, digest_file, digest_tree


def check_validation(path):
    check = json.loads(Path(path).read_text(encoding="utf8"))
    if (check.get("schema") != "hybrid-local-validation-v1" or
            check.get("runtime_validated") is not True or
            check.get("source_fingerprint") != source_fingerprint()):
        raise ValueError("A successful full local hybrid.validate on this exact source is required")
    return check


def restore_guard(checkpoint, expected):
    checkpoint = Path(checkpoint)
    root = checkpoint.parent.parent
    require_metadata(json.loads((root/"policy_metadata.json").read_text(encoding="utf8")), expected)
    report = json.loads((root/"report.json").read_text(encoding="utf8"))
    if (report.get("checkpoint_validated") is not True or
            report.get("checkpoint_name") != checkpoint.name or
            report.get("checkpoint_sha256") != digest_tree(checkpoint)):
        raise ValueError("Restore only a completed, verified checkpoint of the SAME action mode")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=MODES, default="virtual_leg_force")
    p.add_argument("--validation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--restore", type=Path)
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--num-envs", type=int, default=2)
    p.add_argument("--eval-envs", type=int, default=2)
    p.add_argument("--unroll-length", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--expected-platform", choices=("cpu", "gpu"))
    return p


def main():
    if not __debug__:
        raise RuntimeError("Do not disable assertions in the experimental acceptance pipeline")
    args = parser().parse_args()
    if min(args.steps, args.num_envs, args.eval_envs, args.unroll_length) < 1:
        raise ValueError("Steps and environment/rollout counts must be positive")
    check_validation(args.validation)
    contract = metadata(args.mode, scan_geometry(read_geometry()))
    if args.restore:
        restore_guard(args.restore, contract)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/"policy_metadata.json", contract)
    started = time.monotonic()
    try:
        run(args, output, contract, started)
    except Exception as exc:
        write_json(output/"status.json", dict(state="failed", error=f"{type(exc).__name__}: {exc}",
                   elapsed_seconds=time.monotonic()-started, checkpoint_validated=False))
        raise


def run(args, output, contract, started):
    import inspect
    import importlib.metadata
    import jax
    import numpy as np
    from brax.training.agents.ppo import train as brax_ppo
    # Do not silently drop checkpoint options on unsupported Brax versions.
    supported = inspect.signature(brax_ppo.train).parameters
    required = {"save_checkpoint_path"} | ({"restore_checkpoint_path"} if args.restore else set())
    if not required.issubset(supported):
        raise RuntimeError("Installed Brax lacks required save/restore checkpoint support")
    validation = check_validation(args.validation)
    for package, version in validation["runtime_packages"].items():
        if importlib.metadata.version(package) != version:
            raise RuntimeError("Runtime packages changed after validation; rerun hybrid.validate")
    from hybrid.env import HybridLegWheelRobot, hybrid_config
    from hybrid.deployment import HybridPolicy
    from legwheel.training.ppo import get_ppo_config, compute_discounting, train_ppo
    from legwheel.deployment.extract import load, export_params
    from legwheel.utils import save_cpu_checkpoint
    from verify_artifacts import check_parameter_trees, check_losses

    if args.expected_platform and any(d.platform != args.expected_platform for d in jax.devices()):
        raise RuntimeError(f"Unexpected devices: {jax.devices()}")
    cfg = hybrid_config()
    params_cfg = get_ppo_config(cfg, num_evals=2, maxstep_candidate=args.steps,
        save_checkpoint_path=str(output/"checkpoints"), ppo_overrides=dict(
            batch_size=args.num_envs, num_minibatches=1, unroll_length=args.unroll_length, num_updates_per_batch=1))
    params_cfg.seed = args.seed
    params_cfg.num_eval_envs = args.eval_envs
    params_cfg.learning_rate = 1e-4
    params_cfg.reward_scaling = .1
    params_cfg.entropy_cost = 1e-2
    params_cfg.discounting = compute_discounting(cfg.ctrl_dt, .5)
    params_cfg.network_factory.policy_hidden_layer_sizes = [64, 64]
    params_cfg.network_factory.value_hidden_layer_sizes = [512, 512, 512, 512]
    if args.restore:
        params_cfg.restore_checkpoint_path = str(args.restore.resolve())
    (output/"env_config.json").write_text(cfg.to_json_best_effort(indent=2), encoding="utf8")
    (output/"ppo_config.json").write_text(params_cfg.to_json_best_effort(indent=2), encoding="utf8")
    events = []

    def progress(steps, metrics):
        event = dict(steps=int(steps), elapsed_seconds=time.monotonic()-started,
                     metrics={k: float(v) for k, v in metrics.items() if np.asarray(v).size == 1})
        events.append(event)
        write_json(output/"progress.json", events)
        write_json(output/"status.json", dict(state="training", **event, checkpoint_validated=False))
        print(json.dumps(event), flush=True)

    env = HybridLegWheelRobot(args.mode, cfg)
    eval_env = HybridLegWheelRobot(args.mode, hybrid_config())
    training_started = time.monotonic()
    make_inference, params, _ = train_ppo(env=env, eval_env=eval_env, env_cfg=cfg,
                                        ppo_params=params_cfg, progress_fn=progress)
    # Synchronize before recording elapsed: JAX dispatch alone isn't completion.
    jax.block_until_ready(params)
    training_seconds = time.monotonic()-training_started
    save_cpu_checkpoint(params, str(output))
    checkpoints = [p for p in (output/"checkpoints").iterdir() if p.is_dir() and p.name.isdigit()]
    if not checkpoints:
        raise RuntimeError("PPO did not save a numeric checkpoint")
    checkpoint = max(checkpoints, key=lambda p: int(p.name))
    restored = load(str(checkpoint))
    checks = check_parameter_trees(params, restored)
    checks.update(check_losses(events))
    export_path = output/"policy.cpp.txt"
    export_params(str(checkpoint), export_path)
    contract["export_sha256"] = digest_file(export_path)
    write_json(output/"policy_metadata.json", contract)
    before = jax.jit(make_inference(params, deterministic=True))
    after = jax.jit(make_inference(restored, deterministic=True))
    deployed = HybridPolicy(export_path, output/"policy_metadata.json", expected_mode=args.mode)
    step = jax.jit(env.step)
    restore_errors, export_errors, torque_errors = [], [], []
    dynamic_samples = 0
    for seed in range(8):
        key = jax.random.PRNGKey(seed)
        sample = env.reset(key)
        is_dynamic = False
        for _ in range(max(0, seed-3)):
            sample = step(sample, before(sample.obs, key)[0])
            if bool(sample.done):
                sample = env.reset(key)  # Terminal samples are not deployment states.
                is_dynamic = False
            else:
                is_dynamic = True
        dynamic_samples += int(is_dynamic)
        a = np.asarray(before(sample.obs, key)[0])
        b = np.asarray(after(sample.obs, key)[0])
        c = deployed.action(np.asarray(sample.obs["state"]))
        assert all(np.isfinite(x).all() for x in (a, b, c))
        restore_errors.append(float(np.max(np.abs(a-b))))
        export_errors.append(float(np.max(np.abs(a-c))))
        qh, qk = np.asarray(sample.data.qpos[env.hip_q]), np.asarray(sample.data.qpos[env.knee_q])
        torque_c = deployed.torque(c, qh, qk, np.asarray(sample.data.qvel[env.hip_v]),
                                  np.asarray(sample.data.qvel[env.knee_v]))
        jac = env.geometry(sample.data)[1]
        torque_jax, _ = env.torque_command(jp_array(c), jac)
        torque_errors.append(float(np.max(np.abs(torque_c-np.asarray(torque_jax)))))
    if dynamic_samples == 0:
        raise AssertionError("No nonterminal dynamic sample survived for the export/adapter check")
    if max(restore_errors) > 1e-6 or max(export_errors) > 1e-3 or max(torque_errors) > 1e-5:
        raise AssertionError("Checkpoint/export/action-adapter round trip failed")
    if not events or events[-1]["steps"] < params_cfg.num_timesteps:
        raise AssertionError("Requested training progress not reached")
    if args.restore:
        from hybrid.checkpoints import network_pair, network_diagnostics, check_network_update
        old = load(str(args.restore.resolve()))
        # Orbax's metadata-based loader returns a list, while Brax train returns
        # a tuple. Normalize ONLY that bundle wrapper, not the network trees.
        write_json(output/"restore_network_check.json", network_diagnostics(old, params))
        if jax.tree.structure(network_pair(old)) != jax.tree.structure(network_pair(params)):
            raise AssertionError("Restore network tree structure changed; see restore_network_check.json")
        checks.update(check_network_update(old, params))
    report = dict(checkpoint_validated=True, action_mode=args.mode, checkpoint=str(checkpoint),
                  checkpoint_name=checkpoint.name, checkpoint_sha256=digest_tree(checkpoint),
                  export_sha256=contract["export_sha256"], dynamic_adapter_samples=dynamic_samples,
                  requested_steps=args.steps, configured_steps=int(params_cfg.num_timesteps),
                  actual_progress_steps=events[-1]["steps"], num_envs=args.num_envs,
                  seed=args.seed, training_including_compile_seconds=training_seconds,
                  total_elapsed_seconds=time.monotonic()-started,
                  restore_max_abs_error=max(restore_errors), export_max_abs_error=max(export_errors),
                  adapter_max_abs_error=max(torque_errors),
                  optimizer_resume="Network and normalization restored; optimizer reinitialized",
                  scope="Plumbing check only; no balance, convergence or hardware claim", **checks)
    write_json(output/"report.json", report)
    write_json(output/"status.json", dict(state="completed", **report))
    print(json.dumps(report, indent=2), flush=True)
    print("HYBRID_TRAIN_RESTORE_EXPORT_OK", flush=True)


def jp_array(value):
    import jax.numpy as jp
    return jp.asarray(value)


if __name__ == "__main__":
    main()
