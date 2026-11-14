"""Continuous GPU PPO runner for the training console.

A cooperative stop is checked at Python boundaries, never by killing the process.
The time budget includes initialization; final checkpoint/export verification is
separate. Resume restores network parameters and normalization, not optimizer.
"""
import argparse
from collections import deque
from datetime import datetime, timedelta, timezone
from functools import partial
import json
import math
from pathlib import Path
import sys
import time


class StopRequested(Exception):
    pass


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False),
                         encoding="utf8")
    temporary.replace(path)


class RunMonitor:
    """Standard-library lifecycle, independently testable without JAX/CUDA."""

    def __init__(self, args, clock=time.monotonic):
        self.args = args
        self.output = args.run_dir
        self.clock = clock
        self.start = clock()
        self.events = deque(maxlen=2000)
        self.total_events = 0
        self.loss_names = set()
        started = datetime.now(timezone.utc)
        self.state = dict(
            mode=args.mode, run_id=self.output.name, run_dir=str(self.output),
            phase="preparing", stage="imports", started_at=started.isoformat(),
            deadline_at=(started + timedelta(seconds=args.duration_seconds)).isoformat()
                        if args.duration_seconds else None,
            duration_seconds=args.duration_seconds, steps=0, elapsed_seconds=0,
            completed=False, checkpoint=None, export_path=None, report_path=None,
            error=None, stop_reason=None, preview_protocol=1,
            restored_from=str(args.restore) if args.restore else None,
            num_envs=128, eval_envs=2, control_hz=None, obs_dim=None,
            progress_file=str(self.output / "progress.json"),
            metrics_file=str(self.output / "metrics.jsonl"),
            progress_window_limit=2000, progress_window_truncated=False,
            total_progress_events=0, eval_interval_seconds=45,
            optimizer_semantics="Single continuous PPO call; resume resets optimizer",
            resume_restores=["network_parameters", "observation_normalization"]
                            if args.restore else [],
        )
        self.publish()
        atomic_json(self.output / "progress.json", [])
        (self.output / "metrics.jsonl").touch(exist_ok=True)

    @property
    def elapsed(self):
        return self.clock() - self.start

    @property
    def remaining(self):
        return max(0.0, self.args.duration_seconds - self.elapsed) \
            if self.args.duration_seconds else math.inf

    def publish(self, **changes):
        self.state.update(changes)
        self.state["elapsed_seconds"] = self.elapsed
        self.state["remaining_seconds"] = self.remaining if self.args.duration_seconds else None
        atomic_json(self.output / "status.json", self.state)

    def check_stop(self):
        reason = None
        if (self.output / "stop.request").exists():
            reason = "user_stop"
        elif self.remaining <= 0:
            reason = "wall_budget"
        if reason:
            self.publish(phase="stopping", stage="safe_boundary", stop_reason=reason)
            raise StopRequested(reason)

    def event(self, steps, metrics):
        steps = int(steps)
        if self.events and steps < self.events[-1]["steps"]:
            raise ValueError("Progress steps must be monotonic within this run")
        clean = {}
        for key, value in metrics.items():
            value = float(value)
            if not math.isfinite(value):
                raise FloatingPointError(f"Nonfinite metric: {key}")
            clean[key] = value
            if key.startswith("training/") and "loss" in key:
                self.loss_names.add(key)
        event = dict(steps=steps, elapsed_seconds=self.elapsed, metrics=clean)
        # Every callback is retained on disk, even a second update at the same step.
        with (self.output / "metrics.jsonl").open("a", encoding="utf8") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")
        self.total_events += 1
        if self.events and self.events[-1]["steps"] == steps:
            self.events[-1]["metrics"].update(clean)
            self.events[-1]["elapsed_seconds"] = self.elapsed
        else:
            if len(self.events) == self.events.maxlen:
                self.state["progress_window_truncated"] = True
            self.events.append(event)
        atomic_json(self.output / "progress.json", list(self.events))
        self.publish(steps=steps, total_progress_events=self.total_events)


def validate_restore_metadata(checkpoint, mode, obs_dim):
    """Old checkpoints may lack a mode field; dimensions are checked below too."""
    report_path = checkpoint.parent.parent / "report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf8"))
        if report.get("mode") not in (None, mode):
            raise ValueError("Checkpoint mode does not match requested mode")
        if report.get("obs_dim") not in (None, obs_dim):
            raise ValueError("Checkpoint observation dimension does not match environment")


def run_training(args, monitor):
    # Select source before the first ML import; the web server's CPU env is not used.
    source = (args.original_root / "mujoco_legwheel_rl_public"
              if args.mode == "original" else args.torque_root / "src")
    deploy_source = args.original_root / "mujoco_esp32_robot_deploy_public"
    sys.path[:0] = [str(source), str(deploy_source)]
    monitor.check_stop()

    import jax
    import numpy as np
    from unittest.mock import patch
    from brax.training import acting
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import checkpoint as ppo_checkpoint
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as brax_ppo
    import inspect
    import legwheel
    from legwheel import LegWheelRobot, default_config
    from legwheel.training.ppo import compute_discounting, get_ppo_config, train_ppo
    from legwheel.deployment.extract import load, export_params
    from legwheel.utils import save_cpu_checkpoint
    from verify_artifacts import check_parameter_trees

    if not Path(legwheel.__file__).resolve().is_relative_to(source.resolve()):
        raise RuntimeError(f"Wrong legwheel source: {legwheel.__file__}")
    devices = jax.devices()
    if not devices or any(device.platform != "gpu" for device in devices):
        raise RuntimeError(f"GPU required; actual devices: {devices}")
    if "policy_params_fn" not in inspect.signature(brax_ppo.train).parameters:
        raise RuntimeError("Installed Brax lacks the safe policy snapshot callback")
    monitor.publish(stage="environment_setup", devices=[str(d) for d in devices],
                    source=str(Path(legwheel.__file__).resolve()))
    monitor.check_stop()
    cfg = default_config()  # Preserve the original environment semantics.
    overrides = (dict(batch_size=8, num_minibatches=16, unroll_length=75,
                      num_updates_per_batch=4) if args.mode == "original" else
                 dict(batch_size=128, num_minibatches=1, unroll_length=16,
                      num_updates_per_batch=1))
    # Hundreds of millions of callback epochs, with no finite UI event limit.
    # Torque uses four training steps/epoch (8192 env steps), original one (9600).
    epoch_steps = 9600 if args.mode == "original" else 8192
    epochs = 100_000_000
    ppo_cfg = get_ppo_config(cfg, num_evals=epochs + 1,
                            maxstep_candidate=epoch_steps * epochs,
                            save_checkpoint_path=str(args.run_dir / "checkpoints"),
                            ppo_overrides=overrides)
    ppo_cfg.num_timesteps = epoch_steps * epochs
    ppo_cfg.seed = args.seed
    ppo_cfg.learning_rate = 1e-4
    ppo_cfg.reward_scaling = 0.1
    ppo_cfg.entropy_cost = 1e-2
    ppo_cfg.discounting = compute_discounting(cfg.ctrl_dt, 0.5)
    ppo_cfg.network_factory.policy_hidden_layer_sizes = [64, 64]
    ppo_cfg.network_factory.value_hidden_layer_sizes = [512, 512, 512, 512]
    ppo_cfg.num_eval_envs = 2
    ppo_cfg.save_checkpoint_path = None  # Custom 300-second/final saves.
    if args.restore:
        ppo_cfg.restore_checkpoint_path = str(args.restore)
    if ppo_cfg.num_envs != 128:
        raise AssertionError("Expected exactly 128 training environments")
    atomic_json(args.run_dir / "env_config.json", json.loads(cfg.to_json_best_effort()))
    atomic_json(args.run_dir / "ppo_config.json", json.loads(ppo_cfg.to_json_best_effort()))
    env = LegWheelRobot(config=cfg)
    monitor.publish(stage="observation_shape")
    sample = jax.jit(env.reset)(jax.random.PRNGKey(args.seed))
    observation_sizes = {key: int(value.shape[-1]) for key, value in sample.obs.items()}
    obs_dim = observation_sizes["state"]
    expected_dim = 26 if args.mode == "original" else 24
    if obs_dim != expected_dim:
        raise ValueError(f"Unexpected actual actor observation dimension {obs_dim}")
    monitor.publish(control_hz=1.0 / cfg.ctrl_dt, obs_dim=obs_dim,
                    observation_sizes=observation_sizes)
    monitor.check_stop()
    network_factory = partial(ppo_networks.make_ppo_networks, **dict(ppo_cfg.network_factory))
    # Do not call load_config: its optional initializer=None decoding is broken.
    network_config = ppo_checkpoint.network_config(
        observation_sizes, env.action_size, True, network_factory)
    inference_network = network_factory(
        observation_sizes, env.action_size,
        preprocess_observations_fn=running_statistics.normalize)
    make_policy = ppo_networks.make_inference_fn(inference_network)
    latest = None
    if args.restore:
        monitor.publish(stage="loading_checkpoint")
        validate_restore_metadata(args.restore, args.mode, obs_dim)
        restored = load(str(args.restore))
        for key, size in observation_sizes.items():
            if np.asarray(restored[0].mean[key]).shape != (size,):
                raise ValueError(f"Checkpoint normalization shape mismatch: {key}")
            if np.asarray(restored[0].std[key]).shape != (size,):
                raise ValueError(f"Checkpoint normalization shape mismatch: {key}")
        # Validate complete actor AND critic shapes, including exploration output.
        for params, network in ((restored[1], inference_network.policy_network),
                                (restored[2], inference_network.value_network)):
            expected = network.init(jax.random.PRNGKey(0))
            expected_leaves, expected_tree = jax.tree.flatten(expected)
            leaves, tree = jax.tree.flatten(params)
            if tree != expected_tree or any(
                    np.shape(a) != np.shape(b) or not np.isfinite(np.asarray(a)).all()
                    for a, b in zip(leaves, expected_leaves, strict=True)):
                raise ValueError("Checkpoint network architecture/parameters mismatch")
        # Also reject nonfinite normalization, critic and exploration leaves.
        check_parameter_trees(restored, restored)
        latest = (0, restored, make_policy)
    from preview_sync import SnapshotPublisher
    preview_publisher = SnapshotPublisher(args.run_dir, args.mode)

    def service_preview():
        if latest is not None:
            result = preview_publisher.service(latest[0], latest[1])
            if result is not None:
                monitor.publish(preview_snapshot=result, preview_publish_count=preview_publisher.count)

    saved_steps = set()
    last_save = monitor.elapsed
    last_eval_started = -math.inf
    last_eval_duration = 20.0
    real_evaluations = 0
    original_eval = acting.Evaluator.run_evaluation

    def save_snapshot(snapshot):
        steps, params, _ = snapshot
        checkpoint = args.run_dir / "checkpoints" / f"{steps:012d}"
        if steps not in saved_steps:
            ppo_checkpoint.save(str(args.run_dir / "checkpoints"), steps, params, network_config)
            saved_steps.add(steps)
        return checkpoint

    def capture(steps, factory, params):
        nonlocal latest, last_save
        # Copy first: a stop arriving during the just-completed epoch must retain it.
        cpu = jax.tree.map(lambda x: np.asarray(x).copy(), jax.device_get(params))
        latest = (int(steps), cpu, factory)
        service_preview()
        monitor.event(steps, {})
        monitor.publish(phase="training", stage="optimization")
        monitor.check_stop()
        if monitor.elapsed - last_save >= 300:
            monitor.publish(stage="periodic_checkpoint")
            checkpoint = save_snapshot(latest)
            last_save = monitor.elapsed
            monitor.publish(stage="optimization", checkpoint=str(checkpoint))
            monitor.check_stop()

    def progress(steps, metrics):
        service_preview()
        scalar_metrics = {name: float(value) for name, value in metrics.items()
                          if np.asarray(value).size == 1}
        monitor.event(steps, scalar_metrics)
        monitor.check_stop()

    def scheduled_eval(evaluator, params, training_metrics, aggregate_episodes=True):
        nonlocal last_eval_started, last_eval_duration, real_evaluations
        service_preview()
        monitor.check_stop()
        reserve = max(20.0, last_eval_duration * 1.5 + 3)
        if monitor.elapsed - last_eval_started < 45 or monitor.remaining <= reserve:
            return dict(training_metrics)
        last_eval_started = monitor.elapsed
        monitor.publish(phase="training", stage="evaluation")
        result = original_eval(evaluator, params, training_metrics, aggregate_episodes)
        last_eval_duration = monitor.elapsed - last_eval_started
        real_evaluations += 1
        # Let progress record this completed real evaluation before checking stop.
        return result

    ppo_cfg.policy_params_fn = capture
    failure = None
    try:
        monitor.check_stop()
        monitor.publish(phase="training", stage="compiling")
        with patch.object(acting.Evaluator, "run_evaluation", scheduled_eval):
            factory, params, _ = train_ppo(
                env=env, env_cfg=cfg, ppo_params=ppo_cfg, progress_fn=progress)
        capture(monitor.state["steps"], factory, params)
        stop_reason = "step_limit"
    except StopRequested as exc:
        stop_reason = str(exc)
    except KeyboardInterrupt:
        stop_reason = "user_interrupt"
    except Exception as exc:
        stop_reason, failure = "error", repr(exc)
    if latest is None:
        if failure:
            raise RuntimeError(failure)
        monitor.publish(phase="cancelled", stage="no_model_initialized",
                        stop_reason=stop_reason, completed=False)
        atomic_json(args.run_dir / "report.json", dict(monitor.state))
        monitor.publish(report_path=str(args.run_dir / "report.json"))
        return

    service_preview()
    stop_elapsed = monitor.elapsed
    monitor.publish(phase="saving", stage="checkpoint_and_export", steps=latest[0],
                    stop_reason=stop_reason, error=failure)
    checkpoint = save_snapshot(latest)
    steps, params, factory = latest
    monitor.publish(checkpoint=str(checkpoint))
    save_cpu_checkpoint(params, str(args.run_dir))
    recovered = load(str(checkpoint))
    checks = check_parameter_trees(params, recovered)
    checks["finite_training_losses_checked"] = sorted(monitor.loss_names)
    export_path = args.run_dir / "policy.cpp.txt"
    export_params(str(checkpoint), export_path)
    if args.mode == "original":
        from host.policy.rl_policy import RL_Policy
    else:
        from legwheel.deployment.numpy_policy import RL_Policy
    deployed = RL_Policy(path=export_path)
    before = jax.jit(factory(params, deterministic=True))
    after = jax.jit(factory(recovered, deterministic=True))
    mean = np.asarray(params[0].mean["state"])
    std = np.asarray(params[0].std["state"])
    generator = np.random.default_rng(args.seed)
    export_errors, restore_errors = [], []
    for index in range(8):
        obs = (mean + std * generator.uniform(-1, 1, size=mean.shape)).astype(np.float32)
        key = jax.random.PRNGKey(index)
        a = np.asarray(before({"state": obs}, key)[0])
        b = np.asarray(after({"state": obs}, key)[0])
        deployed.clean_history()
        c = deployed.run(obs)
        if not all(np.isfinite(x).all() for x in (a, b, c)):
            raise FloatingPointError("Non-finite exported policy action")
        restore_errors.append(float(np.max(np.abs(a - b))))
        export_errors.append(float(np.max(np.abs(a - c))))
    if max(restore_errors) > 1e-6 or max(export_errors) > 1e-3:
        raise AssertionError("Checkpoint/export action comparison failed")
    report = dict(monitor.state, checkpoint=str(checkpoint), export_path=str(export_path),
                  actual_timesteps=steps, num_eval_envs=2, real_evaluations=real_evaluations,
                  completed=failure is None, phase="failed" if failure else "completed",
                  training_stop_elapsed_seconds=stop_elapsed,
                  finalization_seconds=monitor.elapsed - stop_elapsed,
                  elapsed_seconds=monitor.elapsed,
                  deadline_overrun_seconds=max(0.0, stop_elapsed - args.duration_seconds)
                                            if args.duration_seconds else None,
                  export_max_abs_error=max(export_errors), export_atol=1e-3,
                  restore_max_abs_error=max(restore_errors),
                  export_observations="8 seeded synthetic observations around saved mean/std",
                  deadline_semantics="Cooperative Python boundaries; includes setup/JIT/evaluation; finalization separate",
                  **checks)
    atomic_json(args.run_dir / "report.json", report)
    monitor.publish(phase="failed" if failure else "completed",
                    stage="saved_previous_snapshot" if failure else "verified",
                    completed=failure is None, export_path=str(export_path),
                    report_path=str(args.run_dir / "report.json"),
                    training_stop_elapsed_seconds=stop_elapsed,
                    finalization_seconds=monitor.elapsed - stop_elapsed)
    if failure:
        raise RuntimeError(failure)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("original", "torque200"), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--original-root", type=Path,
                        default=(Path.home() / "projects/tenny-reference"))
    parser.add_argument("--torque-root", type=Path,
                        default=(Path.home() / "projects/tenny-torque200"))
    parser.add_argument("--duration-seconds", type=float, default=0)
    parser.add_argument("--restore", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if not math.isfinite(args.duration_seconds) or args.duration_seconds < 0:
        parser.error("--duration-seconds must be finite and >= 0")
    if not args.run_dir.is_absolute() or not args.run_dir.is_dir():
        parser.error("--run-dir must be an existing absolute directory")
    args.run_dir = args.run_dir.resolve()
    if args.restore:
        if not args.restore.is_absolute() or not args.restore.is_dir():
            parser.error("--restore must be an existing absolute checkpoint directory")
        args.restore = args.restore.resolve()
    return args


def main(argv=None):
    args = parse_args(argv)
    monitor = RunMonitor(args)  # Publish BEFORE any ML imports.
    try:
        run_training(args, monitor)
    except StopRequested as exc:
        monitor.publish(phase="cancelled", stage="no_model_initialized",
                        stop_reason=str(exc), completed=False)
        atomic_json(args.run_dir / "report.json", dict(monitor.state))
        monitor.publish(report_path=str(args.run_dir / "report.json"))
    except BaseException as exc:
        monitor.publish(phase="failed", stage="failed", completed=False, error=repr(exc))
        report_path = args.run_dir / "report.json"
        if not report_path.exists():
            atomic_json(report_path, dict(monitor.state))
        monitor.publish(report_path=str(report_path))
        raise


if __name__ == "__main__":
    main()
