"""One continuous, budgeted PPO run with live metrics and safe final export.

The 900-second wall-clock budget includes imports, setup, compilation and real
evaluation. Parameters are copied at every 8192-step callback. Near the deadline
we stop scheduling work, then save/verify that completed snapshot. Finalization
is reported separately. No optimizer restart occurs inside the run.
"""
import argparse
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import signal
import sys
import time

FAILURE_REPORT = None


class DeadlineReached(Exception):
    pass


class WallBudget:
    def __init__(self, seconds, clock=time.monotonic):
        self.clock = clock
        self.start = clock()
        self.seconds = float(seconds)

    @property
    def elapsed(self):
        return self.clock() - self.start

    @property
    def remaining(self):
        return max(0.0, self.seconds - self.elapsed)

    def check(self, reserve=0.0):
        if self.remaining <= reserve:
            raise DeadlineReached("Wall-clock budget reached")


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf8")
    temporary.replace(path)


def merge_event(events, steps, metrics, elapsed):
    if events and steps < events[-1]["steps"]:
        raise ValueError("Progress steps must be monotonic")
    if events and steps == events[-1]["steps"]:
        events[-1]["metrics"].update(metrics)
        events[-1]["elapsed_seconds"] = elapsed
    else:
        events.append({"steps": steps, "elapsed_seconds": elapsed, "metrics": metrics})


def self_test():
    now = [10.0]
    budget = WallBudget(900, lambda: now[0])
    assert budget.remaining == 900
    now[0] = 909.0
    budget.check()
    try:
        budget.check(reserve=2)
        raise AssertionError("Reserve did not stop work")
    except DeadlineReached:
        pass
    now[0] = 910.0
    try:
        budget.check()
        raise AssertionError("Deadline did not stop work")
    except DeadlineReached:
        pass
    events = []
    merge_event(events, 0, {}, 0)
    merge_event(events, 0, {"eval/avg_episode_length": 40.0}, 1)
    merge_event(events, 8192, {}, 2)
    merge_event(events, 8192, {"training/v_loss": 0.1}, 3)
    assert len(events) == 2 and events[-1]["metrics"]["training/v_loss"] == 0.1
    assert "eval/avg_episode_length" not in events[-1]["metrics"]
    print("TIMED_TRAIN_SYNTHETIC_BUDGET_AND_PROGRESS_OK", flush=True)


def main():
    global FAILURE_REPORT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, default=(Path.home() / "projects/tenny-torque200"))
    ap.add_argument("--duration", type=float, default=900)
    ap.add_argument("--verify-only", action="store_true", help="Save/verify existing B1 parameters; no training or live publication")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not 0 < args.duration <= 900:
        raise ValueError("Duration must be in (0, 900] seconds")
    budget = WallBudget(args.duration)
    started = datetime.now(timezone.utc)
    artifacts = args.base.resolve() / "artifacts"
    prefix = "timed-validation" if args.verify_only else "training"
    run_id = started.strftime(prefix + "-%Y%m%d-%H%M%S")
    output = artifacts / run_id
    output.mkdir(parents=True, exist_ok=False)
    latest_manifest = json.loads((artifacts / "performance-latest.json").read_text())
    restore = Path(latest_manifest["checkpoint"]).resolve()
    if (latest_manifest.get("completed") is not True or not restore.is_relative_to(artifacts) or
            not restore.parent.parent.parent.name.startswith("performance-") or not restore.is_dir()):
        raise ValueError("Expected the already-verified performance B1 checkpoint")
    source_report = json.loads((restore.parent.parent / "report.json").read_text())
    if source_report.get("all_parameters_finite") is not True or source_report.get("parameter_tree_max_abs_error") != 0:
        raise ValueError("Restore checkpoint lacks validation")
    events = []
    state = {"run_id": run_id, "run_dir": str(output), "progress_file": str(output / "progress.json"),
             "phase": "preparing", "stage": "imports", "started_at": started.isoformat().replace("+00:00", "Z"),
             "deadline_at": (started + timedelta(seconds=args.duration)).isoformat().replace("+00:00", "Z"),
             "duration_seconds": args.duration, "steps": 0, "completed": False,
             "checkpoint": None, "error": None, "restored_from": str(restore),
             "num_envs": 128, "eval_envs": 2, "eval_interval_seconds": 45,
             "live_unverified": True}
    publish_enabled = False

    def publish(**changes):
        state.update(changes)
        state["elapsed_seconds"] = budget.elapsed
        state["remaining_seconds"] = budget.remaining
        atomic_json(output / "status.json", state)
        if publish_enabled and not args.verify_only:
            atomic_json(artifacts / "training-status.json", state)
            atomic_json(artifacts / "training-live.json", state)

    FAILURE_REPORT = lambda exc: publish(phase="failed", error=repr(exc))

    import jax
    import numpy as np
    from functools import partial
    from unittest.mock import patch
    from brax.training import acting
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import checkpoint as ppo_checkpoint
    from brax.training.agents.ppo import networks as ppo_networks
    from legwheel import LegWheelRobot, default_config
    from legwheel.training.ppo import compute_discounting, get_ppo_config, train_ppo
    from legwheel.deployment.extract import load, export_params
    from legwheel.deployment.numpy_policy import RL_Policy
    from legwheel.utils import save_cpu_checkpoint
    from verify_artifacts import check_parameter_trees, check_losses
    if any(d.platform != "gpu" for d in jax.devices()):
        raise RuntimeError(f"Expected GPU, got {jax.devices()}")
    cfg = default_config()
    ppo_cfg = get_ppo_config(cfg, num_evals=100001, maxstep_candidate=8192 * 100000,
        save_checkpoint_path=str(output / "checkpoints"),
        ppo_overrides={"batch_size": 128, "num_minibatches": 1, "unroll_length": 16, "num_updates_per_batch": 1})
    ppo_cfg.num_timesteps = 8192 * 100000
    ppo_cfg.learning_rate = 1e-4
    ppo_cfg.reward_scaling = 0.1
    ppo_cfg.entropy_cost = 1e-2
    ppo_cfg.discounting = compute_discounting(cfg.ctrl_dt, 0.5)
    ppo_cfg.network_factory.policy_hidden_layer_sizes = [64, 64]
    ppo_cfg.network_factory.value_hidden_layer_sizes = [512, 512, 512, 512]
    ppo_cfg.num_eval_envs = 2
    ppo_cfg.restore_checkpoint_path = str(restore)
    ppo_cfg.save_checkpoint_path = None  # periodic custom save, not every 8192 steps
    if ppo_cfg.num_envs != 128:
        raise AssertionError(f"Expected 128 environments, got {ppo_cfg.num_envs}")
    atomic_json(output / "env_config.json", json.loads(cfg.to_json_best_effort()))
    atomic_json(output / "ppo_config.json", json.loads(ppo_cfg.to_json_best_effort()))
    atomic_json(output / "progress.json", events)
    publish_enabled = True
    publish(stage="loading_checkpoint")
    fallback_params = load(str(restore))
    # Rebuild from the exact training factory; installed load_config cannot decode
    # optional initializer=None in these otherwise valid saved configurations.
    network_config = ppo_checkpoint.network_config(
        {"state": 24, "privileged_state": 51}, 4, True,
        partial(ppo_networks.make_ppo_networks, **dict(ppo_cfg.network_factory)))
    inference_network = ppo_networks.make_ppo_networks(
        {"state": 24, "privileged_state": 51}, 4,
        preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=(64, 64), value_hidden_layer_sizes=(512, 512, 512, 512),
        policy_obs_key="state", value_obs_key="privileged_state")
    make_policy = ppo_networks.make_inference_fn(inference_network)
    latest = (0, fallback_params, make_policy)
    saved_steps = set()
    last_save = budget.elapsed
    last_eval_started = -1e9
    last_eval_duration = 20.0
    real_evaluations = 0
    original_eval = acting.Evaluator.run_evaluation
    stop_reason = "budget"
    failure = None

    def save_snapshot(snapshot):
        steps, params, _ = snapshot
        checkpoint = output / "checkpoints" / f"{steps:012d}"
        if steps not in saved_steps:
            ppo_checkpoint.save(str(output / "checkpoints"), steps, params, network_config)
            saved_steps.add(steps)
        return checkpoint

    def capture(steps, policy_factory, params):
        nonlocal latest, last_save
        budget.check(reserve=2.0)
        # Keep CPU-owned arrays: later donated GPU buffers cannot invalidate this snapshot.
        cpu = jax.tree.map(lambda x: np.asarray(x).copy(), jax.device_get(params))
        latest = (int(steps), cpu, policy_factory)
        merge_event(events, int(steps), {}, budget.elapsed)
        atomic_json(output / "progress.json", events)
        publish(phase="training", stage="optimization", steps=int(steps))
        if budget.elapsed - last_save >= 45 and budget.remaining > 20:
            checkpoint = save_snapshot(latest)
            last_save = budget.elapsed
            publish(checkpoint=str(checkpoint))

    def progress(steps, metrics):
        keep = ("training/sps", "training/walltime", "training/v_loss", "training/policy_loss",
                "training/entropy_loss", "training/total_loss", "training/kl_mean", "training/learning_rate",
                "eval/avg_episode_length", "eval/std_episode_length", "eval/episode_reward",
                "eval/episode_reward_std", "eval/epoch_eval_time", "eval/sps", "eval/walltime")
        values = {}
        for key in keep:
            if key in metrics:
                value = float(np.asarray(metrics[key]))
                if not math.isfinite(value):
                    raise FloatingPointError(f"Non-finite metric: {key}")
                values[key] = value
        merge_event(events, int(steps), values, budget.elapsed)
        if len(events) > 2000:
            raise RuntimeError("Progress record cap reached")
        atomic_json(output / "progress.json", events)
        publish(phase="training", stage="optimization", steps=int(steps))
        if "eval/avg_episode_length" in values:
            print(json.dumps({"steps": int(steps), "elapsed_seconds": budget.elapsed,
                              "evaluation": values}), flush=True)
        budget.check(reserve=2.0)

    def scheduled_eval(evaluator, params, training_metrics, aggregate_episodes=True):
        nonlocal last_eval_started, last_eval_duration, real_evaluations
        budget.check(reserve=2.0)
        due = budget.elapsed - last_eval_started >= 45
        reserve = max(20.0, last_eval_duration * 1.5 + 3.0)
        if not due or budget.remaining <= reserve:
            return dict(training_metrics)  # no fabricated/forward-filled evaluation values
        last_eval_started = budget.elapsed
        publish(phase="training", stage="evaluation")
        result = original_eval(evaluator, params, training_metrics, aggregate_episodes)
        last_eval_duration = budget.elapsed - last_eval_started
        real_evaluations += 1
        return result

    def alarm(_signum, _frame):
        raise DeadlineReached("Wall-clock deadline signal")

    previous_handler = signal.getsignal(signal.SIGALRM)
    try:
        if args.verify_only:
            stop_reason = "verification_only_no_training"
        else:
            budget.check(reserve=2.0)
            signal.signal(signal.SIGALRM, alarm)
            signal.setitimer(signal.ITIMER_REAL, budget.remaining)
            ppo_cfg.policy_params_fn = capture
            env = LegWheelRobot(config=cfg)
            with patch.object(acting.Evaluator, "run_evaluation", scheduled_eval):
                train_ppo(env=env, env_cfg=cfg, ppo_params=ppo_cfg, progress_fn=progress)
            stop_reason = "step_limit"
    except DeadlineReached:
        stop_reason = "wall_budget"
    except KeyboardInterrupt:
        stop_reason = "user_interrupt"
    except Exception as exc:
        failure = repr(exc)
        stop_reason = "error"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
    stop_elapsed = budget.elapsed
    publish(phase="saving", stage="checkpoint_and_export", steps=latest[0], stop_reason=stop_reason, error=failure)
    try:
        checkpoint = save_snapshot(latest)
        steps, params, factory = latest
        save_cpu_checkpoint(params, str(output))
        restored = load(str(checkpoint))
        checks = check_parameter_trees(params, restored)
        checks.update(check_losses(events))
        export_path = output / "smoke_params.cpp.txt"
        export_params(str(checkpoint), export_path)
        deployed = RL_Policy(path=export_path)
        before = jax.jit(factory(params, deterministic=True))
        after = jax.jit(factory(restored, deterministic=True))
        mean = np.asarray(params[0].mean["state"])
        std = np.asarray(params[0].std["state"])
        generator = np.random.default_rng(0)
        export_errors, restore_errors = [], []
        for index in range(8):
            obs = (mean + std * generator.uniform(-1, 1, size=mean.shape)).astype(np.float32)
            key = jax.random.PRNGKey(index)
            inputs = {"state": obs}
            a, b = np.asarray(before(inputs, key)[0]), np.asarray(after(inputs, key)[0])
            c = deployed.run(obs)
            if not all(np.isfinite(x).all() for x in (a, b, c)):
                raise FloatingPointError("Non-finite exported policy action")
            restore_errors.append(float(np.max(np.abs(a - b))))
            export_errors.append(float(np.max(np.abs(a - c))))
        if max(restore_errors) > 1e-6 or max(export_errors) > 1e-3:
            raise AssertionError("Final checkpoint/export action comparison failed")
        report = {"checkpoint": str(checkpoint), "export_path": str(export_path),
                  "restored_training_from": str(restore), "control_hz": 200, "num_envs": 128, "num_eval_envs": 2,
                  "actual_timesteps": steps, "duration_seconds": args.duration,
                  "training_stop_elapsed_seconds": stop_elapsed,
                  "deadline_overrun_seconds": max(0.0, stop_elapsed - args.duration),
                  "finalization_seconds": budget.elapsed - stop_elapsed, "elapsed_seconds": budget.elapsed,
                  "stop_reason": stop_reason, "error": failure, "real_evaluations": real_evaluations,
                  "parameter_platform": "gpu", "export_max_abs_error": max(export_errors),
                  "restore_max_abs_error": max(restore_errors), "export_atol": 1e-3,
                  "export_observations": "8 reproducible synthetic observations around saved mean/std; numerical check, not robot performance",
                  "deadline_semantics": "Budget includes setup/import/JIT/evaluation. Callback reserves 2s before next epoch; no new eval with <20s remaining. SIGALRM is a fallback at Python return boundaries; final save/verification is separate.",
                  "optimizer_semantics": "One continuous PPO call during this run; initial restore loads network/statistics, not prior optimizer state",
                  **checks}
        atomic_json(output / "report.json", report)
        atomic_json(output / "policy_metadata.json", {
            "variant": "original-robot-torque200", "action_semantics": "absolute_joint_torque",
            "action_order": ["thigh_left", "thigh_right", "wheel_left", "wheel_right"],
            "layer_sizes": [24, 64, 64, 8], "ctrl_dt": cfg.ctrl_dt, "sim_dt": cfg.sim_dt,
            "torque_scales_nm": list(cfg.torque_control.scales), "checkpoint": str(checkpoint)})
        if failure:
            publish(phase="failed", stage="saved_previous_snapshot", checkpoint=str(checkpoint), error=failure)
            raise RuntimeError(failure)
        publish(phase="completed", stage="verified", completed=True, live_unverified=False,
                checkpoint=str(checkpoint), export_path=str(export_path), report_path=str(output / "report.json"),
                training_stop_elapsed_seconds=stop_elapsed, finalization_seconds=budget.elapsed - stop_elapsed)
        if not args.verify_only:
            atomic_json(artifacts / "training-latest.json", state)
        print("TIMED_TRAIN_VERIFIED " + json.dumps(report), flush=True)
    except Exception as exc:
        publish(phase="failed", stage="finalization_failed", error=repr(exc))
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        if FAILURE_REPORT is not None:
            FAILURE_REPORT(exc)
        raise
