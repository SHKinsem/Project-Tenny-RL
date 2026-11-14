#!/usr/bin/env python3
"""Bounded CPU/CUDA device and original MJX throughput check.

This runs fixed zero actions, without a policy or automatic episode resets.
It measures raw environment stepping, not balancing quality or PPO throughput.
Run CPU and CUDA with identical batch size and steps for a useful comparison.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--expected-backend", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args()
    if args.batch_size < 1 or args.steps < 1:
        parser.error("batch-size and steps must be positive")
    if args.output.exists():
        parser.error("output already exists; choose a new report path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Explicit selection must fail if CUDA is unavailable; never silently use CPU.
    os.environ["JAX_PLATFORMS"] = args.expected_backend
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    sys.path.insert(0, str(args.repo.resolve()))
    report = {
        "status": "running", "purpose": "raw MJX zero-action throughput, not policy quality",
        "expected_backend": args.expected_backend, "batch_size": args.batch_size,
        "steps_per_environment": args.steps, "repeats": 3,
        "automatic_episode_reset": False, "configuration_overrides": {},
        "environment_steps_per_repeat": args.batch_size * args.steps,
        "environment": {key: os.environ.get(key) for key in (
            "JAX_PLATFORMS", "XLA_PYTHON_CLIENT_PREALLOCATE",
            "XLA_PYTHON_CLIENT_MEM_FRACTION", "XLA_CLIENT_MEM_FRACTION",
            "JAX_COMPILATION_CACHE_DIR", "XLA_FLAGS")},
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    def stage(name):
        report["stage"] = name
        save()
        print(name, flush=True)

    try:
        stage("initialize_backend")
        import jax
        import jax.numpy as jp
        import numpy as np
        report["versions"] = {name: importlib.metadata.version(name) for name in
                              ("jax", "jaxlib", "mujoco", "mujoco-mjx", "playground")}
        devices = jax.devices()
        report["default_backend"] = jax.default_backend()
        report["devices"] = [{"id": d.id, "platform": d.platform,
                              "device_kind": d.device_kind,
                              "platform_version": d.client.platform_version}
                             for d in devices]
        expected_platform = "gpu" if args.expected_backend == "cuda" else "cpu"
        if not devices or any(d.platform != expected_platform for d in devices):
            raise RuntimeError(f"Unexpected devices: {report['devices']}")
        if args.expected_backend == "cuda" and not any(
                "CUDA" in d.client.platform_version.upper() for d in devices):
            raise RuntimeError("GPU backend did not identify itself as CUDA")
        device = devices[0]
        report["memory_stats_before"] = device.memory_stats()

        stage("matrix_multiply_device_check")
        operand = jax.device_put(np.ones((256, 256), np.float32), device)
        started = time.perf_counter()
        product = jax.jit(lambda a: a @ a)(operand)
        product.block_until_ready()
        report["matrix_first_jit_and_run_seconds"] = time.perf_counter() - started
        report["matrix_result_devices"] = [str(d) for d in product.devices()]
        if any(d.platform != expected_platform for d in product.devices()):
            raise RuntimeError("Matrix result is on the wrong device")
        np.testing.assert_array_equal(np.asarray(product), np.full((256, 256), 256.0))
        report["matrix_check_passed"] = True

        stage("build_original_environment")
        from legwheel import LegWheelRobot, default_config
        cfg = default_config()
        env = LegWheelRobot(config=cfg)
        report["control_dt_seconds"] = float(cfg.ctrl_dt)
        report["physics_dt_seconds"] = float(cfg.sim_dt)
        report["episode_length"] = int(cfg.episode_length)
        report["action_size"] = int(env.action_size)
        keys = jax.random.split(jax.random.PRNGKey(0), args.batch_size)
        keys = jax.device_put(keys, device)
        reset_batch = jax.jit(jax.vmap(lambda key: env.reset(key, False)))
        stage("compile_batched_reset")
        started = time.perf_counter()
        reset_executable = reset_batch.lower(keys).compile()
        report["reset_lower_and_compile_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        initial = reset_executable(keys)
        jax.block_until_ready(initial)
        report["reset_first_execution_seconds"] = time.perf_counter() - started
        report["state_devices"] = [str(d) for d in initial.data.qpos.devices()]
        if any(d.platform != expected_platform for d in initial.data.qpos.devices()):
            raise RuntimeError("MJX state is on the wrong device")
        step_batch = jax.vmap(env.step)
        actions = jp.zeros((args.batch_size, env.action_size), dtype=jp.float32)

        def rollout(state):
            def body(carry, unused):
                current = step_batch(carry, actions)
                finite = (jp.all(jp.isfinite(current.data.qpos))
                          & jp.all(jp.isfinite(current.data.qvel))
                          & jp.all(jp.isfinite(current.reward))
                          & jp.all(jp.isfinite(current.done)))
                for value in current.obs.values():
                    finite = finite & jp.all(jp.isfinite(value))
                # These are done-flagged transitions, not a count of falls/episodes.
                return current, (finite, jp.sum(current.done > 0))
            return jax.lax.scan(body, state, xs=None, length=args.steps)

        stage("compile_batched_rollout")
        started = time.perf_counter()
        rollout_executable = jax.jit(rollout).lower(initial).compile()
        report["rollout_lower_and_compile_seconds"] = time.perf_counter() - started
        stage("warm_up_batched_rollout")
        started = time.perf_counter()
        warm = rollout_executable(initial)
        jax.block_until_ready(warm)
        report["rollout_warmup_seconds"] = time.perf_counter() - started
        if not bool(np.all(np.asarray(warm[1][0]))):
            raise RuntimeError("Non-finite state, observation or reward during warmup")
        del warm

        stage("measure_three_synchronized_rollouts")
        samples = []
        for repeat in range(3):
            started = time.perf_counter()
            result = rollout_executable(initial)
            jax.block_until_ready(result)
            duration = time.perf_counter() - started
            final_state, checks = result
            finite = bool(np.all(np.asarray(checks[0])))
            sample = {"repeat": repeat + 1, "seconds": duration,
                      "environment_steps_per_second": args.batch_size * args.steps / duration,
                      "all_checked_values_finite": finite,
                      "done_flagged_transitions": int(np.asarray(checks[1]).sum()),
                      "final_qpos_checksum": float(np.asarray(final_state.data.qpos).sum()),
                      "final_reward_mean": float(np.asarray(final_state.reward).mean())}
            samples.append(sample)
            report["samples"] = samples
            save()
            print(json.dumps(sample), flush=True)
            if not finite:
                raise RuntimeError("Non-finite values during measured rollout")
            del result, final_state, checks
        median_seconds = statistics.median(item["seconds"] for item in samples)
        report["median_seconds"] = median_seconds
        report["median_environment_steps_per_second"] = args.batch_size * args.steps / median_seconds
        report["memory_stats_after"] = device.memory_stats()
        report["memory_note"] = "JAX allocator statistics; not total board usage or nvidia-smi peak."
        report["timing_note"] = (
            "Lower/compile may load a persistent cache. Warmup excluded from throughput. "
            "Identical initial state reused each repeat; no Python-device stepping loop. "
            "Finite checks are included in the measured scan; no PPO update is included.")
        report["status"] = "passed"
        report["stage"] = "complete"
        save()
        print(f"DEVICE_MJX_BENCHMARK_OK {args.output}", flush=True)
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
        save()
        raise


if __name__ == "__main__":
    main()
