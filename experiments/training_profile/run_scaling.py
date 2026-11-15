"""One isolated, fixed-work PPO scaling case; launched by launch_scaling.py."""
import argparse
from functools import wraps
import gzip
import hashlib
import importlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch

from run_profile import write_json


def case_config(original, *, num_envs, batch_size, warmup_epochs=2, measure_epochs=8):
    """Keep batch/update work fixed while changing only rollout concurrency."""
    if any(type(n) is not int or n <= 0 for n in
           (num_envs, batch_size, warmup_epochs, measure_epochs)):
        raise ValueError("Environment, batch and epoch counts must be positive integers")
    if batch_size % num_envs:
        raise ValueError("The common batch must be divisible by every environment count")
    if original.get("num_resets_per_eval", 0) != 0:
        raise ValueError("The scaling adapter does not support extra epoch resets")
    if original["num_minibatches"] != 1 or original["num_updates_per_batch"] != 1:
        raise ValueError("This comparison requires one minibatch and one update per batch")
    epochs = warmup_epochs + measure_epochs + 2  # one trace + one recovery
    epoch_steps = batch_size * original["unroll_length"] * original["action_repeat"]
    config = dict(original, num_envs=num_envs, batch_size=batch_size,
                  num_timesteps=epoch_steps * epochs, num_evals=epochs + 1,
                  run_evals=False, save_checkpoint_path=None, restore_checkpoint_path=None)
    config.pop("policy_params_fn", None)
    return config, epoch_steps, epochs


def timing_summary(rows):
    values = [row["complete_seconds"] for row in rows if row["kind"] == "measure"]
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("No valid untraced measurement epochs")
    steps = sum(row["env_steps"] for row in rows if row["kind"] == "measure")
    return {"samples": len(values), "env_steps": steps, "seconds": sum(values),
            "steps_per_second": steps / sum(values),
            "median_epoch_seconds": statistics.median(values),
            "min_epoch_seconds": min(values), "max_epoch_seconds": max(values)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--runtime-src", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--measure-epochs", type=int, default=8)
    args = parser.parse_args(argv)
    out = args.output.expanduser().resolve()
    if str(out).startswith("/mnt/") or (out / "result.json").exists():
        parser.error("Use a fresh WSL ext4 output directory")
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    def status(stage, **fields):
        value = {"stage": stage, "elapsed_seconds": time.monotonic() - started, **fields}
        write_json(out / "status.json", value)
        with (out / "events.jsonl").open("a", encoding="utf8") as stream:
            stream.write(json.dumps(value) + "\n")
        print(json.dumps(value), flush=True)

    status("imports")
    sys.path.insert(0, str(args.runtime_src.resolve()))
    import jax
    import numpy as np
    from force_teacher.env import StudentEnv, student_config
    from force_teacher.ppo import train_student, restore_fixed_normalizer, normalizer_max_abs_change
    from force_teacher.contract import restore_guard, tree_digest
    from hybrid.core import source_fingerprint
    from legwheel.deployment.extract import load
    train_module = importlib.import_module("brax.training.agents.ppo.train")
    losses_module = importlib.import_module("brax.training.agents.ppo.losses")
    if jax.default_backend() != "gpu" or len(jax.devices()) != 1:
        raise ValueError("This scaling adapter requires exactly one GPU")
    device = jax.devices()[0]
    status("gpu_ready", device=str(device))
    run = args.run_dir.expanduser().resolve()
    baseline = json.loads((run / "report.json").read_text())
    original = json.loads((run / "ppo_config.json").read_text())
    env_config = json.loads((run / "env_config.json").read_text())
    if json.loads(student_config().to_json_best_effort()) != env_config:
        raise ValueError("Task configuration differs from the verified model")
    checkpoint = Path(baseline["checkpoint"])
    restore_guard(checkpoint, Path.home() / "projects/tenny-reference")
    source_sha, checkpoint_sha = source_fingerprint(), tree_digest(checkpoint)
    if source_sha != baseline["source_fingerprint"] or checkpoint_sha != baseline["checkpoint_sha256"]:
        raise ValueError("Production source/checkpoint identity changed")
    installed = {str(Path(module.__file__).resolve()): hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                 for module in (train_module, losses_module)}
    config, epoch_steps, epochs = case_config(original, num_envs=args.num_envs,
        batch_size=args.batch_size, warmup_epochs=args.warmup_epochs, measure_epochs=args.measure_epochs)
    contract = {"run_id": run.name, "checkpoint_sha256": checkpoint_sha,
        "source_fingerprint": source_sha, "num_envs": args.num_envs,
        "common_batch_size": args.batch_size, "epoch_env_steps": epoch_steps,
        "epochs": epochs, "updates_per_epoch": 1, "config": config,
        "overrides": {key: value for key, value in config.items() if value != original.get(key)},
        "original_config": original, "installed_source_sha256": installed,
        "untraced_measure_epochs": args.measure_epochs, "warmup_epochs": args.warmup_epochs,
        "scope": "original single-device Brax pmap, one complete epoch boundary; no internal stage markers or model saves"}
    write_json(out / "contract.json", contract)
    rows, boundaries, capture_steps, memory = [], [], [], []
    initial = None
    last_state = None

    def memory_sample(stage):
        stats = device.memory_stats()
        memory.append({"stage": stage, "stats": stats,
                       "elapsed_seconds": time.monotonic() - started})
        write_json(out / "device-memory.json", memory)

    def cpu_snapshot(params, label):
        t0 = time.monotonic()
        jax.block_until_ready(params)
        t1 = time.monotonic()
        host = jax.device_get(params)
        t2 = time.monotonic()
        result = jax.tree.map(lambda value: np.asarray(value).copy(), host)
        t3 = time.monotonic()
        boundaries.append({"label": label, "ready_wait_seconds": t1 - t0,
            "device_get_seconds": t2 - t1, "host_copy_seconds": t3 - t2,
            "bytes": sum(value.nbytes for value in jax.tree.leaves(result))})
        write_json(out / "boundary-transfers.json", boundaries)
        return result

    def max_difference(before, after):
        left, left_def = jax.tree.flatten(before)
        right, right_def = jax.tree.flatten(after)
        if left_def != right_def:
            raise ValueError("Model structure changed")
        maximum = 0.
        for old, new in zip(left, right, strict=True):
            if old.shape != new.shape or not np.isfinite(old).all() or not np.isfinite(new).all():
                raise ValueError("Nonfinite or mismatched model arrays")
            if old.size:
                maximum = max(maximum, float(np.max(np.abs(old.astype(np.float64)-new.astype(np.float64)))))
        return maximum

    def tree_hash(value):
        digest = hashlib.sha256()
        for leaf in jax.tree.leaves(value):
            array = np.ascontiguousarray(leaf)
            digest.update(str((array.shape, array.dtype.str)).encode())
            digest.update(array.tobytes())
        return digest.hexdigest()

    def capture(steps, factory, params):
        nonlocal initial
        capture_steps.append(int(steps))
        if int(steps) == 0:
            initial = cpu_snapshot(params, "initial")
            error = max_difference(reference, initial)
            write_json(out / "initial-validation.json", {"parameter_max_abs": error,
                "parameter_sha256": tree_hash(initial)})
            if error != 0:
                raise ValueError("Training did not start from the frozen input parameters")

    real_pmap = jax.pmap
    trace_active = False
    compile_record = None

    def record_pmap(function, *pmap_args, **pmap_kwargs):
        mapped = real_pmap(function, *pmap_args, **pmap_kwargs)
        if getattr(function, "__name__", "") != "training_epoch":
            return mapped
        executable = None

        @wraps(function)
        def epoch(*values, **options):
            nonlocal executable, trace_active, last_state, compile_record
            index = len(rows)
            if executable is None:
                status("inputs_ready")
                t0 = time.monotonic(); jax.block_until_ready((values, options)); t1 = time.monotonic()
                status("lowering")
                t2 = time.monotonic(); lowered = mapped.lower(*values, **options); t3 = time.monotonic()
                status("compiling")
                t4 = time.monotonic(); executable = lowered.compile(); t5 = time.monotonic()
                with gzip.open(out / "compiled-hlo.txt.gz", "wt", encoding="utf8") as stream:
                    stream.write(executable.as_text())
                compile_record = {"input_wait_seconds": t1-t0, "lowering_seconds": t3-t2,
                                  "xla_compile_seconds": t5-t4}
                write_json(out / "compile.json", compile_record)
                memory_sample("compiled")
            trace_index = args.warmup_epochs + args.measure_epochs
            kind = ("warmup" if index < args.warmup_epochs else
                    "measure" if index < trace_index else "trace" if index == trace_index else "recovery")
            if kind == "trace":
                write_json(out / "performance.json", timing_summary(rows))
                status("starting_trace")
                profile_options = jax.profiler.ProfileOptions()
                profile_options.python_tracer_level = 0
                profile_options.host_tracer_level = 1
                profile_options.enable_hlo_proto = True
                jax.profiler.start_trace(str(out / "trace"), create_perfetto_trace=True,
                                         profiler_options=profile_options)
                trace_active = True
            elif trace_active:
                status("stopping_trace")
                jax.profiler.stop_trace()
                trace_active = False
            status("epoch", epoch=index, kind=kind)
            with jax.profiler.StepTraceAnnotation("scaling.host_epoch", step_num=index,
                                                  num_envs=args.num_envs, kind=kind):
                t0 = time.monotonic()
                result = executable(*values, **options)
                t1 = time.monotonic()
                jax.block_until_ready(result)
                t2 = time.monotonic()
            rows.append({"epoch": index, "kind": kind, "env_steps": epoch_steps,
                "submit_seconds": t1-t0, "completion_wait_seconds": t2-t1,
                "complete_seconds": t2-t0})
            write_json(out / "epochs.json", rows)
            last_state = result[1]
            # Only small loss metrics leave the GPU after the timed boundary.
            metrics = jax.device_get(result[2])
            if any(not np.isfinite(value).all() for value in jax.tree.leaves(metrics)):
                raise ValueError("Nonfinite training metrics")
            memory_sample(kind + "." + str(index))
            return result
        return epoch

    status("setup")
    params = load(str(checkpoint))
    reference = cpu_snapshot((restore_fixed_normalizer(params[0]), params[1], params[2]), "input_reference")
    environment = StudentEnv(config=student_config())
    eval_environment = StudentEnv(config=student_config())
    memory_sample("setup")
    try:
        with patch.object(jax, "pmap", record_pmap):
            _, final_params, _ = train_student(env=environment, eval_env=eval_environment,
                env_cfg=environment._config, ppo_params=dict(config, policy_params_fn=capture),
                progress_fn=lambda *a: None, params=params)
        final = cpu_snapshot(final_params, "final")
    finally:
        if trace_active:
            jax.profiler.stop_trace()
    final_physics = cpu_snapshot((last_state.data.qpos, last_state.data.qvel), "final_physics")
    if any(not np.isfinite(value).all() for value in jax.tree.leaves(final_physics)):
        raise ValueError("Nonfinite physics output")
    initial_error = max_difference(reference, initial)
    parameter_change = max_difference(initial[1:], final[1:])
    normalizer_error = normalizer_max_abs_change(initial[0], final[0])
    expected_steps = [index * epoch_steps for index in range(epochs + 1)]
    if capture_steps != expected_steps or len(rows) != epochs:
        raise ValueError("Brax executed unexpected epoch/update work")
    unchanged = source_fingerprint() == source_sha and tree_digest(checkpoint) == checkpoint_sha
    unchanged = unchanged and all(hashlib.sha256(Path(file).read_bytes()).hexdigest() == digest
                                  for file, digest in installed.items())
    performance = timing_summary(rows)
    trace_candidates = sorted((out / "trace").rglob("perfetto_trace.json.gz"))
    if len(trace_candidates) != 1:
        raise ValueError("Expected one captured device trace")
    passed = initial_error == 0 and normalizer_error == 0 and unchanged and parameter_change > 0
    result = {"passed": passed, "num_envs": args.num_envs, "batch_size": args.batch_size,
        "run_id": run.name, "checkpoint_sha256": checkpoint_sha, "epoch_env_steps": epoch_steps,
        "updates_per_epoch": 1, "diagnostic_env_steps": epochs * epoch_steps,
        "initial_parameter_max_abs": initial_error, "initial_parameter_sha256": tree_hash(initial),
        "normalizer_max_abs_change": normalizer_error, "parameter_max_abs_change": parameter_change,
        "production_and_checkpoint_unchanged": unchanged, "compile": compile_record,
        "performance": performance, "trace_epoch_seconds": rows[-2]["complete_seconds"],
        "trace_to_untraced_median_ratio": rows[-2]["complete_seconds"] / performance["median_epoch_seconds"],
        "recovery_epoch_seconds": rows[-1]["complete_seconds"], "memory": memory,
        "boundary_transfers": boundaries, "trace_file": str(trace_candidates[0]),
        "total_wall_seconds": time.monotonic() - started}
    write_json(out / "result.json", result)
    status("complete", passed=passed, steps_per_second=performance["steps_per_second"])
    if not passed:
        raise RuntimeError("Scaling case validation failed; inspect retained result")


if __name__ == "__main__":
    main()
