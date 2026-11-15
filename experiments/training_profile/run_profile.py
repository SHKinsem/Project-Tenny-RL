"""Bounded, isolated profiling of the installed Brax epoch and a frozen Tenny run."""
import argparse
from contextlib import ExitStack, nullcontext
from dataclasses import replace
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

from brax_instrumentation import instrument_brax, named_callable
from trace_summary import summarize_trace


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=lambda x: x.tolist(), allow_nan=False) + "\n",
                    encoding="utf8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--runtime-src", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=6)
    args = parser.parse_args(argv)
    if not 4 <= args.epochs <= 12:
        parser.error("--epochs must be in [4,12] (two warm-up epochs and one untraced final epoch)")
    out = args.output.expanduser().resolve()
    if not out.is_absolute() or str(out).startswith("/mnt/"):
        parser.error("Use an absolute WSL ext4 output directory")
    out.mkdir(parents=True, exist_ok=True)
    if (out / "result.json").exists():
        parser.error("Output already contains a completed profile; choose a new directory")
    started = time.monotonic()

    def status(stage, **fields):
        value = {"stage": stage, "elapsed_seconds": time.monotonic()-started, **fields}
        write_json(out / "status.json", value)
        with (out / "events.jsonl").open("a", encoding="utf8") as stream:
            stream.write(json.dumps(value) + "\n")
        print(json.dumps(value), flush=True)

    status("imports")
    sys.path.insert(0, str(args.runtime_src.resolve()))
    import jax
    import numpy as np
    # Persistent executables may otherwise retain metadata from an older graph.
    jax.config.update("jax_compilation_cache_include_metadata_in_key", True)
    from force_teacher.env import StudentEnv, student_config
    from force_teacher.ppo import train_student
    import force_teacher.networks as force_networks
    from force_teacher.contract import restore_guard, tree_digest
    from hybrid.core import source_fingerprint
    from legwheel.deployment.extract import load
    from mujoco_playground._src import mjx_env
    train_module = importlib.import_module("brax.training.agents.ppo.train")
    losses_module = importlib.import_module("brax.training.agents.ppo.losses")
    run = args.run_dir.expanduser().resolve()
    baseline = json.loads((run / "report.json").read_text())
    source_config = json.loads((run / "ppo_config.json").read_text())
    env_config = json.loads((run / "env_config.json").read_text())
    checkpoint = Path(baseline["checkpoint"])
    original_root = Path.home() / "projects/tenny-reference"
    restore_guard(checkpoint, original_root)
    source_sha = source_fingerprint()
    checkpoint_sha = tree_digest(checkpoint)
    if source_sha != baseline["source_fingerprint"] or checkpoint_sha != baseline["checkpoint_sha256"]:
        raise ValueError("Production source or checkpoint differs from the verified run")
    if jax.default_backend() != "gpu" or len(jax.devices()) != 1:
        raise ValueError("This bounded harness currently requires exactly one GPU")
    actual_config = json.loads(student_config().to_json_best_effort())
    if actual_config != env_config:
        raise ValueError("Current task config differs from the frozen run")
    installed_files = {str(Path(module.__file__).resolve()): hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                       for module in (train_module, losses_module)}
    per_step = source_config["batch_size"] * source_config["num_minibatches"] * source_config["unroll_length"] * source_config["action_repeat"]
    epoch_steps = per_step * math.ceil(source_config["num_timesteps"] /
        (max(source_config["num_evals"]-1, 1) * per_step * max(source_config.get("num_resets_per_eval", 0), 1)))
    config = dict(source_config, num_timesteps=epoch_steps*args.epochs, num_evals=args.epochs+1,
                  run_evals=False, save_checkpoint_path=None, restore_checkpoint_path=None)
    config.pop("policy_params_fn", None)
    write_json(out / "contract.json", {"run_id": run.name, "checkpoint_sha256": checkpoint_sha,
        "source_fingerprint": source_sha, "epoch_env_steps": epoch_steps, "epochs": args.epochs,
        "diagnostic_overrides": {key: config[key] for key in ("num_timesteps", "num_evals", "run_evals", "save_checkpoint_path", "restore_checkpoint_path")},
        "original_ppo_config": source_config, "environment_config": env_config,
        "limits_declared_before_measurement": {"initial_params_max_abs": 0., "first_epoch_params_max_abs": 1e-5},
        "cache_includes_metadata": True,
        "installed_source_sha256": installed_files,
        "scope": "same original pmap graph with named metadata only; no model save, evaluation, per-step callback or stage synchronization"})
    status("gpu_ready", devices=[str(d) for d in jax.devices()], epoch_env_steps=epoch_steps)

    def cpu_tree(value):
        return jax.tree.map(lambda item: np.asarray(item).copy(), jax.device_get(value))

    def compare_tree(before, after, details=False):
        old, old_tree = jax.tree_util.tree_flatten_with_path(before)
        new, new_tree = jax.tree_util.tree_flatten_with_path(after)
        if old_tree != new_tree:
            raise ValueError("Instrumented result tree differs from the original")
        errors = []
        for (key, a), (_, b) in zip(old, new, strict=True):
            a, b = np.asarray(a), np.asarray(b)
            if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError("Nonfinite or mismatched instrumented results")
            errors.append({"path": jax.tree_util.keystr(key), "shape": list(a.shape),
                           "max_abs": float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64))))})
        maximum = max((item["max_abs"] for item in errors), default=0.)
        return {"max_abs": maximum, "leaves": errors} if details else maximum

    def variant(label, marked):
        rows, snapshots, last_state = [], {}, None
        trace_active = False
        real_pmap = jax.pmap
        trace_root = out / "trace"
        manifest = None

        def capture(steps, factory, params):
            steps = int(steps)
            if steps in (0, epoch_steps, epoch_steps*args.epochs):
                snapshots[steps] = cpu_tree(params)

        def record_pmap(function, *pmap_args, **pmap_kwargs):
            compiled = real_pmap(function, *pmap_args, **pmap_kwargs)
            if getattr(function, "__name__", "") != "training_epoch":
                return compiled
            executable = None

            @wraps(function)
            def epoch(*values, **options):
                nonlocal executable, trace_active, last_state
                index = len(rows)
                if executable is None:
                    status(label + ".inputs_ready")
                    t = time.monotonic(); jax.block_until_ready((values, options)); input_wait = time.monotonic()-t
                    status(label + ".lowering")
                    t = time.monotonic(); lowered = compiled.lower(*values, **options); lowering = time.monotonic()-t
                    # XlaComputation.as_hlo_text() omits debug metadata by default.
                    hlo = lowered.as_text(debug_info=True)
                    if marked and "ppo.rollout" not in hlo:
                        raise RuntimeError("Stage metadata is missing before compilation")
                    with gzip.open(out / (label + "-lowered-hlo.txt.gz"), "wt", encoding="utf8") as stream:
                        stream.write(hlo)
                    status(label + ".compiling")
                    t = time.monotonic(); executable = lowered.compile(); compile_seconds = time.monotonic()-t
                    with gzip.open(out / (label + "-compiled-hlo.txt.gz"), "wt", encoding="utf8") as stream:
                        stream.write(executable.as_text())
                    write_json(out / (label + "-compile.json"), {"inputs_ready_seconds": input_wait,
                        "tracing_lowering_seconds": lowering, "xla_compile_seconds": compile_seconds})
                traced = marked and 2 <= index < args.epochs-1
                if traced and not trace_active:
                    status(label + ".start_trace")
                    profile_options = jax.profiler.ProfileOptions()
                    profile_options.python_tracer_level = 0
                    profile_options.host_tracer_level = 1
                    profile_options.enable_hlo_proto = True
                    jax.profiler.start_trace(str(trace_root), create_perfetto_trace=True,
                                             profiler_options=profile_options)
                    trace_active = True
                elif trace_active and not traced:
                    status(label + ".stop_trace")
                    jax.profiler.stop_trace(); trace_active = False
                status(label + ".epoch", epoch=index, traced=traced)
                shadow = None
                if index == 0:
                    shadow = jax.tree.map(lambda item: item.copy(), (values, options))
                    jax.block_until_ready(shadow)
                annotation = jax.profiler.StepTraceAnnotation("ppo.host_epoch", step_num=index, variant=label)
                with annotation:
                    t = time.monotonic()
                    result = executable(*values, **options)
                    submitted = time.monotonic()
                    jax.block_until_ready(result)
                    completed = time.monotonic()
                if shadow is not None:
                    # Same executable, values and keys: measure the GPU's own repeatability.
                    original_params = cpu_tree(result[0].params)
                    original_physics = cpu_tree((result[1].data.qpos, result[1].data.qvel))
                    repeated = jax.block_until_ready(executable(*shadow[0], **shadow[1]))
                    write_json(out / (label + "-reproducibility.json"), {
                        "parameter_max_abs": compare_tree(original_params, cpu_tree(repeated[0].params)),
                        "physics_max_abs": compare_tree(original_physics, cpu_tree((repeated[1].data.qpos, repeated[1].data.qvel))),
                        "extra_diagnostic_env_steps": epoch_steps})
                last_state = result[1]
                row = {"epoch": index, "traced": traced, "env_steps": epoch_steps,
                       "submit_seconds": submitted-t, "completion_wait_seconds": completed-submitted,
                       "complete_seconds": completed-t}
                rows.append(row)
                write_json(out / (label + "-epochs.json"), rows)
                return result
            return epoch

        environment = StudentEnv(config=student_config())
        eval_environment = StudentEnv(config=student_config())
        options = dict(config, policy_params_fn=capture)
        try:
            with ExitStack() as stack:
                if marked:
                    manifest = stack.enter_context(instrument_brax(train_module, losses_module))
                    for method, name in (("_step_control", "ppo.control"), ("_get_reward", "ppo.reward"),
                                         ("_get_obs", "ppo.obs")):
                        stack.enter_context(patch.object(environment, method, named_callable(getattr(environment, method), name)))
                    stack.enter_context(patch.object(mjx_env, "step", named_callable(mjx_env.step, "ppo.physics")))
                    original_factory = force_networks.make_ppo_networks

                    @wraps(original_factory)
                    def factory(*values, **kwargs):
                        network = original_factory(*values, **kwargs)
                        return network.replace(
                            policy_network=replace(network.policy_network, apply=named_callable(network.policy_network.apply, "ppo.policy")),
                            value_network=replace(network.value_network, apply=named_callable(network.value_network.apply, "ppo.value")))
                    stack.enter_context(patch.object(force_networks, "make_ppo_networks", factory))
                stack.enter_context(patch.object(jax, "pmap", record_pmap))
                status(label + ".setup")
                params = load(str(checkpoint))
                _, result_params, _ = train_student(env=environment, eval_env=eval_environment,
                    env_cfg=environment._config, ppo_params=options, progress_fn=lambda *a: None, params=params)
                final_params = cpu_tree(result_params)
                final_physics = cpu_tree((last_state.data.qpos, last_state.data.qvel))
            if len(rows) != args.epochs:
                raise ValueError("Unexpected Brax epoch count; profile work differs from the contract")
        finally:
            if trace_active:
                jax.profiler.stop_trace()
        if manifest is not None:
            write_json(out / "instrumentation.json", manifest)
        return {"epochs": rows, "snapshots": snapshots, "params": final_params, "physics": final_physics}

    plain = variant("original", False)
    marked = variant("marked", True)

    difference = compare_tree
    initial_error = difference(plain["snapshots"][0], marked["snapshots"][0])
    first_error = difference(plain["snapshots"][epoch_steps], marked["snapshots"][epoch_steps])
    write_json(out / "first-epoch-leaf-differences.json", compare_tree(
        plain["snapshots"][epoch_steps], marked["snapshots"][epoch_steps], details=True))
    final_error = difference(plain["params"], marked["params"])
    physics_error = difference(plain["physics"], marked["physics"])
    status("analyzing_trace")
    candidates = sorted((out / "trace").rglob("perfetto_trace.json.gz"))
    if len(candidates) != 1:
        raise ValueError("Expected exactly one collected device trace")
    summary = summarize_trace(candidates[0], window_name="ppo.host_epoch",
        compiled_hlo_path=out / "marked-compiled-hlo.txt.gz",
        annotated_mlir_path=out / "marked-lowered-hlo.txt.gz")
    write_json(out / "trace-summary.json", summary)
    untraced_indices = (1, args.epochs-1)
    untraced_ratios = [marked["epochs"][i]["complete_seconds"] / plain["epochs"][i]["complete_seconds"] for i in untraced_indices]
    traced_ratios = [marked["epochs"][i]["complete_seconds"] / plain["epochs"][i]["complete_seconds"] for i in range(2, args.epochs-1)]
    unchanged = source_fingerprint() == source_sha and tree_digest(checkpoint) == checkpoint_sha
    for file, digest in installed_files.items():
        unchanged = unchanged and hashlib.sha256(Path(file).read_bytes()).hexdigest() == digest
    result = {"passed": initial_error == 0 and first_error <= 1e-5 and unchanged and summary.get("supported") is True
                       and summary.get("classification_coverage_fraction", 0) >= .8,
        "strict_parameter_gate_passed": first_error <= 1e-5,
        "usable_for_device_hotspot_analysis": initial_error == 0 and unchanged and summary.get("supported") is True
                       and summary.get("classification_coverage_fraction", 0) >= .8,
        "run_id": run.name, "checkpoint_sha256": checkpoint_sha, "epoch_env_steps": epoch_steps,
        "diagnostic_env_steps_per_variant": epoch_steps*args.epochs,
        "original_epochs": plain["epochs"], "marked_epochs": marked["epochs"],
        "initial_parameter_max_abs": initial_error, "first_epoch_parameter_max_abs": first_error,
        "final_parameter_max_abs": final_error, "final_physics_max_abs": physics_error,
        "untraced_annotation_time_ratio_median": statistics.median(untraced_ratios),
        "traced_time_ratio_median": statistics.median(traced_ratios),
        "production_and_checkpoint_unchanged": unchanged,
        "total_wall_seconds": time.monotonic()-started,
        "trace_file": str(candidates[0]), "trace_summary": summary}
    write_json(out / "result.json", result)
    status("complete", passed=result["passed"], annotation_ratio=result["untraced_annotation_time_ratio_median"],
           trace_ratio=result["traced_time_ratio_median"], first_epoch_error=first_error)
    if not result["passed"]:
        raise RuntimeError("Profile acceptance failed; inspect result.json without relaxing its limits")


if __name__ == "__main__":
    main()
