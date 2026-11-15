"""Build-independent descriptor export and bounded native evaluation commands."""
import argparse
import json
from pathlib import Path
import statistics
import time

from .api import Evaluator, capabilities


def _json_value(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(file, value):
    file = Path(file).expanduser().resolve()
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps(value, default=_json_value, indent=2, allow_nan=False) + "\n",
                    encoding="utf8")


def _initial_states(file):
    if file is None:
        return None
    file = Path(file).expanduser().resolve()
    if file.suffix.lower() != ".npz":
        value = json.loads(file.read_text(encoding="utf8"))
        if not isinstance(value, list):
            raise ValueError("Initial-state JSON must contain a list of world dictionaries")
        return value
    import numpy as np
    with np.load(file, allow_pickle=False) as arrays:
        if "qpos" not in arrays or arrays["qpos"].ndim != 2:
            raise ValueError("Initial-state NPZ needs qpos[world, coordinate]")
        count = arrays["qpos"].shape[0]
        result = [{"qpos": arrays["qpos"][world].tolist(), "extra": {}} for world in range(count)]
        for key in arrays.files:
            if key == "observation":  # Legacy comparison input, not a reset field.
                continue
            values = arrays[key]
            if values.ndim == 0 or values.shape[0] != count:
                raise ValueError(f"Initial-state field {key} needs one value per world")
            for world, item in enumerate(result):
                if key in ("qpos", "qvel", "command"):
                    item[key] = values[world].tolist()
                else:
                    item["extra"][key] = np.atleast_1d(values[world]).tolist()
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Deterministic native MuJoCo evaluation")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("capabilities", help="Show supported native task and policy kinds")
    export = commands.add_parser("export-run", help="Export a verified hybrid run as a descriptor")
    export.add_argument("run_dir", type=Path)
    export.add_argument("--runtime-src", type=Path)
    export.add_argument("--output", type=Path, required=True)
    evaluate = commands.add_parser("evaluate", help="Evaluate an existing descriptor without training")
    evaluate.add_argument("descriptor", type=Path)
    evaluate.add_argument("--initial", type=Path, help="World dictionaries in JSON or reset arrays in NPZ")
    evaluate.add_argument("--num-envs", type=int)
    evaluate.add_argument("--steps", type=int)
    evaluate.add_argument("--episode-steps", type=int)
    evaluate.add_argument("--control-dt", type=float)
    evaluate.add_argument("--seed", type=int)
    evaluate.add_argument("--repeats", type=int, default=1)
    evaluate.add_argument("--trace-world", type=int, action="append", default=[])
    evaluate.add_argument("--trace-output", type=Path, help="First trial's traces as NPZ (separate from timing JSON)")
    evaluate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "capabilities":
            print(json.dumps(capabilities(), indent=2))
            return
        if args.command == "export-run":
            from .legwheel import descriptor_from_run
            descriptor = descriptor_from_run(args.run_dir, args.runtime_src)
            _write_json(args.output, descriptor)
            print(json.dumps({"descriptor": str(args.output), "metadata": descriptor["metadata"]}))
            return
        if args.repeats < 1:
            parser.error("--repeats must be positive")
        if args.trace_world and args.trace_output is None:
            parser.error("--trace-world requires --trace-output")
        if args.trace_output is not None and not args.trace_world:
            parser.error("--trace-output requires at least one --trace-world")
        initial = _initial_states(args.initial)
        start = time.perf_counter()
        engine = Evaluator(args.descriptor)
        setup_seconds = time.perf_counter() - start
        trials = []
        for index in range(args.repeats):
            start = time.perf_counter()
            result = engine.evaluate(initial, num_envs=args.num_envs, steps=args.steps,
                                     episode_steps=args.episode_steps, control_dt=args.control_dt,
                                     seed=args.seed, trace_worlds=args.trace_world if index == 0 else [])
            seconds = time.perf_counter() - start
            traces = result.pop("traces")
            trials.append({"index": index, "seconds": seconds, "includes_trace": bool(traces), **result})
            if traces:
                import numpy as np
                target = args.trace_output.expanduser().resolve()
                target.parent.mkdir(parents=True, exist_ok=True)
                np.savez(target, **{f"world_{trace['world']}_{key}": value
                                   for trace in traces for key, value in trace.items() if key != "world"})
        measured = [trial["seconds"] for trial in trials if not trial["includes_trace"]]
        report = {"schema_version": 1, "backend": "native_mujoco_cpp", "model": engine.model_info,
                  "metadata": engine.descriptor.get("metadata", {}), "setup_seconds": setup_seconds,
                  "median_seconds_without_trace": statistics.median(measured) if measured else None,
                  "trials": trials}
        _write_json(args.output, report)
        print(json.dumps({"report": str(args.output), "median_seconds_without_trace":
                          report["median_seconds_without_trace"], "trials": len(trials)}))
    except (ValueError, TypeError, RuntimeError, OSError, KeyError) as exc:
        parser.exit(2, f"native_eval: {exc}\n")


if __name__ == "__main__":
    main()
