"""Bounded 128/256/512 environment matrix with a common PPO batch."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import urlopen

from launch import stop_process_group
from run_profile import write_json


def matrix_counts(envs, batch_size):
    if not envs or len(set(envs)) != len(envs) or len(envs) > 6:
        raise ValueError("Provide one to six distinct environment counts")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("Common batch size must be a positive integer")
    if any(type(n) is not int or n <= 0 or batch_size % n for n in envs):
        raise ValueError("Each positive environment count must divide the common batch")
    return list(envs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torque-root", type=Path, default=Path.home() / "projects/tenny-torque200")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--envs", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--measure-epochs", type=int, default=8)
    parser.add_argument("--max-seconds", type=float, default=480)
    parser.add_argument("--case-seconds", type=float, default=240)
    parser.add_argument("--compile-seconds", type=float, default=120)
    args = parser.parse_args(argv)
    if os.name != "posix":
        parser.error("Run this launcher inside WSL/Linux")
    try:
        envs = matrix_counts(args.envs, args.batch_size)
    except ValueError as error:
        parser.error(str(error))
    if not 1 <= args.warmup_epochs <= 4 or not 3 <= args.measure_epochs <= 16:
        parser.error("Use 1..4 warm-up and 3..16 measurement epochs")
    if not 0 < args.max_seconds <= 900 or not 0 < args.case_seconds <= 300 or not 0 < args.compile_seconds <= 180:
        parser.error("Deadlines must be positive and bounded (900/300/180 seconds)")
    import fcntl
    root = args.torque_root.expanduser().resolve()
    work = Path(os.environ.get("TENNY_SCALING_WORKDIR", Path.home() / "projects/tenny-parallel-scaling"))
    out = (args.output or work / ("matrix-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))).expanduser().resolve()
    if str(out).startswith("/mnt/") or (out / "matrix.json").exists():
        parser.error("Use a fresh WSL ext4 output directory")
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root / "src"))
    from hybrid.managed_training import runtime_root
    from hybrid.readiness import prepare_environment
    from hybrid.core import source_fingerprint
    from force_teacher.contract import tree_digest
    environment = prepare_environment(runtime_root(), dict(os.environ, JAX_PLATFORMS="cuda"))
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    receipt = {"envs": envs, "common_batch_size": args.batch_size,
        "max_seconds": args.max_seconds, "case_seconds": args.case_seconds,
        "compile_seconds": args.compile_seconds, "shutdown_grace_seconds": 25,
        "cases": [], "output": str(out)}

    def save():
        write_json(out / "matrix.json", receipt)

    def state():
        with urlopen("http://127.0.0.1:8770/state", timeout=10) as response:
            return json.load(response)

    def gpu_sample(label, stage):
        try:
            query = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,power.draw",
                "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=2)
            fields = query.stdout.strip().splitlines()[0].split(",")
            sample = {"elapsed_seconds": time.monotonic()-started, "num_envs": label, "stage": stage,
                "memory_used_mib": float(fields[0]), "memory_total_mib": float(fields[1]),
                "utilization_percent": float(fields[2]), "power_watts": float(fields[3])}
        except (OSError, ValueError, IndexError, subprocess.TimeoutExpired) as error:
            sample = {"elapsed_seconds": time.monotonic()-started, "num_envs": label,
                      "stage": stage, "unsupported": str(error)}
        with (out / "resources.jsonl").open("a", encoding="utf8") as stream:
            stream.write(json.dumps(sample) + "\n")

    started = time.monotonic()
    with ExitStack() as stack:
        for target in sorted({root, runtime_root()}):
            stream = stack.enter_context((target / ".runner.lock").open("a+b"))
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        before = state()
        if before["training"].get("active"):
            parser.error("Production training is active; matrix was not launched")
        run = (args.run_dir or Path(before["training"]["latest_checkpoint"]).parents[1]).expanduser().resolve()
        report = json.loads((run / "report.json").read_text())
        checkpoint_sha = tree_digest(Path(report["checkpoint"]))
        source_sha = source_fingerprint()
        export_sha = hashlib.sha256(Path(report["export_path"]).read_bytes()).hexdigest()
        if source_sha != report["source_fingerprint"] or checkpoint_sha != report["checkpoint_sha256"]:
            raise ValueError("Input model/source differs from the verified run")
        receipt.update(run_dir=str(run), checkpoint_sha256=checkpoint_sha, source_fingerprint=source_sha,
                       export_sha256=export_sha, run_id=report["run_id"], before_state=before)
        save()
        child = None
        try:
            for count in envs:
                if time.monotonic()-started >= args.max_seconds:
                    receipt["stop_reason"] = "matrix_wall_budget"
                    break
                case_out = out / ("envs-" + str(count))
                case_out.mkdir()
                case = {"num_envs": count, "output": str(case_out)}
                receipt["cases"].append(case)
                with (case_out / "console.log").open("ab") as log:
                    child = subprocess.Popen([sys.executable, str(Path(__file__).with_name("run_scaling.py")),
                        "--run-dir", str(run), "--runtime-src", str(root / "src"), "--output", str(case_out),
                        "--num-envs", str(count), "--batch-size", str(args.batch_size),
                        "--warmup-epochs", str(args.warmup_epochs), "--measure-epochs", str(args.measure_epochs)],
                        cwd=case_out, env=environment, stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                case.update(pid=child.pid)
                save()
                case_started, compile_started, previous_stage, last_sample = time.monotonic(), None, None, -float("inf")
                try:
                    while child.poll() is None:
                        now = time.monotonic()
                        try:
                            current = json.loads((case_out / "status.json").read_text())
                        except (OSError, ValueError):
                            current = {}
                        stage = current.get("stage", "")
                        if stage != previous_stage:
                            print(json.dumps({"num_envs": count, **current}), flush=True)
                            if stage == "lowering":
                                compile_started = now
                            elif stage == "compiling":
                                compile_started = compile_started or now
                            else:
                                compile_started = None
                            previous_stage = stage
                        reason = None
                        if now-started >= args.max_seconds:
                            reason = "matrix_wall_budget"
                        elif now-case_started >= args.case_seconds:
                            reason = "case_wall_budget"
                        elif compile_started is not None and now-compile_started >= args.compile_seconds:
                            reason = "compile_budget"
                        if reason:
                            case["stop_reason"] = reason
                            receipt["stop_reason"] = reason
                            stop_process_group(child)
                            break
                        if now-last_sample >= 1:
                            gpu_sample(count, stage)
                            last_sample = now
                        time.sleep(.25)
                except BaseException:
                    stop_process_group(child)
                    raise
                finally:
                    case.update(exit_code=child.poll(), wall_seconds=time.monotonic()-case_started)
                    if (case_out / "result.json").exists():
                        result = json.loads((case_out / "result.json").read_text())
                        case.update(passed=result["passed"], performance=result["performance"],
                                    initial_parameter_sha256=result["initial_parameter_sha256"])
                    save()
                if child.returncode != 0 or case.get("passed") is not True:
                    receipt.setdefault("stop_reason", "case_failed")
                    break
        finally:
            # Also cover failures between Popen and the inner monitor's try.
            if child is not None and child.poll() is None:
                stop_process_group(child)
            after = state()
            receipt.update(wall_seconds=time.monotonic()-started, after_state=after,
                source_unchanged=source_fingerprint() == source_sha,
                checkpoint_unchanged=tree_digest(Path(report["checkpoint"])) == checkpoint_sha,
                export_unchanged=hashlib.sha256(Path(report["export_path"]).read_bytes()).hexdigest() == export_sha,
                production_run_unchanged=after["training"]["run_id"] == before["training"]["run_id"],
                console_idle=not after["training"]["active"])
            hashes = {case.get("initial_parameter_sha256") for case in receipt["cases"]}
            receipt["same_initial_parameters"] = None not in hashes and len(hashes) == 1
            receipt["passed"] = (len(receipt["cases"]) == len(envs) and
                all(case.get("passed") is True for case in receipt["cases"]) and
                all(receipt[key] for key in ("source_unchanged", "checkpoint_unchanged", "export_unchanged",
                                            "production_run_unchanged", "console_idle", "same_initial_parameters")))
            save()
    print(json.dumps({key: value for key, value in receipt.items() if key not in ("before_state", "after_state")}), flush=True)
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
