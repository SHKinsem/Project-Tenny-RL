"""Five-minute capped GPU PPO throughput probe; no robot-quality evaluation.

Uses performance_train's real PPO and checkpoint/export checks. Only its
Evaluator.run_evaluation is replaced, inside each isolated child, to return
training metrics without expensive eval rollouts. No installed files changed.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request


def web(action=None):
    payload = json.dumps({"action": action}).encode() if action else None
    route = "/control" if action else "/state"
    request = urllib.request.Request("http://127.0.0.1:8770" + route, data=payload,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=3) as response:
        return json.load(response)


def child(args):
    # Load the project's compatibility patch before installing the scoped probe.
    from legwheel.training import ppo  # noqa: F401
    from brax.training import acting
    from unittest.mock import patch
    import performance_train
    sys.argv = ["performance_train.py", "--repo", str(Path(__file__).parent),
                "--output", str(args.output), "--restore", str(args.restore),
                "--steps", "32768", "--batch-size", str(args.envs),
                "--eval-envs", "2", "--expected-platform", "gpu"]

    def metrics_only(self, params, training_metrics, aggregate_episodes=True):
        return dict(training_metrics)

    with patch.object(acting.Evaluator, "run_evaluation", metrics_only):
        performance_train.main()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--envs", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--restore", type=Path)
    args = parser.parse_args()
    if args.child:
        return child(args)
    base = (Path.home() / "projects/tenny-torque200")
    latest = json.loads((base / "artifacts/performance-latest.json").read_text())
    restore = Path(latest["checkpoint"])
    output = base / "artifacts" / time.strftime("throughput-%Y%m%d-%H%M%S")
    output.mkdir()
    print(f"PROBE_OUTPUT {output}", flush=True)
    original_web = None
    results = []
    started = time.monotonic()
    process = None
    try:
        try:
            original_web = web()
            web("suspend")
            time.sleep(0.2)
        except OSError as exc:
            raise RuntimeError("Cannot confirm viewer is suspended; no throughput test started") from exc
        if not web().get("suspended"):
            raise RuntimeError("Viewer did not suspend; no throughput test started")
        for envs in (16, 64, 128):
            remaining = 300 - (time.monotonic() - started)
            if remaining < 15:
                break
            candidate = output / f"envs-{envs}"
            limit = min(100, remaining)
            print(f"START envs={envs} timeout={limit:.0f}s", flush=True)
            round_start = time.monotonic()
            samples = []
            with (output / f"envs-{envs}.log").open("w") as log:
                process = subprocess.Popen([sys.executable, "-u", __file__, "--child",
                                            "--envs", str(envs), "--output", str(candidate),
                                            "--restore", str(restore)], stdout=log, stderr=subprocess.STDOUT)
                timed_out = False
                while process.poll() is None:
                    if time.monotonic() - round_start >= limit:
                        timed_out = True
                        process.terminate()
                        try:
                            process.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        break
                    try:
                        smi = subprocess.run(["/usr/lib/wsl/lib/nvidia-smi",
                            "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, timeout=2, check=True)
                        util, memory = map(float, smi.stdout.strip().splitlines()[0].split(","))
                        samples.append({"seconds": time.monotonic() - round_start,
                                        "gpu_util_percent": util, "gpu_memory_mib": memory})
                    except (OSError, ValueError, subprocess.SubprocessError):
                        pass
                    time.sleep(0.5)
                code = process.returncode
                process = None
            row = {"envs": envs, "elapsed_seconds": time.monotonic() - round_start,
                   "status": "timeout" if timed_out else "passed" if code == 0 else "failed",
                   "exit_code": code, "resource_samples": samples}
            if samples:
                row["device_memory_peak_mib"] = max(s["gpu_memory_mib"] for s in samples)
            if row["status"] == "passed":
                report = json.loads((candidate / "report.json").read_text())
                events = json.loads((candidate / "progress.json").read_text())
                trained = [e for e in events if "training/walltime" in e["metrics"]]
                if len(trained) != 4:
                    raise AssertionError("Expected four measured training intervals")
                warm, final = trained[1], trained[-1]
                seconds = final["metrics"]["training/walltime"] - warm["metrics"]["training/walltime"]
                steps = final["steps"] - warm["steps"]
                row.update(steady_sps=steps / seconds, steady_seconds=seconds, steady_steps=steps,
                           interval_sps=[e["metrics"]["training/sps"] for e in trained[2:]],
                           actual_steps=report["actual_timesteps"], checkpoint=report["checkpoint"],
                           parameter_error=report["parameter_tree_max_abs_error"],
                           export_error=report["export_max_abs_error"],
                           finite_losses=report["finite_training_losses_checked"])
            results.append(row)
            (output / "results.json").write_text(json.dumps(results, indent=2))
            print(json.dumps({k: v for k, v in row.items() if k != "resource_samples"}), flush=True)
            # Never extend the budget to obtain a missing candidate.
            if timed_out:
                break
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if original_web is not None:
            try:
                web("suspend" if original_web["suspended"] else "resume")
            except OSError:
                pass
        passed = [r for r in results if r["status"] == "passed"]
        fastest = max(passed, key=lambda r: r["steady_sps"]) if passed else None
        summary = {"elapsed_seconds": time.monotonic() - started, "hard_budget_seconds": 300,
                   "restore": str(restore), "results": results,
                   "fastest_envs": fastest["envs"] if fastest else None,
                   "note": "True PPO sampling and updates, no eval rollouts. First two intervals excluded. Batch size changes with env count; learning/sample efficiency not established. GPU memory is device-wide."}
        (output / "summary.json").write_text(json.dumps(summary, indent=2))
        if fastest:
            config = json.loads((output / f"envs-{fastest['envs']}" / "ppo_config.json").read_text())
            (output / "fastest-tested-ppo-config.json").write_text(json.dumps(config, indent=2))
        print(f"PROBE_DONE {output} fastest_envs={summary['fastest_envs']}", flush=True)


if __name__ == "__main__":
    main()
