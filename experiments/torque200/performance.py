"""Bounded ABBA training comparison with a separate live web evaluator.

A suspends the evaluator's simulation/rendering, while the browser stays open.
B runs the evaluator with at least one stream client. All four PPO runs restore
the same checkpoint with seed 0. The first two training intervals are excluded.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import threading
import time
import urllib.request


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2), encoding="utf8")
    temp.replace(path)


def request(base, route="/state", action=None):
    body = json.dumps({"action": action}).encode() if action else None
    req = urllib.request.Request(base + route, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as response:
        return json.load(response)


def switch(base, active):
    request(base, "/control", "resume" if active else "suspend")
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        state = request(base)
        if state.get("suspended") == (not active):
            if active and (state.get("clients", 0) < 1 or state.get("paused")):
                raise RuntimeError("Web condition needs an unpaused viewer and a real stream client")
            # Let outstanding work settle before sampling a condition.
            time.sleep(1)
            return request(base)
        time.sleep(0.1)
    raise RuntimeError("Viewer did not enter the requested condition")


def training_processes():
    found = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = (proc / "cmdline").read_bytes().split(b"\0")
            if any(Path(arg.decode(errors="replace")).name == "preview.py" for arg in args):
                found.append(int(proc.name))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    return found


def process_usage(pid):
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
        fields = text[text.rindex(")") + 2:].split()
        ticks = int(fields[11]) + int(fields[12])
        rss = int(fields[21]) * os.sysconf("SC_PAGE_SIZE") / 1024 ** 2
        return {"cpu_seconds": ticks / os.sysconf("SC_CLK_TCK"), "rss_mib": rss}
    except (FileNotFoundError, ProcessLookupError):
        return {}


def monitor(stop, samples, base, pid, started):
    smi = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
    while not stop.is_set():
        row = {"elapsed_seconds": time.monotonic() - started, **process_usage(pid)}
        try:
            result = subprocess.run([smi, "--query-gpu=utilization.gpu,memory.used,power.draw",
                                     "--format=csv,noheader,nounits"],
                                    capture_output=True, text=True, timeout=4, check=True)
            values = [float(x.strip()) for x in result.stdout.strip().splitlines()[0].split(",")]
            row.update(zip(("gpu_util_percent", "gpu_memory_mib", "gpu_power_w"), values))
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            row["gpu_query_error"] = str(exc)
        try:
            row["web"] = request(base)
        except Exception as exc:
            row["web_error"] = str(exc)
        samples.append(row)
        stop.wait(1.0)


def summarize_resources(samples):
    result = {"sample_count": len(samples), "scope": "whole round incl. initial compile, evaluation, saving/export"}
    for key in ("gpu_util_percent", "gpu_memory_mib", "gpu_power_w", "rss_mib"):
        values = [s[key] for s in samples if key in s]
        if values:
            result[key] = {"mean": statistics.mean(values), "max": max(values)}
    cpu = [s for s in samples if "cpu_seconds" in s]
    if len(cpu) > 1:
        result["training_process_cpu_percent_one_core_100"] = (
            100 * (cpu[-1]["cpu_seconds"] - cpu[0]["cpu_seconds"]) /
            (cpu[-1]["elapsed_seconds"] - cpu[0]["elapsed_seconds"]))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, default=Path("/home/jaccob/projects/tenny-torque200"))
    ap.add_argument("--web-url", default="http://127.0.0.1:8770")
    ap.add_argument("--restore", type=Path, required=True)
    args = ap.parse_args()
    if training_processes():
        raise RuntimeError(f"Close the separate native preview before benchmarking: {training_processes()}")
    original_web = request(args.web_url)
    if original_web.get("clients", 0) < 1:
        raise RuntimeError("Open the web viewer before starting the ABBA comparison")
    run_id = time.strftime("performance-%Y%m%d-%H%M%S")
    output = args.base / "artifacts" / run_id
    output.mkdir(parents=True, exist_ok=False)
    runs = []
    try:
        for name, active in (("A1", False), ("B1", True), ("B2", True), ("A2", False)):
            if (output / "STOP_AFTER_ROUND").exists() or (args.base / "artifacts/PERFORMANCE_STOP_AFTER_ROUND").exists():
                atomic_json(output / "stopped.json", {"completed": False, "reason": "requested stop between rounds", "runs": runs})
                atomic_json(args.base / "artifacts/performance-status.json", {"run_id": run_id, "completed": False, "stopped": True, "runs": runs})
                print("TORQUE200_PERFORMANCE_STOPPED " + str(output), flush=True)
                return
            before = switch(args.web_url, active)
            round_dir = output / name
            status = {"run_id": run_id, "completed": False, "round": name,
                      "condition": "web_live" if active else "web_suspended",
                      "progress_file": str(round_dir / "progress.json"),
                      "requested_steps_per_round": 32768, "planned_rounds": 4}
            atomic_json(args.base / "artifacts/performance-status.json", status)
            print(f"START {name} web_active={active}", flush=True)
            started = time.monotonic()
            samples = []
            with (output / f"{name}.log").open("w") as log:
                process = subprocess.Popen([
                    sys.executable, "-u", str(args.base / "src/performance_train.py"),
                    "--repo", str(args.base / "src"), "--output", str(round_dir),
                    "--restore", str(args.restore), "--steps", "32768", "--batch-size", "16",
                    "--eval-envs", "2", "--expected-platform", "gpu",
                ], stdout=log, stderr=subprocess.STDOUT)
                stop = threading.Event()
                sampler = threading.Thread(target=monitor, args=(stop, samples, args.web_url, process.pid, started))
                sampler.start()
                try:
                    code = process.wait(timeout=600)
                except BaseException:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise
                finally:
                    stop.set()
                    sampler.join(timeout=10)
                    atomic_json(output / f"{name}-resources.json", samples)
            if code != 0:
                raise RuntimeError(f"Training {name} exited {code}; see {output / (name + '.log')}")
            after = request(args.web_url)
            web_rows = [s["web"] for s in samples if "web" in s]
            if len(web_rows) < 2 or any(s.get("suspended") != (not active) for s in web_rows):
                raise RuntimeError("Viewer condition changed during measurement")
            if active and (any(s.get("clients", 0) < 1 or s.get("paused") for s in web_rows) or
                           after["frames"] <= before["frames"]):
                raise RuntimeError("Web live phase lost stream client or rendering progress")
            if not active and after["frames"] != before["frames"]:
                raise RuntimeError("Suspended viewer still rendered frames")
            report = json.loads((round_dir / "report.json").read_text())
            events = json.loads((round_dir / "progress.json").read_text())
            trained = [e for e in events if "training/walltime" in e["metrics"]]
            if len(trained) != 4:
                raise AssertionError(f"Expected four training intervals, got {len(trained)}")
            first, last = trained[1], trained[-1]
            stable_steps = last["steps"] - first["steps"]
            stable_seconds = last["metrics"]["training/walltime"] - first["metrics"]["training/walltime"]
            row = {"name": name, "web_active": active, "actual_steps": report["actual_timesteps"],
                   "steady_steps": stable_steps, "steady_training_seconds": stable_seconds,
                   "steady_sps": stable_steps / stable_seconds,
                   "steady_interval_sps": [e["metrics"]["training/sps"] for e in trained[2:]],
                   "first_two_training_intervals_excluded_steps": first["steps"],
                   "elapsed_seconds_including_startup_eval_export": time.monotonic() - started,
                   "resources": summarize_resources(samples),
                   "web_frames": after["frames"] - before["frames"],
                   "checkpoint": report["checkpoint"], "export_max_abs_error": report["export_max_abs_error"],
                   "parameter_tree_max_abs_error": report["parameter_tree_max_abs_error"],
                   "finite_training_losses_checked": report["finite_training_losses_checked"]}
            runs.append(row)
            atomic_json(output / "rounds.json", runs)
            print(json.dumps(row), flush=True)
        def throughput(active):
            selected = [r for r in runs if r["web_active"] == active]
            return sum(r["steady_steps"] for r in selected) / sum(r["steady_training_seconds"] for r in selected)
        baseline, web = throughput(False), throughput(True)
        summary = {"run_id": run_id, "completed": True,
                   "checkpoint": runs[-1]["checkpoint"],
                   "export_path": str(output / "A2/smoke_params.cpp.txt"),
                   "report_path": str(output / "comparison.json"),
                   "control_hz": 200, "total_environment_steps": sum(r["actual_steps"] for r in runs),
                   "baseline_sps": baseline, "web_sps": web,
                   "throughput_change_percent": 100 * (web / baseline - 1),
                   "same_start_checkpoint": str(args.restore), "seed_each_round": 0,
                   "runs": runs,
                   "note": "ABBA, 16 training environments, excludes first two training intervals of each round; A keeps browser open but suspends simulation/rendering. Small workload and two repeats per condition, not convergence or large-batch forecast."}
        atomic_json(output / "comparison.json", summary)
        atomic_json(args.base / "artifacts/performance-latest.json", summary)
        atomic_json(args.base / "artifacts/performance-status.json", summary)
        print("TORQUE200_PERFORMANCE_OK " + str(output), flush=True)
    except BaseException as exc:
        atomic_json(args.base / "artifacts/performance-status.json",
                    {"run_id": run_id, "completed": False, "error": repr(exc), "runs": runs})
        raise
    finally:
        request(args.web_url, "/control", "suspend" if original_web.get("suspended") else "resume")


if __name__ == "__main__":
    main()
