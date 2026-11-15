"""Launch an isolated profile with production GPU settings and explicit deadlines."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.request import urlopen


def stop_process_group(child, interrupt_seconds=15, terminate_seconds=5):
    """Bound cancellation even when a native compiler ignores Python signals."""
    for sig, timeout in ((signal.SIGINT, interrupt_seconds),
                         (signal.SIGTERM, terminate_seconds), (signal.SIGKILL, 5)):
        if child.poll() is not None:
            return
        try:
            os.killpg(child.pid, sig)
        except ProcessLookupError:
            return
        try:
            child.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            pass
    raise RuntimeError("Profile process group did not exit after SIGKILL")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torque-root", type=Path, default=Path.home() / "projects/tenny-torque200")
    parser.add_argument("--run-dir", type=Path, help="Verified input run; defaults to the idle console's latest model")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--max-seconds", type=float, default=300)
    parser.add_argument("--gpu-session-seconds", type=float, default=180)
    parser.add_argument("--compile-seconds", type=float, default=120)
    args = parser.parse_args(argv)
    if os.name != "posix":
        parser.error("Run this launcher inside WSL/Linux")
    if not 4 <= args.epochs <= 12:
        parser.error("--epochs must be in [4,12]")
    for name in ("max_seconds", "gpu_session_seconds", "compile_seconds"):
        if not 0 < getattr(args, name) <= 300:
            parser.error("All deadlines must be positive and at most 300 seconds")
    import fcntl
    root = args.torque_root.expanduser().resolve()
    work = Path(os.environ.get("TENNY_PROFILE_WORKDIR", Path.home() / "projects/tenny-ppo-profile"))
    out = (args.output or work / ("profile-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))).expanduser().resolve()
    if str(out).startswith("/mnt/") or (out / "launch.json").exists():
        parser.error("Use a fresh WSL ext4 output directory")
    out.mkdir(parents=True, exist_ok=True)
    with urlopen("http://127.0.0.1:8770/state", timeout=10) as response:
        state = json.load(response)
    if state["training"].get("active"):
        parser.error("Production training is active; profiling was not launched")
    run = (args.run_dir or Path(state["training"]["latest_checkpoint"]).parents[1]).expanduser().resolve()
    sys.path.insert(0, str(root / "src"))
    from hybrid.managed_training import runtime_root
    from hybrid.readiness import prepare_environment
    environment = prepare_environment(runtime_root(), dict(os.environ, JAX_PLATFORMS="cuda"))
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    receipt = {"run_dir": str(run), "output": str(out), "epochs": args.epochs,
               "max_seconds": args.max_seconds, "gpu_session_seconds": args.gpu_session_seconds,
               "compile_seconds": args.compile_seconds}

    def save_receipt():
        (out / "launch.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf8")

    save_receipt()
    with ExitStack() as stack:
        for target in sorted({root, runtime_root()}):
            stream = stack.enter_context((target / ".runner.lock").open("a+b"))
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (out / "console.log").open("ab") as log:
            child = subprocess.Popen([sys.executable, str(Path(__file__).with_name("run_profile.py")),
                "--run-dir", str(run), "--runtime-src", str(root / "src"), "--output", str(out),
                "--epochs", str(args.epochs)], cwd=out, env=environment, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        receipt["pid"] = child.pid
        save_receipt()
        started = time.monotonic()
        gpu_started = None
        compile_started = None
        previous_stage = None
        try:
            while child.poll() is None:
                now = time.monotonic()
                try:
                    current = json.loads((out / "status.json").read_text(encoding="utf8"))
                except (OSError, ValueError):
                    current = {}
                stage = current.get("stage", "")
                if gpu_started is None and (stage == "gpu_ready" or stage.startswith(("original.", "marked."))):
                    # Conservative if a fast transition passed between polls.
                    gpu_started = now
                if stage != previous_stage:
                    print(json.dumps(current), flush=True)
                    if stage.endswith(".lowering"):
                        compile_started = now
                    elif not stage.endswith(".compiling"):
                        compile_started = None
                    previous_stage = stage
                reason = None
                if now-started > args.max_seconds:
                    reason = "command_wall_budget"
                elif gpu_started is not None and now-gpu_started > args.gpu_session_seconds and stage not in ("analyzing_trace", "complete"):
                    reason = "gpu_session_budget"
                elif compile_started is not None and now-compile_started > args.compile_seconds:
                    reason = "compile_budget"
                if reason:
                    receipt["stop_reason"] = reason
                    stop_process_group(child)
                    break
                time.sleep(.5)
        except BaseException:
            if child.poll() is None:
                stop_process_group(child)
            raise
        finally:
            receipt.update(exit_code=child.poll(), wall_seconds=time.monotonic()-started)
            receipt["deadline_shutdown_grace_seconds"] = 25
            save_receipt()
    print(json.dumps(receipt), flush=True)
    return child.returncode


if __name__ == "__main__":
    raise SystemExit(main())
