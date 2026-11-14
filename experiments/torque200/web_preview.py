"""Process-isolated original / torque200 policy preview; never starts training."""
import argparse
import io
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import time
from control_modes import MODES, mode_spec



def persist_mode(artifacts, mode):
    if mode not in MODES:
        raise ValueError("Unknown mode")
    artifacts.mkdir(parents=True, exist_ok=True)
    target = artifacts / "selected-mode.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps({"mode": mode}), encoding="utf8")
    temporary.replace(target)


def restart_arguments(args, mode):
    if mode not in MODES:
        raise ValueError("Unknown mode")
    return [sys.executable, str(Path(__file__).resolve()), "--artifacts", str(args.artifacts),
            "--original-root", str(args.original_root), "--host", args.host,
            "--port", str(args.port), "--mode", mode]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--mode", choices=MODES, default="original")
    parser.add_argument("--original-root", type=Path, default=(Path.home() / "projects/tenny-reference"))
    args = parser.parse_args()
    persist_mode(args.artifacts, args.mode)
    if mode_spec(args.mode)["hybrid"]:
        from hybrid_web import main as hybrid_main
        return hybrid_main(args)
    from preview_sync import read_snapshot, utc_now
    from training_control import TrainingController, ControlError
    controller = TrainingController(args.artifacts, args.original_root, args.artifacts.parent)
    console_model = controller.latest(args.mode)

    import jax
    import mujoco
    import numpy as np
    from PIL import Image

    # Each change execs a new interpreter: never reuse an imported legwheel package.
    if args.mode == "original":
        source_root = args.original_root / "mujoco_legwheel_rl_public"
        deploy_root = args.original_root / "mujoco_esp32_robot_deploy_public"
        if not (deploy_root / "host/policy/rl_policy.py").is_file():
            raise FileNotFoundError(f"Original deployment not found: {deploy_root}")
        sys.path.insert(0, str(deploy_root))
    else:
        script_root = Path(__file__).resolve().parent
        source_root = script_root if (script_root / "legwheel/__init__.py").is_file() else script_root / "src"
    if not (source_root / "legwheel/__init__.py").is_file():
        raise FileNotFoundError(f"Selected environment not found: {source_root}")
    sys.path.insert(0, str(source_root))
    from legwheel import LegWheelRobot, default_config

    checkpoint = None
    export_file = None
    if args.mode == "torque200":
        from preview import select_checkpoint
        checkpoint = select_checkpoint(args.artifacts)
        for name in ("performance-latest.json", "training-latest.json"):
            manifest = args.artifacts / name
            if manifest.is_file():
                result = json.loads(manifest.read_text())
                candidate = Path(result["checkpoint"]).resolve()
                previous_report = checkpoint.parent.parent / "report.json"
                if (result.get("completed") and candidate.is_dir() and
                        candidate.is_relative_to(args.artifacts.resolve()) and
                        manifest.stat().st_mtime > previous_report.stat().st_mtime):
                    checkpoint = candidate
    if console_model is not None:
        checkpoint = Path(console_model["checkpoint"])
        export_file = Path(console_model["export_path"])
    cfg = default_config()
    cfg.noise_config.level = 0.0
    cfg.push_config.enable = False
    cfg.cmd_config.cmd_prob = 0.0
    env = LegWheelRobot(cfg)
    reset = jax.jit(env.reset)
    if args.mode == "original":
        from host.policy.rl_policy import RL_Policy
        from host.policy.ckpt import POLICY_CPP
        actor = RL_Policy(path=export_file) if export_file is not None else RL_Policy()
        checkpoint = checkpoint if export_file is not None else POLICY_CPP
        step = jax.jit(env.step)

        def advance(state):
            action = actor.run(np.asarray(state.obs["state"]))
            return step(state, action)

        error = None
        model_source = "控制台训练保存的模型" if console_model is not None else "作者预训练模型"
        semantics = "腿位置增量 + 轮速度残差"
        units = ["rad", "rad", "N·m", "N·m"]
    else:
        from legwheel.evaluation.runner import load_policy
        from legwheel.deployment.numpy_policy import RL_Policy
        policy = load_policy(str(checkpoint), deterministic=True)
        actor = RL_Policy(export_file or checkpoint.parent.parent / "smoke_params.cpp.txt")
        step = jax.jit(env.step)

        def advance(state):
            action = actor.run(np.asarray(state.obs["state"]))
            return step(state, action)

        model_source = "全扭矩训练的已保存模型"
        semantics = "四路直接扭矩"
        units = ["N·m"] * 4
    initial = reset(jax.random.PRNGKey(0))
    warm = initial
    for _ in range(3):
        warm = jax.block_until_ready(advance(warm))
    if args.mode == "torque200":
        exported = RL_Policy(export_file or checkpoint.parent.parent / "smoke_params.cpp.txt")
        action, _ = policy(initial.obs, jax.random.PRNGKey(0))
        error = float(np.max(np.abs(np.asarray(action) - exported.run(initial.obs["state"]))))
        if not np.isfinite(error) or error > 1e-3:
            raise AssertionError(f"Web policy/export mismatch {error}")
    else:
        actor.clean_history()
    baseline_actor = actor
    renderer = mujoco.Renderer(env.mj_model, height=500, width=800)
    data = mujoco.MjData(env.mj_model)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = env.mj_model.body("root").id
    camera.azimuth, camera.elevation, camera.distance = 35, -18, 0.65
    lock = threading.Condition()
    shared = {"frames": 0, "clients": 0, "suspended": True, "paused": False,
              "steps": 0, "episode_seconds": 0.0, "resets": 0, "fps": 0.0,
              "terminal": False, "torques": [0.0]*4, "actuator_inputs": [0.0]*4,
              "input_units": units, "mode": args.mode, "control_hz": round(1 / cfg.ctrl_dt),
              "ctrl_dt": cfg.ctrl_dt, "obs_dim": int(initial.obs["state"].shape[-1]),
              "action_semantics": semantics, "model_source": model_source,
              "switching": False, "checkpoint": str(checkpoint),
              "model_run_id": console_model.get("run_id") if console_model else None,
              "policy_export_error": error, "error": None,
              "sync_status": "idle", "sync_error": None,
              "preview_training_steps": console_model.get("actual_timesteps") if console_model else None,
              "preview_synced_at": None, "preview_request_id": None,
              "preview_kind": "saved"}
    baseline_info = {key: shared[key] for key in ("model_source", "checkpoint", "model_run_id", "preview_training_steps")}
    pending_preview = None
    frame = b""
    reset_requested = False
    switch_requested = None
    stopping = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def send(self, content, kind="application/json", status=200):
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            if self.path == "/":
                return self.send(Path(__file__).with_name("dashboard.html").read_bytes(),
                                 "text/html; charset=utf-8")
            if self.path == "/learning-curves":
                self.send_response(302)
                self.send_header("Location", "/#learning")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if self.path == "/learning-data":
                from learning_curves import load_learning_data, load_console_learning_data
                selected_run = controller.status(args.mode)
                if selected_run.get("run_dir"):
                    return self.send(json.dumps(load_console_learning_data(Path(selected_run["run_dir"]), args.mode), allow_nan=False).encode())
                curve_root = args.original_root / "artifacts" if args.mode == "original" else args.artifacts
                return self.send(json.dumps(load_learning_data(curve_root, mode=args.mode), allow_nan=False).encode())
            if self.path == "/state":
                with lock:
                    snapshot = dict(shared)
                manifest = args.artifacts / "performance-latest.json"
                if args.mode == "torque200" and manifest.is_file():
                    try:
                        snapshot["benchmark"] = json.loads(manifest.read_text())
                    except (OSError, ValueError):
                        pass
                snapshot["training"] = controller.status(args.mode)
                return self.send(json.dumps(snapshot).encode())
            if self.path.split("?", 1)[0] != "/stream.mjpg":
                return self.send(b"Not found", "text/plain", 404)
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            with lock:
                shared["clients"] += 1
            sequence = -1
            try:
                while not stopping.is_set():
                    with lock:
                        lock.wait_for(lambda: stopping.is_set() or shared["frames"] != sequence, timeout=1)
                        if stopping.is_set():
                            break
                        jpg, sequence = frame, shared["frames"]
                        if shared["suspended"]:
                            continue
                    if not jpg:
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " +
                                     str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                pass
            finally:
                with lock:
                    shared["clients"] -= 1

        def do_POST(self):
            nonlocal reset_requested, switch_requested, pending_preview
            if self.path not in ("/control", "/training"):
                return self.send(b"Not found", "text/plain", 404)
            # Require a same-origin JSON request; no CORS or arbitrary commands.
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                return self.send(b"JSON required", "text/plain", 415)
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 256:
                    raise ValueError("Invalid request size")
                request = json.loads(self.rfile.read(size))
                if not isinstance(request, dict):
                    raise ValueError("Invalid request")
                action = request["action"]
                if self.path == "/training":
                    try:
                        with lock:
                            if switch_requested is not None:
                                return self.send(b'{"error":"Mode is switching"}', status=409)
                            if action == "start":
                                result = controller.start(args.mode, duration_seconds=request.get("duration_seconds", 0), resume=request.get("resume", False))
                            elif action == "stop":
                                result = controller.stop()
                            else:
                                raise ValueError("Unknown training action")
                        return self.send(json.dumps(result, allow_nan=False).encode())
                    except ControlError as exc:
                        return self.send(json.dumps({"error": str(exc)}).encode(), status=getattr(exc, "status_code", 409))
                    except (ValueError, TypeError) as exc:
                        return self.send(json.dumps({"error": str(exc)}).encode(), status=400)
                with lock:
                    if action == "suspend":
                        controller.cancel_preview(pending_preview)
                        pending_preview = None
                        shared.update(suspended=True, sync_status="idle", sync_error=None)
                    elif action == "resume":
                        if shared["suspended"] and pending_preview is None:
                            try:
                                pending_preview = controller.request_preview(args.mode)
                            except ControlError as exc:
                                shared.update(sync_status="error", sync_error=str(exc))
                                return self.send(json.dumps({"error": str(exc)}).encode(), status=409)
                            pending_preview["started"] = time.monotonic()
                            pending_preview["next_poll"] = 0.0
                            shared.update(sync_status="pending", sync_error=None, paused=False)
                    elif action in ("pause", "unpause"):
                        shared["paused"] = action == "pause"
                    elif action in ("switch-mode", "reload-model"):
                        if controller.status().get("active"):
                            return self.send(json.dumps({"error": "请先停止训练并等待保存结束，再切换模式或加载模型。"}).encode(), status=409)
                        mode = args.mode if action == "reload-model" else request.get("mode")
                        if mode not in MODES:
                            raise ValueError("Unknown mode")
                        if mode != args.mode or action == "reload-model":
                            controller.cancel_preview(pending_preview)
                            pending_preview = None
                            switch_requested = mode
                            shared["switching"] = True
                            shared["suspended"] = True
                    elif action == "reset":
                        reset_requested = True
                    else:
                        raise ValueError("Unknown action")
            except (ValueError, KeyError, TypeError):
                return self.send(b"Invalid control", "text/plain", 400)
            self.send(b'{"ok":true}')

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"WEB_READY http://{args.host}:{args.port}/ export_error={error}", flush=True)
    state = initial
    terminal_until = 0.0
    frames_since = 0
    fps_since = time.perf_counter()
    try:
        while True:
            started = time.perf_counter()
            with lock:
                if switch_requested is not None:
                    break
                if pending_preview is not None and started >= pending_preview["next_poll"]:
                    pending_preview["next_poll"] = started + 0.25
                    request = pending_preview
                    try:
                        replacement = None
                        metadata = None
                        if request["kind"] == "live":
                            run = Path(request["run_dir"])
                            response_path = run / "preview.response.json"
                            response = json.loads(response_path.read_text()) if response_path.is_file() else {}
                            if response.get("request_id") == request["request_id"]:
                                if response.get("error"):
                                    raise ValueError(response["error"])
                                loaded = read_snapshot(run / "preview-policy.npz", args.mode, request["request_id"])
                                if loaded is not None:
                                    replacement, metadata = loaded
                                    metadata.update(model_source="打开画面时同步的训练策略",
                                                    checkpoint=str(run / "preview-policy.npz"))
                            elif started - request["started"] > 180:
                                raise ValueError("同步等待超时；训练可能正在编译或评估，请关闭后重新打开。")
                            else:
                                current = controller.status(args.mode)
                                if not current.get("active") or current.get("run_dir") != request["run_dir"]:
                                    model = controller.latest(args.mode)
                                    if model and model.get("run_id") == request["run_id"]:
                                        request.update(kind="saved", model=model)
                                    else:
                                        raise ValueError("本轮训练已结束，未生成可用策略；请检查训练状态。")
                        else:
                            model = request.get("model")
                            if model:
                                replacement = RL_Policy(path=Path(model["export_path"]))
                                metadata = dict(run_id=model["run_id"], steps=model.get("actual_timesteps"),
                                                checkpoint=model["checkpoint"], model_source="最近保存的训练模型",
                                                synced_at=utc_now())
                            else:
                                replacement = baseline_actor
                                metadata = dict(run_id=baseline_info["model_run_id"],
                                                steps=baseline_info["preview_training_steps"],
                                                checkpoint=baseline_info["checkpoint"],
                                                model_source=baseline_info["model_source"], synced_at=utc_now())
                        if replacement is not None:
                            probe = replacement.run(np.asarray(initial.obs["state"]))
                            if np.shape(probe) != (4,) or not np.isfinite(probe).all():
                                raise ValueError("Invalid preview action")
                            replacement.clean_history()
                            actor = replacement
                            state, terminal_until, frame = initial, 0.0, b""
                            shared.update(suspended=False, sync_status="ready", sync_error=None,
                                          preview_kind=request["kind"], preview_request_id=request["request_id"],
                                          preview_training_steps=metadata["steps"], preview_synced_at=metadata["synced_at"],
                                          checkpoint=metadata["checkpoint"], model_source=metadata["model_source"],
                                          model_run_id=metadata["run_id"], episode_seconds=0.0, terminal=False)
                            controller.cancel_preview(request)
                            pending_preview = None
                    except (OSError, ValueError, KeyError, ControlError) as exc:
                        controller.cancel_preview(request)
                        pending_preview = None
                        shared.update(suspended=True, sync_status="error", sync_error=str(exc))
                active = shared["clients"] > 0 and not shared["suspended"]
                paused = shared["paused"]
                if reset_requested:
                    state = initial
                    actor.clean_history()
                    terminal_until = 0.0
                    shared["resets"] += 1
                    reset_requested = False
            if not active:
                with lock:
                    shared["fps"] = 0.0
                frames_since, fps_since = 0, started
                time.sleep(0.05)
                continue
            if not paused:
                if terminal_until and started >= terminal_until:
                    state = initial
                    terminal_until = 0.0
                    with lock:
                        shared["resets"] += 1
                if not terminal_until:
                    for _ in range(round(0.05 / cfg.ctrl_dt)):
                        state = advance(state)
                        if not all(np.isfinite(np.asarray(v)).all() for v in
                                   (state.data.qpos, state.data.qvel, state.data.ctrl)):
                            raise FloatingPointError("Non-finite simulation state")
                        with lock:
                            shared["steps"] += 1
                        if bool(state.done) or float(state.data.time) >= cfg.episode_time:
                            terminal_until = time.perf_counter() + 0.7
                            break
            data.qpos[:] = np.asarray(state.data.qpos)
            data.qvel[:] = np.asarray(state.data.qvel)
            data.ctrl[:] = np.asarray(state.data.ctrl)
            mujoco.mj_forward(env.mj_model, data)
            renderer.update_scene(data, camera=camera)
            buffer = io.BytesIO()
            Image.fromarray(renderer.render()).save(buffer, format="JPEG", quality=80)
            frames_since += 1
            with lock:
                frame = buffer.getvalue()
                shared.update(frames=shared["frames"] + 1, episode_seconds=float(state.data.time),
                              terminal=bool(terminal_until), torques=data.ctrl.tolist(),
                              actuator_inputs=data.ctrl.tolist())
                if started - fps_since >= 1:
                    shared["fps"] = frames_since / (time.perf_counter() - fps_since)
                    frames_since, fps_since = 0, time.perf_counter()
                lock.notify_all()
            time.sleep(max(0.0, 0.05 - (time.perf_counter() - started)))
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        with lock:
            shared["error"] = str(exc)
        raise
    finally:
        stopping.set()
        with lock:
            lock.notify_all()
        server.shutdown()
        server.server_close()
        renderer.close()
    if switch_requested is not None:
        persist_mode(args.artifacts, switch_requested)
        os.execv(sys.executable, restart_arguments(args, switch_requested))


if __name__ == "__main__":
    main()
