"""Hybrid console using the existing dashboard, with lazy CPU-only preview.

The HTTP service needs no model or ML imports to start. A single background
thread owns the simulator; cancelled opens can never publish a late frame.
"""
import io
import json
import os
from pathlib import Path
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from control_modes import MODES, mode_spec
from training_control import ControlError


def select_teacher_sources(original_root):
    """Select the reference environment before its first import; never swap modules."""
    root = Path(original_root or Path.home() / "projects/tenny-reference").resolve()
    source = root / "mujoco_legwheel_rl_public"
    deployment = root / "mujoco_esp32_robot_deploy_public"
    if not (source / "legwheel/__init__.py").is_file():
        raise FileNotFoundError("Reference environment is unavailable for wheel-torque preview")
    if not (deployment / "host/policy/rl_policy.py").is_file():
        raise FileNotFoundError("Reference policy reader is unavailable for wheel-torque preview")
    imported = sys.modules.get("legwheel")
    if imported is not None and not Path(imported.__file__).resolve().is_relative_to(source):
        raise RuntimeError("Wrong legwheel source is already imported; switch mode in a new process")
    sys.path[:0] = [str(source), str(deployment)]
    return root


def select_torque_sources():
    """Force students use the local torque environment, never the teacher's legwheel."""
    source = Path(__file__).resolve().parent
    if not (source / "legwheel/__init__.py").is_file():
        raise FileNotFoundError("Torque environment is unavailable for force preview")
    imported = sys.modules.get("legwheel")
    if imported is not None and not Path(imported.__file__).resolve().is_relative_to(source / "legwheel"):
        raise RuntimeError("Wrong legwheel source is already imported; switch mode in a new process")
    sys.path.insert(0, str(source))
    return source


class Simulator:
    """Constructed only after an explicit request to open a valid policy."""

    def __init__(self, mode, original_root=None, experiment_config=None):
        spec = mode_spec(mode)
        if mode == "leg_force_bc":
            select_torque_sources()
        elif spec["teacher"]:
            select_teacher_sources(original_root)
        import jax
        import mujoco
        import numpy as np
        from PIL import Image
        self.jax, self.mujoco, self.np, self.Image = jax, mujoco, np, Image
        if mode == "leg_force_bc":
            from force_teacher.env import StudentEnv, student_config
            cfg = student_config(experiment_config=experiment_config)
            environment = StudentEnv
        elif spec["teacher"]:
            from wheel_teacher.env import StudentEnv, student_config
            cfg = student_config()
            environment = StudentEnv
        else:
            from hybrid.env import HybridLegWheelRobot, hybrid_config
            cfg = hybrid_config(spec["control_hz"])
            environment = lambda config: HybridLegWheelRobot(spec["action_mode"], config)
        cfg.noise_config.level = 0.0
        cfg.push_config.enable = False
        cfg.cmd_config.cmd_prob = 0.0
        self.env = environment(cfg)
        self.initial = jax.jit(self.env.reset)(jax.random.PRNGKey(0))
        self.step = jax.jit(self.env.step)
        self.cfg = cfg
        self.renderer = mujoco.Renderer(self.env.mj_model, height=500, width=800)
        self.data = mujoco.MjData(self.env.mj_model)
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.camera.trackbodyid = self.env.mj_model.body("root").id
        self.camera.azimuth, self.camera.elevation, self.camera.distance = 35, -18, .65
        self.reset()

    def reset(self):
        self.state = self.initial
        self.terminal_until = 0.0

    def frame(self, actor, paused=False):
        np = self.np
        reset = 0
        if not paused:
            if self.terminal_until and time.monotonic() >= self.terminal_until:
                self.reset()
                reset = 1
            if not self.terminal_until:
                for _ in range(round(.05 / self.cfg.ctrl_dt)):
                    observation = np.asarray(self.state.obs["state"])
                    action = actor.action(observation) if hasattr(actor, "action") else actor.run(observation)
                    self.state = self.step(self.state, action)
                    if not all(np.isfinite(np.asarray(x)).all() for x in
                               (self.state.data.qpos, self.state.data.qvel, self.state.data.ctrl)):
                        raise FloatingPointError("Nonfinite preview state")
                    if bool(self.state.done) or float(self.state.data.time) >= self.cfg.episode_time:
                        self.terminal_until = time.monotonic() + .7
                        break
        self.data.qpos[:] = np.asarray(self.state.data.qpos)
        self.data.qvel[:] = np.asarray(self.state.data.qvel)
        self.data.ctrl[:] = np.asarray(self.state.data.ctrl)
        self.mujoco.mj_forward(self.env.mj_model, self.data)
        self.renderer.update_scene(self.data, camera=self.camera)
        buffer = io.BytesIO()
        self.Image.fromarray(self.renderer.render()).save(buffer, format="JPEG", quality=80)
        return buffer.getvalue(), dict(episode_seconds=float(self.state.data.time),
            terminal=bool(self.terminal_until), actuator_inputs=self.data.ctrl.tolist(), reset_delta=reset)

    def close(self):
        self.renderer.close()


def load_saved(model, mode, original_root=None):
    spec = mode_spec(mode)
    profile = {}
    if mode == "leg_force_bc":
        select_torque_sources()
        from force_teacher.contract import verified_report
        run = Path(model["run_dir"])
        report = json.loads((run / "report.json").read_text(encoding="utf8"))
        verified_report(run, report, original_root)
        from force_teacher.options import options_from_metadata
        profile = {"preview_training_profile": report.get("training_profile"),
                   "preview_evaluation_policy": report.get("evaluation_policy"),
                   "preview_experiment_config": options_from_metadata(json.loads((run / "policy_metadata.json").read_text(encoding="utf8")))}
        from force_teacher.policy import ForcePolicy
        actor = ForcePolicy(Path(report["export_path"]), run / "policy_metadata.json", original_root)
        description = "最近保存的腿力＋轮扭矩模型"
    elif spec["teacher"]:
        root = select_teacher_sources(original_root)
        from wheel_teacher.contract import verified_report
        run = Path(model["run_dir"])
        report = json.loads((run / "report.json").read_text(encoding="utf8"))
        verified_report(run, report, root)
        from host.policy.rl_policy import RL_Policy
        actor = RL_Policy(path=Path(report["export_path"]))
        description = "最近保存的腿位置＋轮扭矩模型"
    else:
        from hybrid.deployment import HybridPolicy
        actor = HybridPolicy(model["export_path"], Path(model["run_dir"]) / "policy_metadata.json",
                             spec["action_mode"], spec["control_hz"])
        description = "最近保存的混合动作模型"
    return actor, dict(model_source=description, checkpoint=model["checkpoint"], **profile,
                      model_run_id=model.get("run_id"),
                      preview_training_steps=model.get("actual_timesteps"), preview_kind="saved",
                      preview_synced_at=datetime.now(timezone.utc).isoformat())


class Preview:
    def __init__(self, mode, controller, simulator_factory=None, original_root=None):
        spec = mode_spec(mode)
        position_teacher = spec["teacher"] and not spec["hybrid"]
        self.original_root = original_root
        self.mode, self.controller = mode, controller
        self.factory = simulator_factory
        self.simulator_config = None
        self.lock = threading.Condition()
        self.generation = 0
        self.pending = None
        self.actor = self.simulator = None
        self.closed = False
        self.reset_requested = False
        self.jpg = b""
        self.state = dict(mode=mode, control_hz=spec["control_hz"], ctrl_dt=1 / spec["control_hz"],
            obs_dim=spec["obs_dim"], action_semantics="腿位置增量＋轮扭矩" if position_teacher else spec["action_mode"], frames=0, clients=0, fps=0.,
            suspended=True, paused=False, terminal=False, episode_seconds=0., resets=0,
            actuator_inputs=[0.]*4, torques=[0.]*4, input_units=["rad", "rad", "N·m", "N·m"] if position_teacher else ["N·m"]*4, switching=False,
            actuator_description="策略腿命令为虚拟伸展力（N）；下表显示映射后实际发送的执行器扭矩（N·m）。" if spec["action_mode"] == "virtual_leg_force" else "下表显示发送给执行器的控制输入及其单位。",
            model_source="尚未加载腿位置＋轮扭矩模型" if position_teacher else "尚未加载腿力＋轮扭矩模型" if mode == "leg_force_bc" else "尚未加载混合动作模型", checkpoint="", model_run_id=None,
            preview_training_steps=None, preview_synced_at=None, preview_kind="saved",
            preview_training_profile=None, preview_evaluation_policy=None,
            preview_experiment_config=None,
            preview_conditions="标称确定性预览 · 使用模型指令配置 · 训练扰动关闭" if mode == "leg_force_bc" else "零指令 · 无观测噪声 · 无外部推力",
            sync_status="idle", sync_error=None, error=None)

    def snapshot(self):
        with self.lock:
            return dict(self.state)

    def open(self):
        with self.lock:
            if self.pending is not None or not self.state["suspended"]:
                return
            request = self.controller.request_preview(self.mode)
            if request["kind"] == "saved" and not request.get("model"):
                raise ControlError("尚无此模式与频率的可用模型；请先开始训练或完成一次保存。")
            self.generation += 1
            self.pending = dict(request, generation=self.generation, started=time.monotonic())
            self.state.update(sync_status="pending", sync_error=None, paused=False)
            self.lock.notify_all()

    def suspend(self):
        with self.lock:
            request, self.pending = self.pending, None
            self.generation += 1
            self.jpg = b""
            self.state.update(suspended=True, sync_status="idle", sync_error=None, fps=0.)
            self.lock.notify_all()
        self.controller.cancel_preview(request)

    def _resolve(self, request):
        if request["kind"] == "saved":
            return load_saved(request["model"], self.mode, self.original_root)
        from preview_sync import read_snapshot
        run = Path(request["run_dir"])
        response_path = run / "preview.response.json"
        response = json.loads(response_path.read_text()) if response_path.is_file() else {}
        if response.get("request_id") == request["request_id"]:
            if response.get("error"):
                raise ValueError(response["error"])
            loaded = read_snapshot(run / "preview-policy.npz", self.mode, request["request_id"], original_root=self.original_root)
            if loaded is not None:
                actor, metadata = loaded
                profile = {}
                if self.mode == "leg_force_bc":
                    profile = {"preview_training_profile": None, "preview_evaluation_policy": None,
                               "preview_experiment_config": actor.experiment_config}
                    try:
                        status = json.loads((run / "status.json").read_text(encoding="utf8"))
                        profile.update(preview_training_profile=status.get("training_profile"),
                                       preview_evaluation_policy=status.get("evaluation_policy"))
                    except (OSError, ValueError, AttributeError):
                        pass  # Older runs do not attest a training profile.
                return actor, dict(model_source="打开画面时同步的训练策略", checkpoint=str(run / "preview-policy.npz"),
                    model_run_id=metadata["run_id"], preview_training_steps=metadata["steps"],
                    preview_synced_at=metadata["synced_at"], preview_kind="live", **profile)
        if time.monotonic() - request["started"] > 180:
            raise ValueError("等待策略同步超时；关闭后可重新打开，训练继续运行。")
        current = self.controller.status(self.mode)
        if not current.get("active") or current.get("run_dir") != request["run_dir"]:
            model = self.controller.latest(self.mode)
            if model and model.get("run_id") == request["run_id"]:
                return load_saved(model, self.mode, self.original_root)
            raise ValueError("本轮已结束，尚无可用模型；请查看训练状态。")
        return None

    def tick(self):
        """One worker iteration, separated for cancellation/race tests."""
        with self.lock:
            request = self.pending
            generation = self.generation
        try:
            if request is not None:
                resolved = self._resolve(request)
                if resolved is not None:
                    actor, metadata = resolved
                    with self.lock:
                        if generation != self.generation:
                            return
                    config = metadata.get("preview_experiment_config")
                    if self.simulator is None or config != self.simulator_config:
                        if self.simulator is not None:
                            self.simulator.close()
                        self.simulator = (self.factory(self.mode) if self.factory is not None else
                                          Simulator(self.mode, self.original_root, config))
                        self.simulator_config = config
                    self.simulator.reset()
                    with self.lock:
                        if generation != self.generation:
                            return
                        self.actor = actor
                        self.pending = None
                        self.jpg = b""
                        self.state.update(metadata, suspended=False, sync_status="ready", sync_error=None,
                            episode_seconds=0., terminal=False)
                    self.controller.cancel_preview(request)
            with self.lock:
                active = not self.state["suspended"] and self.state["clients"] > 0
                paused, reset = self.state["paused"], self.reset_requested
                self.reset_requested = False
            if not active or self.actor is None:
                return
            if reset:
                self.simulator.reset()
            started = time.monotonic()
            jpg, state = self.simulator.frame(self.actor, paused)
            reset_delta = state.pop("reset_delta", 0)
            with self.lock:
                if generation != self.generation or self.state["suspended"]:
                    return
                self.jpg = jpg
                self.state.update(state, fps=min(20., 1 / max(.00001, time.monotonic() - started)))
                self.state["frames"] += 1
                self.state["resets"] += reset_delta + int(reset)
                self.lock.notify_all()
        except Exception as exc:
            with self.lock:
                if generation != self.generation:
                    return
                self.pending = None
                self.jpg = b""
                self.state.update(suspended=True, sync_status="error", sync_error=str(exc), fps=0.)
            self.controller.cancel_preview(request)

    def run(self):
        try:
            while not self.closed:
                started = time.monotonic()
                self.tick()
                with self.lock:
                    self.lock.wait(timeout=max(.01, .05 - (time.monotonic() - started)))
        finally:
            if self.simulator is not None:
                self.simulator.close()

    def close(self):
        self.suspend()
        with self.lock:
            self.closed = True
            self.lock.notify_all()


class Console:
    def __init__(self, args, controller, preview=None):
        self.args, self.controller = args, controller
        self.preview = preview or Preview(args.mode, controller, original_root=getattr(args, "original_root", None))
        self.switch_mode = None
        self.control_lock = threading.Lock()

    def state(self):
        value = self.preview.snapshot()
        training = self.controller.status(self.args.mode)
        value.update(training=training, preview_available=bool(training.get("latest") or
            (training.get("active") and training.get("active_mode") == self.args.mode)),
            switching=self.switch_mode is not None)
        if self.args.mode == "leg_force_bc":
            from force_teacher.options import option_schema
            schema = option_schema()
            value.update(experiment_schema=schema, experiment_defaults=schema["defaults"],
                         saved_experiment_config=(training.get("latest") or {}).get("experiment_config"))
        return value

    def training(self, request):
        with self.control_lock:
            if self.switch_mode is not None:
                raise ControlError("模式正在切换。")
            if request.get("action") == "start":
                options = {}
                config = request.get("experiment_config")
                if self.args.mode != "leg_force_bc" and config is not None:
                    raise ValueError("此模式不支持腿力实验配置；请切换到腿力＋轮扭矩。")
                if self.args.mode == "leg_force_bc":
                    if request.get("resume", False):
                        if config is not None:
                            raise ValueError("继续训练采用保存模型的配置，请勿提交新实验草稿。")
                    else:
                        from force_teacher.options import normalize_options
                        try:
                            options["experiment_config"] = normalize_options(config)
                        except ValueError as exc:
                            raise ValueError("实验配置无效：" + str(exc)) from exc
                return self.controller.start(self.args.mode, duration_seconds=request.get("duration_seconds", 0),
                    resume=request.get("resume", False), force_revalidate=request.get("force_revalidate", False), **options)
            if request.get("action") == "stop":
                return self.controller.stop()
            raise ValueError("Unknown training action")

    def control(self, request):
        action = request.get("action")
        with self.control_lock:
            if action in ("switch-mode", "reload-model"):
                mode = self.args.mode if action == "reload-model" else request.get("mode")
                if mode not in MODES:
                    raise ValueError("Unknown mode")
                if self.controller.status().get("active"):
                    raise ControlError("请先停止训练并等待保存结束，再切换模式。")
                self.preview.suspend()
                self.switch_mode = mode
            elif action == "resume":
                self.preview.open()
            elif action == "suspend":
                self.preview.suspend()
            elif action in ("pause", "unpause", "reset"):
                with self.preview.lock:
                    if action == "reset":
                        self.preview.reset_requested = True
                    else:
                        self.preview.state["paused"] = action == "pause"
            else:
                raise ValueError("Unknown action")
        return {"ok": True}


def make_handler(console):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def send(self, value, status=200, kind="application/json; charset=utf-8"):
            data = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/":
                return self.send(Path(__file__).with_name("dashboard.html").read_bytes(), kind="text/html; charset=utf-8")
            if self.path == "/learning-curves":
                self.send_response(302)
                self.send_header("Location", "/#learning")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if self.path == "/state":
                return self.send(console.state())
            if self.path == "/learning-data":
                from learning_curves import load_console_learning_data
                selected = console.controller.status(console.args.mode)
                result = load_console_learning_data(Path(selected["run_dir"]), console.args.mode) if selected.get("run_dir") else {
                    "available": False, "metadata": {"message": "此模式与频率尚无训练记录；可直接开始新训练。"}, "points": []}
                return self.send(result)
            if self.path.split("?", 1)[0] != "/stream.mjpg":
                return self.send({"error": "Not found"}, 404)
            preview = console.preview
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            with preview.lock:
                preview.state["clients"] += 1
                sequence = preview.state["frames"]
            try:
                while not preview.closed:
                    with preview.lock:
                        preview.lock.wait_for(lambda: preview.closed or preview.state["frames"] != sequence, timeout=1)
                        jpg, sequence = preview.jpg, preview.state["frames"]
                        if preview.closed or preview.state["suspended"]:
                            break
                    if jpg:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " +
                            str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                pass
            finally:
                with preview.lock:
                    preview.state["clients"] -= 1

        def do_POST(self):
            if self.path not in ("/control", "/training"):
                return self.send({"error": "Not found"}, 404)
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                return self.send({"error": "JSON required"}, 415)
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 8192:
                    raise ValueError("Invalid request size")
                request = json.loads(self.rfile.read(size))
                if not isinstance(request, dict):
                    raise ValueError("Request must be an object")
                result = console.training(request) if self.path == "/training" else console.control(request)
                return self.send(result)
            except ControlError as exc:
                return self.send({"error": str(exc)}, 409)
            except (ValueError, TypeError, KeyError) as exc:
                return self.send({"error": str(exc)}, 400)
    return Handler


def main(args):
    from training_control import TrainingController
    from web_preview import persist_mode, restart_arguments
    controller = TrainingController(args.artifacts, args.original_root, args.artifacts.parent)
    console = Console(args, controller)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(console))
    server.daemon_threads = True
    server.timeout = .25
    worker = threading.Thread(target=console.preview.run, daemon=True)
    worker.start()
    print(f"WEB_READY http://{args.host}:{args.port}/ mode={args.mode} preview=off", flush=True)
    try:
        while console.switch_mode is None:
            server.handle_request()
    except KeyboardInterrupt:
        pass
    finally:
        console.preview.close()
        server.server_close()
        worker.join(timeout=2)
    if console.switch_mode is not None:
        persist_mode(args.artifacts, console.switch_mode)
        os.execv(sys.executable, restart_arguments(args, console.switch_mode))
