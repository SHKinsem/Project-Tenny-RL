"""Training console with bounded, lazy CPU-only policy previews.

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
from workbench_preview import PreviewGroup, dispatch_viewports


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
        if mode in {"leg_force_bc", "torque200"}:
            select_torque_sources()
        elif spec["teacher"] or mode == "original":
            select_teacher_sources(original_root)
        import jax
        import mujoco
        import numpy as np
        from PIL import Image
        self.jax, self.mujoco, self.np, self.Image = jax, mujoco, np, Image
        if mode in {"original", "torque200"}:
            from legwheel import LegWheelRobot, default_config
            cfg = default_config()
            environment = LegWheelRobot
        elif mode == "leg_force_bc":
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
        self._advance_remainder = 0.0
        self._lag_seconds = 0.0
        self._render_key = None
        self._jpg = b""

    def _image_key(self):
        camera = self.camera
        return (self.data.qpos.tobytes(), self.data.qvel.tobytes(), self.data.ctrl.tobytes(),
                camera.type, camera.fixedcamid, camera.trackbodyid,
                tuple(camera.lookat), camera.distance, camera.azimuth, camera.elevation,
                camera.orthographic, self.renderer.width, self.renderer.height,
                tuple(self.renderer.scene.flags), 80)

    def frame(self, actor, paused=False, *, advance_seconds=.05):
        """Advance the playback clock independently of the JPEG refresh budget.

        A slow consumer can cost at most 250ms of simulation in one call. Older
        time is discarded and reported, rather than creating unbounded catch-up.
        """
        np = self.np
        reset, advanced = 0, 0.0
        if not paused:
            if self.terminal_until and time.monotonic() >= self.terminal_until:
                self.reset()
                if hasattr(actor, "clean_history"):
                    actor.clean_history()
                reset = 1
            if not self.terminal_until:
                requested = max(0.0, advance_seconds)
                budget = min(.25, requested)
                self._lag_seconds += requested - budget
                budget += self._advance_remainder
                steps = int((budget + 1e-10) / self.cfg.ctrl_dt)
                self._advance_remainder = max(0., budget - steps * self.cfg.ctrl_dt)
                for _ in range(steps):
                    observation = np.asarray(self.state.obs["state"])
                    action = actor.action(observation) if hasattr(actor, "action") else actor.run(observation)
                    self.state = self.step(self.state, action)
                    advanced += self.cfg.ctrl_dt
                    if not all(np.isfinite(np.asarray(x)).all() for x in
                               (self.state.data.qpos, self.state.data.qvel, self.state.data.ctrl)):
                        raise FloatingPointError("Nonfinite preview state")
                    if bool(self.state.done) or float(self.state.data.time) >= self.cfg.episode_time:
                        self.terminal_until = time.monotonic() + .7
                        self._advance_remainder = 0.0
                        break
        self.data.qpos[:] = np.asarray(self.state.data.qpos)
        self.data.qvel[:] = np.asarray(self.state.data.qvel)
        self.data.ctrl[:] = np.asarray(self.state.data.ctrl)
        key = self._image_key()
        rendered = key != self._render_key
        if rendered:
            self.mujoco.mj_forward(self.env.mj_model, self.data)
            self.renderer.update_scene(self.data, camera=self.camera)
            buffer = io.BytesIO()
            self.Image.fromarray(self.renderer.render()).save(buffer, format="JPEG", quality=80)
            self._jpg, self._render_key = buffer.getvalue(), key
        return self._jpg, dict(episode_seconds=float(self.state.data.time),
            terminal=bool(self.terminal_until), actuator_inputs=self.data.ctrl.tolist(), reset_delta=reset,
            rendered=rendered, advanced_seconds=advanced, lag_seconds=self._lag_seconds)

    def close(self):
        self.renderer.close()


def load_saved(model, mode, original_root=None):
    spec = mode_spec(mode)
    profile = {}
    if mode == "original":
        select_teacher_sources(original_root)
        from host.policy.rl_policy import RL_Policy
        actor = RL_Policy(path=Path(model["export_path"]))
        description = "最近保存的腿位置＋轮速度模型"
    elif mode == "torque200":
        select_torque_sources()
        from legwheel.deployment.numpy_policy import RL_Policy
        actor = RL_Policy(Path(model["export_path"]))
        description = "最近保存的全扭矩模型"
    elif mode == "leg_force_bc":
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


def load_baseline(mode, original_root=None):
    """Keep the original author's baseline available without inventing a run."""
    if mode != "original":
        raise ControlError("尚无此模式与频率的可用模型；请先开始训练或完成一次保存。")
    select_teacher_sources(original_root)
    from host.policy.rl_policy import RL_Policy
    from host.policy.ckpt import POLICY_CPP
    return RL_Policy(), dict(model_source="作者预训练模型", checkpoint=str(POLICY_CPP),
                            model_run_id=None, preview_training_steps=None,
                            preview_kind="baseline", preview_synced_at=datetime.now(timezone.utc).isoformat())


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
        self.release_requested = False
        self._last_frame_at = None
        self._clock_at = None
        self.reset_requested = False
        self.jpg = b""
        self.state = dict(mode=mode, control_hz=spec["control_hz"], ctrl_dt=1 / spec["control_hz"],
            obs_dim=spec["obs_dim"], action_semantics="腿位置增量＋轮速度残差" if mode == "original" else "四路直接扭矩" if mode == "torque200" else "腿位置增量＋轮扭矩" if position_teacher else spec["action_mode"], frames=0, clients=0, fps=0.,
            suspended=True, paused=False, terminal=False, episode_seconds=0., resets=0,
            actuator_inputs=[0.]*4, torques=[0.]*4, input_units=["rad", "rad", "N·m", "N·m"] if position_teacher else ["N·m"]*4, switching=False,
            actuator_description="策略腿命令为虚拟伸展力（N）；下表显示映射后实际发送的执行器扭矩（N·m）。" if spec["action_mode"] == "virtual_leg_force" else "下表显示发送给执行器的控制输入及其单位。",
            model_source="尚未加载腿位置＋轮扭矩模型" if position_teacher else "尚未加载腿力＋轮扭矩模型" if mode == "leg_force_bc" else "尚未加载混合动作模型", checkpoint="", model_run_id=None,
            preview_training_steps=None, preview_synced_at=None, preview_kind="saved",
            preview_training_profile=None, preview_evaluation_policy=None,
            preview_experiment_config=None,
            preview_conditions="标称确定性预览 · 使用模型指令配置 · 训练扰动关闭" if mode == "leg_force_bc" else "零指令 · 无观测噪声 · 无外部推力",
            sync_status="idle", sync_error=None, error=None,
            loading={"active": False, "stage": "idle", "progress": None},
            playback_rate=None, lag_seconds=0.)

    def reset_clock(self):
        """Idle, pause and reconnect never accumulate playback debt."""
        self._clock_at = None

    def snapshot(self):
        with self.lock:
            return {**self.state, "generation": self.generation}

    def loading(self, stage, *, generation=None):
        with self.lock:
            if generation is not None and generation != self.generation:
                return
            self.state["loading"] = {"active": stage not in {"idle", "ready", "error"},
                                     "stage": stage, "progress": None}
            self.lock.notify_all()

    def open(self):
        with self.lock:
            if self.closed:
                raise ControlError("视窗已关闭。")
            if self.pending is not None or not self.state["suspended"]:
                return
            request = self.controller.request_preview(self.mode)
            if request["kind"] == "saved" and not request.get("model") and self.mode != "original":
                self.controller.cancel_preview(request)
                raise ControlError("尚无此模式与频率的可用模型；请先开始训练或完成一次保存。")
            self.generation += 1
            self.pending = dict(request, generation=self.generation, started=time.monotonic())
            self._last_frame_at = None
            self.reset_clock()
            self.state.update(sync_status="pending", sync_error=None, paused=False)
            self.loading("syncing" if request["kind"] == "live" else "loading_policy")
            self.lock.notify_all()

    def suspend(self):
        with self.lock:
            request, self.pending = self.pending, None
            self.generation += 1
            self.release_requested = True
            self.jpg = b""
            self._last_frame_at = None
            self.reset_clock()
            self.state.update(suspended=True, sync_status="idle", sync_error=None, fps=0.)
            self.loading("idle")
            self.lock.notify_all()
        self.controller.cancel_preview(request)

    def _resolve(self, request):
        if request["kind"] == "saved":
            if not request.get("model"):
                return load_baseline(self.mode, self.original_root)
            return load_saved(request["model"], self.mode, self.original_root)
        from preview_sync import read_snapshot
        run = Path(request["run_dir"])
        response_path = run / "preview.response.json"
        response = json.loads(response_path.read_text()) if response_path.is_file() else {}
        if response.get("request_id") == request["request_id"]:
            if response.get("error"):
                raise ValueError(response["error"])
            self.loading("loading_policy", generation=request.get("generation"))
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

    def release(self):
        """Release native rendering resources on the owning worker thread."""
        with self.lock:
            simulator, self.simulator = self.simulator, None
            self.actor = None
            self.release_requested = False
        if simulator is not None:
            try:
                simulator.close()
            except Exception as exc:
                # A failed native teardown must not terminate other viewports'
                # shared worker. Surface it instead of pretending cleanup won.
                with self.lock:
                    self.state["error"] = "释放预览资源失败：" + str(exc)

    def tick(self):
        """One worker iteration, separated for cancellation/race tests."""
        with self.lock:
            request = self.pending
            generation = self.generation
            cleanup = self.release_requested
        try:
            if cleanup:
                self.release()
            if request is not None:
                resolved = self._resolve(request)
                if resolved is not None:
                    actor, metadata = resolved
                    with self.lock:
                        if generation != self.generation or self.closed:
                            return
                    self.loading("loading_environment", generation=generation)
                    config = metadata.get("preview_experiment_config")
                    if self.simulator is None or config != self.simulator_config:
                        if self.simulator is not None:
                            self.simulator.close()
                        self.simulator = (self.factory(self.mode) if self.factory is not None else
                                          Simulator(self.mode, self.original_root, config))
                        self.simulator_config = config
                    self.simulator.reset()
                    if hasattr(actor, "clean_history"):
                        actor.clean_history()
                    with self.lock:
                        if generation != self.generation or self.closed:
                            self.release()
                            return
                        self.actor = actor
                        self.pending = None
                        self.jpg = b""
                        self.reset_clock()
                        self.state.update(metadata, suspended=False, sync_status="ready", sync_error=None,
                            episode_seconds=0., terminal=False)
                        self.loading("first_frame")
                    self.controller.cancel_preview(request)
            with self.lock:
                active = not self.state["suspended"] and self.state["clients"] > 0
                paused, reset = self.state["paused"], self.reset_requested
                if active:
                    self.reset_requested = False
            if not active or self.actor is None:
                self.reset_clock()
                with self.lock:
                    self.state.update(fps=0., playback_rate=None)
                return
            if reset:
                self.simulator.reset()
                if hasattr(self.actor, "clean_history"):
                    self.actor.clean_history()
                self.reset_clock()
            started = time.monotonic()
            interval = max(0., started - self._clock_at) if self._clock_at is not None and not paused else 0.
            self._clock_at = None if paused else started
            jpg, state = self.simulator.frame(self.actor, paused, advance_seconds=interval)
            rendered_at = time.monotonic()
            reset_delta = state.pop("reset_delta", 0)
            rendered = state.pop("rendered", True)
            advanced = state.pop("advanced_seconds", 0.)
            with self.lock:
                if generation != self.generation or self.state["suspended"]:
                    return
                self.jpg = jpg
                frame_interval = rendered_at - (self._last_frame_at if self._last_frame_at is not None else started)
                self.state.update(state, fps=min(20., 1 / max(.00001, frame_interval)) if rendered else 0.,
                                  playback_rate=advanced / interval
                                  if interval > 0 and not paused and not state.get("terminal") else None)
                if rendered:
                    self._last_frame_at = rendered_at
                    self.state["frames"] += 1
                self.state["resets"] += reset_delta + int(reset)
                self.loading("ready")
                self.lock.notify_all()
        except Exception as exc:
            with self.lock:
                if generation != self.generation:
                    return
                self.pending = None
                self.jpg = b""
                self.state.update(suspended=True, sync_status="error", sync_error=str(exc), fps=0.)
                self.loading("error")
            self.release()
            self.controller.cancel_preview(request)

    def run(self):
        try:
            while not self.closed:
                started = time.monotonic()
                self.tick()
                with self.lock:
                    self.lock.wait(timeout=max(.01, .05 - (time.monotonic() - started)))
        finally:
            self.release()

    def close(self):
        self.suspend()
        with self.lock:
            self.closed = True
            self.lock.notify_all()


class Console:
    def __init__(self, args, controller, preview=None):
        self.args, self.controller = args, controller
        factory = lambda coordinated: Preview(args.mode, coordinated, original_root=getattr(args, "original_root", None))
        self.viewports = PreviewGroup(args.mode, controller, factory, initial=preview)
        self.switch_mode = None
        self.control_lock = threading.Lock()

    @property
    def preview(self):
        return self.viewports.primary

    def state(self):
        value = self.preview.snapshot()
        training = self.controller.status(self.args.mode)
        value.update(training=training, preview_available=bool(training.get("latest") or
            (training.get("active") and training.get("active_mode") == self.args.mode)),
            switching=self.switch_mode is not None, viewports=self.viewports.snapshot()["viewports"])
        if self.args.mode == "leg_force_bc":
            from force_teacher.options import option_schema
            schema = option_schema()
            value.update(experiment_schema=schema, experiment_defaults=schema["defaults"],
                         saved_experiment_config=(training.get("latest") or {}).get("experiment_config"))
        if self.args.mode == "original":
            value["preview_available"] = True
        return value

    def training(self, request):
        with self.control_lock:
            if self.switch_mode is not None:
                raise ControlError("模式正在切换。")
            if request.get("action") == "start":
                options = {key: request[key] for key in ("checkpoint", "parameters") if key in request}
                config = request.get("experiment_config")
                if self.args.mode != "leg_force_bc" and config is not None:
                    raise ValueError("此模式不支持腿力实验配置；请切换到腿力＋轮扭矩。")
                if self.args.mode == "leg_force_bc":
                    if request.get("resume", False) or request.get("checkpoint"):
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
            if request.get("action") in {"pause", "resume", "save"}:
                function = getattr(self.controller, request["action"])
                return function(**({"request_id": request["request_id"]} if "request_id" in request else {}))
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
                self.viewports.suspend_all()
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
                    self.preview.reset_clock()
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
            if getattr(self, "command", None) != "HEAD":
                self.wfile.write(data)

        def do_GET(self):
            if dispatch_viewports(self, console.viewports):
                return
            from workbench_web import dispatch
            if dispatch(self, artifacts=getattr(console.args, "artifacts", None), mode=console.args.mode,
                        controller=console.controller):
                return
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
            return self.send({"error": "Not found"}, 404)

        def do_HEAD(self):
            if dispatch_viewports(self, console.viewports):
                return
            from workbench_web import dispatch
            if not dispatch(self, artifacts=getattr(console.args, "artifacts", None), mode=console.args.mode,
                            controller=console.controller):
                self.send_error(404)

        def do_PUT(self):
            if dispatch_viewports(self, console.viewports):
                return
            from workbench_web import dispatch
            if not dispatch(self, artifacts=getattr(console.args, "artifacts", None), mode=console.args.mode,
                            controller=console.controller):
                self.send_error(405)

        do_PATCH = do_PUT
        do_DELETE = do_PUT
        do_OPTIONS = do_PUT

        def do_POST(self):
            if dispatch_viewports(self, console.viewports):
                return
            from workbench_web import dispatch
            if dispatch(self, artifacts=getattr(console.args, "artifacts", None), mode=console.args.mode,
                        controller=console.controller):
                return
            if self.path not in ("/control", "/training"):
                return self.send({"error": "Not found"}, 404)
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                return self.send({"error": "JSON required"}, 415)
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 16384:
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
    # Only this web process owns the simulators. TrainingController gives its
    # detached runner an explicit CUDA environment when the user starts a run.
    os.environ["JAX_PLATFORMS"] = "cpu"
    from training_control import TrainingController
    from web_preview import persist_mode, restart_arguments
    controller = TrainingController(args.artifacts, args.original_root, args.artifacts.parent)
    console = Console(args, controller)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(console))
    server.daemon_threads = True
    server.timeout = .25
    console.viewports.start()
    print(f"WEB_READY http://{args.host}:{args.port}/ mode={args.mode} preview=off", flush=True)
    try:
        while console.switch_mode is None:
            server.handle_request()
    except KeyboardInterrupt:
        pass
    finally:
        console.viewports.close()
        server.server_close()
    if console.switch_mode is not None:
        persist_mode(args.artifacts, console.switch_mode)
        os.execv(sys.executable, restart_arguments(args, console.switch_mode))
