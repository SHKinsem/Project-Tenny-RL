"""Web 接到 sim 本体：GLFW MuJoCo 窗口，命令来自同一套 Operator。

不把仿真画进 control.html。浏览器仍是真机那套页面；机器人在这个窗口里。
在本仓库 `.venv`（`requirements-sim.txt`）里跑，且不要设 MUJOCO_GL（macOS 设成 egl 就没有窗口）。
"""

from __future__ import annotations

import os
import time

import glfw
import jax
import jax.numpy as jp
import mujoco
import numpy as np

from sim.paths import ensure_plant

ensure_plant()
from legwheel import LegWheelRobot, default_config  # noqa: E402

from controller.obs import ObsPack
from policy.rl_policy import RL_Policy
from sim.io import from_mjx, wheels
from web.server import SESSION, serve_in_thread

# 按住 W/S 用满幅线速度，才能压过已经在前进的惯性；A/D 用中等偏航。
KEY_HOLD_VX = 0.2
KEY_HOLD_YAW = 0.5
# 与训练站立指令同一档：height_bar 50 → command[2] = -0.25。
SIM_HEIGHT_BAR = 50.0


def _wrap_pi(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _yaw_from_quat_wxyz(q) -> float:
    w, x, y, z = (float(v) for v in q)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _overlay(lines, grid_pos, font_scale, viewport, context) -> None:
    mujoco.mjr_overlay(font_scale, grid_pos, viewport, "\n".join(lines), "", context)


class MouseCam:
    def __init__(self, cam: mujoco.MjvCamera) -> None:
        self.cam = cam
        self.left = False
        self.middle = False
        self.right = False
        self.lastx = 0.0
        self.lasty = 0.0
        self.last_click = 0.0

    def on_button(self, window, button, act, mods) -> None:  # noqa: ARG002
        pressed = act == glfw.PRESS
        if button == glfw.MOUSE_BUTTON_LEFT:
            self.left = pressed
            if pressed:
                now = glfw.get_time()
                if now - self.last_click < 0.3:
                    mujoco.mjv_defaultCamera(self.cam)
                self.last_click = now
        elif button == glfw.MOUSE_BUTTON_MIDDLE:
            self.middle = pressed
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            self.right = pressed
        self.lastx, self.lasty = glfw.get_cursor_pos(window)

    def on_move(self, window, xpos, ypos) -> None:  # noqa: ARG002
        dx = xpos - self.lastx
        dy = ypos - self.lasty
        cam = self.cam
        if self.right or self.left:
            cam.azimuth += 0.1 * dx
            cam.elevation = max(min(cam.elevation - 0.1 * dy, 90), -90)
        elif self.middle:
            az = np.radians(cam.azimuth)
            el = np.radians(cam.elevation)
            forward = np.array([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)])
            right = np.array([-np.sin(az), np.cos(az), 0.0])
            up = np.cross(right, forward)
            scale = 0.001 * cam.distance
            cam.lookat[0] += scale * (right[0] * dx - up[0] * dy)
            cam.lookat[1] += scale * (right[1] * dx - up[1] * dy)
            cam.lookat[2] += scale * (right[2] * dx - up[2] * dy)
        self.lastx, self.lasty = xpos, ypos

    def on_scroll(self, window, xoffset, yoffset) -> None:  # noqa: ARG002
        self.cam.distance = max(self.cam.distance * (0.9 if yoffset > 0 else 1.1), 0.1)


class KeyCmd:
    """键盘与网页摇杆同一套语义：vx/yaw 必须持续给值，松手即停。

    高度/横滚是滑条，按一下保持。
    """

    def __init__(self, env) -> None:
        cfg = env._config["cmd_config"]
        self.lo = [
            float(cfg["lin_vel_x"][0]),
            float(cfg["ang_vel_yaw"][0]),
            float(cfg["height_value"][0]),
            float(cfg["roll_value"][0]),
        ]
        self.hi = [
            float(cfg["lin_vel_x"][1]),
            float(cfg["ang_vel_yaw"][1]),
            float(cfg["height_value"][1]),
            float(cfg["roll_value"][1]),
        ]
        self.v = [0.0, 0.0, 0.0, 0.0]
        self.held_w = False
        self.held_s = False
        self.held_a = False
        self.held_d = False
        self.window_focused = False
        self._last_joy = [0.0, 0.0]
        self.prev = {
            k: glfw.RELEASE
            for k in (
                glfw.KEY_Q,
                glfw.KEY_E,
                glfw.KEY_B,
                glfw.KEY_N,
                glfw.KEY_SPACE,
                glfw.KEY_X,
            )
        }

    def _edge(self, window, key) -> bool:
        now = glfw.get_key(window, key)
        hit = now == glfw.PRESS and self.prev[key] == glfw.RELEASE
        self.prev[key] = now
        return hit

    def _bump(self, i: int, delta: float, name: str) -> None:
        self.v[i] = min(max(self.v[i] + delta, self.lo[i]), self.hi[i])
        print(f"key {name}={self.v[i]:.3f}", flush=True)

    def _write_sliders(self) -> None:
        with SESSION.lock:
            SESSION.op.height_bar = float(np.clip(-200.0 * self.v[2], 0.0, 100.0))
            SESSION.op.oscillation_bar = float(np.clip(250.0 * self.v[3], -50.0, 50.0))

    def poll(self, window) -> str | None:
        self.window_focused = bool(glfw.get_window_attrib(window, glfw.FOCUSED))
        if not self.window_focused:
            self.held_w = self.held_s = self.held_a = self.held_d = False
            for key in self.prev:
                self.prev[key] = glfw.get_key(window, key)
            return None
        self.held_w = glfw.get_key(window, glfw.KEY_W) == glfw.PRESS
        self.held_s = glfw.get_key(window, glfw.KEY_S) == glfw.PRESS
        self.held_a = glfw.get_key(window, glfw.KEY_A) == glfw.PRESS
        self.held_d = glfw.get_key(window, glfw.KEY_D) == glfw.PRESS
        if glfw.get_key(window, glfw.KEY_X) == glfw.PRESS:
            return "quit"
        if self._edge(window, glfw.KEY_SPACE):
            return "reset"
        if self._edge(window, glfw.KEY_Q):
            self._bump(2, 0.1, "height")
            self._write_sliders()
        if self._edge(window, glfw.KEY_E):
            self._bump(2, -0.1, "height")
            self._write_sliders()
        if self._edge(window, glfw.KEY_B):
            self._bump(3, 0.1, "roll")
            self._write_sliders()
        if self._edge(window, glfw.KEY_N):
            self._bump(3, -0.1, "roll")
            self._write_sliders()
        return None

    def merge(self, now_ms: int) -> tuple[list[float], bool]:
        with SESSION.lock:
            web = list(SESSION.op.command(now_ms))
            joy = SESSION.op.live_joystick(now_ms)
            turn_on = SESSION.op.turn_on_edge()
        cmd = list(web)
        self.v[2], self.v[3] = web[2], web[3]
        # 键盘只在按住时盖摇杆。MuJoCo 有焦点但不能把摇杆清零，否则网页下拉会被丢掉，机器人仍按上一档前进。
        if self.held_s:
            cmd[0] = -KEY_HOLD_VX
        elif self.held_w:
            cmd[0] = KEY_HOLD_VX
        if self.held_d:
            cmd[1] = -KEY_HOLD_YAW
        elif self.held_a:
            cmd[1] = KEY_HOLD_YAW
        self._last_joy = joy
        return cmd, turn_on


def _apply_k(hold, packer) -> None:
    op = SESSION.op
    hold.sinyaw_scale = op.get_k("sinyaw_scale", 1.0)
    hold.hold_error_clip_enable = op.get_k("hold_error_clip_enable", 1.0) > 0.5
    hold.hold_error_clip_ds = op.get_k("hold_error_clip_ds", 3.0)
    hold.hold_error_clip_dpsi = op.get_k("hold_error_clip_dpsi", 1.0)
    hold.pitch_max = op.get_k("pitch_max", hold.pitch_max)
    packer.cfg.sinpitch_scale = op.get_k("sinpitch_scale", packer.cfg.sinpitch_scale)
    packer.cfg.sinroll_scale = op.get_k("sinroll_scale", packer.cfg.sinroll_scale)
    packer.cfg.pitch_zero_low = op.get_k("pitch_zero_low", packer.cfg.pitch_zero_low)
    packer.cfg.pitch_zero_high = op.get_k("pitch_zero_high", packer.cfg.pitch_zero_high)
    alpha = op.get_k("wheel_vel_smoothing_alpha(0,1)", packer.cfg.wheel_vel_smoothing_alpha)
    packer.cfg.wheel_vel_smoothing_alpha = 0.0 if alpha < 0.0 else 1.0 if alpha > 1.0 else alpha


def _sync_web(env, state, cmd: list[float], hold_out: dict) -> None:
    frame = from_mjx(env, state)
    with SESSION.lock:
        SESSION.q_l, SESSION.q_r = frame.q_l, frame.q_r
        SESSION.yaw = frame.yaw_rad
        SESSION.pitch = frame.pitch_rad
        SESSION.roll = frame.roll_rad
        SESSION.last_out = hold_out
        SESSION.last_cmd = list(cmd)
        SESSION.op.debug = (
            f"cmd: {cmd[0]:.4f} {cmd[1]:.4f} {cmd[2]:.4f} {cmd[3]:.4f}\n"
            f"ds_true:{SESSION.hold.ds_true:.6f} dpsi_true:{SESSION.hold.dpsi_true:.6f} "
            f"s0:{SESSION.hold.position_start:.6f} yaw0:{SESSION.hold.yaw_sin_zero:.6f}\n"
        )


def run(
    host: str = "127.0.0.1",
    port: int = 8765,
    *,
    noise: bool = False,
    random_init: bool = False,
    hold_clip_wheel: float = 5.0,
    hold_clip_sinyaw: float = 1.0,
) -> None:
    env_cfg = default_config()
    env_cfg.noise_config.level = 1.0 if noise else 0.0
    env_cfg.push_config.enable = False
    env_cfg.cmd_config.cmd_prob = 0.0
    env_cfg.hold_error_clip.enable = True
    env_cfg.hold_error_clip.wheel_sinpos = float(hold_clip_wheel)
    env_cfg.hold_error_clip.sinyaw = float(hold_clip_sinyaw)
    env = LegWheelRobot(env_cfg)

    with SESSION.lock:
        SESSION.op.height_bar = SIM_HEIGHT_BAR

    httpd = serve_in_thread(host, port, ticker=False)
    url = f"http://{host}:{port}/"
    print(f"浏览器打开 {url}  （真机同一套 Control 页）", flush=True)
    print("正在加载策略并编译，随后弹出 MuJoCo 窗口…", flush=True)
    try:
        import webbrowser

        if os.environ.get("SIM_NO_BROWSER") != "1":
            webbrowser.open(url)
    except Exception:
        pass

    glfw_started = False
    try:
        jit_reset = jax.jit(env.reset)
        jit_step = jax.jit(env.step)

        policy = RL_Policy()
        packer = ObsPack()
        print(policy.checkpoint_comment, flush=True)
        print("策略走 src/policy/rl_policy.cpp（与真机同一份），观测走共享 ObsPack", flush=True)

        rng = jax.random.PRNGKey(0)
        state = jit_reset(rng, random_init)
        frame0 = from_mjx(env, state)
        with SESSION.lock:
            _apply_k(SESSION.hold, packer)
            hold0 = SESSION.hold.step([0.0, 0.0, 0.0, 0.0], frame0, turn_on=True)
        obs0 = np.asarray(packer.pack(hold0["cmd"], frame0, hold0), dtype=np.float32)
        ctrl = jp.asarray(policy.run(obs0), dtype=jp.float32)
        state = jit_step(state, ctrl)
        rng = jax.random.PRNGKey(0)
        state = jit_reset(rng, random_init)
        policy.clean_history()
        packer.reset()
        with SESSION.lock:
            SESSION.hold.hold_anchors_inited = False
        mj_data = mujoco.MjData(env.mj_model)
        zero_ctrl = jp.zeros(env.action_size)

        if not glfw.init():
            raise RuntimeError("GLFW initialization failed")
        glfw_started = True
        window = glfw.create_window(1200, 900, "LegWheel Web + Sim", None, None)
        if not window:
            raise RuntimeError("GLFW window creation failed")
        glfw.make_context_current(window)
        glfw.swap_interval(1)
        print("GLFW window ready: LegWheel Web + Sim", flush=True)
        print("  keys: hold W forward  S back (release = stop)  A/D yaw  Q/E height  B/N roll  SPACE reset  X exit", flush=True)
        keys = KeyCmd(env)

        cam = mujoco.MjvCamera()
        opt = mujoco.MjvOption()
        opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        opt.frame = mujoco.mjtFrame.mjFRAME_WORLD
        scene = mujoco.MjvScene(env.mj_model, maxgeom=10000)
        context = mujoco.MjrContext(env.mj_model, mujoco.mjtFontScale.mjFONTSCALE_150)
        mujoco.mjv_defaultCamera(cam)
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_BODY, "root")
        cam.azimuth = 30
        cam.elevation = -20
        cam.distance = 0.8

        mouse = MouseCam(cam)
        glfw.set_mouse_button_callback(window, mouse.on_button)
        glfw.set_cursor_pos_callback(window, mouse.on_move)
        glfw.set_scroll_callback(window, mouse.on_scroll)

        sim_accumulator = 0.0
        last_render_sim_time = 0.0
        sim_timestep = float(env.mj_model.opt.timestep * env._config.action_repeat)
        render_interval = 1.0 / 30.0
        wall0 = time.perf_counter()
        command_vx = 0.0

        def reset_now(reason: str) -> None:
            nonlocal rng, state, sim_accumulator, last_render_sim_time, wall0, command_vx
            command_vx = 0.0
            rng = jax.random.PRNGKey(0)
            state = jit_reset(rng, random_init)
            policy.clean_history()
            packer.reset()
            sim_accumulator = 0.0
            last_render_sim_time = 0.0
            wall0 = time.perf_counter()
            with SESSION.lock:
                SESSION.hold.hold_anchors_inited = False
            print(reason, flush=True)

        while not glfw.window_should_close(window):
            glfw.poll_events()
            action = keys.poll(window)
            if action == "quit":
                break
            if action == "reset":
                reset_now("reset")

            now = time.perf_counter()
            sim_accumulator += now - wall0
            wall0 = now
            steps = 0
            cmd = list(SESSION.last_cmd)
            while sim_accumulator >= sim_timestep:
                cmd, turn_on = keys.merge(SESSION.now_ms())
                is_on = SESSION.op.switcher
                if turn_on:
                    reset_now("switch ON -> reset")
                if not is_on:
                    # 与固件 is_on / shutdown 同语义：关掉就不出力。
                    cmd = [0.0, 0.0, 0.0, 0.0]
                    turn_on = False
                    info = dict(state.info)
                    info["command"] = jp.zeros(4)
                    state = jit_step(state.replace(info=info), zero_ctrl)
                    frame = from_mjx(env, state)
                    with SESSION.lock:
                        _apply_k(SESSION.hold, packer)
                        hold_out = SESSION.hold.step(cmd, frame, turn_on=False)
                else:
                    info = dict(state.info)
                    info["command"] = jp.asarray(cmd)
                    state = state.replace(info=info)
                    frame = from_mjx(env, state)
                    delta_vx = cmd[0] - command_vx
                    max_delta_vx = SESSION.op.get_k("command_vx_rate", 0.2) * sim_timestep
                    command_vx += max(-max_delta_vx, min(max_delta_vx, delta_vx))
                    cmd[0] = command_vx
                    with SESSION.lock:
                        _apply_k(SESSION.hold, packer)
                        hold_out = SESSION.hold.step(cmd, frame, turn_on=turn_on)
                        obs = np.asarray(
                            packer.pack(hold_out["cmd"], frame, hold_out),
                            dtype=np.float32,
                        )
                    act = policy.run(obs)
                    packer.note_policy_action(act, frame)
                    state = jit_step(state, jp.asarray(act, dtype=jp.float32))
                _sync_web(env, state, cmd, hold_out)
                sim_accumulator -= sim_timestep
                steps += 1
                if turn_on or steps > 10:
                    sim_accumulator = 0.0
                    break

            mj_data.qpos = np.array(state.data.qpos)
            mj_data.qvel = np.array(state.data.qvel)
            mj_data.time = float(state.data.time)
            mujoco.mj_forward(env.mj_model, mj_data)

            current_sim_time = float(state.data.time)
            if current_sim_time - last_render_sim_time < render_interval:
                continue

            viewport = mujoco.MjrRect(0, 0, *glfw.get_framebuffer_size(window))
            mujoco.mjv_updateScene(
                env.mj_model, mj_data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene
            )
            mujoco.mjr_render(viewport, scene, context)

            q_l, q_r = wheels(env, mj_data)
            s_act = 0.5 * (q_l + q_r)
            s_des = SESSION.hold.position_start
            yaw_act = _yaw_from_quat_wxyz(mj_data.qpos[3:7])
            pos = np.asarray(state.data.qpos[:3])
            status = [
                f"cmd  vx {cmd[0]:+.2f}  yaw {cmd[1]:+.2f}  h {cmd[2]:+.2f}  roll {cmd[3]:+.2f}",
                f"joy y {keys._last_joy[1]:+.2f}  keys W{int(keys.held_w)}S{int(keys.held_s)}  focus {int(keys.window_focused)}",
                f"switcher {'ON' if SESSION.op.switcher else 'OFF (limp)'}",
                f"yaw act {_wrap_pi(yaw_act):+.2f} rad",
                f"pos s  act {s_act:+.2f}  des {s_des:+.2f}  err {s_act - s_des:+.2f}",
                f"xyz [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]",
                f"sim {current_sim_time:.2f}s  steps/frame {steps}",
            ]
            _overlay(
                status,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                mujoco.mjtFontScale.mjFONTSCALE_150,
                viewport,
                context,
            )
            _overlay(
                [
                    "hold W = forward  S = back  (release = stop)",
                    "Q/E = height   B/N = roll",
                    "SPACE = reset   X = exit",
                    "Web stick: hold to move, release to stop",
                    "Web switch ON = policy + reset, OFF = limp",
                    "Mouse drag / wheel = camera",
                ],
                mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                mujoco.mjtFontScale.mjFONTSCALE_150,
                viewport,
                context,
            )
            glfw.swap_buffers(window)
            last_render_sim_time = current_sim_time
    finally:
        SESSION._stop = True
        httpd.shutdown()
        httpd.server_close()
        if glfw_started:
            glfw.terminate()
