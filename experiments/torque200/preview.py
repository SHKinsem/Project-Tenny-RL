"""Live checkpoint evaluation using the exact torque200 MJX environment.

No training. CPU inference/physics, separate 30 Hz OpenGL display. Simulated
control dt stays 5 ms even when the computer cannot sustain real time.
"""
import argparse
import json
from pathlib import Path
import time

import glfw
import jax
import mujoco
import numpy as np

from legwheel import LegWheelRobot, default_config
from legwheel.evaluation.runner import load_policy


def select_checkpoint(artifacts):
    # Only accept runs that completed parameter/export verification.
    for run in sorted(artifacts.glob("smoke-*"), reverse=True):
        if not (run / "parameter-verification.json").is_file():
            continue
        report = run / "resumed/report.json"
        if report.is_file():
            checkpoint = Path(json.loads(report.read_text())["checkpoint"])
            if checkpoint.is_dir():
                return checkpoint
    raise FileNotFoundError("No verified torque200 checkpoint; run the smoke launcher first")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--seconds", type=float, default=0, help="0: until window closes")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint or select_checkpoint(args.artifacts)
    print(f"CHECKPOINT {checkpoint}", flush=True)
    cfg = default_config()
    cfg.noise_config.level = 0.0
    cfg.push_config.enable = False
    cfg.cmd_config.cmd_prob = 0.0
    env = LegWheelRobot(cfg)
    policy = load_policy(str(checkpoint), deterministic=True)
    reset = jax.jit(env.reset)

    @jax.jit
    def advance(state):
        action, _ = policy(state.obs, jax.random.PRNGKey(0))
        return env.step(state, action)

    print("Compiling CPU simulation; the window opens after warm-up...", flush=True)
    initial = reset(jax.random.PRNGKey(0))
    warm = initial
    for _ in range(3):
        warm = jax.block_until_ready(advance(warm))
    from legwheel.deployment.numpy_policy import RL_Policy
    exported = RL_Policy(path=checkpoint.parent.parent / "smoke_params.cpp.txt")
    action, _ = policy(initial.obs, jax.random.PRNGKey(0))
    export_error = float(np.max(np.abs(np.asarray(action) - exported.run(initial.obs["state"]))))
    if export_error > 1e-3 or not np.isfinite(export_error):
        raise AssertionError(f"Loaded viewer policy differs from verified export: {export_error}")
    print(f"VIEWER_EXPORT_MAX_ERROR {export_error}", flush=True)
    state = initial
    data = mujoco.MjData(env.mj_model)
    if not glfw.init():
        raise RuntimeError("GLFW initialization failed; check WSLg")
    window = glfw.create_window(1000, 750, "Torque200 | short-training checkpoint", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Cannot open WSLg window")
    glfw.make_context_current(window)
    glfw.swap_interval(0)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = env.mj_model.body("root").id
    camera.azimuth, camera.elevation, camera.distance = 35, -18, 0.65
    option = mujoco.MjvOption()
    scene = mujoco.MjvScene(env.mj_model, maxgeom=10000)
    context = mujoco.MjrContext(env.mj_model, mujoco.mjtFontScale.mjFONTSCALE_150)
    paused, reset_requested = False, False

    def key_callback(_window, key, _scancode, action, _mods):
        nonlocal paused, reset_requested
        if action != glfw.PRESS:
            return
        if key in (glfw.KEY_ESCAPE, glfw.KEY_X):
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_SPACE:
            paused = not paused
        elif key == glfw.KEY_R:
            reset_requested = True

    def scroll_callback(_window, _dx, dy):
        camera.distance = float(np.clip(camera.distance * 0.9 ** dy, 0.15, 3.0))

    glfw.set_key_callback(window, key_callback)
    glfw.set_scroll_callback(window, scroll_callback)
    started = time.perf_counter()
    frames = steps = resets = episode_steps = 0
    physics_seconds = render_seconds = 0.0
    budget = 0.0
    terminal_until = 0.0
    completed = []
    screenshot_saved = False
    print("PREVIEW_READY | SPACE pause | R reset | scroll zoom | X exit", flush=True)
    try:
        while not glfw.window_should_close(window):
            frame_start = time.perf_counter()
            if args.seconds and frame_start - started >= args.seconds:
                break
            glfw.poll_events()
            if reset_requested or (terminal_until and frame_start >= terminal_until and not paused):
                state = initial
                resets += 1
                episode_steps = 0
                budget = terminal_until = 0.0
                reset_requested = False
            if not paused and not terminal_until:
                budget += 1.0 / 30.0
                tick = time.perf_counter()
                while budget + 1e-9 >= cfg.ctrl_dt:
                    state = advance(state)
                    # Host read synchronizes GPU/CPU before timing and termination checks.
                    if not all(np.isfinite(np.asarray(value)).all() for value in
                               (state.data.qpos, state.data.qvel, state.info["applied_torque"])):
                        raise FloatingPointError("Non-finite simulation state")
                    steps += 1
                    episode_steps += 1
                    budget -= cfg.ctrl_dt
                    if bool(state.done) or episode_steps >= cfg.episode_length:
                        completed.append({"steps": episode_steps, "seconds": episode_steps * cfg.ctrl_dt,
                                          "terminated": bool(state.done)})
                        terminal_until = time.perf_counter() + 0.7
                        budget = 0.0
                        break
                physics_seconds += time.perf_counter() - tick
            tick = time.perf_counter()
            data.qpos[:] = np.asarray(state.data.qpos)
            data.qvel[:] = np.asarray(state.data.qvel)
            data.ctrl[:] = np.asarray(state.info["applied_torque"])
            data.time = float(state.data.time)
            mujoco.mj_forward(env.mj_model, data)
            width, height = glfw.get_framebuffer_size(window)
            viewport = mujoco.MjrRect(0, 0, width, height)
            mujoco.mjv_updateScene(env.mj_model, data, option, None, camera,
                                  mujoco.mjtCatBit.mjCAT_ALL, scene)
            mujoco.mjr_render(viewport, scene, context)
            status = "PAUSED" if paused else ("EPISODE ENDED - resetting" if terminal_until else "RUNNING")
            compute_ratio = steps * cfg.ctrl_dt / max(physics_seconds, 1e-6)
            lines = ["TORQUE200 - SHORT TRAINING, NOT CONVERGED", checkpoint.parent.parent.parent.name,
                     f"{status} | resets {resets}",
                     f"Control: 200 Hz simulated | display target: 30 fps",
                     f"Episode: {episode_steps * cfg.ctrl_dt:.3f} s | compute capacity: {compute_ratio:.2f}x",
                     "Torque N.m (thigh L/R, wheel L/R):",
                     "  " + "  ".join(f"{v:+.4f}" for v in data.ctrl),
                     "Zero command | no observation noise / external push",
                     "SPACE pause | R reset | scroll zoom | X exit"]
            mujoco.mjr_overlay(mujoco.mjtFontScale.mjFONTSCALE_150,
                               mujoco.mjtGridPos.mjGRID_TOPLEFT, viewport, "\n".join(lines), "", context)
            if frames >= 10 and not screenshot_saved:
                from PIL import Image
                rgb = np.empty((height, width, 3), dtype=np.uint8)
                mujoco.mjr_readPixels(rgb, None, viewport, context)
                Image.fromarray(np.flipud(rgb)).save(args.output / "preview.png")
                screenshot_saved = True
            glfw.swap_buffers(window)
            render_seconds += time.perf_counter() - tick
            frames += 1
            time.sleep(max(0.0, 1.0 / 30.0 - (time.perf_counter() - frame_start)))
    finally:
        report = {"checkpoint": str(checkpoint), "devices": [str(d) for d in jax.devices()],
                  "control_hz_simulated": 200, "display_target_fps": 30,
                  "wall_seconds_excluding_compile": time.perf_counter() - started,
                  "frames": frames, "steps": steps, "resets": resets,
                  "viewer_export_max_abs_error": export_error,
                  "physics_seconds": physics_seconds, "render_seconds": render_seconds,
                  "completed_episodes": completed, "screenshot_saved": screenshot_saved,
                  "evaluation": "deterministic mean action; zero command; no noise/push/domain randomization",
                  "note": "Compute capacity excludes rendering, deliberate waits and resets; not training throughput"}
        (args.output / "preview-report.json").write_text(json.dumps(report, indent=2))
        context.free()
        glfw.destroy_window(window)
        glfw.terminate()
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
