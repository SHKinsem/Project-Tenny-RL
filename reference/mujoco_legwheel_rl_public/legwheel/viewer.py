"""交互式策略查看器:GLFW 窗口 + 鼠标相机 + 键盘控制命令,可录像。

原 keyboard.py 迁移,仅参数化 checkpoint 路径与移除 cwd 相关 XML 覆盖。
"""

import mujoco
import numpy as np
import termios
import tty
import sys
import os
import jax
import glfw
import cv2
from datetime import datetime
from legwheel import LegWheelRobot, default_config
from legwheel.evaluation.runner import load_policy
import gc  # Add garbage collection
import threading
import queue
import time  # For precise frame timing

# Set numpy print options for better readability
np.set_printoptions(precision=3, suppress=True)

def get_key():
    """Capture keyboard input."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd, key = sys.stdin.read(1))
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return key

# Mouse callback functions for camera control
def mouse_button_callback(window, button, act, mods):
    global button_left, button_middle, button_right, lastx, lasty, button_action_time

    # Update button state
    if button == glfw.MOUSE_BUTTON_LEFT:
        button_left = act == glfw.PRESS
        if button_left:
            # Check for double-click (reset view)
            current_time = glfw.get_time()
            if current_time - button_action_time < 0.3:  # 300ms threshold for double-click
                mujoco.mjv_defaultCamera(cam)
            button_action_time = current_time
            
    elif button == glfw.MOUSE_BUTTON_MIDDLE:
        button_middle = act == glfw.PRESS
    elif button == glfw.MOUSE_BUTTON_RIGHT:
        button_right = act == glfw.PRESS

    # Update mouse position
    lastx, lasty = glfw.get_cursor_pos(window)

def mouse_move_callback(window, xpos, ypos):
    global button_left, button_middle, button_right, lastx, lasty

    # Compute mouse displacement
    dx = xpos - lastx
    dy = ypos - lasty

    if button_right:
        # Pan (shift camera target position)
        cam.azimuth += 0.1 * dx
        cam.elevation = max(min(cam.elevation - 0.1 * dy, 90), -90)
    elif button_left:
        # Rotate (move camera around)
        cam.azimuth += 0.1 * dx
        cam.elevation = max(min(cam.elevation - 0.1 * dy, 90), -90)
    elif button_middle:
        # Move camera target along the plane
        forward = np.array([
            np.cos(np.radians(cam.azimuth)) * np.cos(np.radians(cam.elevation)),
            np.sin(np.radians(cam.azimuth)) * np.cos(np.radians(cam.elevation)),
            np.sin(np.radians(cam.elevation))
        ])
        right = np.array([
            -np.sin(np.radians(cam.azimuth)), 
            np.cos(np.radians(cam.azimuth)),
            0
        ])
        up = np.cross(right, forward)
        
        scaling = 0.001 * cam.distance
        cam.lookat[0] += scaling * (right[0] * dx - up[0] * dy)
        cam.lookat[1] += scaling * (right[1] * dx - up[1] * dy)
        cam.lookat[2] += scaling * (right[2] * dx - up[2] * dy)

    # Save current mouse position for next update
    lastx = xpos
    lasty = ypos

def scroll_callback(window, xoffset, yoffset):
    global cam
    # Adjust camera distance with scroll wheel
    cam.distance *= 0.9 if yoffset > 0 else 1.1
    # Ensure minimum distance
    cam.distance = max(cam.distance, 0.1)

def add_overlay_text(img, text, position, font_scale=0.5, color=(255, 255, 255), thickness=1):
    """Add text overlay to an image."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    
    # Add black background for better readability
    bg_rect_tl = (position[0], position[1] - text_size[1] - 5)
    bg_rect_br = (position[0] + text_size[0] + 10, position[1] + 5)
    cv2.rectangle(img, bg_rect_tl, bg_rect_br, (0, 0, 0), -1)
    
    # Add text
    cv2.putText(img, text, (position[0] + 5, position[1]), font, font_scale, color, thickness)
    return img

def format_overlay_strings():
    """Group overlay text into separate sections for proper display."""
    # Replace with new implementation - returning multiple sections
    return {
        'status': [],  # Robot state and commands
        'controls': [], # Control instructions
        'recording': []  # Recording information
    }

def _wrap_pi(angle):
    """把角度折到 (-pi, pi]."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _yaw_from_quat_wxyz(q):
    """MuJoCo 自由关节四元数 (w,x,y,z) → yaw (绕世界 z, rad)。"""
    w, x, y, z = (float(v) for v in q)
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def render_overlay_lines(lines, grid_pos, font_scale, viewport, context, line_spacing=1):
    """Helper function to render multiple lines with proper spacing."""
    # Combine all lines into a single string with proper newlines
    # This ensures proper spacing between lines
    combined_text = "\n".join(lines)
    
    # Render all lines at once with a single call
    mujoco.mjr_overlay(font_scale, grid_pos, viewport, combined_text, "", context)

def main(ckpt_path: str, noise: bool = False, random_init: bool = False,
         hold_clip_wheel: float = 5.0, hold_clip_sinyaw: float = 1.0):
    # Initialize global mouse/camera variables
    global button_left, button_middle, button_right, lastx, lasty, button_action_time, cam
    button_left = False
    button_middle = False
    button_right = False
    lastx = 0
    lasty = 0
    button_action_time = 0
    
    # Video recording variables
    is_recording = False
    video_writer = None
    video_path = None
    record_start_sim_time = 0.0  # Changed to simulation time
    
    env_cfg = default_config()
    # 必须在 jit 之前写入:jax.jit 会把 config 标量固化进计算图
    env_cfg.noise_config.level = 1.0 if noise else 0.0
    env_cfg.push_config.enable = False
    env_cfg.cmd_config.cmd_prob = 0.0
    # 推理:锚点不动,观测误差越界则 clip;带宽由 view.sh 传入,须在 jit 前写入
    env_cfg.hold_error_clip.enable = True
    env_cfg.hold_error_clip.wheel_sinpos = float(hold_clip_wheel)
    env_cfg.hold_error_clip.sinyaw = float(hold_clip_sinyaw)
    env = LegWheelRobot(env_cfg)

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    jit_prepare = jax.jit(env.prepare_inference)

    # Create MjData for visualization - this is what was missing
    mj_data = mujoco.MjData(env.mj_model)

    # Load checkpoint for inference
    # 走 runner.load_policy:brax 0.14.2 反序列化 kernel init=None 会 KeyError,需回退重建
    print(f"Loading checkpoint from: {ckpt_path}")
    jit_inference_fn = jax.jit(load_policy(ckpt_path))

    rng = jax.random.PRNGKey(0)
    state = jit_reset(rng, random_init)
    yaw_des = _yaw_from_quat_wxyz(np.array(state.data.qpos[3:7]))

    gyro_sid = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "root_gyro")
    gyro_adr = int(env.mj_model.sensor_adr[gyro_sid]) if gyro_sid >= 0 else -1

    # Initialize GLFW and create window for visualization
    if not glfw.init():
        raise Exception("GLFW initialization failed")
    
    window = glfw.create_window(1200, 900, "LegWheel Robot Control", None, None)
    if not window:
        glfw.terminate()
        raise Exception("GLFW window creation failed")
    
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    # Initialize MuJoCo visualization structures
    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = True
    opt.frame = mujoco.mjtFrame.mjFRAME_WORLD
    
    scene = mujoco.MjvScene(env.mj_model, maxgeom=10000)
    context = mujoco.MjrContext(env.mj_model, mujoco.mjtFontScale.mjFONTSCALE_150)

    # Initialize camera
    mujoco.mjv_defaultCamera(cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    root_body_id = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_BODY, 'root')
    cam.trackbodyid = root_body_id
    cam.lookat = [0., 0., 0.]
    cam.azimuth = 30
    cam.elevation = -20
    cam.distance = 0.8

    print(f"obs noise: {'ON (level=1.0)' if noise else 'OFF (level=0.0)'}")
    print(f"random init: {'ON (random orientation)' if random_init else 'OFF (qpos0 on ground)'}")
    print(
        "hold error clip: ON "
        f"(Δs<={float(env._hold_error_clip_wheel):.3f}, "
        f"Δψ<={float(env._hold_error_clip_sinyaw):.3f})"
    )
    print("Use keys to control the robot:")
    print("  W/S: Increase/Decrease linear velocity (lin_vel_x)")
    print("  A/D: Increase/Decrease angular velocity (ang_vel_yaw)")
    print("  Q/E: Adjust thigh joint range (height_value)")
    print("  B/N: Adjust roll balance (roll_value)")
    print("  SPACE: Reset environment")
    print("  X: Exit")

    # Set input mode for better keyboard response
    glfw.set_input_mode(window, glfw.STICKY_KEYS, glfw.TRUE)
    
    # Initialize previous key states to avoid repeat actions
    prev_key_states = {
        glfw.KEY_W: glfw.RELEASE,
        glfw.KEY_S: glfw.RELEASE,
        glfw.KEY_A: glfw.RELEASE,
        glfw.KEY_D: glfw.RELEASE,
        glfw.KEY_Q: glfw.RELEASE,
        glfw.KEY_E: glfw.RELEASE,
        glfw.KEY_R: glfw.RELEASE,  # For recording toggle
        glfw.KEY_B: glfw.RELEASE,  
        glfw.KEY_N: glfw.RELEASE,  
        glfw.KEY_SPACE: glfw.RELEASE,  # For environment reset
        glfw.KEY_X: glfw.RELEASE
    }

    command = {'lin_vel_x': 0.0, 'ang_vel_yaw': 0.0, 'height_value': 0.0, 'roll_value': 0.0}

    # Register mouse callbacks
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, mouse_move_callback)
    glfw.set_scroll_callback(window, scroll_callback)

    print("  R: Start/Stop video recording")

    # Create video directory if it doesn't exist
    video_dir = os.path.join(os.getcwd(), 'videos')
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)

    # Allocate a pixel buffer for reading from GPU
    viewport_width, viewport_height = glfw.get_framebuffer_size(window)
    rgb_buffer = np.zeros((viewport_height, viewport_width, 3), dtype=np.uint8)

    # Real-time synchronization variables
    sim_time = 0.0  # Track simulation time
    time_factor = 1.0  # 1.0 = real-time, 2.0 = 2x speed, 0.5 = half speed
    
    # Rendering control variables
    render_fps = 30  # How many frames per second to render
    render_interval = 1.0 / render_fps  # Simulation time between renders
    last_render_sim_time = 0.0  # Last render time in simulation time
    render_skipped = 0  # Count skipped renders for debugging
    
    # Simulation control variables
    sim_accumulator = 0.0  # Time accumulator for simulation
    sim_timestep = env.mj_model.opt.timestep * env._config.action_repeat  # Simulation step size
    fixed_dt = sim_timestep  # Use fixed time delta for deterministic simulation

    while not glfw.window_should_close(window):
        # Process events at the beginning of each frame
        glfw.poll_events()
        
        # Only process keyboard input if window is in focus
        if glfw.get_window_attrib(window, glfw.FOCUSED):
            # Handle recording toggle
            if glfw.get_key(window, glfw.KEY_R) == glfw.PRESS and prev_key_states[glfw.KEY_R] == glfw.RELEASE:
                if is_recording:
                    # Stop recording
                    is_recording = False
                    if video_writer is not None:
                        video_writer.release()
                        video_writer = None
                        print(f"Video saved to {video_path}")
                else:
                    # Start recording
                    try:
                        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                        video_path = os.path.join(video_dir, f'legwheel_recording_{timestamp}.mp4')
                        
                        # Initialize video writer with the same FPS as the render rate
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        video_writer = cv2.VideoWriter(video_path, fourcc, render_fps,
                                                      (viewport_width, viewport_height))
                        
                        if not video_writer.isOpened():
                            raise RuntimeError("Failed to open video writer")
                            
                        is_recording = True
                        record_start_sim_time = sim_time  # Use simulation time
                        print(f"Recording started: {video_path}")
                    except Exception as e:
                        print(f"Failed to start recording: {e}")
                        if video_writer is not None:
                            video_writer.release()
                            video_writer = None
            
            # Check for key presses with state change detection
            if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS and prev_key_states[glfw.KEY_W] == glfw.RELEASE:
                command['lin_vel_x'] = min(command['lin_vel_x'] + 0.1, env._config['cmd_config']['lin_vel_x'][1])
                print(f"Updated command: lin_vel_x={command['lin_vel_x']:.3f}")
            
            if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS and prev_key_states[glfw.KEY_S] == glfw.RELEASE:
                command['lin_vel_x'] = max(command['lin_vel_x'] - 0.1, env._config['cmd_config']['lin_vel_x'][0])
                print(f"Updated command: lin_vel_x={command['lin_vel_x']:.3f}")
            
            if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS and prev_key_states[glfw.KEY_A] == glfw.RELEASE:
                command['ang_vel_yaw'] = min(command['ang_vel_yaw'] + 0.1, env._config['cmd_config']['ang_vel_yaw'][1])
                print(f"Updated command: ang_vel_yaw={command['ang_vel_yaw']:.3f}")
            
            if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS and prev_key_states[glfw.KEY_D] == glfw.RELEASE:
                command['ang_vel_yaw'] = max(command['ang_vel_yaw'] - 0.1, env._config['cmd_config']['ang_vel_yaw'][0])
                print(f"Updated command: ang_vel_yaw={command['ang_vel_yaw']:.3f}")
            
            if glfw.get_key(window, glfw.KEY_Q) == glfw.PRESS and prev_key_states[glfw.KEY_Q] == glfw.RELEASE:
                command['height_value'] = min(command['height_value'] + 0.1, env._config['cmd_config']['height_value'][1])
                print(f"Updated command: height_value={command['height_value']:.3f}")
            
            if glfw.get_key(window, glfw.KEY_E) == glfw.PRESS and prev_key_states[glfw.KEY_E] == glfw.RELEASE:
                command['height_value'] = max(command['height_value'] - 0.1, env._config['cmd_config']['height_value'][0])
                print(f"Updated command: height_value={command['height_value']:.3f}")
                
            # Adjust roll value with B/N keys instead of toggling
            if glfw.get_key(window, glfw.KEY_B) == glfw.PRESS and prev_key_states[glfw.KEY_B] == glfw.RELEASE:
                command['roll_value'] = min(command['roll_value'] + 0.1, env._config['cmd_config']['roll_value'][1])
                print(f"Updated command: roll_value={command['roll_value']:.3f}")
                
            if glfw.get_key(window, glfw.KEY_N) == glfw.PRESS and prev_key_states[glfw.KEY_N] == glfw.RELEASE:
                command['roll_value'] = max(command['roll_value'] - 0.1, env._config['cmd_config']['roll_value'][0])
                print(f"Updated command: roll_value={command['roll_value']:.3f}")
            
            if glfw.get_key(window, glfw.KEY_X) == glfw.PRESS:
                break
                
            # Add environment reset with SPACE key
            if glfw.get_key(window, glfw.KEY_SPACE) == glfw.PRESS and prev_key_states[glfw.KEY_SPACE] == glfw.RELEASE:
                print("Resetting environment...")
                rng = jax.random.PRNGKey(0)
                state = jit_reset(rng, random_init)
                
                # Reset simulation time trackers
                sim_time = 0.0
                sim_accumulator = 0.0
                last_render_sim_time = 0.0
                
                yaw_des = _yaw_from_quat_wxyz(np.array(state.data.qpos[3:7]))
                # Keep current command settings (don't reset those)
                print(f"Environment reset with current commands: {command}")
            
            # Update previous key states
            for key in prev_key_states:
                prev_key_states[key] = glfw.get_key(window, key)
        
        # Use fixed time step for deterministic simulation
        sim_accumulator += fixed_dt * time_factor
        
        # Run as many simulation steps as needed to catch up
        steps_taken = 0
        while sim_accumulator >= sim_timestep:
            # Update state using inference function
            act_rng, rng = jax.random.split(rng)
            cmd_array = np.array([command['lin_vel_x'], command['ang_vel_yaw'], 
                                command['height_value'], command['roll_value']])
            info = dict(state.info)
            info['command'] = jax.numpy.array(cmd_array)
            state = jit_prepare(state.replace(info=info))
            ctrl, _ = jit_inference_fn(state.obs, act_rng)
            state = jit_step(state, ctrl)
            yaw_des += command['ang_vel_yaw'] * sim_timestep

            # Update simulation time accumulator
            sim_accumulator -= sim_timestep
            sim_time += sim_timestep  # Track total simulation time
            steps_taken += 1
            
            # Don't let the simulation get too far behind (avoid spiral of death)
            if steps_taken > 10:  # Max steps per frame
                sim_accumulator = 0
                break
        
        # Get current simulation time from state
        current_sim_time = float(state.data.time)
        
        # Extract data from JAX state to NumPy
        np_robot_pos = np.array(state.data.qpos[:3])
        np_robot_vel = np.array(state.data.qvel[:3])
        
        # Update MuJoCo data for visualization
        mj_data.qpos = np.array(state.data.qpos)
        mj_data.qvel = np.array(state.data.qvel)
        mj_data.time = float(state.data.time)
        mujoco.mj_forward(env.mj_model, mj_data)
        
        # Determine if it's time to render based on simulation time
        time_since_last_render = current_sim_time - last_render_sim_time
        should_render = time_since_last_render >= render_interval
        
        if should_render:
            # Update scene and render
            viewport = mujoco.MjrRect(0, 0, *glfw.get_framebuffer_size(window))
            mujoco.mjv_updateScene(env.mj_model, mj_data, opt, None, cam, 
                                mujoco.mjtCatBit.mjCAT_ALL, scene)
            
            # Render the scene (only once)
            mujoco.mjr_render(viewport, scene, context)
            
            yaw_act = _yaw_from_quat_wxyz(mj_data.qpos[3:7])
            yaw_err = _wrap_pi(yaw_act - yaw_des)
            if gyro_adr >= 0:
                yaw_rate_act = float(mj_data.sensordata[gyro_adr + 2])
            else:
                yaw_rate_act = float(mj_data.qvel[5])
            yaw_rate_des = command['ang_vel_yaw']
            yaw_rate_err = yaw_rate_act - yaw_rate_des

            s_act = 0.5 * (float(mj_data.qpos[9]) + float(mj_data.qpos[12]))  # 同号前进,见 get_hold_position
            s_des = float(np.asarray(state.info["position_start"]).reshape(-1)[0])
            s_err = s_act - s_des

            # Prepare overlay data
            status_text = [
                f"Lin Vel X: {command['lin_vel_x']:.2f} m/s",
                (f"Yaw:     act {_wrap_pi(yaw_act):+.2f}  "
                 f"des {_wrap_pi(yaw_des):+.2f}  err {yaw_err:+.2f} rad"),
                (f"YawRate: act {yaw_rate_act:+.2f}  "
                 f"des {yaw_rate_des:+.2f}  err {yaw_rate_err:+.2f} rad/s"),
                (f"Pos s:   act {s_act:+.2f}  "
                 f"des {s_des:+.2f}  err {s_err:+.2f} rad"),
                f"Thigh Joint: {command['height_value']:.2f} rad",
                f"Roll Value: {command['roll_value']:.2f} rad",
                f"Pos: [{np_robot_pos[0]:.2f}, {np_robot_pos[1]:.2f}, {np_robot_pos[2]:.2f}]",
                f"Vel: [{np_robot_vel[0]:.2f}, {np_robot_vel[1]:.2f}, {np_robot_vel[2]:.2f}]",
                "",  # Empty line for separation
                f"Sim Time: {current_sim_time:.2f}s",
                f"Time Factor: {time_factor:.1f}x",
                f"Render Interval: {render_interval:.3f}s",
                f"Steps Per Frame: {steps_taken}"
            ]
            
            # Add overlay text
            render_overlay_lines(
                status_text, 
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                mujoco.mjtFontScale.mjFONTSCALE_150, 
                viewport, 
                context
            )
            
            # Display controls at bottom left
            control_text = [
                "Controls:",
                "W/S = Linear Velocity",
                "A/D = Angular Velocity",
                "Q/E = Thigh Joint Range", 
                "B/N = Roll Balance",
                "SPACE = Reset Environment",
                "R = Record Video",
                "X = Exit"
            ]
            
            # For bottom positioning, reverse the order so text grows upward
            render_overlay_lines(
                control_text,
                mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                mujoco.mjtFontScale.mjFONTSCALE_150, 
                viewport, 
                context
            )
            
            # Display recording indicator if recording
            if is_recording:
                record_time = current_sim_time - record_start_sim_time  # Use simulation time
                rec_text = f"RECORDING: {record_time:.1f}s"
                mujoco.mjr_overlay(
                    mujoco.mjtFontScale.mjFONTSCALE_150,
                    mujoco.mjtGridPos.mjGRID_TOPRIGHT, 
                    viewport, 
                    rec_text, 
                    "", 
                    context
                )
            
            # Handle video recording - BEFORE swapping buffers
            if is_recording and video_writer is not None:
                try:
                    # Create a fresh buffer for each frame
                    buffer = np.zeros((viewport_height, viewport_width, 3), dtype=np.uint8)
                    
                    # Read pixels from the current render - before buffer swap
                    mujoco.mjr_readPixels(buffer, None, viewport, context)
                    
                    # Process with a safer approach
                    frame = np.flipud(buffer.copy())
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    
                    # Write the frame if dimensions match
                    if frame.shape[0] == viewport_height and frame.shape[1] == viewport_width:
                        video_writer.write(frame)
                    else:
                        print(f"Frame dimensions mismatch: expected {viewport_width}x{viewport_height}, got {frame.shape[1]}x{frame.shape[0]}")
                        
                except Exception as e:
                    print(f"Error during recording: {e}")
                    import traceback
                    traceback.print_exc()
                    is_recording = False
                    if video_writer is not None:
                        video_writer.release()
                        video_writer = None
            
            # Swap buffers ONLY ONCE per frame
            glfw.swap_buffers(window)
            
            # Reset the render timer using simulation time
            last_render_sim_time = current_sim_time
            render_skipped = 0
        else:
            # Skip rendering for this frame
            render_skipped += 1
        
        # Update the last simulation time
        last_sim_time = current_sim_time
        
    # Make sure to close the video writer if still recording
    if is_recording and video_writer is not None:
        try:
            video_writer.release()
            print(f"Video saved to {video_path}")
        except Exception as e:
            print(f"Error finalizing video: {e}")

    glfw.terminate()

if __name__ == "__main__":
    main()
