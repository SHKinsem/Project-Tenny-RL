"""评估视频渲染:OSD 状态叠加、速度/推力 3D 箭头与统一回放渲染。"""

import os
import logging
from typing import Optional

import numpy as np
import jax
import mujoco
import mediapy as media

logger = logging.getLogger(__name__)


def overlay_state_info(
    frames: list,
    rollout: list,
    render_every: int,
    step_actions: Optional[list] = None,
    step_rewards: Optional[list] = None,
    wheel_friction_per_frame: Optional[list] = None,
    rng_seed: Optional[int] = None,
) -> list:
    """在视频帧左上角叠加机器人状态和控制信息，方便分析失败原因。"""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        logger.warning("Pillow 未安装，跳过 OSD 叠加 (pip install Pillow)")
        return frames

    # 查找等宽字体
    font_size = 16
    font = None
    for fp in [
        "/System/Library/Fonts/SFNSMono.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Monaco.dfont",
        "/Library/Fonts/Courier New.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    ]:
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, font_size)
                break
            except Exception:
                continue
    if font is None:
        try:
            font = ImageFont.load_default(size=font_size)
        except TypeError:
            font = ImageFont.load_default()

    # obs 字段布局（与 legwheel_env 的 _get_obs 保持一致）
    # 基础 26 维；last_act_obs=True 时末尾多 4 维 last_act，共 30 维
    obs_fields_base = [
        ("command", 4), ("gyro", 3), ("sinyaw", 1),
        ("sinpitch", 1), ("sinroll", 1), ("joints", 4),
        ("velocity", 4), ("linvel", 2), ("pid_hist", 4), ("pid_error", 2),
    ]
    base_obs_dim = sum(l for _, l in obs_fields_base)  # 26

    # 预计算累计奖励
    cum_rewards = [0.0]
    if step_rewards:
        for r in step_rewards:
            cum_rewards.append(cum_rewards[-1] + r)

    line_height = font_size + 5
    margin = 8
    total_rollout_steps = len(rollout) - 1  # 排除初始状态

    overlaid = []
    total_frames = len(frames)
    # 进度打印：每 5% 输出一次
    progress_stride = max(1, total_frames // 20) if total_frames > 0 else 1
    for fi, frame in enumerate(frames):
        if total_frames > 0 and (fi == 0 or fi == total_frames - 1 or fi % progress_stride == 0):
            pct = int(round((fi + 1) / total_frames * 100))
            logger.info("OSD 叠加进度: %d%% (%d/%d)", pct, fi + 1, total_frames)
        ri = fi * render_every  # 对应 rollout 的索引
        if ri >= len(rollout):
            overlaid.append(frame)
            continue

        state = rollout[ri]
        try:
            obs_np = np.array(jax.device_get(state.obs["state"]))
        except Exception:
            overlaid.append(frame)
            continue

        # 根据实际 obs 长度判断是否包含 last_act
        obs_fields = list(obs_fields_base)
        if len(obs_np) > base_obs_dim:
            obs_fields.append(("last_act", 4))

        # 解析 obs 向量
        fields = {}
        pos = 0
        for name, length in obs_fields:
            if pos + length <= len(obs_np):
                fields[name] = obs_np[pos:pos + length]
            pos += length

        # ── 构建显示文本行 ──
        lines = []
        lines.append(f"Step: {ri:4d} / {total_rollout_steps}")

        if rng_seed is not None:
            lines.append(f"Seed: {int(rng_seed)}")

        if "sinpitch" in fields and "sinroll" in fields:
            sp = float(fields["sinpitch"][0])
            sr = float(fields["sinroll"][0])
            lines.append(f"sinP:{sp:+.4f}  sinR:{sr:+.4f}")

        if "command" in fields:
            c = fields["command"]
            lines.append(f"Cmd: vx={c[0]:+.2f} vyaw={c[1]:+.2f} h={c[2]:.2f} p={c[3]:.2f}")

        if step_actions is not None and ri > 0 and (ri - 1) < len(step_actions):
            a = np.asarray(step_actions[ri - 1])
            if len(a) >= 4:
                lines.append(f"Act: {a[0]:+.3f} {a[1]:+.3f} {a[2]:+.3f} {a[3]:+.3f}")

        # 最终控制值（经过 PID 等处理后实际施加到执行器的值）
        try:
            ctrl_np = np.array(jax.device_get(state.data.ctrl))
            if len(ctrl_np) >= 4:
                lines.append(f"Ctrl:{ctrl_np[0]:+.3f} {ctrl_np[1]:+.3f} {ctrl_np[2]:+.3f} {ctrl_np[3]:+.3f}")
        except Exception:
            pass

        if "gyro" in fields:
            g = fields["gyro"]
            lines.append(f"Gyro: {g[0]:+.3f} {g[1]:+.3f} {g[2]:+.3f}")

        if "linvel" in fields:
            v = fields["linvel"]
            lines.append(f"LinV: {v[0]:+.3f} {v[1]:+.3f}")

        if "joints" in fields:
            j = fields["joints"]
            lines.append(f"Jnt:  {j[0]:+.3f} {j[1]:+.3f} {j[2]:+.3f} {j[3]:+.3f}")

        if "velocity" in fields:
            jv = fields["velocity"]
            lines.append(f"JntV: {jv[0]:+.3f} {jv[1]:+.3f} {jv[2]:+.3f} {jv[3]:+.3f}")

        if "last_act" in fields:
            la = fields["last_act"]
            lines.append(f"LstA: {la[0]:+.3f} {la[1]:+.3f} {la[2]:+.3f} {la[3]:+.3f}")

        # 轮子-地面摩擦力（切向）信息：来自渲染阶段的 MuJoCo contact force
        if wheel_friction_per_frame is not None and fi < len(wheel_friction_per_frame):
            wf = wheel_friction_per_frame[fi] or {}
            fl = float(wf.get("fric_left", 0.0))
            fr = float(wf.get("fric_right", 0.0))
            ft = float(wf.get("fric_total", fl + fr))
            lines.append(f"Fric: L={fl:6.1f}  R={fr:6.1f}  T={ft:6.1f}  (N)")

            # 轮子/地板接触面的摩擦系数（优先显示滑动摩擦系数）
            mu_floor = wf.get("mu_floor", None)
            mu_l = wf.get("mu_wheel_left", None)
            mu_r = wf.get("mu_wheel_right", None)
            if mu_floor is not None or mu_l is not None or mu_r is not None:
                mf = float(mu_floor) if mu_floor is not None else 0.0
                ml = float(mu_l) if mu_l is not None else 0.0
                mr = float(mu_r) if mu_r is not None else 0.0
                # 可选：接触里实际使用的摩擦系数（contact.friction[0]）平均值
                muc_l = wf.get("mu_contact_left", None)
                muc_r = wf.get("mu_contact_right", None)
                if muc_l is not None or muc_r is not None:
                    cl = float(muc_l) if muc_l is not None else 0.0
                    cr = float(muc_r) if muc_r is not None else 0.0
                    lines.append(f"Mu: floor={mf:.3f}  wL={ml:.3f}  wR={mr:.3f}  (con L={cl:.3f} R={cr:.3f})")
                else:
                    lines.append(f"Mu: floor={mf:.3f}  wL={ml:.3f}  wR={mr:.3f}")

        # Push 信息（始终显示，便于回放/对齐事件）
        try:
            push_vec = np.array(jax.device_get(state.info["push"]))
            push_norm = float(np.linalg.norm(push_vec))
            tag = "PUSH!" if push_norm > 1e-6 else "PUSH"
            lines.append(
                f"{tag} dir=({push_vec[0]:+.3f},{push_vec[1]:+.3f}) |f|={push_norm:.3f}"
            )
        except Exception:
            lines.append("PUSH: n/a")

        rew = float(jax.device_get(state.reward))
        cum_r = cum_rewards[min(ri, len(cum_rewards) - 1)] if cum_rewards else 0.0
        lines.append(f"Rew:{rew:+.3f}  Cum:{cum_r:.2f}")

        done = bool(jax.device_get(state.done) > 0)
        if done:
            lines.append("!! TERMINATED !!")

        # ── 绘制半透明背景 + 文字 ──
        img_arr = np.array(frame, copy=True)
        dummy_img = Image.fromarray(img_arr)
        dummy_draw = ImageDraw.Draw(dummy_img)
        text_widths = [dummy_draw.textlength(line, font=font) for line in lines]
        bg_w = min(int(max(text_widths) + margin * 2), img_arr.shape[1]) if text_widths else 100
        bg_h = min(margin + line_height * len(lines) + margin, img_arr.shape[0])

        # numpy 混合实现半透明背景
        roi = img_arr[:bg_h, :bg_w].astype(np.float32)
        roi = roi * 0.3 + np.array([15, 15, 15], dtype=np.float32) * 0.7
        img_arr[:bg_h, :bg_w] = roi.astype(np.uint8)

        img = Image.fromarray(img_arr)
        draw = ImageDraw.Draw(img)
        y = margin
        for line in lines:
            if "TERMINATED" in line:
                color = (255, 80, 80)
            elif line.startswith("PUSH"):
                color = (80, 255, 255)
            else:
                color = (255, 255, 80)
            draw.text((margin, y), line, fill=color, font=font)
            y += line_height

        overlaid.append(np.array(img))

    logger.info("OSD 叠加完成 (%d 帧)", len(overlaid))
    return overlaid


def rotation_z_to_dir(direction: np.ndarray) -> np.ndarray:
    """计算将 +Z 轴旋转到 direction 方向的 3x3 旋转矩阵 (Rodrigues)。"""
    z = np.array([0.0, 0.0, 1.0])
    d = direction / (np.linalg.norm(direction) + 1e-12)
    v = np.cross(z, d)
    s = np.linalg.norm(v)
    c = np.dot(z, d)
    if s < 1e-8:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]],
                    [v[2], 0, -v[0]],
                    [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


def render_with_push_arrows(
    env,
    rollout: list,
    render_every: int,
    height: int,
    width: int,
    camera,
    scene_option,
    wheel_friction_per_frame: Optional[list] = None,
) -> list:
    """
    逐帧渲染，绘制 3D 箭头：
      - 绿色箭头：(vx, vy) 速度向量，始终显示，长度正比于速度大小
      - 红色箭头：push 推力向量，仅在 push 生效时显示
    """
    renderer = mujoco.Renderer(env.mj_model, height=height, width=width)
    mj_data = mujoco.MjData(env.mj_model)

    # 预取轮子与地面 geom id（用于计算摩擦力）
    # 注：若 XML 中名称不存在，mj_name2id 会返回 -1，此时自动跳过。
    wheel_left_geom_id = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "wheel_left_geom_shell")
    wheel_right_geom_id = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "wheel_right_geom_shell")
    floor_geom_id = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    def _geom_mu_slide(geom_id: int) -> Optional[float]:
        if geom_id < 0:
            return None
        try:
            return float(env.mj_model.geom_friction[geom_id, 0])
        except Exception:
            return None

    def _accum_wheel_friction(data: mujoco.MjData) -> dict:
        # Surface friction coefficients (slide): always report if geom ids exist.
        out = {
            "mu_floor": _geom_mu_slide(floor_geom_id),
            "mu_wheel_left": _geom_mu_slide(wheel_left_geom_id),
            "mu_wheel_right": _geom_mu_slide(wheel_right_geom_id),
        }

        if floor_geom_id < 0 or (wheel_left_geom_id < 0 and wheel_right_geom_id < 0):
            out.update({"fric_left": 0.0, "fric_right": 0.0, "fric_total": 0.0, "ncon": 0})
            return out

        fric_left = 0.0
        fric_right = 0.0
        ncon = 0
        mu_contact_left_sum = 0.0
        mu_contact_right_sum = 0.0
        mu_contact_left_n = 0
        mu_contact_right_n = 0
        cf = np.zeros(6, dtype=np.float64)
        for ci in range(int(data.ncon)):
            con = data.contact[ci]
            g1 = int(con.geom1)
            g2 = int(con.geom2)

            side = None
            if wheel_left_geom_id >= 0 and ((g1 == wheel_left_geom_id and g2 == floor_geom_id) or (g2 == wheel_left_geom_id and g1 == floor_geom_id)):
                side = "left"
            elif wheel_right_geom_id >= 0 and ((g1 == wheel_right_geom_id and g2 == floor_geom_id) or (g2 == wheel_right_geom_id and g1 == floor_geom_id)):
                side = "right"

            if side is None:
                continue

            cf[:] = 0.0
            mujoco.mj_contactForce(env.mj_model, data, ci, cf)
            tangential = float(np.linalg.norm(cf[0:2]))  # 接触坐标系切向摩擦力大小

            # MuJoCo contact 摩擦系数（若可用，取 friction[0] 作为滑动摩擦系数）
            try:
                mu_c0 = float(con.friction[0])
            except Exception:
                mu_c0 = None
            if side == "left":
                fric_left += tangential
                if mu_c0 is not None:
                    mu_contact_left_sum += mu_c0
                    mu_contact_left_n += 1
            else:
                fric_right += tangential
                if mu_c0 is not None:
                    mu_contact_right_sum += mu_c0
                    mu_contact_right_n += 1
            ncon += 1

        fric_total = fric_left + fric_right
        out.update({
            "fric_left": fric_left,
            "fric_right": fric_right,
            "fric_total": fric_total,
            "ncon": ncon,
            "mu_contact_left": (mu_contact_left_sum / mu_contact_left_n) if mu_contact_left_n > 0 else None,
            "mu_contact_right": (mu_contact_right_sum / mu_contact_right_n) if mu_contact_right_n > 0 else None,
        })
        return out

    sampled_states = rollout[::render_every]
    frames = []
    total_frames = len(sampled_states)
    # 进度打印：每 5% 输出一次
    progress_stride = max(1, total_frames // 20) if total_frames > 0 else 1

    for fi, state in enumerate(sampled_states):
        if total_frames > 0 and (fi == 0 or fi == total_frames - 1 or fi % progress_stride == 0):
            pct = int(round((fi + 1) / total_frames * 100))
            logger.info("渲染进度: %d%% (%d/%d)", pct, fi + 1, total_frames)
        # 将 mjx 状态拷贝到 mj_data
        qpos = np.array(jax.device_get(state.data.qpos))
        qvel = np.array(jax.device_get(state.data.qvel))
        mj_data.qpos[:] = qpos
        mj_data.qvel[:] = qvel
        mj_data.time = float(jax.device_get(state.data.time))
        mujoco.mj_forward(env.mj_model, mj_data)

        # 记录该帧轮-地摩擦力（切向）
        if wheel_friction_per_frame is not None:
            try:
                wheel_friction_per_frame.append(_accum_wheel_friction(mj_data))
            except Exception:
                wheel_friction_per_frame.append({"fric_left": 0.0, "fric_right": 0.0, "fric_total": 0.0, "ncon": 0})

        renderer.update_scene(mj_data, camera, scene_option)

        scene = renderer.scene
        root_pos = mj_data.xpos[env._body_id].copy()

        # ── (vx, vy) 速度向量箭头（始终显示） ──
        try:
            vel_xy = qvel[:2]  # world-frame (vx, vy)
            vel_norm = float(np.linalg.norm(vel_xy))
            if vel_norm > 1e-4 and scene.ngeom < scene.maxgeom:
                vel_dir = np.array([vel_xy[0], vel_xy[1], 0.0])
                vel_dir = vel_dir / (np.linalg.norm(vel_dir) + 1e-12)

                # 箭头长度与速度成正比，最小 0.04，最大 0.5
                vel_arrow_len = float(np.clip(vel_norm * 0.3, 0.04, 0.5))
                vel_arrow_radius = 0.006

                vel_arrow_mid = root_pos + np.array([0, 0, 0.05]) + vel_dir * vel_arrow_len / 2
                vel_mat = rotation_z_to_dir(vel_dir)

                mujoco.mjv_initGeom(
                    scene.geoms[scene.ngeom],
                    type=mujoco.mjtGeom.mjGEOM_ARROW,
                    size=np.array([vel_arrow_radius, vel_arrow_radius, vel_arrow_len / 2]),
                    pos=vel_arrow_mid,
                    mat=vel_mat.flatten(),
                    rgba=np.array([0.15, 1.0, 0.15, 0.85], dtype=np.float32),
                )
                scene.ngeom += 1
        except Exception:
            pass

        # ── push 推力箭头（仅 push 生效时显示） ──
        try:
            push = np.array(jax.device_get(state.info["push"]))
            push_norm = float(np.linalg.norm(push))
            if push_norm > 1e-6 and scene.ngeom < scene.maxgeom:
                # push 方向 (XY 平面 -> 3D)
                push_dir = np.array([push[0], push[1], 0.0])
                push_dir = push_dir / np.linalg.norm(push_dir)

                # 箭头长度与 push 大小成正比，最小 0.08，最大 0.4
                arrow_length = float(np.clip(push_norm * 0.6, 0.08, 0.4))
                arrow_radius = 0.008

                # 箭头中点 = root 位置 + 方向偏移
                arrow_mid = root_pos + np.array([0, 0, 0.02]) + push_dir * arrow_length / 2

                # 旋转矩阵：将默认 +Z 对齐到 push_dir
                mat = rotation_z_to_dir(push_dir)

                mujoco.mjv_initGeom(
                    scene.geoms[scene.ngeom],
                    type=mujoco.mjtGeom.mjGEOM_ARROW,
                    size=np.array([arrow_radius, arrow_radius, arrow_length / 2]),
                    pos=arrow_mid,
                    mat=mat.flatten(),
                    rgba=np.array([1.0, 0.15, 0.15, 0.85], dtype=np.float32),
                )
                scene.ngeom += 1
        except Exception:
            pass

        frames.append(renderer.render())

    renderer.close()
    return frames


def render_replay_video(
    env,
    rollout: list,
    step_actions: Optional[list],
    step_rewards: Optional[list],
    save_video_path: Optional[str],
    height: int = 720,
    width: int = 1280,
    rng_seed: Optional[int] = None,
):
    """统一管理回放视频渲染、OSD 叠加与保存/播放。"""
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(env.mj_model, camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    root_body_id = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_BODY, "root")
    camera.trackbodyid = root_body_id
    camera.lookat = [0.0, 0.0, 0.0]
    camera.azimuth = 30
    camera.elevation = -20
    camera.distance = 0.8

    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = True
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTOBJ] = True
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_COM] = True
    scene_option.frame = mujoco.mjtFrame.mjFRAME_WORLD

    render_every = 1 #max(1, len(rollout) // 600)
    wheel_friction_per_frame: list[dict] = []
    frames = render_with_push_arrows(
        env,
        rollout,
        render_every,
        height=height,
        width=width,
        camera=camera,
        scene_option=scene_option,
        wheel_friction_per_frame=wheel_friction_per_frame,
    )
    frames = overlay_state_info(
        frames,
        rollout,
        render_every,
        step_actions=step_actions,
        step_rewards=step_rewards,
        wheel_friction_per_frame=wheel_friction_per_frame,
        rng_seed=rng_seed,
    )
    fps_value = 1.0 / env.dt # / render_every
    logger.info("帧率: %.1f FPS, 共 %d 帧", fps_value, len(frames))

    if save_video_path:
        os.makedirs(os.path.dirname(save_video_path) or ".", exist_ok=True)
        media.write_video(save_video_path, frames, fps=fps_value)
        logger.info("视频已保存: %s", save_video_path)
    else:
        media.show_video(frames, fps=fps_value)
