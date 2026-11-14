#!/usr/bin/env python
"""baseline 4e8 全量 ckpt 的 60s 部署效果演示视频:脚本化命令切换 + 随机推力。

- 策略: tuning/run1/trials/confirm-baseline-s0/ckpt/000408576000
  (= 代码默认参数 4e8 全量训练,deploy 口径第一名);
- deterministic 推理;观测噪声开(env 默认),随机推力开(env 默认);
  域随机化不开(纯可视化演示,非评测);
- 命令编排: 12 段 x 5s = 60s,覆盖 lin_vel/yaw/height/roll 单维度与组合;
  通过把 command_change_interval/cmd_start_delay 调到极大禁止 env 自动重采样,
  每段开始时强制写入脚本命令;
- 摔倒自动重置接着演,摔倒时刻记入 falls.txt;
- 可视化叠加(1080p 版新增):
  顶部:时间 + 当前命令;
  左上面板:实测 lin_vel / yaw_rate / 左右腿高 / roll / pitch / 轮扭矩(附命令值);
  底部:lin_vel 与 yaw_rate 的滚动跟踪曲线(绿=实测,红=命令,近 5s 窗口);
  推力:被随机推力击中的瞬间,顶部红色闪现 "PUSH! 幅值 m/s @ 方向角",持续 1s;
- 输出: test/baseline_demo_60s/baseline_demo_60s_s{NN}[_1080p].mp4 (50fps)
  + schedule.txt + falls_s{NN}.txt。

用法: .venv-local/bin/python tools/record_baseline_demo.py [--seed 7] [--width 1920 --height 1080]
       [--ckpt 路径] [--out 输出目录] [--tag 文件名前缀]
新语义环境(hold-position-heading)下遥测面板含停车保持误差行(共模 Δs mm / Δψ deg)。
"""

import argparse
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")  # 无头服务器走 EGL 渲染
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jp
import mujoco
import numpy as np

CKPT = "tuning/run1/trials/confirm-baseline-s0/ckpt/000408576000"
OUT_DIR = "test/baseline_demo_60s"
TAG = "baseline_demo_60s"
FPS = 50            # dt=0.01, render_every=2
DURATION_S = 60.0
SEG_S = 5.0

# (label, [lin_vel_x, ang_vel_yaw, height, roll])
SCHEDULE = [
    ("stand",          [0.0, 0.0, -0.25, 0.0]),
    ("forward 0.2",    [0.2, 0.0, -0.25, 0.0]),
    ("backward -0.2",  [-0.2, 0.0, -0.25, 0.0]),
    ("yaw left 1.5",   [0.0, 1.5, -0.25, 0.0]),
    ("yaw right -1.5", [0.0, -1.5, -0.25, 0.0]),
    ("crouch -0.5",    [0.0, 0.0, -0.5, 0.0]),
    ("stand tall 0.0", [0.0, 0.0, 0.0, 0.0]),
    ("roll left 0.2",  [0.0, 0.0, -0.25, 0.2]),
    ("roll right -0.2",[0.0, 0.0, -0.25, -0.2]),
    ("fwd+yaw",        [0.2, 1.0, -0.25, 0.0]),
    ("combo",          [0.15, -1.0, -0.4, 0.15]),
    ("stand",          [0.0, 0.0, -0.25, 0.0]),
]

PUSH_FLASH_S = 1.0     # 推力提示显示时长
STRIP_WIN = 250        # 滚动曲线窗口(帧数,= 5s)

# 随机命令编排的采样范围,与 legwheel/config.py cmd_config 一致
RANDOM_CMD_RANGES = [(-0.2, 0.2), (-1.5, 1.5), (-0.5, 0.0), (-0.2, 0.2)]
RANDOM_STAND_PROB = 0.15  # 每段 15% 概率为站立(零速停车)


def random_schedule(n_seg, seed):
    """按 env cmd_config 采样范围随机生成 n_seg 段命令(每段 5s,seed 可复现)。"""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_seg):
        if rng.random() < RANDOM_STAND_PROB:
            out.append(("stand", [0.0, 0.0, -0.25, 0.0]))
            continue
        cmd = [round(float(rng.uniform(lo, hi)), 2) for lo, hi in RANDOM_CMD_RANGES]
        out.append((f"v{cmd[0]:+.2f} y{cmd[1]:+.2f} h{cmd[2]:+.2f} r{cmd[3]:+.2f}", cmd))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=7, help="随机种子(影响初始状态/推力/噪声)")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--duration", type=float, default=DURATION_S,
                    help="演示时长(秒),须为 5 的倍数;调试用")
    ap.add_argument("--ckpt", type=str, default=CKPT, help="checkpoint 路径")
    ap.add_argument("--out", type=str, default=OUT_DIR, help="输出目录")
    ap.add_argument("--tag", type=str, default=TAG, help="输出文件名前缀")
    ap.add_argument("--random-cmds", action="store_true",
                    help="命令编排改为按 env 采样范围随机生成(每 5s 一段,seed 决定)")
    args = ap.parse_args()
    seed = args.seed
    W, H = args.width, args.height
    ckpt = args.ckpt
    out_dir = args.out
    tag = args.tag

    from legwheel import LegWheelRobot, default_config
    from legwheel.evaluation.runner import load_policy

    cfg = default_config()
    cfg.cmd_config.command_change_interval = 10 ** 6  # 禁止自动重采样命令(>6000 即可,int32 内)
    cfg.cmd_start_delay_s = 200.0  # 20000 步后才触发首次采样,60s 演示内不触发
    env = LegWheelRobot(cfg)
    dt = env.dt
    n_steps = int(args.duration / dt)
    seg_steps = int(SEG_S / dt)
    n_seg = int(args.duration // SEG_S)
    assert n_steps == n_seg * seg_steps
    if args.random_cmds:
        schedule = random_schedule(n_seg, seed)
    else:
        assert n_seg <= len(SCHEDULE), "固定编排仅 12 段,更长时长请用 --random-cmds"
        schedule = SCHEDULE[:n_seg]

    inference = jax.jit(load_policy(os.path.abspath(ckpt), deterministic=True))
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    @jax.jit
    def telem(data, info):
        """每帧遥测:lin_vel/yaw_rate/关节/sinroll/sinpitch/push/ctrl/世界速度/偏航角,
        以及停车保持误差(共模 Δs 与 Δψ;新语义锚点 position_start/sinyaw_start)。"""
        qw, qx, qy, qz = data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6]
        yaw = jp.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy ** 2 + qz ** 2))
        # Δψ:sinyaw 锚点反解为角度(sinyaw=sin(yaw*0.5)/0.5),与当前 yaw 求差( wrap )
        yaw0 = 2 * jp.arcsin(jp.clip(0.5 * info["sinyaw_start"], -1.0, 1.0))
        dpsi = jp.arctan2(jp.sin(yaw - yaw0), jp.cos(yaw - yaw0))
        ds = env.get_hold_position(data) - info["position_start"]
        return (env.get_local_linvel_x(data),
                env.get_gyro_newest(data)[2],
                data.qpos[env._qpos_wanted_joints],
                env._getSinRoll_newest(data),
                env._getSinPitch_newest(data),
                info["push"],
                data.ctrl,
                data.qvel[0:2],
                yaw,
                ds,
                dpsi)

    # 相机:跟踪 root,视角同 scripts/visualize.py
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(env.mj_model, camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = mujoco.mj_name2id(
        env.mj_model, mujoco.mjtObj.mjOBJ_BODY, "root")
    camera.azimuth = 30
    camera.elevation = -20
    camera.distance = 0.8
    scene_option = mujoco.MjvOption()

    rng = jax.random.PRNGKey(seed)
    state = jit_reset(rng, True)
    falls = []
    rollout = []          # 每 2 步存 1 档(对应 50fps)
    frame_cmd = []        # 每帧 (t, label, cmd)
    frame_telem = []      # 每帧遥测 dict
    push_events = []      # (t, magnitude, theta_deg)
    print(f"[demo] start: {n_steps} steps, seed={seed}, ckpt={ckpt}", flush=True)
    for step in range(n_steps):
        seg = step // seg_steps
        label, cmd = schedule[seg]
        if step % seg_steps == 0:
            state = state.replace(
                info={**state.info, "command": jp.array(cmd, dtype=jp.float32)})
            print(f"[demo] t={step*dt:5.1f}s -> {label} {cmd}", flush=True)
        act_rng, rng = jax.random.split(rng)
        act, _ = inference(state.obs, act_rng)
        state = jit_step(state, act)
        push_vec = np.asarray(jax.device_get(state.info["push"]))
        if np.linalg.norm(push_vec) > 0:
            mag = float(np.linalg.norm(push_vec))
            deg = float(np.degrees(np.arctan2(push_vec[1], push_vec[0])))
            push_events.append((step * dt, mag, deg))
            print(f"[demo] t={step*dt:5.1f}s PUSH {mag:.2f} m/s @ {deg:.0f}deg",
                  flush=True)
        if bool(jax.device_get(state.done)):
            falls.append(step * dt)
            print(f"[demo] t={step*dt:5.1f}s FALL ({label}), reset", flush=True)
            rng, reset_rng = jax.random.split(rng)
            state = jit_reset(reset_rng, True)
            state = state.replace(
                info={**state.info, "command": jp.array(cmd, dtype=jp.float32)})
        if step % 2 == 0:
            lv, gyro_z, jnts, sinroll, sinpitch, _, ctrl, wvel, yaw, ds, dpsi = \
                jax.device_get(telem(state.data, state.info))
            rollout.append(state)
            frame_cmd.append((step * dt, label, cmd))
            frame_telem.append(dict(
                lin_vel=float(np.asarray(lv).reshape(-1)[0]),
                yaw_rate=float(gyro_z),
                thigh_l=float(jnts[0]), thigh_r=float(jnts[2]),
                sinroll=float(sinroll), sinpitch=float(sinpitch),
                tq_l=float(ctrl[2]), tq_r=float(ctrl[3]),
                wvel_x=float(wvel[0]), wvel_y=float(wvel[1]),
                yaw=float(yaw),
                ds_mm=float(ds) * float(env._config.wheel_radius) * 1000.0,
                dpsi_deg=float(np.degrees(dpsi)),
            ))

    print(f"[demo] rollout done, falls={len(falls)} at {falls}, "
          f"pushes={len(push_events)}", flush=True)

    # ---- 渲染 + 可视化叠加 + 写视频(逐块渲染,避免一次性占内存) ----
    import cv2
    from PIL import Image, ImageDraw, ImageFont

    font_top = ImageFont.truetype("DejaVuSans.ttf", max(16, H // 42))
    font_panel = ImageFont.truetype("DejaVuSans.ttf", max(13, H // 50))
    font_push = ImageFont.truetype("DejaVuSans.ttf", max(20, H // 30))

    def draw_strip(d, box, hist, cmd_key, act_key, ylim, title):
        """滚动跟踪曲线:红=命令,绿=实测。box=(x0,y0,x1,y1)。"""
        x0, y0, x1, y1 = box
        d.rectangle(box, fill=(10, 10, 20), outline=(90, 90, 90))
        d.text((x0 + 6, y0 + 2), title, font=font_panel, fill=(220, 220, 220))
        mid = (y0 + y1) / 2
        d.line([x0, mid, x1, mid], fill=(60, 60, 60))
        n = len(hist)
        if n < 2:
            return

        def pt(i, v):
            x = x0 + (x1 - x0) * i / (STRIP_WIN - 1)
            y = mid - (y1 - y0) * 0.45 * np.clip(v / ylim, -1, 1)
            return (x, y)

        for key, color in ((cmd_key, (255, 90, 90)), (act_key, (90, 255, 90))):
            pts = [pt(STRIP_WIN - n + i, rec[key]) for i, rec in enumerate(hist)]
            d.line(pts, fill=color, width=2)

    VEL_REF = 0.5  # m/s,对应罗盘满半径

    def arrow(d, cx, cy, ang, length, color, width=4, head=12):
        """世界系角度 ang(ccw,从 +x 起),屏幕 y 翻转;从 (cx,cy) 画箭头。"""
        ex = cx + length * np.cos(ang)
        ey = cy - length * np.sin(ang)
        d.line([cx, cy, ex, ey], fill=color, width=width)
        a1, a2 = ang + np.pi * 0.82, ang - np.pi * 0.82
        p1 = (ex + head * np.cos(a1), ey - head * np.sin(a1))
        p2 = (ex + head * np.cos(a2), ey - head * np.sin(a2))
        d.polygon([(ex, ey), p1, p2], fill=color)

    def draw_compass(d, tm, cmd, t):
        """右下罗盘:白=机身朝向,红=cmd 速度,绿=实际速度,橙=push(闪现)。"""
        R = H // 9
        cx, cy = W - R - 30, H - R - 30
        d.ellipse([cx - R, cy - R, cx + R, cy + R], fill=(10, 10, 20, 180),
                  outline=(90, 90, 90))
        yaw = tm["yaw"]
        scale = (R - 15) / VEL_REF
        # 机身朝向(白)
        arrow(d, cx, cy, yaw, R - 25, (240, 240, 240), width=3, head=10)
        # cmd 线速度(红):沿朝向,后退取反
        cmd_v = cmd[0]
        if abs(cmd_v) > 0.01:
            ang = yaw if cmd_v > 0 else yaw + np.pi
            arrow(d, cx, cy, ang, min(abs(cmd_v), VEL_REF) * scale,
                  (255, 90, 90), width=4)
        # 实际速度(绿):世界系 qvel[:2]
        vx, vy = tm["wvel_x"], tm["wvel_y"]
        spd = float(np.hypot(vx, vy))
        if spd > 0.02:
            arrow(d, cx, cy, np.arctan2(vy, vx), min(spd, VEL_REF) * scale,
                  (90, 255, 90), width=4)
        # cmd 偏航率(红弧):ccw 为正,满 1.5 rad/s 对应 120° 扫掠
        cmd_yaw = cmd[1]
        if abs(cmd_yaw) > 0.05:
            sweep = np.clip(cmd_yaw / 1.5, -1, 1) * np.radians(120)
            angs = np.linspace(yaw, yaw + sweep, 17)
            r = R - 6
            pts = [(cx + r * np.cos(a), cy - r * np.sin(a)) for a in angs]
            d.line(pts, fill=(255, 90, 90), width=3)
            arrow(d, *pts[-2], angs[-1] + np.pi / 2 * np.sign(sweep),
                  0.1, (255, 90, 90), width=1, head=10)
        # push(橙,闪现 PUSH_FLASH_S 秒)
        recent = [e for e in push_events if 0 <= t - e[0] <= PUSH_FLASH_S]
        if recent:
            _, mag, deg = recent[-1]
            arrow(d, cx, cy, np.radians(deg), min(mag, VEL_REF) * scale,
                  (255, 170, 40), width=7, head=16)
        # 图例
        d.text((cx - R, cy - R - (H // 50 + 10)),
               "white=heading red=cmd green=vel orange=push",
               font=font_panel, fill=(200, 200, 200))

    def overlay(img, j):
        t, label, cmd = frame_cmd[j]
        tm = frame_telem[j]
        d = ImageDraw.Draw(img, "RGBA")
        # 顶部:时间 + 命令
        txt = (f"t={t:4.1f}s  [{label}]  "
               f"v={cmd[0]:+.2f} yaw={cmd[1]:+.2f} h={cmd[2]:+.2f} roll={cmd[3]:+.2f}")
        d.rectangle([4, 4, 12 + 11 * len(txt), 14 + H // 42], fill=(0, 0, 0, 200))
        d.text((10, 6), txt, font=font_top, fill=(255, 255, 120))
        # 左上遥测面板
        lines = [
            f"lin_vel  {tm['lin_vel']:+.2f}  (cmd {cmd[0]:+.2f})",
            f"yaw_rate {tm['yaw_rate']:+.2f}  (cmd {cmd[1]:+.2f})",
            f"height L {tm['thigh_l']:+.2f} R {tm['thigh_r']:+.2f}  (cmd {cmd[2]:+.2f})",
            f"roll     {tm['sinroll']:+.3f}  (cmd {cmd[3]:+.2f})   pitch {tm['sinpitch']:+.3f}",
            f"torque L {tm['tq_l']:+.3f}  R {tm['tq_r']:+.3f}",
            f"hold ds {tm['ds_mm']:+.1f} mm   dpsi {tm['dpsi_deg']:+.1f} deg",
        ]
        lh = H // 50 + 8
        py0 = 20 + H // 42
        d.rectangle([4, py0, 560, py0 + lh * len(lines) + 10], fill=(0, 0, 0, 140))
        for k, ln in enumerate(lines):
            d.text((12, py0 + 6 + k * lh), ln, font=font_panel,
                   fill=(200, 230, 255))
        # 推力闪现(最近 PUSH_FLASH_S 秒内的事件)
        recent = [e for e in push_events if 0 <= t - e[0] <= PUSH_FLASH_S]
        if recent:
            _, mag, deg = recent[-1]
            ptxt = f"PUSH! {mag:.2f} m/s @ {deg:.0f} deg"
            tw = 20 * len(ptxt) * (H // 30) // 36
            d.rectangle([W // 2 - tw // 2 - 10, 4, W // 2 + tw // 2 + 10,
                         16 + H // 30], fill=(120, 0, 0, 220))
            d.text((W // 2 - tw // 2, 8), ptxt, font=font_push,
                   fill=(255, 80, 80))
        # 底部滚动曲线(近 STRIP_WIN 帧)
        lo = max(0, j - STRIP_WIN + 1)
        hist = [dict(frame_telem[k],
                     cmd_v=frame_cmd[k][2][0], cmd_yaw=frame_cmd[k][2][1])
                for k in range(lo, j + 1)]
        ch_h, ch_w = H // 7, W // 4 - 20
        y0 = H - ch_h - 10
        draw_strip(d, (10, y0, 10 + ch_w, H - 10), hist,
                   "cmd_v", "lin_vel", 0.25, "lin_vel (green=actual, red=cmd)")
        draw_strip(d, (30 + ch_w, y0, 30 + 2 * ch_w, H - 10), hist,
                   "cmd_yaw", "yaw_rate", 1.6, "yaw_rate (green=actual, red=cmd)")
        # 右下罗盘箭头
        draw_compass(d, tm, cmd, t)
        return img

    os.makedirs(out_dir, exist_ok=True)
    res_tag = "" if (W, H) == (960, 540) else f"_{H}p"
    out_mp4 = os.path.join(
        out_dir, f"{tag}_s{seed:02d}{res_tag}.mp4")
    writer = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*"mp4v"),
                             FPS, (W, H))
    chunk = 250
    for i in range(0, len(rollout), chunk):
        frames = env.render(rollout[i:i + chunk], height=H, width=W,
                            camera=camera, scene_option=scene_option)
        for j, fr in enumerate(frames):
            img = overlay(Image.fromarray(np.asarray(fr)), i + j)
            writer.write(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR))
        print(f"[demo] rendered {min(i + chunk, len(rollout))}/{len(rollout)}",
              flush=True)
    writer.release()

    with open(os.path.join(out_dir, "schedule.txt"), "a") as fp:
        fp.write(f"seed={seed} {'random' if args.random_cmds else 'fixed'} schedule:\n")
        for k, (label, cmd) in enumerate(schedule):
            fp.write(f"{k*SEG_S:4.0f}-{(k+1)*SEG_S:4.0f}s  {label:36s} {cmd}\n")
    with open(os.path.join(out_dir, f"falls_s{seed:02d}{res_tag}.txt"), "w") as fp:
        fp.write(f"seed={seed} res={W}x{H} falls: {len(falls)}\n")
        for t in falls:
            fp.write(f"  t={t:.2f}s\n")
        fp.write(f"pushes: {len(push_events)}\n")
        for t, mag, deg in push_events:
            fp.write(f"  t={t:.2f}s  {mag:.2f} m/s @ {deg:.0f} deg\n")
    print(f"[demo] video -> {out_mp4}", flush=True)


if __name__ == "__main__":
    main()
