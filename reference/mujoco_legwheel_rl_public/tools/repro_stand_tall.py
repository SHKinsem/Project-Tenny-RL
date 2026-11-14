#!/usr/bin/env python
"""复现 viewer:站稳后腿伸直 → 跑动 + 左右腿不等长。

口径对齐 view.sh:noise=0, push=0, cmd_prob=0, deterministic 策略。
序列:settle 3s @ height=-0.25,再 step 到 0.0(伸直) 4s。
对照:同样序列但 noise=1(训练默认)。

输出 test/repro_stand_tall/:csv + 曲线图。
"""
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.pop("MUJOCO_GL", None)  # mac 上 egl 非法,本脚本不渲染

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax
import jax.numpy as jp

CKPT = "tuning/run1/trials/confirm-baseline-s0/ckpt/000408576000"
OUT = "test/repro_stand_tall"
DT = 0.01
SETTLE_S = 6.0
EXTEND_S = 4.0
SEED = 0  # 与 viewer 首次 reset 同 PRNGKey(0)


def _yaw_from_quat(q):
    w, x, y, z = (float(v) for v in q)
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def rollout(env, inference, jit_reset, jit_step, schedule, noise, label, random_reset):
    """schedule: list of (cmd4, duration_s, phase). viewer 同款每步写 command。"""
    rng = jax.random.PRNGKey(SEED)
    state = jit_reset(rng, random_reset)
    rows = []
    t = 0.0

    def telem(st, t, phase, c, act):
        qpos = np.asarray(st.data.qpos)
        thighs = np.asarray(st.data.qpos[env._qpos_thigh_joints])
        gyro = np.asarray(env.get_gyro_newest(st.data))
        lin_x = float(np.asarray(env.get_local_linvel_x(st.data)).reshape(-1)[0])
        sinroll = float(np.asarray(env._getSinRoll_newest(st.data)))
        sinpitch = float(np.asarray(env._getSinPitch_newest(st.data)))
        ctrl = np.asarray(st.data.ctrl)
        act = np.asarray(act).reshape(-1)
        wsin = np.asarray(env.get_wheels_sinposition(st.data)).reshape(-1)
        s_act = float(np.asarray(env.get_hold_position(st.data)).reshape(-1)[0])
        s0 = float(np.asarray(st.info["position_start"]).reshape(-1)[0])
        syn = float(np.asarray(env._getSinYaw_newest(st.data)).reshape(-1)[0])
        syn0 = float(np.asarray(st.info["sinyaw_start"]).reshape(-1)[0])
        last_tgt = np.asarray(st.info["last_target_wheel_vel"]).reshape(-1)
        return dict(
            t=t, phase=phase, noise=int(noise),
            cmd_h=float(c[2]), cmd_vx=float(c[0]), cmd_roll=float(c[3]),
            thigh_L=float(thighs[0]), thigh_R=float(thighs[1]),
            split=float(thighs[0] - thighs[1]),
            mean_h=float(np.mean(thighs)),
            lin_vel=lin_x,
            yaw_rate=float(gyro[2]),
            yaw=_yaw_from_quat(qpos[3:7]),
            roll=sinroll, pitch=sinpitch,
            x=float(qpos[0]), y=float(qpos[1]), z=float(qpos[2]),
            act_th_L=float(act[0]), act_th_R=float(act[1]),
            act_wh_L=float(act[2]), act_wh_R=float(act[3]),
            ctrl_th_L=float(ctrl[0]), ctrl_th_R=float(ctrl[1]),
            tq_L=float(ctrl[2]), tq_R=float(ctrl[3]),
            wsin_L=float(wsin[0]), wsin_R=float(wsin[1]),
            s_act=s_act, s0=s0, ds=s_act - s0,
            syn=syn, syn0=syn0, syn_err=syn - syn0,
            tgt_wh_L=float(last_tgt[0]), tgt_wh_R=float(last_tgt[1]),
            done=float(np.asarray(st.done)),
        )

    step_i = 0
    fell = False
    for cmd, duration, phase in schedule:
        cmd = np.asarray(cmd, dtype=np.float32)
        n = int(round(duration / DT))
        for _ in range(n):
            state = state.replace(info={**state.info, "command": jp.array(cmd)})
            act_rng, rng = jax.random.split(rng)
            act, _ = inference(state.obs, act_rng)
            state = jit_step(state, act)
            row = telem(state, t, phase, cmd, act)
            rows.append(row)
            if step_i % 100 == 0:
                print(f"  [{label}] t={row['t']:5.2f} {phase:14s} "
                      f"thL={row['thigh_L']:+.3f} thR={row['thigh_R']:+.3f} "
                      f"split={row['split']:+.3f} vx={row['lin_vel']:+.3f} "
                      f"wz={row['yaw_rate']:+.3f}", flush=True)
            t += DT
            step_i += 1
            if row["done"] > 0:
                # 终止(摔倒/数值异常):记录摔倒时刻并中断本 case,
                # 避免把躺平轨迹统计成"假完美"数据
                print(f"  [{label}] !! 终止(摔倒) @ step {step_i} "
                      f"(t={row['t']:.2f}s, phase={phase}),中断本 case", flush=True)
                fell = True
                break
        if fell:
            break
    return rows


def summarize(rows, tag):
    # 存活判定:首个 done 行即终止时刻,统计只用其前的存活段,
    # 终止行本身(已摔倒的状态)不计入
    fell_idx = next((i for i, r in enumerate(rows) if r["done"] > 0), None)
    if fell_idx is None:
        print(f"\n==== {tag} 末 1s 统计 [存活:全程 {len(rows)} 步未终止] ====")
        alive_rows = rows
    else:
        fall_t = rows[fell_idx]["t"]
        print(f"\n==== {tag} 末 1s 统计 [!! 摔倒 @ step {fell_idx + 1} "
              f"(t={fall_t:.2f}s, phase={rows[fell_idx]['phase']}),"
              f"以下仅统计存活段({fell_idx} 步)] ====")
        alive_rows = rows[:fell_idx]
    if not alive_rows:
        print("  (无存活段可统计)")
        return
    arr = {k: np.array([r[k] for r in alive_rows]) for k in
           ("t", "cmd_h", "thigh_L", "thigh_R", "split", "mean_h", "lin_vel",
            "yaw_rate", "roll", "act_th_L", "act_th_R", "act_wh_L",
            "act_wh_R", "ctrl_th_L", "ctrl_th_R",
            "wsin_L", "wsin_R", "ds", "syn_err")}
    phases = np.array([r["phase"] for r in alive_rows])
    for ph in dict.fromkeys(r["phase"] for r in alive_rows):
        m = phases == ph
        if not np.any(m):
            continue
        # 该阶段最后 1s
        t = arr["t"][m]
        tail = m & (arr["t"] >= t.max() - 1.0)
        def st(name):
            v = arr[name][tail]
            return f"{v.mean():+.4f}±{v.std():.4f}  (min {v.min():+.3f} max {v.max():+.3f})"
        print(f"  [{ph}]")
        print(f"    thigh_L  {st('thigh_L')}")
        print(f"    thigh_R  {st('thigh_R')}")
        print(f"    split    {st('split')}   |split|mean={np.abs(arr['split'][tail]).mean():.4f}")
        print(f"    mean_h   {st('mean_h')}")
        # thigh 跟踪:实际 mean_h vs 指令 cmd_h
        cmd_h = arr["cmd_h"][tail]
        h_err = arr["mean_h"][tail] - cmd_h
        print(f"    cmd_h    {cmd_h.mean():+.4f}  ->  跟踪误差 mean_h-cmd_h "
              f"{h_err.mean():+.4f}  (|err|mean={np.abs(h_err).mean():.4f})")
        print(f"    lin_vel  {st('lin_vel')}")
        print(f"    yaw_rate {st('yaw_rate')}")
        print(f"    roll     {st('roll')}")
        print(f"    ds       {st('ds')}")
        print(f"    syn_err  {st('syn_err')}")
        print(f"    act_th   L {arr['act_th_L'][tail].mean():+.3f}  R {arr['act_th_R'][tail].mean():+.3f}")
        print(f"    act_wh   L {arr['act_wh_L'][tail].mean():+.3f}  R {arr['act_wh_R'][tail].mean():+.3f}")
        print(f"    ctrl_th  L {arr['ctrl_th_L'][tail].mean():+.3f}  R {arr['ctrl_th_R'][tail].mean():+.3f}")
        xy = np.array([[r["x"], r["y"]] for r, keep in zip(alive_rows, tail) if keep])
        if len(xy):
            disp = np.hypot(xy[-1, 0] - xy[0, 0], xy[-1, 1] - xy[0, 1])
            print(f"    xy drift over last 1s: {disp:.4f} m")


def plot(rows_by_tag, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
    for tag, rows in rows_by_tag.items():
        t = np.array([r["t"] for r in rows])
        axes[0].plot(t, [r["thigh_L"] for r in rows], label=f"{tag} L")
        axes[0].plot(t, [r["thigh_R"] for r in rows], "--", label=f"{tag} R")
        axes[0].plot(t, [r["cmd_h"] for r in rows], "k:", alpha=0.5)
        axes[1].plot(t, [r["split"] for r in rows], label=tag)
        axes[2].plot(t, [r["lin_vel"] for r in rows], label=tag)
        axes[3].plot(t, [r["yaw_rate"] for r in rows], label=tag)
        axes[4].plot(t, [r["ds"] for r in rows], label=tag)
    for ax, ylab in zip(axes, ("thigh L/R (rad)", "split L-R (rad)",
                               "lin_vel x (m/s)", "yaw rate (rad/s)",
                               "Δs (rad)")):
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
    axes[0].axhline(0.0, color="0.5", lw=0.5)
    axes[-1].set_xlabel("t (s)")
    axes[0].set_title("random reset: extend vs walk-then-extend")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"saved {path}")


def main():
    from legwheel import LegWheelRobot, default_config
    from legwheel.evaluation.runner import load_policy

    os.makedirs(OUT, exist_ok=True)
    ckpt = os.path.abspath(CKPT)
    print(f"ckpt={ckpt}", flush=True)

    def make_env(noise_on):
        cfg = default_config()
        cfg.noise_config.level = 1.0 if noise_on else 0.0
        cfg.push_config.enable = False
        cfg.cmd_config.cmd_prob = 0.0
        return LegWheelRobot(cfg)

    inference = jax.jit(load_policy(ckpt, deterministic=True))
    print("\n---- viewer_noise0 env (push=0,cmd_prob=0) ----", flush=True)
    env = make_env(False)
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    cases = [
        ("extend_no_walk", True, [
            ([0.0, 0.0, -0.25, 0.0], 3.0, "settle_-0.25"),
            ([0.0, 0.0, 0.0, 0.0], 4.0, "extend_0.0"),
        ]),
        ("extend_after_walk", True, [
            ([0.0, 0.0, -0.25, 0.0], 3.0, "settle_-0.25"),
            ([0.2, 0.0, -0.25, 0.0], 2.0, "fwd_0.2"),
            ([0.0, 0.0, -0.25, 0.0], 1.5, "stop"),
            ([0.0, 0.0, 0.0, 0.0], 4.0, "extend_0.0"),
        ]),
    ]
    rows_by_tag = {}
    for tag, rnd, schedule in cases:
        print(f"\n---- {tag} random={rnd} ----", flush=True)
        rows = rollout(env, inference, jit_reset, jit_step, schedule,
                       noise=False, label=tag, random_reset=rnd)
        rows_by_tag[tag] = rows
        header = list(rows[0].keys())
        csv_path = os.path.join(OUT, f"{tag}.csv")
        with open(csv_path, "w") as f:
            f.write(",".join(header) + "\n")
            for r in rows:
                f.write(",".join(str(r[k]) for k in header) + "\n")
        print(f"saved {csv_path}")
        summarize(rows, tag)

    plot(rows_by_tag, os.path.join(OUT, "walk_then_extend.png"))


if __name__ == "__main__":
    main()
