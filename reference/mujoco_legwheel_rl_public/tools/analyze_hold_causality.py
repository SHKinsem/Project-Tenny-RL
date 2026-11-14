#!/usr/bin/env python
"""hold-position-heading 改动的因果分析(E1 反事实奖励 / E2 机动污染 / E3 静止收益)。

E1 反事实奖励对照(隔离"定义性下降" vs "行为性下降"):
  旧策略轨迹由 tools/e1_old_rollout.py 在旧代码(/tmp/lw_base)下录制;
  新策略轨迹由本脚本 --mode e1new 录制;--mode e1report 对两条轨迹离线
  重算新旧两套 hold 公式(其余分项取自在线 reward,公式无关),输出四方格:
    旧策略×旧公式 / 旧策略×新公式 / 新策略×旧公式 / 新策略×新公式
  旧公式(74648c2):per-wheel 锚点 sin(q*0.1)/0.1 空间,误差=mean(diffL,diffR),
    锚点在 |vx|>margin 或 |yaw|>margin 时刷新;门控=vx,yaw 都在死区。
  新公式(当前):共模 s=(qL+qR)/2 原始弧度,锚点仅 |vx|>margin 刷新;门控=vx 死区。
  (sinyaw 项两版公式一致:Δψ=sin(yaw*0.5)/0.5 相对锚点,σ=0.1,yaw 死区门控。)

E2 机动污染测试(新代码,规范初始化,noise=0,push=0,deterministic):
  (a) 强制蹲起(脚本 thigh 斜坡,policy 轮 / 脚本轮两版)
  (b) 强制 roll split  (c) 强制差速 yaw 旋转 (d) 静止对照(policy)
  逐帧记录 Δs、Δψ、机身平移(qpos x/y 真值),回答:机动本身是否污染轮里程计。

E3 静止收益核算:门控激活占比(解析+实测)与"不动 vs 跟踪"收益差(E1 数据)。

用法:
  .venv/bin/python tools/analyze_hold_causality.py --mode e1new --out /tmp/e1_new_traj.npz
  .venv/bin/python tools/analyze_hold_causality.py --mode e1report \
      --old /tmp/e1_old_traj_rand1.npz --new /tmp/e1_new_traj.npz
  .venv/bin/python tools/analyze_hold_causality.py --mode e2
"""
import argparse
import os
import sys

CKPT = "ckpts/holdrec-step2/000403660800"


# ─────────────────────────── E1: 新侧轨迹录制 ───────────────────────────
def run_e1new(args):
    import numpy as np
    import jax
    import jax.numpy as jp

    from legwheel import LegWheelRobot, default_config
    from legwheel.evaluation.runner import load_policy

    cfg = default_config()
    env = LegWheelRobot(cfg)
    inference = jax.jit(load_policy(os.path.abspath(args.ckpt), deterministic=True))
    num_envs, steps = args.num_envs, int(env._config.episode_length)

    vmap_reset = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))
    vmap_step = jax.jit(jax.vmap(env.step))
    vmap_infer = jax.jit(jax.vmap(inference))
    vmap_sinyaw = jax.jit(jax.vmap(env._getSinYaw_newest))
    q_w = [int(env._qpos_wanted_joints[1]), int(env._qpos_wanted_joints[3])]

    rng = jax.random.PRNGKey(args.seed)
    states = vmap_reset(jax.random.split(rng, num_envs), bool(args.random_init))
    act_rng = jax.random.PRNGKey(0)

    rec = {k: [] for k in
           ("cmd", "qL", "qR", "sinyaw", "reward", "done",
            "term_wsin", "term_sinyaw", "anch_s", "anch_y")}
    for i in range(steps):
        rec["cmd"].append(np.asarray(jax.device_get(states.info["command"])))
        act_rng, sub = jax.random.split(act_rng)
        acts, _ = vmap_infer(states.obs, jax.random.split(sub, num_envs))
        states = vmap_step(states, acts)
        rec["qL"].append(np.asarray(jax.device_get(states.data.qpos[:, q_w[0]])))
        rec["qR"].append(np.asarray(jax.device_get(states.data.qpos[:, q_w[1]])))
        rec["sinyaw"].append(np.asarray(jax.device_get(vmap_sinyaw(states.data))))
        rec["reward"].append(np.asarray(jax.device_get(states.reward)))
        rec["done"].append(np.asarray(jax.device_get(states.done)))
        rec["term_wsin"].append(np.asarray(jax.device_get(
            states.metrics["reward/tracking_wheels_sinposition"])))
        rec["term_sinyaw"].append(np.asarray(jax.device_get(
            states.metrics["reward/tracking_sinyaw"])))
        rec["anch_s"].append(np.asarray(jax.device_get(states.info["position_start"])))
        rec["anch_y"].append(np.asarray(jax.device_get(states.info["sinyaw_start"])))
        if i % 100 == 0:
            print(f"  step {i}/{steps}", flush=True)

    arrs = {k: np.stack(v) for k, v in rec.items()}
    np.savez(args.out, **arrs,
             zero_keep_scale=float(env._config.reward_config.zero_keep_scale),
             sigma_wheel=float(env._config.reward_config.sigma_wheel),
             sigma=float(env._config.reward_config.sigma),
             cmd_scale_yaw=float(env._cmd_scales[3]),
             margin_vx=float(env._config.cmd_lin_vel_x_zero_margin),
             margin_yaw=float(env._config.cmd_ang_vel_zero_margin),
             wheel_sin_scale=float(env._config.wheel_sin_scale))
    print(f"saved {args.out}; shapes: reward {arrs['reward'].shape}")


# ─────────────────── E1/E3: 离线公式重算与四方格 ───────────────────
def _alive_mask(done):
    """done (T,N) -> alive (T,N),终止步本身计入(与 batch_test 一致)。"""
    import numpy as np

    T, N = done.shape
    alive = np.ones((T, N), dtype=bool)
    for n in range(N):
        idx = np.nonzero(done[:, n] > 0)[0]
        if len(idx):
            alive[idx[0] + 1:, n] = False
    return alive


def _sim_anchor(vals, moving):
    """离线锚点模拟:anchor_0=vals[0];moving 步刷新为当前值(锚定本步奖励)。"""
    import numpy as np

    T = vals.shape[0]
    anch = np.empty_like(vals)
    a = vals[0].copy()
    for t in range(T):
        anch[t] = a
        a = np.where(moving[t], vals[t], a)
    return anch


def _hold_terms(qL, qR, sinyaw, cmd, p, formula):
    """离线计算一套公式的 (pos_term, yaw_term) 逐 step 值(未乘 zero_keep_scale)。"""
    import numpy as np

    vx, yaw = cmd[..., 0], cmd[..., 1]
    if formula == "old":
        wsc = p["wheel_sin_scale"]
        wL = np.sin(qL * wsc) / wsc
        wR = np.sin(qR * wsc) / wsc
        moving = (np.abs(vx) > p["margin_vx"]) | (np.abs(yaw) > p["margin_yaw"])
        aL = _sim_anchor(wL, moving)
        aR = _sim_anchor(wR, moving)
        diff = 0.5 * ((wL - aL) + (wR - aR))
        gate = (np.abs(vx) <= p["margin_vx"]) & (np.abs(yaw) <= p["margin_yaw"])
        pos_term = gate * np.exp(-np.abs(diff) / p["sigma_wheel"])
    else:
        s = 0.5 * (qL + qR)
        moving = np.abs(vx) > p["margin_vx"]
        a = _sim_anchor(s, moving)
        gate = np.abs(vx) <= p["margin_vx"]
        pos_term = gate * np.exp(-np.abs(s - a) / p["sigma_wheel"])
    # sinyaw 项两版一致
    moving_y = np.abs(yaw) > p["margin_yaw"]
    ay = _sim_anchor(sinyaw, moving_y)
    gate_y = np.abs(yaw) <= p["margin_yaw"]
    yaw_term = gate_y * np.exp(-np.abs(sinyaw - ay) / (p["sigma"] * p["cmd_scale_yaw"]))
    return pos_term, yaw_term


def run_e1report(args):
    import numpy as np

    old = np.load(args.old)
    new = np.load(args.new)
    p = {k: float(old[k]) for k in
         ("zero_keep_scale", "sigma_wheel", "sigma", "cmd_scale_yaw",
          "margin_vx", "margin_yaw", "wheel_sin_scale")}
    ZK = p["zero_keep_scale"]

    def cells(traj, tag):
        alive = _alive_mask(traj["done"])
        n_envs = alive.shape[1]
        ep_len = alive.sum(axis=0)
        online_total = (traj["reward"] * alive).sum(axis=0).mean()
        online_zk = ZK * ((traj["term_wsin"] + traj["term_sinyaw"]) * alive
                          ).sum(axis=0).mean()
        other = online_total - online_zk  # 公式无关部分(在线实测)
        out = {"tag": tag, "online_total": online_total, "other": other,
               "ep_len": ep_len.mean(), "surv": float((ep_len >= 600).mean())}
        for formula in ("old", "new"):
            pos_t, yaw_t = _hold_terms(traj["qL"], traj["qR"], traj["sinyaw"],
                                       traj["cmd"], p, formula)
            pos_sum = (pos_t * alive).sum(axis=0).mean()
            yaw_sum = (yaw_t * alive).sum(axis=0).mean()
            out[formula] = dict(
                pos_sum=pos_sum, yaw_sum=yaw_sum,
                act_pos=(np.abs(traj["cmd"][..., 0]) <= p["margin_vx"]
                         ).__mul__(alive).sum() / n_envs,
                zk=ZK * (pos_sum + yaw_sum),
                total=other + ZK * (pos_sum + yaw_sum))
            # 同版公式离线 vs 在线校验
            online_pos = (traj["term_wsin"] * alive).sum(axis=0).mean()
            out[f"online_pos_sum_{formula}"] = online_pos
        return out

    co = cells(old, "旧策略 base2-s0@1.15e8")
    cn = cells(new, "新策略 holdrec-step2@5e8")

    print("\n════════ E1 四方格(episode 总 reward,均值) ════════")
    print(f"{'':22s} {'旧公式(per-wheel sin 锚点)':>26s} {'新公式(共模 Δs 锚点)':>24s}")
    for c in (co, cn):
        print(f"{c['tag']:22s} {c['old']['total']:26.1f} {c['new']['total']:24.1f}")
    print("\n── 分解 ──")
    for c in (co, cn):
        print(f"[{c['tag']}] 存活 {c['ep_len']:.0f}/600 (存活率 {c['surv']*100:.0f}%)  "
              f"公式无关部分(在线) {c['other']:.1f}")
        for f in ("old", "new"):
            d = c[f]
            print(f"   {f} 公式: 位置项和 {d['pos_sum']:6.1f} (激活 {d['act_pos']:.0f} 步)  "
                  f"朝向项和 {d['yaw_sum']:6.1f}  zero_keep 小计 {d['zk']:7.1f}  "
                  f"→ 总分 {d['total']:.1f}")
        print(f"   (校验: 在线位置项和 {c['online_pos_sum_old']:.1f})")

    print("\n════════ E3 静止收益核算 ════════")
    T = old["done"].shape[0]
    for name, traj in (("旧策略", old), ("新策略", new)):
        alive = _alive_mask(traj["done"])
        gate_old = ((np.abs(traj["cmd"][..., 0]) <= p["margin_vx"])
                    & (np.abs(traj["cmd"][..., 1]) <= p["margin_yaw"]))
        gate_new = np.abs(traj["cmd"][..., 0]) <= p["margin_vx"]
        print(f"[{name}] 门控激活占比(存活步): 旧(both 死区) "
              f"{(gate_old & alive).sum() / alive.sum() * 100:.1f}%  "
              f"新(仅 vx 死区) {(gate_new & alive).sum() / alive.sum() * 100:.1f}%")
    # 解析值:cmd_prob=0.9,vx 死区占比 0.000195/0.2,yaw 死区 0.01/1.5
    p_zero = 0.1
    p_vx_dz = 0.9 * 0.000195 / 0.2
    p_yaw_dz = 0.9 * 0.01 / 1.5
    print(f"解析: 旧门控≈{(p_zero + p_vx_dz * p_yaw_dz / 0.9) * 100:.1f}%  "
          f"新位置门控≈{(p_zero + p_vx_dz) * 100:.1f}%  "
          f"新朝向门控≈{(p_zero + p_yaw_dz) * 100:.1f}%  (每步满分均 1×10,不变)")


# ─────────────────────────── E2: 机动污染探针 ───────────────────────────
def run_e2(args):
    """机动本身是否污染轮里程计 Δs/Δψ(新代码,规范初始化,noise=0,push=0)。

    真值口径:不用根机身 x/y(机身绕接触点俯仰会产生假平移),而用
    **轮子世界系滚转角**(xmat 局部 X 在世界 XZ 面的投影角)作"真实滚动",
    污染 = 关节角变化 Δq − 世界滚转角变化 Δφ_w(关节转了但轮没真滚)。
    每个探针先验证机动真的执行了(thigh 行程),再看 Δs 漂移。
    """
    import numpy as np
    import jax
    import jax.numpy as jp

    from legwheel import LegWheelRobot, default_config
    from legwheel.evaluation.runner import load_policy

    cfg = default_config()
    cfg.noise_config.level = 0.0
    cfg.push_config.enable = False
    cfg.cmd_config.cmd_prob = 0.0
    env = LegWheelRobot(cfg)
    inference = jax.jit(load_policy(os.path.abspath(args.ckpt), deterministic=True))
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    q_w = [int(env._qpos_wanted_joints[1]), int(env._qpos_wanted_joints[3])]
    w_ids = (env._left_wheel_id, env._right_wheel_id)
    R = float(env._config.wheel_radius)

    def yaw_of(qpos):
        w, x, y, z = (float(v) for v in qpos[3:7])
        return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def wheel_world_roll(data):
        """轮在世界系的滚转角(绕轮轴):局部 X 轴在世界 XZ 面投影角。
        仅在机身无 yaw 的机动中有绝对意义;crouch/split/静止适用。"""
        out = []
        for wid in w_ids:
            W = np.asarray(data.xmat[wid])
            out.append(np.arctan2(-W[2, 0], W[0, 0]))
        return np.unwrap(np.array(out))

    def probe(name, cmd, act_override, steps=600, windows=()):
        rng = jax.random.PRNGKey(0)
        state = jit_reset(rng, False)
        rows = []
        prev_wr = None
        for i in range(steps):
            state = state.replace(
                info={**state.info, "command": jp.array(cmd, dtype=jp.float32)})
            rng, ar = jax.random.split(rng)
            act, _ = inference(state.obs, ar)
            act = np.asarray(act).copy()
            if act_override is not None:
                act = act_override(i, act, np.asarray(state.data.qpos))
            state = jit_step(state, jp.asarray(act))
            qpos = np.asarray(state.data.qpos)
            wr = wheel_world_roll(state.data)
            if prev_wr is not None:
                wr = prev_wr + (wr - prev_wr + np.pi) % (2 * np.pi) - np.pi
            prev_wr = wr
            rows.append(dict(
                t=i * 0.01, qL=float(qpos[q_w[0]]), qR=float(qpos[q_w[1]]),
                wrL=float(wr[0]), wrR=float(wr[1]),
                yaw=yaw_of(qpos),
                sinpitch=float(env._getSinPitch_newest(state.data)),
                thL=float(qpos[env._qpos_thigh_joints[0]]),
                thR=float(qpos[env._qpos_thigh_joints[1]]),
                done=bool(state.done)))
            if rows[-1]["done"]:
                print(f"[{name}] 终止 @ step {i + 1} (t={i * 0.01:.2f}s)")
                break
        # ── 分析 ──
        if len(rows) < 20:
            print(f"[{name}] 数据太短({len(rows)} 步),跳过\n")
            return
        qL = np.array([r["qL"] for r in rows])
        qR = np.array([r["qR"] for r in rows])
        thL = np.array([r["thL"] for r in rows])
        thR = np.array([r["thR"] for r in rows])
        wrL = np.unwrap(np.array([r["wrL"] for r in rows]))
        wrR = np.unwrap(np.array([r["wrR"] for r in rows]))
        yaw = np.array([r["yaw"] for r in rows])
        ds_joint = 0.5 * ((qL - qL[0]) + (qR - qR[0]))       # 关节共模(里程计用的量)
        ds_world = 0.5 * ((wrL - wrL[0]) + (wrR - wrR[0]))   # 真实滚动共模
        pollut = ds_joint - ds_world                          # 关节转了但轮没真滚
        dyaw = (yaw - yaw[0] + np.pi) % (2 * np.pi) - np.pi
        print(f"\n[{name}] {len(rows)} 步 (存活={'是' if not rows[-1]['done'] else '否'})  "
              f"thigh 行程 L[{thL.min():+.3f},{thL.max():+.3f}] "
              f"R[{thR.min():+.3f},{thR.max():+.3f}]  yaw {np.degrees(dyaw[-1]):+.1f}°")
        print(f"  全程: 关节 Δs {ds_joint[-1]:+.3f} rad ({ds_joint[-1]*R*1000:+.1f} mm)  "
              f"真实滚动 Δs_w {ds_world[-1]:+.3f} ({ds_world[-1]*R*1000:+.1f} mm)  "
              f"污染 {pollut[-1]:+.3f} rad ({pollut[-1]*R*1000:+.1f} mm)")
        for lo, hi, tag in windows:
            if lo >= len(rows):
                continue
            hi = min(hi, len(rows))
            seg = slice(lo, hi)
            print(f"  [{tag} step {lo}-{hi}] thigh均值 {(thL[seg].mean() + thR[seg].mean()) / 2:+.3f}  "
                  f"关节Δs {ds_joint[hi - 1] - ds_joint[lo]:+.3f} rad  "
                  f"真滚Δs_w {ds_world[hi - 1] - ds_world[lo]:+.3f}  "
                  f"污染 {(pollut[hi - 1] - pollut[lo]) * R * 1000:+.1f} mm  "
                  f"新式位置项(锚点冻窗口起点) {np.exp(-abs(ds_joint[hi - 1] - ds_joint[lo]) / 0.5):.3f}")

    H = [0.0, 0.0, -0.25, 0.0]
    W = [(0, 100, "静止"), (100, 300, "机动"), (300, 400, "保持"), (400, 600, "回位")]
    # (d) 静止对照:policy + 蹲姿指令(已知存活但冻结,thigh 不动)
    probe("d_static_policy(h=-0.25)", H, None, windows=W)

    def ramp(i, pts):
        """pts: [(step, val), ...] 分段线性。"""
        if i <= pts[0][0]:
            return pts[0][1]
        for (s0, v0), (s1, v1) in zip(pts, pts[1:]):
            if i <= s1:
                return v0 + (v1 - v0) * (i - s0) / (s1 - s0)
        return pts[-1][1]

    def thigh_servo(q_star_L, q_star_R):
        """thigh 通道闭环脚本:按当前关节角算 delta 动作,逼近期望轨迹。"""
        def ov(i, act, qpos):
            th = qpos[env._qpos_thigh_joints]
            act[0] = np.clip((q_star_L(i) - th[0]) / 0.25, -1.0, 1.0)
            act[1] = np.clip((q_star_R(i) - th[1]) / 0.25, -1.0, 1.0)
            return act
        return ov

    # (a) 强制蹲起:thigh 期望轨迹 0→-0.5→保持→0,policy 负责轮(平衡)
    q_crouch = [(0, 0.0), (100, 0.0), (300, -0.5), (400, -0.5), (600, 0.0)]
    probe("a_crouch_policy轮", H,
          thigh_servo(lambda i: ramp(i, q_crouch), lambda i: ramp(i, q_crouch)),
          windows=W)
    # (b) 强制 roll split:L 0→-0.125→0,R 反向,policy 轮
    q_sl = [(0, 0.0), (100, 0.0), (200, -0.125), (300, -0.125), (400, 0.0), (600, 0.0)]
    q_sr = [(0, 0.0), (100, 0.0), (200, 0.125), (300, 0.125), (400, 0.0), (600, 0.0)]
    probe("b_roll_split_policy轮", H,
          thigh_servo(lambda i: ramp(i, q_sl), lambda i: ramp(i, q_sr)),
          windows=W)
    # (c) policy 执行 yaw 指令(理想差速下共模应≈0;位置门控仍激活)
    probe("c2_yaw_cmd1.5_policy", [0.0, 1.5, -0.25, 0.0], None, steps=300,
          windows=[(0, 100, "转向"), (100, 300, "转向")])



def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["e1new", "e1report", "e2"], required=True)
    parser.add_argument("--ckpt", default=CKPT)
    parser.add_argument("--old", default="/tmp/e1_old_traj_rand1.npz")
    parser.add_argument("--new", default="/tmp/e1_new_traj.npz")
    parser.add_argument("--out", default="/tmp/e1_new_traj.npz")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-init", type=int, default=1)
    args = parser.parse_args()

    if args.mode == "e2":
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.pop("MUJOCO_GL", None)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    if args.mode == "e1new":
        run_e1new(args)
    elif args.mode == "e1report":
        run_e1report(args)
    else:
        run_e2(args)


if __name__ == "__main__":
    main()
