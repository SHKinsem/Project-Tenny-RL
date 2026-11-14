#!/usr/bin/env python
"""baseline 60s 演示的 action rate 采集与分析(不重渲染,只采数据+出图)。

复刻 record_baseline_demo.py 的 rollout(rng 用法逐行一致),因此对同 seed
重现完全相同的轨迹/推力/摔倒(脚本会校验与 falls_sNN_1080p.txt 一致)。

action rate 口径与 legwheel/evaluation/action_rate.py 一致:
  rate/步 = mean(|Δa|) + mean(|Δ²a|)  (对策略原始输出的 4 维动作取均值)
参考值:run2 deploy 评测 baseline = 0.371/步,run2 硬门槛 0.35。

新语义环境(hold-position-heading)下同时采集停车保持误差(共模 Δs / Δψ),
输出"站立段/停车后/推力后恢复"统计(见 main 的 hold precision 段)。

输出:
- <out>/action_rate/action_rate_sNN.npz (逐步原始数据)
- docs/images/<tag>_action_rate_{timeline,segments,dims,events}.png
- docs/images/<tag>_hold_timeline.png (新语义:Δs/Δψ 时间线)
- 终端打印汇总统计

用法:
  .venv-local/bin/python tools/analyze_demo_action_rate.py [seed ...]
  .venv/bin/python tools/analyze_demo_action_rate.py \
      --ckpt ckpts/newobs-norecovery/000403660800 \
      --out test/hold_position_heading_demo_60s --tag holdpos_demo 1 2 3
"""

import argparse
import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.1")

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(TOOLS_DIR))  # repo root
sys.path.insert(0, TOOLS_DIR)

import jax
import jax.numpy as jp
import numpy as np

from record_baseline_demo import SCHEDULE, CKPT, OUT_DIR  # noqa: E402

DIM_NAMES = ["thigh_L", "thigh_R", "wheel_L", "wheel_R"]
FPS_DT = 0.01
SEG_S = 5.0
DURATION_S = 60.0
# run2 评测参考(docs/deploy_unweighted_compare.md / tuning/metrics.py)
AR_BASELINE_RUN2 = 0.371
AR_GATE_RUN2 = 0.35
RESET_GUARD_STEPS = 50  # reset 后 0.5s 内的差分跨 episode 边界,统计时剔除
# 运行期可配(main 的 --ckpt/--out/--tag 覆盖)
CKPT_USE = CKPT
OUT_USE = OUT_DIR
TAG_USE = "baseline_demo"
WHEEL_RADIUS = 0.0195


def collect_one(seed: int) -> dict:
    """复刻 record_baseline_demo 的 rollout,返回逐步动作/推力/摔倒/停车保持误差。"""
    from legwheel import LegWheelRobot, default_config
    from legwheel.evaluation.runner import load_policy

    cfg = default_config()
    cfg.cmd_config.command_change_interval = 10 ** 6
    cfg.cmd_start_delay_s = 200.0
    env = LegWheelRobot(cfg)
    dt = env.dt
    assert abs(dt - FPS_DT) < 1e-9, f"dt={dt}"
    n_steps = int(DURATION_S / dt)
    seg_steps = int(SEG_S / dt)

    inference = jax.jit(load_policy(os.path.abspath(CKPT_USE), deterministic=True))
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    @jax.jit
    def hold_telem(data, info):
        """停车保持误差:共模 Δs(轮弧度)与 Δψ(rad,由 sinyaw 锚点反解)。"""
        qw, qx, qy, qz = data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6]
        yaw = jp.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy ** 2 + qz ** 2))
        yaw0 = 2 * jp.arcsin(jp.clip(0.5 * info["sinyaw_start"], -1.0, 1.0))
        dpsi = jp.arctan2(jp.sin(yaw - yaw0), jp.cos(yaw - yaw0))
        ds = env.get_hold_position(data) - info["position_start"]
        return ds, dpsi

    rng = jax.random.PRNGKey(seed)
    state = jit_reset(rng, True)
    acts = np.zeros((n_steps, env.action_size), dtype=np.float32)
    dones = np.zeros(n_steps, dtype=bool)
    push_mag = np.zeros(n_steps, dtype=np.float32)
    push_deg = np.zeros(n_steps, dtype=np.float32)
    is_push = np.zeros(n_steps, dtype=bool)
    hold_ds = np.zeros(n_steps, dtype=np.float32)    # 共模 Δs (rad)
    hold_dpsi = np.zeros(n_steps, dtype=np.float32)  # Δψ (rad)
    falls = []
    for step in range(n_steps):
        seg = step // seg_steps
        _, cmd = SCHEDULE[seg]
        if step % seg_steps == 0:
            state = state.replace(
                info={**state.info, "command": jp.array(cmd, dtype=jp.float32)})
        act_rng, rng = jax.random.split(rng)
        act, _ = inference(state.obs, act_rng)
        state = jit_step(state, act)
        acts[step] = np.asarray(jax.device_get(act))
        ds, dpsi = jax.device_get(hold_telem(state.data, state.info))
        hold_ds[step] = float(ds)
        hold_dpsi[step] = float(dpsi)
        push_vec = np.asarray(jax.device_get(state.info["push"]))
        if np.linalg.norm(push_vec) > 0:
            is_push[step] = True
            push_mag[step] = float(np.linalg.norm(push_vec))
            push_deg[step] = float(np.degrees(np.arctan2(push_vec[1], push_vec[0])))
        if bool(jax.device_get(state.done)):
            falls.append(step)
            dones[step] = True
            rng, reset_rng = jax.random.split(rng)
            state = jit_reset(reset_rng, True)
            state = state.replace(
                info={**state.info, "command": jp.array(cmd, dtype=jp.float32)})
        if step % 1000 == 0:
            print(f"[ar] seed={seed} step {step}/{n_steps}", flush=True)
    return dict(seed=seed, acts=acts, dones=dones, falls=np.array(falls),
                is_push=is_push, push_mag=push_mag, push_deg=push_deg,
                hold_ds=hold_ds, hold_dpsi=hold_dpsi)


def verify_replay(res: dict) -> str:
    """校验复刻轨迹的摔倒/推力与录像批 falls_sNN_1080p.txt 一致。"""
    ref = os.path.join(OUT_USE, "1080p_viz", f"falls_s{res['seed']:02d}_1080p.txt")
    if not os.path.exists(ref):
        return "no-ref"
    txt = open(ref).read()
    ref_falls = [float(l.split("t=")[1].split("s")[0])
                 for l in txt.splitlines() if l.strip().startswith("t=")
                 and "m/s" not in l]
    ref_pushes = [l for l in txt.splitlines() if "m/s" in l]
    ok_f = len(ref_falls) == len(res["falls"])
    ok_p = len(ref_pushes) == int(res["is_push"].sum())
    return "OK" if (ok_f and ok_p) else f"MISMATCH(falls {len(res['falls'])} vs {len(ref_falls)}, pushes {int(res['is_push'].sum())} vs {len(ref_pushes)})"


def step_metrics(acts: np.ndarray, dones: np.ndarray):
    """逐步 c1=mean|Δa|, c2=mean|Δ²a|;跨 reset 边界的差分置 NaN。"""
    c1 = np.abs(acts[1:] - acts[:-1]).mean(axis=1)
    d2 = acts[2:] - 2 * acts[1:-1] + acts[:-2]
    c2 = np.abs(d2).mean(axis=1)
    t1 = np.arange(1, len(acts))
    t2 = np.arange(2, len(acts))
    # reset 后 GUARD 步内剔除(c1/c2 跨 episode 或含站起瞬态)
    guard = np.zeros(len(acts), dtype=bool)
    for f in np.where(dones)[0]:
        guard[f:f + RESET_GUARD_STEPS] = True
    c1[guard[t1]] = np.nan
    c2[guard[t2]] = np.nan
    rate = np.full(len(acts), np.nan)
    rate[2:] = c1[1:] + c2  # c1[t-1] 与 c2[t-2] 对齐到步 t
    dim_d1 = np.abs(acts[1:] - acts[:-1])  # (T-1, 4) 逐维一阶差分
    return c1, c2, rate, dim_d1, guard


def hold_precision_stats(all_res):
    """停车保持精度统计(新语义):站立段/停车后/推力后恢复。

    返回打印用字典;口径:|Δs| 换算 mm(×wheel_radius),|Δψ| 换算 deg。
    reset 后 GUARD 步剔除(锚点刚重置,含站起瞬态)。
    """
    seg_steps = int(SEG_S / FPS_DT)
    n_steps = int(DURATION_S / FPS_DT)
    out = {}
    for res in all_res:
        guard = np.zeros(n_steps, dtype=bool)
        for f in np.where(res["dones"])[0]:
            guard[f:f + RESET_GUARD_STEPS] = True
        res["_guard"] = guard
        res["_ds_mm"] = np.abs(res["hold_ds"]) * WHEEL_RADIUS * 1000.0
        res["_dpsi_deg"] = np.degrees(np.abs(res["hold_dpsi"]))

    def agg(mask_fn):
        ds_all, dp_all = [], []
        for res in all_res:
            m = mask_fn(res) & ~res["_guard"]
            ds_all.append(res["_ds_mm"][m])
            dp_all.append(res["_dpsi_deg"][m])
        ds = np.concatenate(ds_all)
        dp = np.concatenate(dp_all)
        return (float(ds.mean()), float(np.percentile(ds, 95)),
                float(dp.mean()), float(np.percentile(dp, 95)))

    # 站立段: 0-5s 与 55-60s
    stand_mask = lambda res: (np.arange(n_steps) < seg_steps) | \
                             (np.arange(n_steps) >= 11 * seg_steps)
    out["stand"] = agg(stand_mask)
    # 停车后: combo(50-55s)->站立(55-60s) 过渡后 2s
    post_mask = lambda res: (np.arange(n_steps) >= 11 * seg_steps) & \
                            (np.arange(n_steps) < 11 * seg_steps + 200)
    out["post_stop"] = agg(post_mask)
    # 各零速度指令段(锚点冻结段): stand / crouch / stand tall / roll left / roll right
    out["by_seg"] = {}
    for si, (label, cmd) in enumerate(SCHEDULE):
        if abs(cmd[0]) > 1e-6 or abs(cmd[1]) > 1e-6:
            continue
        seg_mask = lambda res, si=si: (np.arange(n_steps) >= si * seg_steps) & \
                                      (np.arange(n_steps) < (si + 1) * seg_steps)
        out["by_seg"][label] = agg(seg_mask)
    # 推力后恢复: 仅统计锚点冻结(零 vx/yaw 指令)时段内的推力
    cmd_arr = np.array([cmd for _, cmd in SCHEDULE])  # (12, 4)
    seg_of_step = np.minimum(np.arange(n_steps) // seg_steps, len(SCHEDULE) - 1)
    gate = (np.abs(cmd_arr[seg_of_step, 0]) <= 1e-6) & \
           (np.abs(cmd_arr[seg_of_step, 1]) <= 1e-6)
    peaks_ds, resids_ds, peaks_dp, resids_dp = [], [], [], []
    for res in all_res:
        for p in np.where(res["is_push"] & gate)[0]:
            if p + 200 >= n_steps or res["_guard"][p]:
                continue
            if res["dones"][p:p + 200].any():
                continue  # 推倒了的不计入恢复统计(另见摔倒分析)
            win_ds = res["_ds_mm"][p:p + 200]
            win_dp = res["_dpsi_deg"][p:p + 200]
            peaks_ds.append(float(win_ds.max()))
            resids_ds.append(float(win_ds[-1]))
            peaks_dp.append(float(win_dp.max()))
            resids_dp.append(float(win_dp[-1]))
    out["push_recovery"] = dict(
        n=len(peaks_ds),
        ds_peak=float(np.mean(peaks_ds)) if peaks_ds else float("nan"),
        ds_resid=float(np.mean(resids_ds)) if resids_ds else float("nan"),
        dp_peak=float(np.mean(peaks_dp)) if peaks_dp else float("nan"),
        dp_resid=float(np.mean(resids_dp)) if resids_dp else float("nan"),
    )
    return out


def print_hold_precision(hp):
    print("\n[hold] 停车保持精度(|Δs| mm / |Δψ| deg, 均值 / p95):")
    for name, key in (("站立段(0-5s,55-60s)", "stand"),
                      ("停车后 2s(combo→stand)", "post_stop")):
        ds_m, ds_p, dp_m, dp_p = hp[key]
        print(f"  {name:26s} Δs {ds_m:6.1f} / {ds_p:6.1f} mm   "
              f"Δψ {dp_m:5.2f} / {dp_p:5.2f} deg")
    print("  零速度指令段分解:")
    for label, (ds_m, ds_p, dp_m, dp_p) in hp["by_seg"].items():
        print(f"    {label:16s} Δs {ds_m:6.1f} / {ds_p:6.1f} mm   "
              f"Δψ {dp_m:5.2f} / {dp_p:5.2f} deg")
    pr = hp["push_recovery"]
    print(f"  推力后恢复(零指令窗口内推力, n={pr['n']}): "
          f"Δs 峰值 {pr['ds_peak']:.1f} mm → 2s 后 {pr['ds_resid']:.1f} mm;  "
          f"Δψ 峰值 {pr['dp_peak']:.2f} deg → 2s 后 {pr['dp_resid']:.2f} deg")


def plot_hold_timeline(all_res, img_dir):
    """Δs(mm)与 Δψ(deg)的 60s 时间线(种子均值)+ 推力/摔倒标注。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_steps = int(DURATION_S / FPS_DT)
    ts = np.arange(n_steps) * FPS_DT
    ds = np.array([np.where(r["_guard"], np.nan, r["_ds_mm"]) for r in all_res])
    dp = np.array([np.where(r["_guard"], np.nan, r["_dpsi_deg"]) for r in all_res])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    for ax, arr, unit, color in ((ax1, ds, "mm", "tab:blue"),
                                 (ax2, dp, "deg", "tab:green")):
        mu = np.nanmean(arr, axis=0)
        sd = np.nanstd(arr, axis=0)
        ax.plot(ts, mu, color=color, lw=1.2)
        ax.fill_between(ts, np.maximum(mu - sd, 0), mu + sd, color=color, alpha=0.2)
        for si, (label, _) in enumerate(SCHEDULE):
            ax.axvline(si * SEG_S, color="gray", ls=":", lw=0.8)
        for res in all_res:
            ax.plot(res["falls"] * FPS_DT,
                    np.full(len(res["falls"]), np.nanmax(mu) * 1.05),
                    "x", color="red", ms=8, mew=2)
            ax.plot(np.where(res["is_push"])[0] * FPS_DT,
                    np.zeros(int(res["is_push"].sum())) - np.nanmax(mu) * 0.05,
                    "|", color="orange", ms=6)
        ax.set_ylabel(f"|hold error| ({unit})")
        ax.set_ylim(bottom=-np.nanmax(mu) * 0.08)
    ax1.set_title("Hold precision: |Δs| position error (mean over seeds 1-11 "
                  "after stand-up guard)")
    ax2.set_title("|Δψ| heading error")
    ax2.set_xlabel("time (s)")
    for si, (label, _) in enumerate(SCHEDULE):
        ax1.text(si * SEG_S + 0.15, np.nanmax(ds) * 0.92, label, fontsize=7,
                 color="dimgray")
    fig.tight_layout()
    fig.savefig(os.path.join(img_dir, f"{TAG_USE}_hold_timeline.png"), dpi=150)
    plt.close(fig)


def main():
    global CKPT_USE, OUT_USE, TAG_USE
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=str, default=CKPT)
    ap.add_argument("--out", type=str, default=OUT_DIR)
    ap.add_argument("--tag", type=str, default="baseline_demo",
                    help="图片文件名前缀(docs/images/<tag>_*.png)")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("seeds", type=int, nargs="*")
    args = ap.parse_args()
    CKPT_USE, OUT_USE, TAG_USE = args.ckpt, args.out, args.tag
    ar_data_dir = os.path.join(OUT_USE, "action_rate")
    do_plot = not args.no_plot
    seeds = args.seeds or list(range(1, 12))
    os.makedirs(ar_data_dir, exist_ok=True)
    all_res = []
    for seed in seeds:
        npz = os.path.join(ar_data_dir, f"action_rate_s{seed:02d}.npz")
        if os.path.exists(npz):
            res = dict(np.load(npz))
            res["seed"] = seed
            print(f"[ar] seed={seed} loaded from {npz}", flush=True)
        else:
            res = collect_one(seed)
            msg = verify_replay(res)
            print(f"[ar] seed={seed} replay-vs-video: {msg} "
                  f"(falls={len(res['falls'])}, pushes={int(res['is_push'].sum())})",
                  flush=True)
            np.savez_compressed(npz, **res)
        all_res.append(res)
    if not do_plot:
        return

    # ---- 停车保持精度(新语义) ----
    hp = hold_precision_stats(all_res)
    print_hold_precision(hp)

    # ---- 汇总统计 ----
    T = len(all_res[0]["acts"])
    ts = np.arange(T) * FPS_DT
    rates, c1s, c2s, dim_means = [], [], [], []
    seg_rates = {label: [] for label, _ in SCHEDULE}
    seg_of_step = lambda s: SCHEDULE[min(s // int(SEG_S / FPS_DT), len(SCHEDULE) - 1)][0]
    for res in all_res:
        c1, c2, rate, dim_d1, guard = step_metrics(res["acts"], res["dones"])
        rates.append(rate)
        c1s.append(np.nanmean(c1))
        c2s.append(np.nanmean(c2))
        dim_means.append(np.nanmean(dim_d1[~guard[1:]], axis=0))
        for si, (label, _) in enumerate(SCHEDULE):
            lo, hi = si * int(SEG_S / FPS_DT), (si + 1) * int(SEG_S / FPS_DT)
            seg_rates[label].append(np.nanmean(rate[lo:hi]))
    rates = np.array(rates)
    mean_rate = np.nanmean(rates)
    print(f"[ar] overall rate={mean_rate:.3f}/步 "
          f"(c1={np.mean(c1s):.3f} c2={np.mean(c2s):.3f}), "
          f"run2 baseline 参考 {AR_BASELINE_RUN2}", flush=True)
    print(f"[ar] per-dim mean|Δa|: "
          + " ".join(f"{n}={v:.4f}" for n, v in
                     zip(DIM_NAMES, np.mean(dim_means, axis=0))), flush=True)
    # 事件窗口:push 后 0.5s / fall 前 1s 的 rate 提升
    win_push, win_fall = [], []
    for res, rate in zip(all_res, rates):
        for p in np.where(res["is_push"])[0]:
            win_push.append(np.nanmean(rate[p:p + 50]))
        for f in res["falls"]:
            win_fall.append(np.nanmean(rate[max(0, f - 100):f]))
    print(f"[ar] push 后 0.5s 均值={np.nanmean(win_push):.3f}, "
          f"fall 前 1s 均值={np.nanmean(win_fall):.3f}", flush=True)

    # ---- 出图 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img_dir = "docs/images"
    os.makedirs(img_dir, exist_ok=True)
    smooth = lambda x, w=20: np.convolve(np.nan_to_num(x), np.ones(w) / w, "same")

    # 图1: 60s 时间线(均值±std,命令分段/推力/摔倒标注)
    fig, ax = plt.subplots(figsize=(14, 5))
    mu = np.nanmean(rates, axis=0)
    sd = np.nanstd(rates, axis=0)
    ax.plot(ts, smooth(mu), color="tab:blue", lw=1.5, label="mean over seeds (0.2s smooth)")
    ax.fill_between(ts, smooth(mu - sd), smooth(mu + sd), color="tab:blue", alpha=0.2,
                    label="±1 std")
    for si, (label, _) in enumerate(SCHEDULE):
        ax.axvline(si * SEG_S, color="gray", ls=":", lw=0.8)
        ax.text(si * SEG_S + 0.15, ax.get_ylim()[1] * 0.02 + np.nanmax(mu + sd) * 1.02,
                label, rotation=0, fontsize=7, color="dimgray")
    for res in all_res:
        ax.plot(res["falls"] * FPS_DT, np.full(len(res["falls"]), np.nanmax(mu) * 1.12),
                "x", color="red", ms=8, mew=2)
        ax.plot(np.where(res["is_push"])[0] * FPS_DT,
                np.zeros(int(res["is_push"].sum())) - np.nanmax(mu) * 0.06,
                "|", color="orange", ms=6)
    ax.axhline(AR_BASELINE_RUN2, color="green", ls="--", lw=1,
               label=f"run2 deploy baseline = {AR_BASELINE_RUN2}")
    ax.plot([], [], "x", color="red", ms=8, mew=2, label="fall")
    ax.plot([], [], "|", color="orange", ms=6, label="push")
    ax.set_xlim(0, DURATION_S)
    ax.set_ylim(bottom=-np.nanmax(mu) * 0.09)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("action rate / step")
    ax.set_title("Action rate over 60s demo (seeds 1-11, deterministic + noise + push)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{img_dir}/{TAG_USE}_action_rate_timeline.png", dpi=150)
    plt.close(fig)

    # 图2: 按命令分段的均值(12 段去重为 10 类,stand 合并)
    labels = [l for l, _ in SCHEDULE]
    uniq = []
    for l in labels:
        if l not in uniq:
            uniq.append(l)
    # stand 出现在首/尾两段,合并统计
    vals = []
    for l in uniq:
        idxs = [i for i, ll in enumerate(labels) if ll == l]
        per_seed = []
        for res, rate in zip(all_res, rates):
            seg_vals = [rate[i * 500:(i + 1) * 500] for i in idxs]
            per_seed.append(np.nanmean(np.concatenate(seg_vals)))
        vals.append((np.mean(per_seed), np.std(per_seed)))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    xs = np.arange(len(uniq))
    ax.bar(xs, [v[0] for v in vals], yerr=[v[1] for v in vals], capsize=3,
           color="tab:blue", alpha=0.8)
    ax.axhline(AR_BASELINE_RUN2, color="green", ls="--", lw=1,
               label=f"run2 deploy baseline = {AR_BASELINE_RUN2}")
    ax.axhline(mean_rate, color="tab:red", ls="-.", lw=1,
               label=f"demo overall = {mean_rate:.3f}")
    ax.set_xticks(xs)
    ax.set_xticklabels(uniq, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("action rate / step")
    ax.set_title("Action rate by command segment (mean ± std over seeds)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{img_dir}/{TAG_USE}_action_rate_segments.png", dpi=150)
    plt.close(fig)

    # 图3: 逐维贡献(总体 + 按段堆叠 thigh/wheel)
    dim_mu = np.mean(dim_means, axis=0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.bar(DIM_NAMES, dim_mu, color=["tab:blue"] * 2 + ["tab:orange"] * 2)
    ax1.set_ylabel("mean |Δa| / step")
    ax1.set_title("Per-dimension first-difference contribution")
    th = np.zeros(len(uniq))
    wh = np.zeros(len(uniq))
    for i, l in enumerate(uniq):
        idxs = [k for k, ll in enumerate(labels) if ll == l]
        for res in all_res:
            _, _, _, dim_d1, guard = step_metrics(res["acts"], res["dones"])
            mask = np.zeros(len(res["acts"]), dtype=bool)
            for k in idxs:
                mask[k * 500:(k + 1) * 500] = True
            d = dim_d1[mask[1:] & ~guard[1:]]
            th[i] += np.nanmean(d[:, :2].sum(axis=1))
            wh[i] += np.nanmean(d[:, 2:].sum(axis=1))
    th /= len(all_res)
    wh /= len(all_res)
    ax2.bar(xs, th, label="thigh L+R", color="tab:blue")
    ax2.bar(xs, wh, bottom=th, label="wheel L+R", color="tab:orange")
    ax2.set_xticks(xs)
    ax2.set_xticklabels(uniq, rotation=25, ha="right", fontsize=8)
    ax2.set_ylabel("Σ per-group mean |Δa| / step")
    ax2.set_title("Thigh vs wheel contribution by segment")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{img_dir}/{TAG_USE}_action_rate_dims.png", dpi=150)
    plt.close(fig)

    # 图4: 事件对齐曲线(push 起始 / fall 前)
    pre, post = 50, 100
    push_curves, fall_curves = [], []
    for res, rate in zip(all_res, rates):
        for p in np.where(res["is_push"])[0]:
            if p - pre >= 0 and p + post < T:
                push_curves.append(rate[p - pre:p + post])
        for f in res["falls"]:
            if f - 200 >= 0:
                fall_curves.append(rate[f - 200:f + 10])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    t_ev = (np.arange(pre + post) - pre) * FPS_DT
    pc = np.array(push_curves)
    ax1.plot(t_ev, np.nanmean(pc, axis=0), color="tab:orange")
    ax1.fill_between(t_ev, np.nanpercentile(pc, 25, axis=0),
                     np.nanpercentile(pc, 75, axis=0), color="tab:orange", alpha=0.25)
    ax1.axvline(0, color="red", ls="--", lw=1)
    ax1.set_title(f"Push-aligned action rate (n={len(pc)})")
    ax1.set_xlabel("time from push onset (s)")
    ax1.set_ylabel("action rate / step")
    t_f = (np.arange(210) - 200) * FPS_DT
    fc = np.array(fall_curves)
    ax2.plot(t_f, np.nanmean(fc, axis=0), color="tab:red")
    ax2.fill_between(t_f, np.nanpercentile(fc, 25, axis=0),
                     np.nanpercentile(fc, 75, axis=0), color="tab:red", alpha=0.25)
    ax2.axvline(0, color="k", ls="--", lw=1)
    ax2.set_title(f"Fall-aligned action rate (n={len(fc)})")
    ax2.set_xlabel("time from fall (s)")
    fig.tight_layout()
    fig.savefig(f"{img_dir}/{TAG_USE}_action_rate_events.png", dpi=150)
    plt.close(fig)
    plot_hold_timeline(all_res, img_dir)
    print(f"[ar] figures -> {img_dir}/{TAG_USE}_action_rate_*.png, "
          f"{TAG_USE}_hold_timeline.png", flush=True)


if __name__ == "__main__":
    main()
