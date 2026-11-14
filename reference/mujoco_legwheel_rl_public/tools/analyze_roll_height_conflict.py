#!/usr/bin/env python
"""roll 与 height 指令的冲突量化:共用 thigh 行程的可行域分析。

原理:roll 由左右 thigh 差动实现(Δthigh/2 ≈ k·sinroll),height 跟踪要求
mean(thigh)=cmd_height,而 thigh 行程被 action_range 限制在 [-0.5, 0]。
当 |h + δ| 或 |h − δ| 超出行程(δ = k·|sinroll|)时,两项无法同时满分。

方法:策略在环 rollout(与 measure_eval_dynamics 同配置),记录每步
command、thigh 关节角、sinroll、tracking_height_jnt / roll_cost,
然后:
  1) 回归耦合系数 k = (|Δthigh|/2) / |sinroll|(roll 激活窗口);
  2) 用 k 计算每个指令对 (h, φ) 的最小可达 height 误差(行程裁剪),
     得到 height 项的实际上限 E[exp(-err_min/σ)];
  3) 条件达成率:roll 激活/未激活窗口下 height 项的均值差异(反之亦然)。

用法:
  CUDA_VISIBLE_DEVICES=0 python tools/analyze_roll_height_conflict.py \
      --ckpt /abs/path/to/ckpt --out tuning/run1/reports/roll_height_conflict.png
"""

import argparse
import logging
import os
import sys

import jax
import jax.numpy as jp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

log = logging.getLogger("conflict")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--base-seed", type=int, default=42)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from legwheel.evaluation.runner import load_policy, make_env

    env, cfg = make_env(push=True)
    jit_inference = jax.jit(load_policy(args.ckpt))
    num_envs = args.num_envs
    T = int(cfg.episode_length)
    sigma = float(cfg.reward_config.sigma)
    height_scale_cmd = 0.25  # cmd_scales[2],height 半量程
    thigh_lo, thigh_hi = float(cfg.action_range.thigh[0]), float(cfg.action_range.thigh[1])

    env_rngs = jax.random.split(jax.random.PRNGKey(args.base_seed), num_envs)
    vmap_reset = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))
    vmap_step = jax.jit(jax.vmap(env.step))
    thigh_ids = np.asarray(jax.device_get(env._qpos_thigh_joints))
    vmap_sinroll = jax.jit(jax.vmap(env._getSinRoll_newest))

    @jax.jit
    def vmap_inference(obs, rngs):
        return jax.vmap(jit_inference)(obs, rngs)

    states = vmap_reset(env_rngs, True)
    rec = {k: np.zeros((T, num_envs), dtype=np.float32)
           for k in ("cmd_h", "cmd_r", "thigh_l", "thigh_r", "sinroll",
                     "h_jnt", "roll_cost")}
    dones = np.zeros((T, num_envs), dtype=bool)

    act_rng = jax.random.PRNGKey(0)
    for t in range(T):
        act_rng, sub = jax.random.split(act_rng)
        actions, _ = vmap_inference(states.obs, jax.random.split(sub, num_envs))
        states = vmap_step(states, actions)
        cmd = np.asarray(jax.device_get(states.info["command"]))
        qpos = np.asarray(jax.device_get(states.data.qpos))
        rec["cmd_h"][t] = cmd[:, 2]
        rec["cmd_r"][t] = cmd[:, 3]
        rec["thigh_l"][t] = qpos[:, thigh_ids[0]]
        rec["thigh_r"][t] = qpos[:, thigh_ids[1]]
        rec["sinroll"][t] = np.asarray(jax.device_get(vmap_sinroll(states.data)))
        m = states.metrics
        rec["h_jnt"][t] = np.asarray(
            jax.device_get(m["reward/tracking_height_jnt"]))
        rec["roll_cost"][t] = np.asarray(jax.device_get(m["reward/roll_cost"]))
        dones[t] = np.asarray(jax.device_get(states.done)) > 0

    # 只用首次 done 前的步(策略正常行为)
    has_done = dones.any(axis=0)
    ep_len = np.where(has_done, np.argmax(dones, axis=0) + 1, T)
    mask = np.zeros((T, num_envs), dtype=bool)
    for e in range(num_envs):
        mask[: int(ep_len[e]), e] = True

    cmd_h = rec["cmd_h"][mask]
    cmd_r = rec["cmd_r"][mask]
    delta = np.abs(rec["thigh_l"] - rec["thigh_r"])[mask] / 2  # 差动半幅
    sinroll = rec["sinroll"][mask]
    h_jnt = rec["h_jnt"][mask]
    roll_cost = rec["roll_cost"][mask]

    # 1) 耦合系数 k:roll 激活窗口内 delta / |sinroll| 的中位斜率
    roll_on = np.abs(cmd_r) > 0.02
    k = np.median(delta[roll_on] / np.maximum(np.abs(sinroll[roll_on]), 1e-3))
    log.info("耦合系数 k = Δthigh半幅/|sinroll| ≈ %.3f rad(中位回归,n=%d)",
             k, int(roll_on.sum()))

    # 2) 行程裁剪导致的 height 最小误差与实际上限
    #    需要 mean=h、split=±δ(δ=k|φ|),thigh∈[lo,hi] → h∈[lo+δ, hi−δ]
    d_need = k * np.abs(cmd_r)
    h_lo = thigh_lo + d_need
    h_hi = thigh_hi - d_need
    err_min = np.maximum(h_lo - cmd_h, 0) + np.maximum(cmd_h - h_hi, 0)
    sig_h = sigma * height_scale_cmd
    ceiling = np.exp(-err_min / sig_h)
    infeasible = err_min > 1e-6
    log.info("指令不可同时满足(行程裁剪)的步占比: %.1f%%",
             100.0 * infeasible.mean())
    log.info("height_jnt 几何上限: E[exp(-err_min/σ)] = %.3f(满分 1) → episode 上限 ≈ %.0f",
             ceiling.mean(), 600 * ceiling.mean())

    # 3) 条件达成率
    log.info("height_jnt 条件均值: roll激活=%.3f  roll≈0=%.3f",
             h_jnt[roll_on].mean(), h_jnt[~roll_on].mean())
    h_extreme = (cmd_h < thigh_lo + 0.1) | (cmd_h > thigh_hi - 0.1)
    log.info("roll_cost 条件均值: height极端=%.3f  height中段=%.3f",
             roll_cost[h_extreme].mean(), roll_cost[~h_extreme].mean())

    # ---- 绘图 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ax = axes[0]
    idx = np.random.default_rng(0).choice(
        int(roll_on.sum()), size=min(20000, int(roll_on.sum())), replace=False)
    ax.scatter(np.abs(sinroll[roll_on])[idx], delta[roll_on][idx],
               s=1, alpha=0.2)
    xs = np.linspace(0, 0.25, 50)
    ax.plot(xs, k * xs, color="r", label=f"fit k={k:.3f}")
    ax.set_xlabel("|sinroll|"); ax.set_ylabel("|thigh split| / 2 (rad)")
    ax.set_title("roll-thigh coupling"); ax.legend(); ax.grid(alpha=0.3)
    ax = axes[1]
    ax.hist(err_min[roll_on], bins=50, color="tab:blue", edgecolor="k",
            alpha=0.7)
    ax.set_xlabel("min height error from range clip (rad)")
    ax.set_title(f"infeasible steps: {100.0*infeasible.mean():.1f}%")
    ax.grid(alpha=0.3)
    ax = axes[2]
    labels = ["h_jnt|roll_on", "h_jnt|roll_off", "roll|h_extreme", "roll|h_mid"]
    vals = [h_jnt[roll_on].mean(), h_jnt[~roll_on].mean(),
            roll_cost[h_extreme].mean(), roll_cost[~h_extreme].mean()]
    ax.bar(labels, vals, color=["tab:red", "tab:green", "tab:red", "tab:green"])
    ax.axhline(ceiling.mean(), ls="--", color="k",
               label=f"geometric ceiling {ceiling.mean():.3f}")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=20)
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    log.info("图已保存: %s", args.out)


if __name__ == "__main__":
    main()
