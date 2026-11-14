#!/usr/bin/env python
"""带策略的真实批量 rollout,采集失败动力学与门控项激活数据。

背景:batch_context 离线重放跨进程不是 bit-identical(实测仅 28/212 失败步数
复现),凡基于离线重放的轨迹统计均不可信。本脚本把策略放回回路重新跑
(与 batch_test 同配置:4096 envs、推力开、base_seed 42),逐步记录真实序列:
sinpitch、推力、done、command、门控分项每步值。

输出:
- 失败动力学:对齐失败时刻的 sinpitch 轨迹、发散时间分布、失败前 1s 推力占比;
- 门控项:sinyaw/sinposition 的每 episode 激活步数与激活步均值(真实达成率)。

用法:
  CUDA_VISIBLE_DEVICES=0 python tools/measure_eval_dynamics.py \
      --ckpt tuning/run1/trials/confirm-baseline-s0/ckpt/000408576000 \
      --out tuning/run1/reports/failure_dynamics_v2.png
"""

import argparse
import logging
import os
import sys

import jax
import jax.numpy as jp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

log = logging.getLogger("evaldyn")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--base-seed", type=int, default=42)
    ap.add_argument("--window", type=int, default=300)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from legwheel.evaluation.runner import load_policy, make_env

    env, cfg = make_env(push=True)
    inference_fn = load_policy(args.ckpt)
    jit_inference = jax.jit(inference_fn)
    episode_length = int(cfg.episode_length)
    num_envs = args.num_envs
    ang_margin = float(cfg.cmd_ang_vel_zero_margin)
    lin_margin = float(cfg.cmd_lin_vel_x_zero_margin)

    key = jax.random.PRNGKey(args.base_seed)
    env_rngs = jax.random.split(key, num_envs)
    vmap_reset = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))
    vmap_step = jax.jit(jax.vmap(env.step))
    vmap_sinpitch = jax.jit(jax.vmap(env._getSinPitch_newest))

    @jax.jit
    def vmap_inference(obs, rngs):
        return jax.vmap(jit_inference)(obs, rngs)

    states = vmap_reset(env_rngs, True)
    T = episode_length
    sinpitch = np.zeros((T, num_envs), dtype=np.float32)
    push_norm = np.zeros((T, num_envs), dtype=np.float32)
    dones = np.zeros((T, num_envs), dtype=bool)
    act_yaw = np.zeros((T, num_envs), dtype=bool)
    act_both = np.zeros((T, num_envs), dtype=bool)
    sinyaw_val = np.zeros((T, num_envs), dtype=np.float32)
    sinpos_val = np.zeros((T, num_envs), dtype=np.float32)

    act_rng = jax.random.PRNGKey(0)
    for t in range(T):
        if t % 100 == 0:
            log.info("step %d/%d, 已失败 %d", t, T, int(dones.any(axis=0).sum()))
        act_rng, sub = jax.random.split(act_rng)
        actions, _ = vmap_inference(states.obs, jax.random.split(sub, num_envs))
        states = vmap_step(states, actions)
        sinpitch[t] = np.asarray(jax.device_get(vmap_sinpitch(states.data)))
        push_norm[t] = np.asarray(jax.device_get(
            jp.linalg.norm(states.info["push"], axis=-1)))
        dones[t] = np.asarray(jax.device_get(states.done)) > 0
        cmd = np.asarray(jax.device_get(states.info["command"]))
        act_yaw[t] = np.abs(cmd[:, 1]) <= ang_margin
        act_both[t] = act_yaw[t] & (np.abs(cmd[:, 0]) <= lin_margin)
        m = states.metrics
        sinyaw_val[t] = np.asarray(jax.device_get(m["reward/tracking_sinyaw"]))
        sinpos_val[t] = np.asarray(
            jax.device_get(m["reward/tracking_wheels_sinposition"]))

    # ---- 每个 env 首次 done 即失败步(与 batch_test 口径一致)----
    has_done = dones.any(axis=0)
    fail_step = np.where(has_done, np.argmax(dones, axis=0) + 1, T)
    n_fail = int(has_done.sum())
    log.info("失败 %d/%d (%.2f%%),失败步中位数 %d",
             n_fail, num_envs, 100.0 * n_fail / num_envs,
             int(np.median(fail_step[has_done])) if n_fail else -1)

    # ---- 失败动力学(真实轨迹)----
    W = args.window
    aligned, div_times, push_near = [], [], 0
    for e in np.nonzero(has_done)[0]:
        end = int(fail_step[e])
        seg = sinpitch[max(0, end - W):end, e]
        aligned.append(np.concatenate(
            [np.full(W - len(seg), np.nan, dtype=np.float32), seg]))
        above = np.nonzero(seg > -0.3)[0]
        div_times.append(len(seg) - 1 - above[-1] if len(above) else len(seg))
        if push_norm[max(0, end - 101):end, e].max() > 1e-6:
            push_near += 1
    aligned = np.array(aligned)
    div_times = np.array(div_times)
    if n_fail:
        n_fast = int((div_times < 100).sum())
        log.info("发散时间: median=%d 步, <1s 占比 %.1f%%",
                 int(np.median(div_times)), 100.0 * n_fast / n_fail)
        log.info("失败前 1s 内有推力: %.1f%% (%d/%d)",
                 100.0 * push_near / n_fail, push_near, n_fail)

    # ---- 门控项激活(每个 env 取首次 done 前)----
    ay = np.array([act_yaw[:int(fail_step[e]), e].sum() for e in range(num_envs)])
    ab = np.array([act_both[:int(fail_step[e]), e].sum() for e in range(num_envs)])
    sv = np.array([sinyaw_val[:int(fail_step[e]), e].sum() for e in range(num_envs)])
    pv = np.array([sinpos_val[:int(fail_step[e]), e].sum() for e in range(num_envs)])
    log.info("episode 长度 mean=%.1f", fail_step.mean())
    log.info("[sinyaw] 激活步数 mean=%.1f,激活步均值 %.3f;分项和 mean=%.2f",
             ay.mean(), sv.sum() / max(ay.sum(), 1), sv.mean())
    log.info("[sinpos] 激活步数 mean=%.1f,激活步均值 %.3f;分项和 mean=%.2f",
             ab.mean(), pv.sum() / max(ab.sum(), 1), pv.mean())
    log.info("解析上限: 600 × %.4f = %.1f", 0.1 + 0.9 * ang_margin / 1.5,
             600 * (0.1 + 0.9 * ang_margin / 1.5))

    # ---- 绘图 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t_ax = (np.arange(W) - W) * 0.01
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ax = axes[0]
    if n_fail:
        ax.plot(t_ax, np.nanmedian(aligned, axis=0), color="tab:red",
                label="median")
        ax.fill_between(t_ax, np.nanpercentile(aligned, 25, axis=0),
                        np.nanpercentile(aligned, 75, axis=0),
                        color="tab:red", alpha=0.25, label="25-75%")
    ax.axhline(-0.6, ls="--", color="k", lw=1)
    ax.axhline(-0.3, ls=":", color="gray", lw=1)
    ax.set_xlabel("time to failure (s)"); ax.set_ylabel("sinpitch")
    ax.set_title(f"aligned trajectories (n={n_fail}, REAL rollout)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax = axes[1]
    if n_fail:
        ax.hist(div_times * 0.01, bins=30, color="tab:blue", edgecolor="k",
                alpha=0.7)
    ax.axvline(1.0, ls="--", color="r", label="1s (horizon ~0.72s)")
    ax.set_xlabel("divergence time: -0.3 crossing -> failure (s)")
    ax.set_title(f"fast(<1s): {100.0*n_fast/n_fail:.0f}%" if n_fail else "")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax = axes[2]
    if n_fail:
        ax.hist(fail_step[has_done] * 0.01, bins=30, color="tab:gray",
                edgecolor="k", alpha=0.7)
    ax.set_xlabel("failure step (s)")
    ax.set_title(f"failure times; push<1s: {100.0*push_near/max(n_fail,1):.0f}%")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    log.info("图已保存: %s", args.out)


if __name__ == "__main__":
    main()
