#!/usr/bin/env python
"""门控 reward 项(sinyaw/sinposition)激活步数实测:验证结构性上限估算。

用 batch_test 落盘的 batch_context(全体 env 的 reset rng + 动作序列)vmap 重放,
逐步记录 command 与门控分项的每步值,统计:
  - 每个 episode 的激活步数(门控条件成立的步数)分布;
  - 激活步上的平均分项值(真实 per-active-step 达成率);
  - 对比解析估算(600 × (1-cmd_prob + cmd_prob×死区占比) ≈ 64)。

用法:
  CUDA_VISIBLE_DEVICES=0 python tools/measure_gated_activation.py \
      --context tuning/run1/batch_test/confirm-baseline-s0/batch_context_XXX.npz
"""

import argparse
import logging
import os
import sys

import jax
import jax.numpy as jp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

log = logging.getLogger("gated")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from legwheel.evaluation.runner import make_env

    ctx = np.load(args.context)
    env_rngs = jp.array(ctx["env_rngs"])
    all_actions = ctx["all_actions"]
    total_steps, num_envs = all_actions.shape[0], all_actions.shape[1]
    log.info("num_envs=%d steps=%d", num_envs, total_steps)

    env, cfg = make_env(push=True)
    lin_margin = float(cfg.cmd_lin_vel_x_zero_margin)
    ang_margin = float(cfg.cmd_ang_vel_zero_margin)
    log.info("lin_margin=%g ang_margin=%g cmd_prob=%g interval=%d",
             lin_margin, ang_margin, cfg.cmd_config.cmd_prob,
             cfg.cmd_config.command_change_interval)

    vmap_reset = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))
    vmap_step = jax.jit(jax.vmap(env.step))

    states = vmap_reset(env_rngs, True)
    active_yaw = np.zeros((total_steps, num_envs), dtype=bool)
    active_both = np.zeros((total_steps, num_envs), dtype=bool)
    sinyaw_val = np.zeros((total_steps, num_envs), dtype=np.float32)
    sinpos_val = np.zeros((total_steps, num_envs), dtype=np.float32)
    dones = np.zeros((total_steps, num_envs), dtype=bool)

    for t in range(total_steps):
        states = vmap_step(states, jp.array(all_actions[t]))
        cmd = np.asarray(jax.device_get(states.info["command"]))
        active_yaw[t] = np.abs(cmd[:, 1]) <= ang_margin
        active_both[t] = active_yaw[t] & (np.abs(cmd[:, 0]) <= lin_margin)
        m = states.metrics
        sinyaw_val[t] = np.asarray(jax.device_get(m["reward/tracking_sinyaw"]))
        sinpos_val[t] = np.asarray(
            jax.device_get(m["reward/tracking_wheels_sinposition"]))
        dones[t] = np.asarray(jax.device_get(states.done)) > 0
    log.info("重放完成")

    # 只统计每个 env 首次 done 之前的步(与 eval 的 episode 口径一致)
    first_done = np.argmax(dones, axis=0)  # 无 done 时为 0,需特判
    has_done = dones.any(axis=0)
    ep_len = np.where(has_done, first_done + 1, total_steps)

    ay = np.array([active_yaw[:L, e].sum() for e, L in enumerate(ep_len)])
    ab = np.array([active_both[:L, e].sum() for e, L in enumerate(ep_len)])
    sv = np.array([sinyaw_val[:L, e].sum() for e, L in enumerate(ep_len)])
    pv = np.array([sinpos_val[:L, e].sum() for e, L in enumerate(ep_len)])

    log.info("episode 长度: mean=%.1f (满 %d)", ep_len.mean(), total_steps)
    log.info("[sinyaw] 激活步数/episode: mean=%.1f median=%d p5=%d p95=%d",
             ay.mean(), int(np.median(ay)), int(np.percentile(ay, 5)),
             int(np.percentile(ay, 95)))
    log.info("[sinyaw] 分项和/episode: mean=%.2f → 激活步均值 %.3f(满分 1)",
             sv.mean(), sv.sum() / max(ay.sum(), 1))
    log.info("[sinpos] 激活步数/episode: mean=%.1f median=%d",
             ab.mean(), int(np.median(ab)))
    log.info("[sinpos] 分项和/episode: mean=%.2f → 激活步均值 %.3f(满分 1)",
             pv.mean(), pv.sum() / max(ab.sum(), 1))
    log.info("解析估算上限: 600 × (0.1 + 0.9×%.4f) = %.1f",
             ang_margin / 1.5, 600 * (0.1 + 0.9 * ang_margin / 1.5))


if __name__ == "__main__":
    main()
