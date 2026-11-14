#!/usr/bin/env python
"""投影 vs baseline 在 roll-height 冲突窗口内的行为对照(策略在环)。

动机:投影 reward(h_ref = clip(h_cmd, -0.5+δ, 0-δ))理论上应当
"保 roll 的同时给出可行的 height 目标",曲线应不劣于 baseline;
但 1e8 对比显示两条曲线几乎重合。本工具定位原因:

两个策略(baseline ckpt 与 projection ckpt,相同 base-seed 指令流)各跑
一个 episode 批,逐步记录 command / thigh 关节 / 车身 sinroll,
**全部误差离线重算**(两个口径:h_cmd 原始参考 / h_ref 投影参考),
在冲突窗口(err_min>0,即指令对超出 thigh 行程可行域)内对比:
  - roll 误差 |sinroll - cmd_roll|(proj 是否真的更保 roll?)
  - height 误差(对 h_cmd:物理高度质量;对 h_ref:新考核口径)
  - 实际差动半幅 δ_real vs 指令需要 δ_need(策略是否做足了 split?
    proj 是否过度下蹲?)
  - 两项 reward 之和的窗口内均值

同时打印理论上限:当前指令分布下冲突所蕴含的 reward 量级
(冲突步占比、平均可豁免误差、对 episode reward 的天花板贡献)。

用法:
  CUDA_VISIBLE_DEVICES=0 python tools/compare_projection_windows.py \
      --base-ckpt tuning/run1/trials/baseline-s0/ckpt/000105062400 \
      --proj-ckpt tuning/run2/trials/baseproj-s0/ckpt/000105062400
"""

import argparse
import logging
import os
import sys

import jax
import jax.numpy as jp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

log = logging.getLogger("cmp_window")

K_COUPLING = 0.433  # roll_thigh_coupling 实测值


def rollout(env, jit_inference, num_envs, seed):
    T = int(env._config.episode_length)
    env_rngs = jax.random.split(jax.random.PRNGKey(seed), num_envs)
    vmap_reset = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))
    vmap_step = jax.jit(jax.vmap(env.step))
    thigh_ids = np.asarray(jax.device_get(env._qpos_thigh_joints))
    vmap_sinroll = jax.jit(jax.vmap(env._getSinRoll_newest))

    @jax.jit
    def vmap_inference(obs, rngs):
        return jax.vmap(jit_inference)(obs, rngs)

    states = vmap_reset(env_rngs, True)
    rec = {k: np.zeros((T, num_envs), dtype=np.float32)
           for k in ("cmd_h", "cmd_r", "thigh_l", "thigh_r", "sinroll")}
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
        dones[t] = np.asarray(jax.device_get(states.done)) > 0

    # 只用首次 done 前的步
    has_done = dones.any(axis=0)
    ep_len = np.where(has_done, np.argmax(dones, axis=0) + 1, T)
    mask = np.zeros((T, num_envs), dtype=bool)
    for e in range(num_envs):
        mask[: int(ep_len[e]), e] = True
    return {k: v[mask] for k, v in rec.items()}


def analyze(name, rec, sig_h, sig_r, thigh_lo, thigh_hi):
    cmd_h, cmd_r = rec["cmd_h"], rec["cmd_r"]
    mean_t = (rec["thigh_l"] + rec["thigh_r"]) / 2
    split_half = np.abs(rec["thigh_l"] - rec["thigh_r"]) / 2
    sinroll = rec["sinroll"]

    d_need = K_COUPLING * np.abs(cmd_r)
    h_lo, h_hi = thigh_lo + d_need, thigh_hi - d_need
    err_min = np.maximum(h_lo - cmd_h, 0) + np.maximum(cmd_h - h_hi, 0)
    conflict = err_min > 1e-6
    h_ref = np.clip(cmd_h, h_lo, h_hi)

    err_h_cmd = np.abs(mean_t - cmd_h)      # 物理口径:对原始指令
    err_h_ref = np.abs(mean_t - h_ref)      # 投影口径:对可行参考
    err_roll = np.abs(sinroll - cmd_r)

    r_h_cmd = np.exp(-err_h_cmd / sig_h)
    r_h_ref = np.exp(-err_h_ref / sig_h)
    r_roll = np.exp(-err_roll / sig_r)

    log.info("== %s ==", name)
    log.info("  冲突步占比: %.1f%% (n=%d/%d)", 100 * conflict.mean(),
             int(conflict.sum()), conflict.size)
    log.info("  [全程]  roll_err=%.4f  h_err(to cmd)=%.4f  r_h_cmd=%.3f  r_roll=%.3f",
             err_roll.mean(), err_h_cmd.mean(), r_h_cmd.mean(), r_roll.mean())
    cw = conflict
    log.info("  [冲突窗] roll_err=%.4f  h_err_cmd=%.4f  h_err_ref=%.4f",
             err_roll[cw].mean(), err_h_cmd[cw].mean(), err_h_ref[cw].mean())
    log.info("  [冲突窗] r_h_cmd=%.3f  r_h_ref=%.3f  r_roll=%.3f",
             r_h_cmd[cw].mean(), r_h_ref[cw].mean(), r_roll[cw].mean())
    log.info("  [冲突窗] split半幅: 实际=%.4f 需要=%.4f  mean(thigh): 实际=%.4f h_ref=%.4f h_cmd=%.4f",
             split_half[cw].mean(), d_need[cw].mean(),
             mean_t[cw].mean(), h_ref[cw].mean(), cmd_h[cw].mean())
    log.info("  [非冲突] roll_err=%.4f  h_err_cmd=%.4f  r_roll=%.3f  r_h_cmd=%.3f",
             err_roll[~cw].mean(), err_h_cmd[~cw].mean(),
             r_roll[~cw].mean(), r_h_cmd[~cw].mean())
    return dict(conflict=conflict, r_h_cmd=r_h_cmd, r_h_ref=r_h_ref,
                r_roll=r_roll, err_min=err_min)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt", required=True)
    ap.add_argument("--proj-ckpt", required=True)
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--base-seed", type=int, default=42)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from legwheel.evaluation.runner import load_policy, make_env

    env, cfg = make_env(push=True)
    sigma = float(cfg.reward_config.sigma)
    sig_h = sigma * 0.25   # cmd_scales[2]: height 半量程
    sig_r = sigma * 0.1    # cmd_scales[3]: roll 半量程
    thigh_lo = float(cfg.action_range.thigh[0])
    thigh_hi = float(cfg.action_range.thigh[1])
    height_w = float(cfg.reward_config.height_scale)
    roll_w = float(cfg.reward_config.roll_scale)

    # ---- 理论上限(指令分布直接积分,与策略无关)----
    rng = np.random.default_rng(0)
    n = 2_000_000
    h = rng.uniform(-0.5, 0.0, n)
    r = rng.uniform(-0.2, 0.2, n)
    d = K_COUPLING * np.abs(r)
    err_min = np.maximum(thigh_lo + d - h, 0) + np.maximum(h - (thigh_hi - d), 0)
    cw = err_min > 1e-6
    forgive = 1.0 - np.exp(-err_min / sig_h)
    log.info("[理论] 冲突步占比 %.1f%%,平均可豁免 height 误差 %.4f rad",
             100 * cw.mean(), err_min[cw].mean() if cw.any() else 0.0)
    log.info("[理论] height_jnt 每步 reward 天花板增益: %.4f (raw) -> "
             "x height_scale=%.1f -> %.4f/步, 约占 episode 总 reward(≈7.9/步) %.2f%%",
             forgive.mean(), height_w, forgive.mean() * height_w,
             100 * forgive.mean() * height_w / 7.9)

    # ---- 策略在环对照 ----
    for label, ckpt in (("baseline", args.base_ckpt), ("projection", args.proj_ckpt)):
        jit_inference = jax.jit(load_policy(ckpt))
        rec = rollout(env, jit_inference, args.num_envs, args.base_seed)
        analyze(label, rec, sig_h, sig_r, thigh_lo, thigh_hi)


if __name__ == "__main__":
    main()
