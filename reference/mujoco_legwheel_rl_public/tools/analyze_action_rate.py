#!/usr/bin/env python
"""action_rate 分解实测:这个惩罚项为什么这么高、该从哪压。

定义(rewards.py::_cost_action_rate,作用于策略归一化原始输出 a∈[-1,1]^4):
    action_rate = mean_j(|a_t − a_{t−1}|) + mean_j(|a_t − 2a_{t−1} + a_{t−2}|)
tail 值 ≈ 1.36/步(762/episode),惩罚权重 −0.5 → −0.68/步,约占 8.6%。

本工具回答三个分解问题(策略在环,逐步记录策略原始输出):
  1) 一阶差分(c1)与二阶差分(c2)各占多少;
  2) thigh(关节 0/1)与 wheel(关节 2/3)各占多少
     (两者物理 scale 差 8.5 倍:0.25 rad vs 0.0295);
  3) 观测噪声贡献多少:同策略在 noise off + push off 下重跑
     (thigh 速度观测是纯噪声维,若策略对其有非零权重,输出必抖)。

用法:
  CUDA_VISIBLE_DEVICES=0 python tools/analyze_action_rate.py \
      --ckpt /abs/path/to/ckpt/000105062400
"""

import argparse
import logging
import os
import sys

import jax
import jax.numpy as jp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

log = logging.getLogger("action_rate")


def rollout_actions(env, jit_inference, num_envs, seed, randomized=False):
    T = int(env._config.episode_length) if hasattr(env, "_config") else int(env.unwrapped._config.episode_length)
    env_rngs = jax.random.split(jax.random.PRNGKey(seed), num_envs)
    if randomized:
        # BraxDomainRandomizationVmapWrapper 内部已 vmap,domain_randomize 的
        # rng 参数按 brax 约定传 batched keys(与 ppo.train 一致)
        import functools

        from mujoco_playground import wrapper as playground_wrapper

        from legwheel.randomization import domain_randomize

        rand_fn = functools.partial(
            domain_randomize,
            rng=jax.random.split(jax.random.PRNGKey(123), num_envs))
        env = playground_wrapper.BraxDomainRandomizationVmapWrapper(env, rand_fn)
        reset = jax.jit(env.reset)
        step = jax.jit(env.step)
    else:
        vmap_reset = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))
        step = jax.jit(jax.vmap(env.step))
        reset = lambda rngs: vmap_reset(rngs, True)  # noqa: E731

    @jax.jit
    def vmap_inference(obs, rngs):
        return jax.vmap(jit_inference)(obs, rngs)

    states = reset(env_rngs)
    acts = np.zeros((T, num_envs, 4), dtype=np.float32)
    dones = np.zeros((T, num_envs), dtype=bool)
    act_rng = jax.random.PRNGKey(0)
    for t in range(T):
        act_rng, sub = jax.random.split(act_rng)
        actions, _ = vmap_inference(states.obs, jax.random.split(sub, num_envs))
        acts[t] = np.asarray(jax.device_get(actions))
        states = step(states, actions)
        dones[t] = np.asarray(jax.device_get(states.done)) > 0

    has_done = dones.any(axis=0)
    ep_len = np.where(has_done, np.argmax(dones, axis=0) + 1, T)
    return acts, ep_len


def report(name, acts, ep_len):
    """acts: (T, N, 4) 策略原始输出;逐 env 取首次 done 前的步算差分。"""
    T, N, _ = acts.shape
    c1_all, c2_all = [], []
    c1_tw = [[], []]
    c2_tw = [[], []]
    for e in range(N):
        L = int(ep_len[e])
        if L < 3:
            continue
        a = acts[1:L, e]              # (L-1, 4),从第 1 步起有 last
        d1 = np.abs(a[1:] - a[:-1])               # (L-2, 4) 一阶
        d2 = np.abs(a[2:] - 2 * a[1:-1] + a[:-2])  # (L-3, 4) 二阶
        c1_all.append(d1.mean(axis=1))
        c2_all.append(d2.mean(axis=1))
        c1_tw[0].append(d1[:, :2].mean(axis=1))
        c1_tw[1].append(d1[:, 2:].mean(axis=1))
        c2_tw[0].append(d2[:, :2].mean(axis=1))
        c2_tw[1].append(d2[:, 2:].mean(axis=1))
    c1 = np.concatenate(c1_all)
    c2 = np.concatenate(c2_all)
    log.info("== %s ==", name)
    log.info("  action_rate = %.3f/步 (c1 %.3f + c2 %.3f, 占比 %.0f%%/%.0f%%)",
             c1.mean() + c2.mean(), c1.mean(), c2.mean(),
             100 * c1.mean() / (c1.mean() + c2.mean()),
             100 * c2.mean() / (c1.mean() + c2.mean()))
    for gi, gname in enumerate(("thigh", "wheel")):
        c1g = np.concatenate(c1_tw[gi])
        c2g = np.concatenate(c2_tw[gi])
        log.info("  %s: c1 %.3f  c2 %.3f  (c1 分位 p50 %.2f / p90 %.2f / p99 %.2f)",
                 gname, c1g.mean(), c2g.mean(),
                 np.percentile(c1g, 50), np.percentile(c1g, 90),
                 np.percentile(c1g, 99))
    return c1.mean() + c2.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--base-seed", type=int, default=42)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from legwheel.evaluation.runner import load_policy, make_env

    # load_policy 支持两种口径:deterministic(部署口径,均值动作)vs
    # stochastic(训练 eval 口径,brax ppo.train 的 deterministic_eval=False
    # 默认值,采样动作)。
    det_inference = jax.jit(load_policy(args.ckpt, deterministic=True))
    sto_inference = jax.jit(load_policy(args.ckpt, deterministic=False))

    # 训练 eval 完全同款:域随机化 + push + 噪声(确定性策略 = 部署口径)
    env_r, _ = make_env(push=True)
    acts_r, ep_len_r = rollout_actions(env_r, det_inference, args.num_envs,
                                       args.base_seed, randomized=True)
    rate_rand = report("部署口径 (deterministic + randomized + push + noise)",
                       acts_r, ep_len_r)

    # 训练 eval 真实口径:随机采样策略(brax deterministic_eval=False)
    if sto_inference is not None:
        env_s, _ = make_env(push=True)
        acts_s, ep_len_s = rollout_actions(env_s, sto_inference,
                                           args.num_envs, args.base_seed,
                                           randomized=True)
        rate_sto = report("训练 eval 口径 (stochastic + randomized + push + noise)",
                          acts_s, ep_len_s)
        log.info("训练报表口径(stochastic)=%.3f vs 部署口径(deterministic)=%.3f "
                 "→ 策略采样噪声(σ)贡献 %.0f%%",
                 rate_sto, rate_rand,
                 100 * (rate_sto - rate_rand) / max(rate_sto, 1e-9))

    # 干净环境(确定性):noise 关、push 关 → 策略自身增益/结构造成的抖动
    env_clean, _ = make_env(push=False,
                            overrides={"noise_config.level": 0.0})
    acts_c, ep_len_c = rollout_actions(env_clean, det_inference,
                                       args.num_envs, args.base_seed)
    rate_clean = report("干净环境 (deterministic, push off, noise off)",
                        acts_c, ep_len_c)

    log.info("部署口径分解:干净 %.3f | +噪声+push+随机化 %.3f",
             rate_clean, rate_rand)


if __name__ == "__main__":
    main()
