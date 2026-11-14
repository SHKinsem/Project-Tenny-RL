#!/usr/bin/env python
"""失败动力学诊断:快失稳(A) vs 慢累积(B) —— 判定近视策略(0.72s 视野)是否原理充分。

⚠️ 已作废(2026-08-27):batch_context 离线重放跨进程不是 bit-identical
(XLA matmul 算法/GPU 非确定性;实测仅 28/212 失败步数复现,其余 ~1.4s 提前摔倒),
本脚本基于离线重放的轨迹统计不可信。请改用策略在环的
tools/measure_eval_dynamics.py(其结果见 tuning/run1/reports/failure_dynamics_v2.png)。

方法:用 batch_test 落盘的 batch_context(全体 env 的 reset rng + 动作序列)
vmap 重放整个批次一次,记录每步每 env 的 sinpitch 与推力;对每条失败 episode,
以失败时刻为 t=0 对齐,统计:
  1) 对齐平均轨迹:失败前是平坦后急转(A)还是缓慢爬升(B);
  2) 发散时间:sinpitch 从穿越 -0.3(中带)到 -0.6(失败阈值)的步数分布;
  3) 失败与推力的时序关系(失败前 1s 内是否有推力)。

用法:
  CUDA_VISIBLE_DEVICES=0 python tools/analyze_failure_dynamics.py \
      --context tuning/run1/batch_test/confirm-baseline-s0/batch_context_XXX.npz \
      --failures tuning/run1/batch_test/confirm-baseline-s0/failures_XXX.pkl \
      --out tuning/run1/reports/failure_dynamics.png
"""

import argparse
import logging
import os
import sys

import jax
import jax.numpy as jp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

log = logging.getLogger("faildyn")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", required=True)
    ap.add_argument("--failures", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=int, default=300, help="失败前分析窗口(步,10ms/步)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from legwheel.evaluation.failures import load_failures
    from legwheel.evaluation.runner import make_env

    ctx = np.load(args.context)
    env_rngs = jp.array(ctx["env_rngs"])
    all_actions = ctx["all_actions"]  # (T, num_envs, act)
    total_steps, num_envs = all_actions.shape[0], all_actions.shape[1]
    failures, _stats = load_failures(args.failures)
    log.info("num_envs=%d steps=%d failures=%d", num_envs, total_steps, len(failures))

    env, _cfg = make_env(push=True)
    vmap_reset = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))
    vmap_step = jax.jit(jax.vmap(env.step))
    vmap_sinpitch = jax.jit(jax.vmap(env._getSinPitch_newest))

    states = vmap_reset(env_rngs, True)
    sinpitch = np.zeros((total_steps, num_envs), dtype=np.float32)
    push_norm = np.zeros((total_steps, num_envs), dtype=np.float32)
    for t in range(total_steps):
        states = vmap_step(states, jp.array(all_actions[t]))
        sinpitch[t] = np.asarray(jax.device_get(vmap_sinpitch(states.data)))
        push_norm[t] = np.asarray(jax.device_get(
            jp.linalg.norm(states.info["push"], axis=-1)))
    log.info("重放完成")

    # ---- 逐失败分析(失败步 fail_step 对应重放索引 fail_step-1)----
    W = args.window
    aligned = []       # 对齐到失败时刻的 sinpitch 序列(W 长)
    div_times = []     # 从 |穿越 -0.3| 到失败的步数
    push_near = 0      # 失败前 100 步内有推力
    for f in failures:
        end = f.fail_step          # 重放序列索引 end-1 = 失败步
        seg = sinpitch[max(0, end - W):end, f.env_idx]  # 长度 ≤ W,末位≈失败阈值
        pad = np.full(W - len(seg), np.nan, dtype=np.float32)
        aligned.append(np.concatenate([pad, seg]))
        above = np.nonzero(seg > -0.3)[0]            # 中带以上的最后位置
        div_times.append(len(seg) - 1 - above[-1] if len(above) else len(seg))
        if push_norm[max(0, end - 101):end, f.env_idx].max() > 1e-6:
            push_near += 1
    aligned = np.array(aligned)
    div_times = np.array(div_times)

    n_fast = int((div_times < 100).sum())
    log.info("发散时间(<1s 为快失稳): median=%d 步, <100步占比 %.1f%% (%d/%d)",
             int(np.median(div_times)), 100.0 * n_fast / len(div_times),
             n_fast, len(div_times))
    log.info("失败前 1s 内有推力: %.1f%% (%d/%d)",
             100.0 * push_near / len(failures), push_near, len(failures))
    for lag in (200, 100, 50, 20, 10):
        log.info("  失败前 %3d 步: sinpitch 中位数 %.3f",
                 lag, np.nanmedian(aligned[:, W - lag]))

    # ---- 绘图 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = (np.arange(W) - W) * 0.01  # 秒,0 = 失败时刻
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ax = axes[0]
    med = np.nanmedian(aligned, axis=0)
    p25 = np.nanpercentile(aligned, 25, axis=0)
    p75 = np.nanpercentile(aligned, 75, axis=0)
    ax.plot(t, med, color="tab:red", label="median")
    ax.fill_between(t, p25, p75, color="tab:red", alpha=0.25, label="25-75%")
    ax.axhline(-0.6, ls="--", color="k", lw=1, label="fail threshold -0.6")
    ax.axhline(-0.3, ls=":", color="gray", lw=1, label="mid band -0.3")
    ax.set_xlabel("time to failure (s)")
    ax.set_ylabel("sinpitch")
    ax.set_title(f"aligned trajectories (n={len(failures)})")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax = axes[1]
    ax.hist(div_times * 0.01, bins=30, color="tab:blue", edgecolor="k", alpha=0.7)
    ax.axvline(1.0, ls="--", color="r", label="1s (policy horizon ~0.72s)")
    ax.set_xlabel("divergence time: -0.3 crossing -> failure (s)")
    ax.set_ylabel("count")
    ax.set_title(f"fast(<1s): {100.0*n_fast/len(div_times):.0f}%")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax = axes[2]
    ax.hist([f.fail_step * 0.01 for f in failures], bins=30,
            color="tab:gray", edgecolor="k", alpha=0.7)
    ax.set_xlabel("failure step (s)")
    ax.set_title(f"failure time distribution; push<1s before: "
                 f"{100.0*push_near/len(failures):.0f}%")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    log.info("图已保存: %s", args.out)


if __name__ == "__main__":
    main()
