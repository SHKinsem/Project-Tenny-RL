#!/usr/bin/env python
"""run2 终版多方对比图:run1 baseline vs base2 vs cand000 vs cand026(4e8 × 3 seeds)。

数据来源:各方 trials 的 metrics.csv(训练过程 eval 曲线)。
输出(默认 tuning/run2/reports):
- run2_final_curves.png             训练曲线 2x2(RMSE / episode reward / 存活 / 能耗每步)
- run2_final_reward_term_curves.png reward 分项训练过程曲线(3x4,逐分项 mean±std)
- run2_final_tail_terms.json        尾部 3 eval 各分项均值(供报告表格)

注意:run1 baseline 的 height 两项为投影前旧口径(reward 语义不同,仅参考);
其余分项与 RMSE 同口径可比。
"""

import argparse
import csv
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ERR2_LIN = "eval/episode_track/err2_lin_vel"
ERR2_ANG = "eval/episode_track/err2_ang_vel"
EP_LEN = "eval/avg_episode_length"
EPISODE_REWARD = "eval/episode_reward"
ENERGY = "eval/episode_reward/energy_cost"
TERM_PREFIX = "eval/episode_reward/"

GROUPS = {
    "run1-baseline": "tuning/run1/trials/confirm-baseline-s{0,1,2}",
    "base2": "tuning/run2/trials/confirm2-base2-s{0,1,2}",
    "cand000": "tuning/run2/trials/cand000-s{0,1,2}",
    "cand026": "tuning/run2/trials/cand026-s{0,1,2}",
}

TERMS = [
    "tracking_lin_vel", "tracking_ang_vel", "tracking_height_jnt",
    "tracking_height_action", "tracking_sinyaw", "tracking_wheels_sinposition",
    "pitch_cost", "roll_cost", "energy_cost", "action_rate",
    "pitch_velocity_penalty", "termination_penalty",
]


def load_metrics_csv(path):
    with open(path, newline="") as f:
        return [
            {k: float(v) for k, v in row.items() if v not in ("", None)}
            for row in csv.DictReader(f)
        ]


def group_runs(pattern):
    dirs = sorted(d for d in __import__("glob").glob(pattern.replace("{0,1,2}", "*")))
    return [load_metrics_csv(os.path.join(d, "metrics.csv")) for d in dirs]


def group_sigmas(pattern):
    """各 run 的 (sigma, sigma_wheel),从 overrides.json 读,缺省 0.5。"""
    dirs = sorted(d for d in __import__("glob").glob(pattern.replace("{0,1,2}", "*")))
    out = []
    for d in dirs:
        sigma, sigma_wheel = 0.5, 0.5
        ov_path = os.path.join(d, "overrides.json")
        if os.path.exists(ov_path):
            env = json.load(open(ov_path)).get("env", {})
            sigma = env.get("reward_config.sigma", sigma)
            sigma_wheel = env.get("reward_config.sigma_wheel", sigma_wheel)
        out.append((sigma, sigma_wheel))
    return out


def series(rows, fn):
    return [fn(r) for r in rows if fn(r) is not None]


def rmse_of(r):
    if ERR2_LIN in r and EP_LEN in r and r[EP_LEN]:
        return math.sqrt((r[ERR2_LIN] + r[ERR2_ANG]) / r[EP_LEN])
    return None


def mean_std(series_list):
    n = min(len(s) for s in series_list)
    mean, std = [], []
    for i in range(n):
        vals = [s[i] for s in series_list]
        m = sum(vals) / len(vals)
        mean.append(m)
        std.append(math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals)))
    return mean, std


def plot_band(ax, steps, series_list, label, color):
    mean, std = mean_std(series_list)
    xs = steps[: len(mean)]
    ax.plot(xs, mean, label=label, color=color)
    ax.fill_between(xs, [m - s for m, s in zip(mean, std)],
                    [m + s for m, s in zip(mean, std)], color=color, alpha=0.15)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="tuning/run2/reports")
    args = ap.parse_args()

    colors = {"run1-baseline": "gray", "base2": "tab:blue",
              "cand000": "tab:orange", "cand026": "tab:green"}
    data = {name: group_runs(pat) for name, pat in GROUPS.items()}
    for name, runs in data.items():
        assert runs and all(runs), f"{name} 缺 metrics.csv"

    # ---- 2x2 主曲线 ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    panels = [
        ("RMSE (normalized lin+ang)", lambda r: rmse_of(r)),
        ("episode reward", lambda r: r.get(EPISODE_REWARD)),
        ("avg episode length (survival)", lambda r: r.get(EP_LEN)),
        ("energy per step", lambda r: (r[ENERGY] / r[EP_LEN])
         if ENERGY in r and r.get(EP_LEN) else None),
    ]
    for ax, (title, fn) in zip(axes.flat, panels):
        for name, runs in data.items():
            ss = [series(rows, fn) for rows in runs]
            steps = [r["num_steps"] / 1e8 for r in runs[0]]
            plot_band(ax, steps, ss, name, colors[name])
        ax.set_title(title)
        ax.set_xlabel("steps (1e8)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "run2_final_curves.png"), dpi=130)
    plt.close(fig)

    # ---- 分项过程曲线 3x4 ----
    fig, axes = plt.subplots(3, 4, figsize=(17, 10))
    for ax, term in zip(axes.flat, TERMS):
        key = TERM_PREFIX + term
        for name, runs in data.items():
            ss = [series(rows, lambda r, k=key: r.get(k)) for rows in runs]
            ss = [s for s in ss if s]
            if not ss:
                continue
            steps = [r["num_steps"] / 1e8 for r in runs[0]]
            plot_band(ax, steps, ss, name, colors[name])
        ax.set_title(term, fontsize=9)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
    axes.flat[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "run2_final_reward_term_curves.png"), dpi=130)
    plt.close(fig)

    # ---- 无权重总 reward 曲线(分项原始值带符号求和) ----
    # 权重只取正负号:跟踪/姿势类为正,惩罚类为负。注意各组 σ 不同,
    # 左图受"尺子松紧"影响(见报告第 5 节);右图为 σ 无关的纯物理惩罚项。
    POS = ["tracking_lin_vel", "tracking_ang_vel", "tracking_height_jnt",
           "tracking_height_action", "tracking_sinyaw",
           "tracking_wheels_sinposition", "pitch_cost", "roll_cost"]
    NEG = ["energy_cost", "action_rate", "pitch_velocity_penalty",
           "termination_penalty"]

    def unweighted_sum(r):
        total = sum(r.get(TERM_PREFIX + t, 0.0) for t in POS)
        total -= sum(r.get(TERM_PREFIX + t, 0.0) for t in NEG)
        return total

    def phys_penalty_per_step(r):
        if not r.get(EP_LEN):
            return None
        pen = sum(r.get(TERM_PREFIX + t, 0.0)
                  for t in ["energy_cost", "action_rate",
                            "pitch_velocity_penalty"])
        return pen / r[EP_LEN]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for name, runs in data.items():
        steps = [r["num_steps"] / 1e8 for r in runs[0]]
        plot_band(axes[0], steps,
                  [series(rows, unweighted_sum) for rows in runs],
                  name, colors[name])
        plot_band(axes[1], steps,
                  [series(rows, phys_penalty_per_step) for rows in runs],
                  name, colors[name])
    axes[0].set_title("unweighted total reward / episode (own σ)")
    axes[1].set_title("σ-free physical penalties / step (lower better)")
    for ax in axes:
        ax.set_xlabel("steps (1e8)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "run2_final_unweighted_sum.png"), dpi=130)
    plt.close(fig)

    # ---- 固定 σ=0.5 重打分的无权重总 reward 曲线 ----
    # exp(-err/σ) ⇒ s_fixed = s_own^(σ_own/0.5)(cmd_scale 在指数比中约去,
    # 推导见报告第 5 节)。对每集均值做变换是 Jensen 近似,但各组同公式、公平。
    # σ 无关的惩罚项(energy/action_rate/pitch_velocity)原样保留。
    SIGMA_POS = ["tracking_lin_vel", "tracking_ang_vel", "tracking_height_jnt",
                 "tracking_height_action", "tracking_sinyaw",
                 "pitch_cost", "roll_cost"]
    SIGMAW_POS = ["tracking_wheels_sinposition"]
    RAW_NEG = ["energy_cost", "action_rate", "pitch_velocity_penalty",
               "termination_penalty"]

    def fixed_sigma_sum(r, sigma, sigma_wheel, per_step=False):
        ep_len = r.get(EP_LEN)
        if not ep_len:
            return None
        total = 0.0
        for terms, s_own in ((SIGMA_POS, sigma), (SIGMAW_POS, sigma_wheel)):
            expo = s_own / 0.5
            for t in terms:
                s = max(r.get(TERM_PREFIX + t, 0.0) / ep_len, 1e-9)
                total += s ** expo
        for t in RAW_NEG:
            total -= r.get(TERM_PREFIX + t, 0.0) / ep_len
        return total if per_step else total * ep_len

    sigmas = {name: group_sigmas(pat) for name, pat in GROUPS.items()}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for name, runs in data.items():
        steps = [r["num_steps"] / 1e8 for r in runs[0]]
        for ax, per_step in zip(axes, (False, True)):
            ss = [series(rows, lambda r, sg=sigmas[name][i], ps=per_step:
                         fixed_sigma_sum(r, sg[0], sg[1], ps))
                  for i, rows in enumerate(runs)]
            plot_band(ax, steps, ss, name, colors[name])
    axes[0].set_title("unweighted total reward / episode (fixed σ=0.5)")
    axes[1].set_title("unweighted total reward / step (fixed σ=0.5)")
    for ax in axes:
        ax.set_xlabel("steps (1e8)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "run2_final_fixedsigma_unweighted.png"), dpi=130)
    plt.close(fig)

    # ---- 尾部 3 eval 分项均值(供报告表格) ----
    tail = {}
    for name, runs in data.items():
        terms = {}
        for term in TERMS:
            key = TERM_PREFIX + term
            vals = []
            for rows in runs:
                vs = [r[key] for r in rows[-3:] if key in r]
                if vs:
                    vals.append(sum(vs) / len(vs))
            if vals:
                terms[term] = {
                    "mean": sum(vals) / len(vals),
                    "std": math.sqrt(
                        sum((v - sum(vals) / len(vals)) ** 2 for v in vals)
                        / len(vals)),
                }
        tail[name] = terms
    with open(os.path.join(args.out, "run2_final_tail_terms.json"), "w") as fp:
        json.dump(tail, fp, indent=2)
    print("已输出:",
          "run2_final_curves.png, run2_final_reward_term_curves.png,",
          "run2_final_unweighted_sum.png, run2_final_fixedsigma_unweighted.png",
          "run2_final_tail_terms.json →", args.out)


if __name__ == "__main__":
    main()
