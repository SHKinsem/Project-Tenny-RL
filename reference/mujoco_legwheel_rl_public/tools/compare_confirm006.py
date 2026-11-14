#!/usr/bin/env python
"""cand006(search2-006 配置)4e8 全量确认对比图:cand006 vs base2 vs run1 baseline。

数据来源:各方 trials 的 metrics.csv(训练过程 eval 曲线)。
输出(默认 tuning/run2/reports):
- confirm006_curves.png  2x3:RMSE / episode reward / 存活 /
  固定σ无权重和每步 / action_rate 每步 / energy 每步
- confirm006_tail.json   尾部 3 eval 汇总(含固定σ无权重和、RMSE)
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
TERM_PREFIX = "eval/episode_reward/"

SIGMA_POS = ["tracking_lin_vel", "tracking_ang_vel", "tracking_height_jnt",
             "tracking_height_action", "tracking_sinyaw",
             "pitch_cost", "roll_cost"]
SIGMAW_POS = ["tracking_wheels_sinposition"]
RAW_NEG = ["energy_cost", "action_rate", "pitch_velocity_penalty",
           "termination_penalty"]

GROUPS = {
    "run1-baseline": "tuning/run1/trials/confirm-baseline-s{0,1,2}",
    "base2": "tuning/run2/trials/confirm2-base2-s{0,1,2}",
    "cand006": "tuning/run2/trials/cand006-s{0,1,2}",
}
SIGMAS = {"run1-baseline": (0.5, 0.5), "base2": (0.5, 0.5),
          "cand006": (0.5782202498733064, 0.6966602525368217)}


def load_metrics_csv(path):
    with open(path, newline="") as f:
        return [{k: float(v) for k, v in row.items() if v not in ("", None)}
                for row in csv.DictReader(f)]


def group_runs(pattern):
    dirs = sorted(d for d in __import__("glob").glob(
        pattern.replace("{0,1,2}", "*")))
    return [load_metrics_csv(os.path.join(d, "metrics.csv")) for d in dirs]


def series(rows, fn):
    return [fn(r) for r in rows if fn(r) is not None]


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


def rmse_of(r):
    if ERR2_LIN in r and r.get(EP_LEN):
        return math.sqrt((r[ERR2_LIN] + r[ERR2_ANG]) / r[EP_LEN])
    return None


def fixed_sigma_per_step(r, sigma, sigma_wheel):
    ep = r.get(EP_LEN)
    if not ep:
        return None
    tot = 0.0
    for terms, so in ((SIGMA_POS, sigma), (SIGMAW_POS, sigma_wheel)):
        for t in terms:
            s = max(r.get(TERM_PREFIX + t, 0.0) / ep, 1e-9)
            tot += s ** (so / 0.5)
    for t in RAW_NEG:
        tot -= r.get(TERM_PREFIX + t, 0.0) / ep
    return tot


def per_step(key):
    return lambda r: (r[key] / r[EP_LEN]) if key in r and r.get(EP_LEN) else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="tuning/run2/reports")
    args = ap.parse_args()

    colors = {"run1-baseline": "gray", "base2": "tab:blue",
              "cand006": "tab:red"}
    data = {name: group_runs(pat) for name, pat in GROUPS.items()}
    for name, runs in data.items():
        assert runs and all(runs), f"{name} 缺 metrics.csv"

    panels = [
        ("RMSE (normalized lin+ang)", rmse_of),
        ("episode reward", lambda r: r.get(EPISODE_REWARD)),
        ("avg episode length", lambda r: r.get(EP_LEN)),
        ("unweighted sum/step (fixed σ=0.5)", None),  # 特殊处理
        ("action_rate / step", per_step(TERM_PREFIX + "action_rate")),
        ("energy / step", per_step(TERM_PREFIX + "energy_cost")),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(17, 8))
    for ax, (title, fn) in zip(axes.flat, panels):
        for name, runs in data.items():
            steps = [r["num_steps"] / 1e8 for r in runs[0]]
            if fn is None:
                sg = SIGMAS[name]
                ss = [series(rows, lambda r, g=sg:
                             fixed_sigma_per_step(r, g[0], g[1]))
                      for rows in runs]
            else:
                ss = [series(rows, fn) for rows in runs]
            plot_band(ax, steps, ss, name, colors[name])
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("steps (1e8)")
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "confirm006_curves.png"), dpi=130)
    plt.close(fig)

    # ---- 分项过程曲线 3x4(固定 σ=0.5 重打分) ----
    # exp 型分项按 s^(σ_own/0.5) 换算到统一尺子(Jensen 近似,各组同公式);
    # σ 无关惩罚项(energy/action_rate/pitch_velocity/termination)原样保留。
    TERMS = SIGMA_POS + SIGMAW_POS + RAW_NEG

    def term_fixed(r, term, sg):
        ep = r.get(EP_LEN)
        key = TERM_PREFIX + term
        if not ep or key not in r:
            return None
        if term in RAW_NEG:
            return r[key]
        s_own = sg[1] if term in SIGMAW_POS else sg[0]
        s = max(r[key] / ep, 1e-9)
        return (s ** (s_own / 0.5)) * ep

    fig, axes = plt.subplots(3, 4, figsize=(17, 10))
    for ax, term in zip(axes.flat, TERMS):
        for name, runs in data.items():
            ss = [s for s in
                  (series(rows, lambda r, t=term, g=SIGMAS[name]:
                          term_fixed(r, t, g)) for rows in runs)
                  if s]
            if not ss:
                continue
            steps = [r["num_steps"] / 1e8 for r in runs[0]]
            plot_band(ax, steps, ss, name, colors[name])
        ax.set_title(term, fontsize=9)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
    axes.flat[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "confirm006_term_curves.png"), dpi=130)
    plt.close(fig)

    # 尾部 3 eval 汇总
    tail = {}
    for name, runs in data.items():
        sg = SIGMAS[name]
        per_seed = []
        for rows in runs:
            t3 = rows[-3:]
            fs = [fixed_sigma_per_step(r, sg[0], sg[1]) for r in t3]
            rm = [rmse_of(r) for r in t3]
            ar = [r[TERM_PREFIX + "action_rate"] / r[EP_LEN] for r in t3]
            per_seed.append({
                "fixed_sigma_unweighted_per_step": sum(fs) / len(fs),
                "rmse": sum(rm) / len(rm),
                "action_rate_per_step": sum(ar) / len(ar),
            })
        agg = {}
        for k in per_seed[0]:
            vals = [p[k] for p in per_seed]
            m = sum(vals) / len(vals)
            agg[k] = {"mean": m,
                      "std": math.sqrt(sum((v - m) ** 2 for v in vals)
                                       / len(vals))}
        tail[name] = agg
    with open(os.path.join(args.out, "confirm006_tail.json"), "w") as fp:
        json.dump(tail, fp, indent=2)
    print("已输出: confirm006_curves.png, confirm006_term_curves.png, confirm006_tail.json →", args.out)


if __name__ == "__main__":
    main()
