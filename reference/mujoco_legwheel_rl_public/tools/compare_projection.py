#!/usr/bin/env python
"""roll-height 投影 reward 重设计的基线对比(供 docs/projection_baseline_compare.md 引用)。

对比对象:
- 原始 baseline:tuning/run1/trials/baseline-s{0..4}(旧代码,无投影,1e8 步,5 seeds)
- 投影版:      tuning/run2/trials/baseproj-s{0..N}(baseline 参数 + height 投影开,1e8 步)

两者 PPO/网络/映射/权重全部一致,唯一差异是
reward_config.height_feasible_projection(True vs False)。

输出(--out 目录,默认 tuning/run2/reports):
- projection_curves.png       训练曲线 2x2(RMSE / episode reward / 存活 / 能耗)
- projection_term_curves.png  reward 分项训练过程曲线(逐分项 mean±std 对比)
- projection_compare.md       尾部指标表 + reward 分项表(markdown 片段)
"""

import argparse
import csv
import glob
import math
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ERR2_LIN = "eval/episode_track/err2_lin_vel"
ERR2_ANG = "eval/episode_track/err2_ang_vel"
EP_LEN = "eval/avg_episode_length"
EPISODE_REWARD = "eval/episode_reward"
ENERGY = "eval/episode_reward/energy_cost"
TERM_PREFIX = "eval/episode_reward/"

# 展示的分项(去掉 std 与 termination_penalty(恒 0))
TERMS = [
    "tracking_lin_vel", "tracking_ang_vel", "tracking_sinyaw",
    "tracking_wheels_sinposition", "pitch_cost", "roll_cost",
    "tracking_height_jnt", "tracking_height_action", "energy_cost",
    "action_rate", "pitch_velocity_penalty",
]


def load_metrics_csv(path):
    with open(path, newline="") as f:
        return [
            {k: float(v) for k, v in row.items() if v not in ("", None)}
            for row in csv.DictReader(f)
        ]


def eval_series(rows):
    out = {"steps": [], "rmse": [], "reward": [], "ep_len": [], "energy": []}
    for r in rows:
        if ERR2_LIN not in r or EP_LEN not in r or not r[EP_LEN]:
            continue
        out["steps"].append(r["num_steps"] / 1e8)
        out["rmse"].append(math.sqrt((r[ERR2_LIN] + r[ERR2_ANG]) / r[EP_LEN]))
        out["reward"].append(r[EPISODE_REWARD])
        out["ep_len"].append(r[EP_LEN])
        out["energy"].append(r[ENERGY] / r[EP_LEN])
    return out


def term_series(rows, term):
    key = TERM_PREFIX + term
    steps, vals, lens = [], [], []
    for r in rows:
        if key not in r or EP_LEN not in r or not r[EP_LEN]:
            continue
        steps.append(r["num_steps"] / 1e8)
        vals.append(r[key])
        lens.append(r[EP_LEN])
    return steps, vals, lens


def mean_std(series_list):
    n = min(len(s) for s in series_list)
    mean, std = [], []
    for i in range(n):
        vals = [s[i] for s in series_list]
        m = sum(vals) / len(vals)
        mean.append(m)
        std.append(math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals)))
    return mean, std


def tail3(series):
    return sum(series[-3:]) / 3.0


def collect_runs(pattern):
    """匹配 trial 目录(只收 <prefix>-s<数字>,排除 -hung 等现场目录),
    返回 {seed: dir}。"""
    dirs = {}
    for d in sorted(glob.glob(pattern)):
        m = re.search(r"-s(\d+)$", d)
        if m and os.path.exists(os.path.join(d, "metrics.csv")):
            dirs[int(m.group(1))] = d
    return dirs


def plot_group(ax, series_list, key, color, label):
    """画一组 seeds:mean 线 + std 带;n=1 时只画线。"""
    if len(series_list) > 1:
        m, s = mean_std([b[key] for b in series_list])
        steps = series_list[0]["steps"][:len(m)]
        ax.plot(steps, m, color=color, label=label)
        ax.fill_between(steps, [a - b for a, b in zip(m, s)],
                        [a + b for a, b in zip(m, s)], color=color, alpha=0.2)
        for b in series_list:
            ax.plot(b["steps"], b[key], color=color, alpha=0.15, linewidth=0.8)
    else:
        ax.plot(series_list[0]["steps"], series_list[0][key], color=color,
                label=label)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-glob", default="tuning/run1/trials/baseline-s*")
    ap.add_argument("--proj-glob", default="tuning/run2/trials/baseproj-s*")
    ap.add_argument("--out", default="tuning/run2/reports")
    args = ap.parse_args()

    base_map = collect_runs(args.base_glob)
    proj_map = collect_runs(args.proj_glob)
    assert base_map, f"no baseline trials matched {args.base_glob}"
    assert proj_map, f"no projection trials matched {args.proj_glob}"
    matched_seeds = sorted(set(base_map) & set(proj_map))
    assert matched_seeds, "no common seeds between baseline and projection"
    print(f"baseline runs: {sorted(base_map)}")
    print(f"projection runs: {sorted(proj_map)}")
    print(f"matched seeds for curves/paired analysis: {matched_seeds}")

    # 尾部表用全量 seeds(baseline n=5 估计更稳);
    # 曲线与配对分析用交集 seeds——不同 seed 集合求均值在早期高方差阶段
    # 会造成"组成性假象"(baseline 多出的 fast seed 抬高均值线)。
    base_rows = [load_metrics_csv(os.path.join(base_map[s], "metrics.csv"))
                 for s in sorted(base_map)]
    proj_rows = [load_metrics_csv(os.path.join(proj_map[s], "metrics.csv"))
                 for s in sorted(proj_map)]
    base_rows_m = [load_metrics_csv(os.path.join(base_map[s], "metrics.csv"))
                   for s in matched_seeds]
    proj_rows_m = [load_metrics_csv(os.path.join(proj_map[s], "metrics.csv"))
                   for s in matched_seeds]

    base_series = [eval_series(r) for r in base_rows]
    proj_series = [eval_series(r) for r in proj_rows]
    base_series_m = [eval_series(r) for r in base_rows_m]
    proj_series_m = [eval_series(r) for r in proj_rows_m]

    os.makedirs(args.out, exist_ok=True)

    seed_str = ",".join(str(s) for s in matched_seeds)
    base_label = f"baseline run1 (seeds {seed_str})"
    proj_label = f"projection on (seeds {seed_str})"

    # ---------- 图 1:总览 2x2 ----------
    panels = [("rmse", "RMSE (tail metric, lower=better)"),
              ("reward", "episode reward"),
              ("ep_len", "avg episode length"),
              ("energy", "energy / step")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (key, title) in zip(axes.flat, panels):
        plot_group(ax, base_series_m, key, "tab:blue", base_label)
        plot_group(ax, proj_series_m, key, "tab:red", proj_label)
        ax.set_title(title)
        ax.set_xlabel("steps (1e8)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("height-projection reward redesign vs original baseline (1e8 steps)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "projection_curves.png"), dpi=140)
    plt.close(fig)

    # ---------- 图 2:分项曲线 ----------
    ncol = 4
    nrow = math.ceil(len(TERMS) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(16, 3.2 * nrow))
    for ax, term in zip(axes.flat, TERMS):
        for rows, color, label in ((base_rows_m, "tab:blue", "baseline"),
                                   (proj_rows_m, "tab:red", "projection")):
            ts = [term_series(r, term) for r in rows]
            if len(ts) > 1:
                m, s = mean_std([t[1] for t in ts])
                steps = ts[0][0][:len(m)]
                ax.plot(steps, m, color=color, label=label)
                ax.fill_between(steps, [a - b for a, b in zip(m, s)],
                                [a + b for a, b in zip(m, s)], color=color, alpha=0.2)
            else:
                ax.plot(ts[0][0], ts[0][1], color=color, label=label)
        ax.set_title(term, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    for ax in list(axes.flat)[len(TERMS):]:
        ax.axis("off")
    fig.suptitle("reward term curves (episode sum): baseline vs projection (mean+-std)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "projection_term_curves.png"), dpi=140)
    plt.close(fig)

    # ---------- md 表 ----------
    def tail3_of(series_list, key):
        vals = [tail3(b[key]) for b in series_list]
        m = sum(vals) / len(vals)
        sd = math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))
        return m, sd

    lines = []
    lines.append("## 尾部指标(最后 3 次 eval 平均)\n")
    lines.append("| 指标 | baseline run1 (n=%d) | 投影版 (n=%d) | 变化 |"
                 % (len(base_series), len(proj_series)))
    lines.append("|---|---|---|---|")
    rows_md = [("RMSE ↓", "rmse"), ("episode reward ↑", "reward"),
               ("avg episode length ↑", "ep_len"), ("energy/step", "energy")]
    for name, key in rows_md:
        m, sd = tail3_of(base_series, key)
        p, psd = tail3_of(proj_series, key)
        delta = (p - m) / m * 100 if m else float("nan")
        lines.append(f"| {name} | {m:.4f} ± {sd:.4f} | {p:.4f} ± {psd:.4f} | {delta:+.1f}% |")

    lines.append("")
    lines.append("## reward 分项尾部(episode 和,最后 3 次 eval 平均)\n")
    lines.append("| 分项 | baseline mean±std | 投影版 mean±std | 变化 |")
    lines.append("|---|---|---|---|")
    for term in TERMS:
        vals = [tail3(term_series(r, term)[1]) for r in base_rows]
        m = sum(vals) / len(vals)
        sd = math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))
        pvals = [tail3(term_series(r, term)[1]) for r in proj_rows]
        p = sum(pvals) / len(pvals)
        psd = math.sqrt(sum((v - p) ** 2 for v in pvals) / len(pvals))
        delta = (p - m) / m * 100 if m else float("nan")
        lines.append(f"| {term} | {m:.2f} ± {sd:.2f} | {p:.2f} ± {psd:.2f} | {delta:+.1f}% |")

    # per-step 口径的 height 项(消除存活时长影响)
    lines.append("")
    lines.append("## height 分项 per-step 口径(÷episode_length,消除存活差异)\n")
    lines.append("| 分项 | baseline mean | 投影版 mean | 变化 |")
    lines.append("|---|---|---|---|")
    for term in ("tracking_height_jnt", "tracking_height_action", "roll_cost"):
        bm, pm = [], []
        for r in base_rows:
            _, v, l = term_series(r, term)
            bm.append(sum(v[-3:]) / sum(l[-3:]))
        for r in proj_rows:
            _, v, l = term_series(r, term)
            pm.append(sum(v[-3:]) / sum(l[-3:]))
        m = sum(bm) / len(bm)
        p = sum(pm) / len(pm)
        delta = (p - m) / m * 100 if m else float("nan")
        lines.append(f"| {term}/step | {m:.4f} | {p:.4f} | {delta:+.1f}% |")

    # 同 seed 配对差值:逐 eval 点的 (proj_s - baseline_s),回答
    # "前期投影曲线是否系统性低于 baseline"——跨 seed 方差远大于处理效应时,
    # 只有配对差值能说明问题。
    lines.append("")
    lines.append("## 同 seed 配对差值(投影 − baseline,mean±std over %d 对)\n"
                 % len(matched_seeds))
    lines.append("| steps(1e8) | Δreward | Δep_len | ΔRMSE |")
    lines.append("|---|---|---|---|")
    n_eval = min(len(base_series_m[0]["steps"]), len(proj_series_m[0]["steps"]))
    for i in range(1, n_eval):
        steps = base_series_m[0]["steps"][i]
        dr = [proj_series_m[j]["reward"][i] - base_series_m[j]["reward"][i]
              for j in range(len(matched_seeds))]
        dl = [proj_series_m[j]["ep_len"][i] - base_series_m[j]["ep_len"][i]
              for j in range(len(matched_seeds))]
        dm = [proj_series_m[j]["rmse"][i] - base_series_m[j]["rmse"][i]
              for j in range(len(matched_seeds))]

        def ms(v):
            m = sum(v) / len(v)
            sd = math.sqrt(sum((x - m) ** 2 for x in v) / len(v))
            return m, sd

        mr, sdr = ms(dr)
        ml, sdl = ms(dl)
        mm, sdm = ms(dm)
        lines.append(f"| {steps:.2f} | {mr:+.0f} ± {sdr:.0f} | "
                     f"{ml:+.1f} ± {sdl:.1f} | {mm:+.3f} ± {sdm:.3f} |")

    with open(os.path.join(args.out, "projection_compare.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nfigures & md written to {args.out}")


if __name__ == "__main__":
    main()
