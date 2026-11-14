#!/usr/bin/env python
"""阶段三确认组的 baseline vs cand010 对比图表生成(供 docs/tuning_run1_results.md 引用)。

数据来源:tuning/run1/trials/{confirm-baseline,cand010}-s{0,1,2}/ 下的
metrics.csv(训练过程 eval 曲线)、reward_terms.json(尾部 3 eval 的 reward 分项)、
config_resolved.json(生效配置)。

输出(--out 目录,默认 tuning/run1/reports):
- stage3_curves.png       训练曲线 2x2(RMSE / episode reward / 存活 / 能耗)
- stage3_reward_terms.png reward 分项分组柱状图
- stage3_reward_term_curves.png reward 分项训练过程曲线(3x4,逐分项 mean±std)
- stage3_compare.md       参数差异表 + reward 分项表(markdown 片段)
"""

import argparse
import csv
import glob
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


def load_metrics_csv(path):
    with open(path, newline="") as f:
        return [
            {k: float(v) for k, v in row.items() if v not in ("", None)}
            for row in csv.DictReader(f)
        ]


def eval_series(rows):
    """每次 eval 一条记录:steps(1e8)、RMSE、reward、存活、能耗/步。"""
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


def mean_std(series_list):
    """逐点对齐求 mean/std(各 seed 的 eval 次数一致)。"""
    n = len(series_list[0])
    mean, std = [], []
    for i in range(n):
        vals = [s[i] for s in series_list]
        m = sum(vals) / len(vals)
        mean.append(m)
        std.append(math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals)))
    return mean, std


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def fmt(v):
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return f"{v:.6g}"
    if isinstance(v, list):
        return "[" + ", ".join(fmt(x) for x in v) + "]"
    return str(v)


def diff_configs(base_cfg, cand_cfg):
    """递归 diff,返回 (path, base, cand) 列表,跳过 meta 与路径类键。"""
    b, c = flatten(base_cfg), flatten(cand_cfg)
    rows = []
    for k in sorted(set(b) | set(c)):
        if k.startswith("meta.") or k.startswith("randomization."):
            continue
        if "checkpoint_path" in k or k.startswith("env._"):
            continue
        if b.get(k) == c.get(k):
            continue
        bv, cv = b.get(k), c.get(k)
        change = ""
        if isinstance(bv, (int, float)) and isinstance(cv, (int, float)) and bv:
            change = f"{(cv - bv) / abs(bv) * 100:+.1f}%"
        rows.append((k, fmt(bv), fmt(cv), change))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", default="tuning/run1/trials")
    ap.add_argument("--base", default="confirm-baseline")
    ap.add_argument("--cand", default="cand010")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="tuning/run1/reports")
    args = ap.parse_args()

    groups = {}
    for name, prefix in (("baseline", args.base), ("cand010", args.cand)):
        trials = []
        for s in range(args.seeds):
            d = os.path.join(args.trials, f"{prefix}-s{s}")
            rows = load_metrics_csv(os.path.join(d, "metrics.csv"))
            trials.append(
                {
                    "rows": rows,
                    "series": eval_series(rows),
                    "terms": json.load(open(os.path.join(d, "reward_terms.json"))),
                    "config": json.load(open(os.path.join(d, "config_resolved.json"))),
                }
            )
        groups[name] = trials

    # ---- 曲线图 ----
    panels = [
        ("rmse", "tracking RMSE (normalized)"),
        ("reward", "episode reward"),
        ("ep_len", "avg episode length (steps)"),
        ("energy", "energy cost / step"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    colors = {"baseline": "tab:gray", "cand010": "tab:red"}
    for ax, (key, ylabel) in zip(axes.flat, panels):
        for name, trials in groups.items():
            steps = trials[0]["series"]["steps"]
            mean, std = mean_std([t["series"][key] for t in trials])
            ax.plot(steps, mean, color=colors[name], label=name)
            ax.fill_between(
                steps,
                [m - s for m, s in zip(mean, std)],
                [m + s for m, s in zip(mean, std)],
                color=colors[name],
                alpha=0.25,
            )
        ax.set_ylabel(ylabel)
        ax.set_xlabel("env steps (1e8)")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(
        f"Stage-3 confirm: {args.base} vs {args.cand}"
        f" (mean +/- std, {args.seeds} seeds, 4e8 steps)"
    )
    fig.tight_layout()
    curves_png = os.path.join(args.out, "stage3_curves.png")
    fig.savefig(curves_png, dpi=140)

    # ---- reward 分项 ----
    term_names = sorted(
        k for k in groups["baseline"][0]["terms"] if not k.endswith("_std")
    )
    table_rows = []
    for t in term_names:
        row = [t]
        for name in ("baseline", "cand010"):
            vals = [tr["terms"][t] for tr in groups[name]]
            m = sum(vals) / len(vals)
            sd = math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))
            row.append((m, sd))
        bm, cm = row[1][0], row[2][0]
        delta = f"{(cm - bm) / abs(bm) * 100:+.1f}%" if bm else ""
        table_rows.append((t, row[1], row[2], delta))

    fig, ax = plt.subplots(figsize=(11, 5))
    chart_terms = [
        t
        for t in term_names
        if any(tr["terms"][t] for g in groups.values() for tr in g)
    ]
    x = list(range(len(chart_terms)))
    w = 0.38
    for i, name in enumerate(("baseline", "cand010")):
        means = [
            sum(tr["terms"][t] for tr in groups[name]) / args.seeds
            for t in chart_terms
        ]
        ax.bar(
            [xi + (i - 0.5) * w for xi in x],
            means,
            width=w,
            color=colors[name],
            label=name,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(chart_terms, rotation=45, ha="right", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("reward term (tail-3 eval mean, log scale)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    terms_png = os.path.join(args.out, "stage3_reward_terms.png")
    fig.savefig(terms_png, dpi=140)

    # ---- reward 分项训练过程曲线(3x4 small multiples)----
    def term_series(rows, term):
        """逐 eval 的分项值,与 eval_series 相同的行过滤,保证 x 轴对齐。"""
        out = []
        for r in rows:
            if ERR2_LIN not in r or EP_LEN not in r or not r[EP_LEN]:
                continue
            out.append(r.get(TERM_PREFIX + term, float("nan")))
        return out

    ncols = 4
    nrows = math.ceil(len(term_names) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(16, 3.2 * nrows), sharex=True)
    for ax, term in zip(axes.flat, term_names):
        for name, trials in groups.items():
            steps = trials[0]["series"]["steps"]
            mean, std = mean_std([term_series(t["rows"], term) for t in trials])
            ax.plot(steps, mean, color=colors[name], label=name)
            ax.fill_between(
                steps,
                [m - s for m, s in zip(mean, std)],
                [m + s for m, s in zip(mean, std)],
                color=colors[name],
                alpha=0.25,
            )
        ax.set_title(term, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    for ax in axes.flat[len(term_names):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("env steps (1e8)")
    fig.suptitle(
        f"reward terms over training: {args.base} vs {args.cand}"
        f" (mean +/- std, {args.seeds} seeds)"
    )
    fig.tight_layout()
    term_curves_png = os.path.join(args.out, "stage3_reward_term_curves.png")
    fig.savefig(term_curves_png, dpi=140)

    # ---- markdown 片段 ----
    param_rows = diff_configs(
        groups["baseline"][0]["config"], groups["cand010"][0]["config"]
    )
    lines = [
        "### 参数差异(baseline vs cand010,config_resolved.json 全量 diff)",
        "",
        "| 参数 | baseline | cand010 | 变化 |",
        "|---|---|---|---|",
    ]
    lines += [f"| `{k}` | {b} | {c} | {chg} |" for k, b, c, chg in param_rows]
    lines += [
        "",
        "### reward 分项对比(尾部 3 eval 均值,3 seeds 的 mean ± std)",
        "",
        "| 分项 | baseline | cand010 | Δ 均值 |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {t} | {bm:.2f} ± {bs:.2f} | {cm:.2f} ± {cs:.2f} | {d} |"
        for t, (bm, bs), (cm, cs), d in table_rows
    ]
    lines += [
        "",
        "![训练曲线](stage3_curves.png)",
        "",
        "![reward 分项](stage3_reward_terms.png)",
        "",
        "![reward 分项训练曲线](stage3_reward_term_curves.png)",
        "",
    ]
    md_path = os.path.join(args.out, "stage3_compare.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))

    print(f"wrote {curves_png}")
    print(f"wrote {terms_png}")
    print(f"wrote {term_curves_png}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
