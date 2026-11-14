#!/usr/bin/env python
"""run2 免费复核:固定 σ=0.5 无权重和口径下重排名全部 search2 trials。

判别问题:旧搜索目标(RMSE + ar 罚,且 σ 参与搜索)是否埋掉了好 trial?
方法:对每个 trial 的尾部 3 次 eval,把 exp 型分项按 s^(σ_own/0.5) 换算到
固定 σ=0.5(cmd_scale 在指数比中约去,Jensen 近似但各组同公式),σ 无关
惩罚项原样减入,得到"固定 σ 无权重和/步";与 RMSE(σ 无关)联合散点。

输出(tuning/run2/reports):
- run2_rerank_unweighted.png  散点:RMSE × 固定σ无权重和,颜色=σ,形状=状态
- run2_rerank_unweighted.json 全量排名数据
"""

import csv
import glob
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

TERM_PREFIX = "eval/episode_reward/"
EP_LEN = "eval/avg_episode_length"
ERR2_LIN = "eval/episode_track/err2_lin_vel"
ERR2_ANG = "eval/episode_track/err2_ang_vel"
SIGMA_POS = ["tracking_lin_vel", "tracking_ang_vel", "tracking_height_jnt",
             "tracking_height_action", "tracking_sinyaw",
             "pitch_cost", "roll_cost"]
SIGMAW_POS = ["tracking_wheels_sinposition"]
RAW_NEG = ["energy_cost", "action_rate", "pitch_velocity_penalty",
           "termination_penalty"]

TRIALS = "tuning/run2/trials/search2-*"
REFS = {  # 参照点:base2 R0(1e8,2 seeds,σ=0.5)
    "base2-s0": "tuning/run2/trials/base2-s0",
    "base2-s1": "tuning/run2/trials/base2-s1",
}
NOTE = ["search2-006", "search2-004", "search2-042", "search2-043",
        "search2-000", "search2-026"]  # 标注:被埋亮点 + 两个中选 cand


def load(path):
    with open(path, newline="") as f:
        return [{k: float(v) for k, v in row.items() if v not in ("", None)}
                for row in csv.DictReader(f)]


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


def rmse(r):
    if ERR2_LIN in r and r.get(EP_LEN):
        return math.sqrt((r[ERR2_LIN] + r[ERR2_ANG]) / r[EP_LEN])
    return None


def eval_trial(d):
    mc = os.path.join(d, "metrics.csv")
    if not os.path.exists(mc):
        return None
    rows = load(mc)
    if len(rows) < 3:
        return None
    sigma, sw = 0.5, 0.5
    ovp = os.path.join(d, "overrides.json")
    if os.path.exists(ovp):
        env = json.load(open(ovp)).get("env", {})
        sigma = env.get("reward_config.sigma", sigma)
        sw = env.get("reward_config.sigma_wheel", sw)
    tail = rows[-3:]
    fs = [fixed_sigma_per_step(r, sigma, sw) for r in tail]
    rm = [rmse(r) for r in tail]
    if None in fs or None in rm:
        return None
    completed = os.path.exists(os.path.join(d, "result.json"))
    return {
        "name": os.path.basename(d),
        "steps": rows[-1]["num_steps"] / 1e8,
        "sigma": sigma,
        "sigma_wheel": sw,
        "fixed_sigma_unweighted_per_step": sum(fs) / 3,
        "rmse": sum(rm) / 3,
        "status": "completed" if completed else "pruned",
    }


def main():
    out_dir = "tuning/run2/reports"
    trials = [r for d in sorted(glob.glob(TRIALS)) if (r := eval_trial(d))]
    refs = [r for d in REFS.values() if (r := eval_trial(d))]

    fig, ax = plt.subplots(figsize=(9, 6.5))
    for status, marker, label in (("completed", "o", "completed"),
                                  ("pruned", "x", "pruned (ASHA)")):
        sub = [t for t in trials if t["status"] == status]
        sc = ax.scatter([t["rmse"] for t in sub],
                        [t["fixed_sigma_unweighted_per_step"] for t in sub],
                        c=[t["sigma"] for t in sub], cmap="RdYlGn_r",
                        vmin=0.25, vmax=1.0, marker=marker, s=55,
                        alpha=0.85)
    ax.scatter([r["rmse"] for r in refs],
               [r["fixed_sigma_unweighted_per_step"] for r in refs],
               marker="*", s=260, c="tab:blue", edgecolors="k", zorder=5)
    for t in trials:
        if t["name"] in NOTE:
            off = {"search2-042": (8, -16), "search2-043": (-40, -4)}.get(
                t["name"], (6, 6))
            ax.annotate(t["name"].replace("search2-", "#"),
                        (t["rmse"], t["fixed_sigma_unweighted_per_step"]),
                        textcoords="offset points", xytext=off, fontsize=9)
    fig.colorbar(sc, ax=ax, label="sigma used in search (0.5 = default)")
    ax.set_xlabel("RMSE (tail @1e8, lower better)")
    ax.set_ylabel("unweighted sum / step at fixed sigma=0.5 (higher better)")
    ax.set_title("run2 search2 trials re-ranked under fixed-sigma unweighted criterion")
    handles = [Line2D([], [], marker="o", ls="", c="gray", label="completed"),
               Line2D([], [], marker="x", ls="", c="gray", label="pruned (ASHA)"),
               Line2D([], [], marker="*", ls="", c="tab:blue", ms=14,
                      label="base2 (R0, 1e8)")]
    ax.legend(handles=handles, loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "run2_rerank_unweighted.png"), dpi=130)
    plt.close(fig)

    trials.sort(key=lambda t: -t["fixed_sigma_unweighted_per_step"])
    with open(os.path.join(out_dir, "run2_rerank_unweighted.json"), "w") as fp:
        json.dump({"trials": trials, "refs": refs}, fp, indent=2)
    print("已输出: run2_rerank_unweighted.png, run2_rerank_unweighted.json →",
          out_dir)


if __name__ == "__main__":
    main()
