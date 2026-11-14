#!/usr/bin/env python
"""deploy 口径逐 checkpoint 实测:5 个 4e8 确认组的无权重总 reward 曲线。

对每个 trial 的存档 ckpt(每 2 档取 1 + 末档,共 11 点)跑 deploy_evaluate
(deterministic + 域随机化 + push + 噪声 + 固定 σ=0.5),输出:
- tuning/run2/reports/deploy_curves.json  逐点数据(可续跑)
- tuning/run2/reports/deploy_curves.png   无权重和/步 + ar 部署口径曲线

用法:
  python tools/deploy_curves.py --shard 0   # GPU0 跑一半 run
  python tools/deploy_curves.py --shard 1   # GPU1 跑另一半
  python tools/deploy_curves.py --plot      # 数据齐后出图
"""

import argparse
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GROUPS = {
    "run1-baseline": "tuning/run1/trials/confirm-baseline-s{0,1,2}",
    "base2": "tuning/run2/trials/confirm2-base2-s{0,1,2}",
    "cand000": "tuning/run2/trials/cand000-s{0,1,2}",
    "cand026": "tuning/run2/trials/cand026-s{0,1,2}",
    "cand006": "tuning/run2/trials/cand006-s{0,1,2}",
}
OUT_JSON = "tuning/run2/reports/deploy_curves.json"


def shard_json(shard):
    return f"tuning/run2/reports/deploy_curves.s{shard}.json"
OUT_PNG = "tuning/run2/reports/deploy_curves.png"


def list_runs():
    runs = []
    for name, pat in GROUPS.items():
        for d in sorted(glob.glob(pat.replace("{0,1,2}", "*"))):
            runs.append((name, d))
    return runs


def pick_ckpts(d):
    subs = sorted(glob.glob(os.path.join(d, "ckpt", "[0-9]" * 12)))
    picked = subs[::2]
    if subs[-1] not in picked:
        picked.append(subs[-1])
    return picked


def measure(shard: int):
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.1")
    from legwheel.evaluation.action_rate import deploy_evaluate
    from legwheel.tuning.resolve import build_resolved_config

    sj = shard_json(shard)
    cache = json.load(open(sj)) if os.path.exists(sj) else {}
    runs = list_runs()
    todo = [(name, d) for i, (name, d) in enumerate(runs) if i % 2 == shard]
    for name, d in todo:
        ov = json.load(open(os.path.join(d, "overrides.json")))
        _, env_cfg, _ = build_resolved_config(
            ov.get("env", {}), ov.get("ppo", {}), {}, 20, 4e8, 0)
        for ck in pick_ckpts(d):
            key = f"{name}|{os.path.basename(d)}|{os.path.basename(ck)}"
            if key in cache:
                continue
            try:
                res = deploy_evaluate(env_cfg, os.path.abspath(ck))
                cache[key] = {
                    "group": name, "run": os.path.basename(d),
                    "steps": int(os.path.basename(ck)),
                    "action_rate": res["action_rate"],
                    "avg_episode_length": res["avg_episode_length"],
                    "nan_steps": res.get("nan_steps", 0),
                    "attainment": res["attainment"],
                    "unweighted_sum_per_step":
                        sum(res["attainment"].values()) - res["action_rate"],
                }
                print(f"[shard{shard}] {key} uw="
                      f"{cache[key]['unweighted_sum_per_step']:.3f}",
                      flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[shard{shard}] {key} FAILED: {e}", flush=True)
            with open(sj, "w") as fp:
                json.dump(cache, fp)
    print(f"[shard{shard}] done", flush=True)


def plot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cache = {}
    for shard in (0, 1):
        sj = shard_json(shard)
        if os.path.exists(sj):
            cache.update(json.load(open(sj)))
    colors = {"run1-baseline": "gray", "base2": "tab:blue",
              "cand000": "tab:orange", "cand026": "tab:green",
              "cand006": "tab:red"}
    data = {}
    for rec in cache.values():
        data.setdefault(rec["group"], {}).setdefault(
            rec["run"], []).append(rec)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for name, runs in data.items():
        xs_all, uw, ar = [], [], []
        for recs in runs.values():
            recs.sort(key=lambda r: r["steps"])
            xs_all.append([r["steps"] / 1e8 for r in recs])
            uw.append([r["unweighted_sum_per_step"] for r in recs])
            ar.append([r["action_rate"] for r in recs])
        xs = xs_all[0]
        for ax, ys, title in (
                (axes[0], uw, "deploy unweighted sum / step (fixed σ=0.5)"),
                (axes[1], ar, "deploy action_rate / step")):
            n = min(len(y) for y in ys)
            mean = [sum(y[i] for y in ys) / len(ys) for i in range(n)]
            std = [math.sqrt(sum((y[i] - mean[i]) ** 2 for y in ys) / len(ys))
                   for i in range(n)]
            ax.plot(xs[:n], mean, label=name, color=colors[name])
            ax.fill_between(xs[:n], [m - s for m, s in zip(mean, std)],
                            [m + s for m, s in zip(mean, std)],
                            color=colors[name], alpha=0.15)
    axes[0].legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel("steps (1e8)")
        ax.grid(alpha=0.3)
    fig.suptitle("deploy-eval curves: deterministic + domain rand + push + noise, fixed sigma=0.5", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130)
    print("已输出:", OUT_PNG)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard", type=int, default=None, choices=(0, 1))
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    if args.plot:
        plot()
    elif args.shard is not None:
        measure(args.shard)
    else:
        ap.error("需要 --shard 0/1 或 --plot")
