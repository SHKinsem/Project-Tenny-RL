#!/usr/bin/env python
"""netstruct 确认轮逐 ckpt 实测:P6/P7 的 deploy 口径无权重和曲线(对照 A0)。

与 tools/deploy_curves.py 同口径(deterministic + 域随机化 + push + 噪声 + 固定
σ=0.5),每 2 档取 1 + 末档共 11 点;baseline A0 不重测,直接合并
tuning/run2/reports/deploy_curves.s{0,1}.json 里的 run1-baseline 数据。

用法:
  python tools/netstruct_curves.py --shard 0   # GPU0 跑一半
  python tools/netstruct_curves.py --shard 1   # GPU1 跑另一半
  python tools/netstruct_curves.py --plot      # 数据齐后出图(自动并入 baseline)
"""

import argparse
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GROUPS = {
    "P6": "tuning/netstruct/trials/P6-cs*",
    "P7": "tuning/netstruct/trials/P7-cs*",
}
BASELINE_SHARDS = "tuning/run2/reports/deploy_curves.s%d.json"
OUT_PNG = "tuning/netstruct/reports/deploy_curves.png"


def shard_json(shard):
    return f"tuning/netstruct/reports/curves.s{shard}.json"


def list_runs():
    runs = []
    for name, pat in GROUPS.items():
        for d in sorted(glob.glob(pat)):
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
    os.makedirs(os.path.dirname(sj), exist_ok=True)
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
                    "attainment": res["attainment"],
                    "unweighted_sum_per_step":
                        sum(res["attainment"].values()) - res["action_rate"],
                }
                print(f"[shard{shard}] {key} uw="
                      f"{cache[key]['unweighted_sum_per_step']:.3f}", flush=True)
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
    # 并入 run1-baseline(已测,同口径)
    for shard in (0, 1):
        bs = BASELINE_SHARDS % shard
        if os.path.exists(bs):
            for k, v in json.load(open(bs)).items():
                if v["group"] == "run1-baseline":
                    cache[f"baseline|{k}"] = {**v, "group": "A0"}

    colors = {"A0": "gray", "P6": "tab:red", "P7": "tab:green"}
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
    fig.suptitle("netstruct confirm: deploy-eval curves (deterministic + rand + push + noise, σ=0.5)",
                 fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
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
