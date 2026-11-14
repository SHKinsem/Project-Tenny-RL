#!/usr/bin/env python
"""离线重估:用固定 σ 部署口径重测已完成 trial 的达成率,并回写 result.json。

背景(gate-R1 第三轮后修正):reward 达成率若用 trial 自己的 σ 计算,
被搜索的 σ 会移动评估球门(widen σ → 达成率虚高)。本工具对每个 trial 的
最终 ckpt 跑 deploy_evaluate(deterministic + 域随机化 + push + 噪声,
sigma/sigma_wheel 强制 baseline 值),回写:
  result.json: attainment → 固定 σ 实测;原训练 eval 口径移入
               attainment_train_eval;constraints/objective_value 按新口径重算
  base2.json:  attainment_fixed_sigma(逐 seed 重测的均值,作约束参照)

用法(双卡各跑一半):
  CUDA_VISIBLE_DEVICES=0 python tools/reval_fixed_sigma.py --study run2 --shard 0/2
  CUDA_VISIBLE_DEVICES=1 python tools/reval_fixed_sigma.py --study run2 --shard 1/2
"""

import argparse
import glob
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

log = logging.getLogger("reval")


def reval_one(trial_dir: str, base2_ref: dict, energy_budget) -> dict:
    """重测单个 trial,回写 result.json,返回更新后的 summary。"""
    from legwheel.config import apply_config_overrides, default_config
    from legwheel.evaluation.action_rate import deploy_evaluate, latest_ckpt
    from legwheel.tuning.metrics import (
        evaluate_run2_constraints,
        read_metrics_csv,
        run2_soft_penalty,
    )

    result_path = os.path.join(trial_dir, "result.json")
    with open(result_path) as fp:
        result = json.load(fp)
    if result.get("status") != "complete":
        return result
    with open(os.path.join(trial_dir, "overrides.json")) as fp:
        env_overrides = json.load(fp)["env"]

    ckpt = latest_ckpt(os.path.join(trial_dir, "ckpt"))
    env_cfg = apply_config_overrides(default_config(), env_overrides)
    deploy_res = deploy_evaluate(env_cfg, ckpt)

    # 口径切换:训练 eval 达成率(随 σ 移动)降为参照,固定 σ 实测为主口径
    if "attainment_train_eval" not in result and result.get("attainment"):
        result["attainment_train_eval"] = result["attainment"]
    result["attainment"] = deploy_res["attainment"]
    result["action_rate_deploy_per_step"] = deploy_res["action_rate"]

    rows = read_metrics_csv(os.path.join(trial_dir, "metrics.csv"))
    res = evaluate_run2_constraints(
        rows, deploy_res["action_rate"], base2_ref, energy_budget,
        attainment=deploy_res["attainment"])
    result["constraints"] = res["checks"]
    result["violated_constraints"] = res["violated"]
    if result.get("rmse_tail3") is not None:
        penalty = run2_soft_penalty(res["checks"])
        result["constraint_penalty"] = penalty
        result["objective_value"] = result["rmse_tail3"] + penalty
    with open(result_path, "w") as fp:
        json.dump(result, fp, indent=2)
    log.info("%s: rmse=%.4f ar=%.3f syn=%.1f%% violated=%s",
             os.path.basename(trial_dir), result.get("rmse_tail3") or -1,
             deploy_res["action_rate"],
             100 * deploy_res["attainment"].get("tracking_sinyaw", 0),
             res["violated"])
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study", default="run2")
    ap.add_argument("--shard", default="0/1", help="k/N:第 k 片,共 N 片")
    ap.add_argument("--num-envs", type=int, default=1024)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    study_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "tuning", args.study)
    k, n = (int(x) for x in args.shard.split("/"))

    with open(os.path.join(study_dir, "stage0.json")) as fp:
        energy_budget = json.load(fp)["energy_budget"]

    # 1) 先重测 base2(每个 shard 0 负责),生成固定 σ 参照
    base2_path = os.path.join(study_dir, "base2.json")
    with open(base2_path) as fp:
        base2 = json.load(fp)
    if "attainment_fixed_sigma" not in base2:
        if k != 0:
            raise SystemExit("base2.json 尚无 attainment_fixed_sigma,请先跑 shard 0")
        from legwheel.config import apply_config_overrides, default_config
        from legwheel.evaluation.action_rate import deploy_evaluate, latest_ckpt
        from legwheel.tuning.space import BASE2_ENV_OVERRIDES

        env_cfg = apply_config_overrides(
            default_config(), dict(BASE2_ENV_OVERRIDES))
        acc = {}
        for d in sorted(glob.glob(os.path.join(study_dir, "trials", "base2-s*"))):
            res = deploy_evaluate(env_cfg, latest_ckpt(os.path.join(d, "ckpt")),
                                  num_envs=args.num_envs)
            for t, v in res["attainment"].items():
                acc.setdefault(t, []).append(v)
        base2["attainment_fixed_sigma"] = {
            t: sum(vs) / len(vs) for t, vs in acc.items()}
        with open(base2_path, "w") as fp:
            json.dump(base2, fp, indent=2)
        log.info("base2 固定 σ 达成率: %s",
                 {t: round(v, 4) for t, v in base2["attainment_fixed_sigma"].items()})

    base2_ref = {"attainment": base2["attainment_fixed_sigma"]}

    # 2) 分片重测 search2 trials
    trials = sorted(glob.glob(os.path.join(study_dir, "trials", "search2-*")))
    my = [d for i, d in enumerate(trials) if i % n == k]
    log.info("shard %d/%d: %d 个 trial", k, n, len(my))
    for d in my:
        try:
            reval_one(d, base2_ref, energy_budget)
        except Exception:  # noqa: BLE001 - 单个失败不阻塞其他
            log.exception("reval 失败: %s", d)


if __name__ == "__main__":
    main()
