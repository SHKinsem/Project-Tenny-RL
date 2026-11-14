"""单 trial 训练进程入口:python -m legwheel.tuning.worker --input <input.json>

由 scripts/tune.py 以子进程拉起(计划 6.5.3 子进程隔离):
- 崩溃隔离:单 trial OOM/JIT 失败只死本进程,study 记 FAILED,搜索继续;
- 显存干净:进程退出即释放全部 GPU 显存,trial 间无碎片累积;
- 留痕(计划 2.5):config_resolved.json / overrides.json / metrics.csv /
  reward_terms.json / result.json / worker.log / worker_stdout.log;
- 剪枝:父进程轮询 metrics.csv 调 trial.report/should_prune,判定剪枝时发
  SIGTERM,本进程落盘 status.json(pruned) 后退出。
"""

import argparse
import csv
import functools
import logging
import os
import signal
import sys
import time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="trial 输入 JSON(父进程落盘)")
    args = ap.parse_args()

    import json

    with open(args.input) as fp:
        spec = json.load(fp)

    trial_dir = spec["trial_dir"]
    os.makedirs(trial_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        filename=os.path.join(trial_dir, "worker.log"),
        filemode="w",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )

    from legwheel.randomization import domain_randomize
    from legwheel.training.ppo import curriculum_learning
    from legwheel.tuning.metrics import reward_terms, summarize_rows
    from legwheel.tuning.resolve import build_resolved_config, git_info, write_json

    seed = int(spec["seed"])
    num_evals = int(spec["num_evals"])
    maxstep = float(spec["maxstep"])
    env_overrides = spec.get("env_overrides") or {}
    ppo_overrides = spec.get("ppo_overrides") or {}

    started = time.time()
    resolved, env_cfg, rand_cfg = build_resolved_config(
        env_overrides, ppo_overrides, spec.get("randomization_overrides"),
        num_evals, maxstep, seed,
        meta=dict(
            git_info(),
            seed=seed,
            maxstep=maxstep,
            num_evals=num_evals,
            gpu=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        ),
    )
    ckpt_dir = os.path.abspath(os.path.join(trial_dir, "ckpt"))
    resolved["ppo"]["save_checkpoint_path"] = ckpt_dir
    write_json(os.path.join(trial_dir, "config_resolved.json"), resolved)
    write_json(os.path.join(trial_dir, "overrides.json"), {
        "env": env_overrides,
        "ppo": ppo_overrides,
        "randomization": spec.get("randomization_overrides") or {},
    })

    randomizer = functools.partial(domain_randomize, cfg=rand_cfg)

    # --- metrics.csv:brax 每次 eval 触发 progress 回调,追加一行 ---
    csv_path = os.path.join(trial_dir, "metrics.csv")
    eval_rows: list[dict] = []
    csv_state = {"writer": None, "file": None}

    def progress_fn(num_steps, metrics) -> None:
        row = {"num_steps": int(num_steps)}
        for k, v in metrics.items():
            try:
                row[k] = float(v)
            except (TypeError, ValueError):
                pass
        eval_rows.append(row)
        if csv_state["writer"] is None:
            fields = ["num_steps"] + sorted(k for k in row if k != "num_steps")
            csv_state["file"] = open(csv_path, "w", newline="")  # noqa: SIM115
            csv_state["writer"] = csv.DictWriter(
                csv_state["file"], fieldnames=fields, extrasaction="ignore")
            csv_state["writer"].writeheader()
        csv_state["writer"].writerow(row)
        csv_state["file"].flush()

    def on_sigterm(signum, frame) -> None:  # noqa: ARG001
        write_json(os.path.join(trial_dir, "status.json"), {
            "status": "pruned",
            "at_step": eval_rows[-1]["num_steps"] if eval_rows else 0,
            "n_evals": len(eval_rows),
        })
        if csv_state["file"] is not None:
            csv_state["file"].flush()
        os._exit(0)

    signal.signal(signal.SIGTERM, on_sigterm)

    logging.info("trial start: seed=%d num_evals=%d maxstep=%g", seed, num_evals, maxstep)
    try:
        curriculum_learning(
            iter_num=1,
            num_evals=num_evals,
            maxstep_candidate=maxstep,
            save_checkpoint_path=ckpt_dir,
            env_cfg=env_cfg,
            ppo_overrides=ppo_overrides,
            seed=seed,
            progress_fn=progress_fn,
            randomization_fn=randomizer,
        )
    except Exception as exc:  # noqa: BLE001 - 崩溃隔离:记 FAILED 后退出
        logging.exception("trial failed")
        write_json(os.path.join(trial_dir, "status.json"), {
            "status": "failed",
            "error": repr(exc),
            "n_evals": len(eval_rows),
            "wall_time_s": time.time() - started,
        })
        raise SystemExit(1)

    # --- 汇总:主指标(尾部 3 eval 归一化 RMSE)+ 约束/辅助指标 ---
    from legwheel.config import default_config

    summary = summarize_rows(eval_rows, default_config().reward_config)
    summary["status"] = "complete"
    summary["wall_time_s"] = time.time() - started

    # --- run2:达成率 + 部署口径 action_rate 加测 + 硬约束判定 ---
    from legwheel.tuning.metrics import (
        evaluate_run2_constraints,
        per_step_values,
        tail3_attainment,
    )

    if eval_rows:
        # 训练 eval(stochastic + trial 自身 σ)口径,仅供参照,不作约束判定
        summary["attainment_train_eval"] = tail3_attainment(eval_rows)
        # 仪表板:stochastic 报表口径 per-step(仅观察,不作判定)
        summary["per_step_stochastic"] = per_step_values(eval_rows[-1])

    ar_deploy = None
    att_fixed = None
    if spec.get("measure_deploy_action_rate"):
        from legwheel.evaluation.action_rate import deploy_evaluate, latest_ckpt

        ckpt = latest_ckpt(ckpt_dir)
        if ckpt is None:
            logging.warning("未找到 ckpt 子目录,跳过部署口径加测")
        else:
            try:
                deploy_res = deploy_evaluate(env_cfg, ckpt)
                ar_deploy = deploy_res["action_rate"]
                # 固定 σ 实测达成率(不随 trial 的 σ 搜索值移动,gate-R1 修正);
                # 作为约束判定与报告的 attainment 主口径
                att_fixed = deploy_res["attainment"]
                summary["attainment"] = att_fixed
            except Exception:  # noqa: BLE001 - 测量失败按违反约束保守处理
                logging.exception("deploy 加测失败")
    summary["action_rate_deploy_per_step"] = ar_deploy

    constraints_ref = spec.get("constraints_ref")
    if constraints_ref is not None:
        from legwheel.tuning.metrics import run2_soft_penalty

        res = evaluate_run2_constraints(
            eval_rows, ar_deploy, constraints_ref.get("base2"),
            constraints_ref.get("energy_budget"), attainment=att_fixed)
        summary["constraints"] = res["checks"]
        if res["violated"]:
            summary["violated_constraints"] = res["violated"]
        # gate-R1 决议:R1 搜索用软约束(rmse_tail3 保留真实值,目标值加罚),
        # 只有死亡/发散(无 RMSE)才由父进程记 WORST;R2 判定仍用硬门槛
        if summary.get("rmse_tail3") is not None:
            penalty = run2_soft_penalty(res["checks"])
            summary["constraint_penalty"] = penalty
            summary["objective_value"] = summary["rmse_tail3"] + penalty
            if penalty > 0:
                logging.info("软约束罚 %.4f(违反: %s)→ objective %.4f",
                             penalty, res["violated"], summary["objective_value"])

    write_json(os.path.join(trial_dir, "result.json"), summary)
    if eval_rows:
        write_json(os.path.join(trial_dir, "reward_terms.json"),
                   reward_terms(eval_rows[-1]))
    if csv_state["file"] is not None:
        csv_state["file"].close()
    logging.info("trial done: %s", {k: v for k, v in summary.items() if k != "rmse_curve"})


if __name__ == "__main__":
    main()
