#!/usr/bin/env python3
"""调参流水线入口(见 docs/hyperparam_tuning_plan.md)。

子命令(每阶段一次调用,阶段间即人工核准门 gate-1/2/3):
    python scripts/tune.py baseline --study NAME            # 阶段 0:噪声底
    python scripts/tune.py screen   --study NAME [--dry-run] # 阶段 1:PB 筛选
    python scripts/tune.py search   --study NAME [--dry-run] # 阶段 2:TPE+ASHA
    python scripts/tune.py confirm  --study NAME             # 阶段 3:全量确认

存储结构(计划 2.5):tuning/<study>/{study.db, gpu*.lock, trials/<id>/, reports/};
所有 worker 共享 tuning/jax_cache(JAX 编译缓存,计划工程改造 #6)。
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import deque

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from legwheel.tuning.metrics import (  # noqa: E402
    read_metrics_csv,
    rmse_curve,
)
from legwheel.tuning.scheduler import GPUSlots, launch_worker  # noqa: E402
from legwheel.tuning.space import (  # noqa: E402
    ENERGY_BUDGET_RATIO,
    FACTORS,
    MIN_EPISODE_LENGTH,
    WORST,
    main_effects,
    optuna_distributions,
    pb_design,
    pb_row_values,
    split_overrides,
)

JAX_CACHE_DIR = os.path.join(_PROJECT_ROOT, "tuning", "jax_cache")

log = logging.getLogger("tune")


# ---------------------------------------------------------------------------
# 通用调度引擎
# ---------------------------------------------------------------------------


def append_runs_jsonl(study_dir: str, row: dict) -> None:
    with open(os.path.join(study_dir, "runs.jsonl"), "a") as fp:
        fp.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def read_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path) as fp:
        return json.load(fp)


def collect_trial_row(study_dir: str, stage: str, trial_id: str, rc: int) -> dict:
    """worker 结束后汇总一个 trial 的落盘痕迹为一行索引。"""
    trial_dir = os.path.join(study_dir, "trials", str(trial_id))
    result = read_json(os.path.join(trial_dir, "result.json")) or {}
    status = read_json(os.path.join(trial_dir, "status.json")) or {}
    overrides = read_json(os.path.join(trial_dir, "overrides.json")) or {}
    row = {
        "trial_id": trial_id,
        "stage": stage,
        "exit_code": rc,
        "status": result.get("status") or status.get("status") or "failed",
    }
    for k in ("rmse_tail3", "avg_episode_length_tail3", "energy_per_step_tail3",
              "episode_reward_tail3", "n_evals", "wall_time_s", "seed",
              "fixed_score", "action_rate_deploy_per_step",
              "objective_value", "constraint_penalty"):
        if result.get(k) is not None:
            row[k] = result[k]
    for k in ("attainment", "attainment_train_eval", "violated_constraints",
              "constraints"):
        if result.get(k) is not None:
            row[k] = result[k]
    if status.get("status") == "pruned":
        row["status"] = "pruned"
        row["n_evals"] = status.get("n_evals")
    if status.get("status") == "failed":
        row["error"] = status.get("error")
    row["overrides"] = overrides
    return row


def run_scheduler(study_dir: str, gpu_ids, next_job, poll_interval: float = 10.0):
    """双卡调度循环(计划 6.5.4):每卡 1 trial,空闲卡锁取得后才拉起 worker。

    next_job(): 返回 {"trial_id", "spec", "poll"(可选), "on_done"} 或 None(无活)。
    """
    slots = GPUSlots(study_dir, gpu_ids)
    running = {}  # gpu -> {"proc", "job"}
    exhausted = False
    try:
        while True:
            for gpu in slots.gpu_ids:
                if gpu in running or exhausted:
                    continue
                if not slots.acquire(gpu):
                    continue
                job = next_job()
                if job is None:
                    exhausted = True
                    slots.release(gpu)
                    continue
                # 把 trial_dir 写回 job["spec"](poll 回调据此读 metrics.csv;
                # launch_worker 内只改副本,不会回写)
                job["spec"]["trial_dir"] = os.path.abspath(
                    os.path.join(study_dir, "trials", str(job["trial_id"])))
                proc = launch_worker(
                    gpu, study_dir, job["trial_id"], job["spec"],
                    JAX_CACHE_DIR, sys.executable)
                running[gpu] = {"proc": proc, "job": job}
            for gpu in list(running):
                st = running[gpu]
                job = st["job"]
                if job.get("poll") is not None:
                    try:
                        job["poll"](job, st["proc"])
                    except Exception:  # noqa: BLE001 - 轮询异常不应杀死调度器
                        log.exception("poll callback failed for %s", job["trial_id"])
                rc = st["proc"].poll()
                if rc is not None:
                    row = collect_trial_row(
                        study_dir, job.get("stage", "?"), job["trial_id"], rc)
                    if job.get("on_done") is not None:
                        job["on_done"](job, row)
                    append_runs_jsonl(study_dir, row)
                    log.info("trial %s finished: status=%s rmse=%s",
                             job["trial_id"], row["status"],
                             row.get("rmse_tail3"))
                    slots.release(gpu)
                    del running[gpu]
            if exhausted and not running:
                break
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        log.warning("中断:终止所有运行中的 worker")
        for st in running.values():
            st["proc"].terminate()
        raise
    finally:
        slots.release_all()


def static_jobs(jobs: list):
    q = deque(jobs)

    def next_job():
        return q.popleft() if q else None

    return next_job


# ---------------------------------------------------------------------------
# 子命令
# ---------------------------------------------------------------------------


def _spec(seed, num_evals, maxstep, env_ov=None, ppo_ov=None):
    return {
        "seed": seed,
        "num_evals": num_evals,
        "maxstep": maxstep,
        "env_overrides": env_ov or {},
        "ppo_overrides": ppo_ov or {},
    }


def cmd_baseline(args) -> None:
    """阶段 0:噪声底估计(baseline ×N seeds,计划第 2 节)。"""
    from legwheel.tuning.report import write_stage0

    study_dir = _study_dir(args)
    seeds = [int(s) for s in args.seeds.split(",")]
    jobs = [
        {
            "trial_id": f"baseline-s{s}",
            "stage": "baseline",
            "spec": _spec(s, args.num_evals, args.maxstep),
        }
        for s in seeds
    ]
    run_scheduler(study_dir, _gpus(args), static_jobs(jobs), args.poll_interval)
    results = {}
    for s in seeds:
        r = read_json(os.path.join(
            study_dir, "trials", f"baseline-s{s}", "result.json"))
        if r is None:
            raise RuntimeError(f"baseline seed {s} 无 result.json,无法估计噪声底")
        results[s] = r
    out = write_stage0(study_dir, results)
    print(f"[baseline] σ_noise={out['sigma_noise']:.4f}, "
          f"RMSE 均值={out['baseline_rmse_mean']:.4f}, "
          f"能耗预算={out['energy_budget']:.4f} "
          f"(报告: {study_dir}/reports/stage0_noise.md)")


def cmd_screen(args) -> None:
    """阶段 1:Plackett-Burman 主效应筛选(计划第 3 节)。"""
    from legwheel.tuning.report import write_stage1

    study_dir = _study_dir(args)
    factors = FACTORS
    X = pb_design(factors)
    n_runs = X.shape[0]
    print(f"[screen] 因子数 k={len(factors)},PB run 数 N={n_runs}")

    if args.dry_run:
        for i in range(n_runs):
            values = pb_row_values(factors, X[i])
            env_ov, ppo_ov = split_overrides(factors, values)
            print(f"--- run {i}: env={json.dumps(env_ov)} ppo={json.dumps(ppo_ov)}")
        return

    stage0 = read_json(os.path.join(study_dir, "stage0.json"))
    if stage0 is None:
        raise RuntimeError("未找到 stage0.json,请先运行 tune.py baseline 估计噪声底")
    sigma_noise = stage0["sigma_noise"]

    jobs = []
    for i in range(n_runs):
        values = pb_row_values(factors, X[i])
        env_ov, ppo_ov = split_overrides(factors, values)
        jobs.append({
            "trial_id": f"screen-{i:03d}",
            "stage": "screen",
            "spec": _spec(0, args.num_evals, args.maxstep, env_ov, ppo_ov),
        })
    run_scheduler(study_dir, _gpus(args), static_jobs(jobs), args.poll_interval)

    ys, run_table = {}, []
    for i in range(n_runs):
        r = read_json(os.path.join(
            study_dir, "trials", f"screen-{i:03d}", "result.json"))
        if r is None or r.get("rmse_tail3") is None:
            ys[i] = WORST
            run_table.append({"run": i, "y": WORST, "status": "死亡/失败(记 WORST)"})
        else:
            ys[i] = r["rmse_tail3"]
            run_table.append({"run": i, "y": r["rmse_tail3"], "status": "完成"})
    import numpy as np

    effects = main_effects(factors, X, np.array([ys[i] for i in range(n_runs)]))
    write_stage1(study_dir, factors, effects, ys, run_table, sigma_noise, n_runs,
                 top_k=args.top_k)
    print(f"[screen] 报告: {study_dir}/reports/stage1_screening.md; "
          f"gate-1 请核准 {study_dir}/reports/stage1_space.yaml 后再运行 search")


def cmd_search(args) -> None:
    """阶段 2:Optuna TPE + ASHA 剪枝(计划第 4 节,study 统一 direction=minimize)。"""
    import optuna

    from legwheel.tuning.report import write_stage2
    from legwheel.tuning.space import load_search_space_yaml

    study_dir = _study_dir(args)
    yaml_path = os.path.join(study_dir, "reports", "stage1_space.yaml")
    if args.dry_run and not os.path.exists(yaml_path):
        factors = FACTORS  # 无 gate-1 产物时用全因子验证采样/解析链路
    else:
        factors = load_search_space_yaml(yaml_path)
    dists = optuna_distributions(factors)
    print(f"[search] 搜索因子数: {len(factors)}: "
          f"{sorted(f.name for f in factors)}")

    if args.dry_run:
        sampler = optuna.samplers.TPESampler(seed=0)
        tmp = optuna.create_study(sampler=sampler, direction="minimize")
        for _ in range(3):
            trial = tmp.ask(fixed_distributions=dists)
            env_ov, ppo_ov = split_overrides(factors, trial.params)
            print(f"--- sample: params={trial.params}\n"
                  f"    env={json.dumps(env_ov)}\n    ppo={json.dumps(ppo_ov)}")
        return

    stage0 = read_json(os.path.join(study_dir, "stage0.json")) or {}
    energy_budget = stage0.get("energy_budget")
    if energy_budget is None:
        log.warning("无 stage0.json,能耗硬约束停用")

    sampler = optuna.samplers.TPESampler(
        n_startup_trials=args.n_startup, seed=0)
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=args.min_resource, max_resource=args.num_evals,
        reduction_factor=3)
    storage = f"sqlite:///{os.path.join(study_dir, 'study.db')}"
    study = optuna.create_study(
        study_name="stage2", storage=storage, sampler=sampler, pruner=pruner,
        direction="minimize", load_if_exists=True)

    terminal = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)

    def next_job():
        n_done = len(study.get_trials(deepcopy=False, states=terminal))
        if n_done >= args.n_trials:
            return None
        trial = study.ask(fixed_distributions=dists)
        env_ov, ppo_ov = split_overrides(factors, trial.params)
        trial_id = f"search-{trial.number:03d}"
        state = {"n_reported": 0}

        def poll(job, proc):
            rows = read_metrics_csv(os.path.join(
                job["spec"]["trial_dir"], "metrics.csv"))
            while state["n_reported"] < len(rows):
                state["n_reported"] += 1
                # rung 处用已累计 eval 的均值(早期没有尾部 3 点,计划第 2 节)
                curve = rmse_curve(rows[: state["n_reported"]])
                if not curve:
                    continue
                trial.report(sum(curve) / len(curve), step=state["n_reported"])
                if trial.should_prune():
                    log.info("trial %d pruned at eval %d",
                             trial.number, state["n_reported"])
                    proc.terminate()
                    return

        def on_done(job, row):
            if row["status"] == "complete" and row.get("rmse_tail3") is not None:
                value = row["rmse_tail3"]
                # 硬约束:存活与能耗,违反即最差值 WORST(计划第 2 节)
                if (row.get("avg_episode_length_tail3") or 0) < MIN_EPISODE_LENGTH:
                    value = WORST
                if (energy_budget is not None
                        and (row.get("energy_per_step_tail3") or 0) > energy_budget):
                    value = WORST
                study.tell(trial, value)
            elif row["status"] == "pruned":
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            else:
                study.tell(trial, state=optuna.trial.TrialState.FAIL)

        return {
            "trial_id": trial_id,
            "stage": "search",
            "spec": _spec(0, args.num_evals, args.maxstep, env_ov, ppo_ov),
            "poll": poll,
            "on_done": on_done,
        }

    run_scheduler(study_dir, _gpus(args), next_job, args.poll_interval)

    # gate-2 审查材料:top-k + 分项对比 + 异常标记 + 可视化
    trial_rows = []
    for t in study.get_trials(deepcopy=False, states=terminal):
        trial_id = f"search-{t.number:03d}"
        r = collect_trial_row(study_dir, "search", trial_id, 0)
        r["params"] = t.params
        terms = read_json(os.path.join(
            study_dir, "trials", trial_id, "reward_terms.json"))
        if terms:
            r["reward_terms"] = terms
        trial_rows.append(r)
    baseline = _baseline_summary(study_dir)
    write_stage2(study, study_dir, trial_rows, baseline)
    print(f"[search] 报告: {study_dir}/reports/stage2_topk.md; "
          f"gate-2 核准后再运行 confirm")


def cmd_confirm(args) -> None:
    """阶段 3:全量确认 top-k + baseline × 3 seeds(计划第 5 节)。"""
    import optuna

    from legwheel.tuning.report import write_stage3
    from legwheel.tuning.space import load_search_space_yaml

    study_dir = _study_dir(args)
    factors = load_search_space_yaml(
        os.path.join(study_dir, "reports", "stage1_space.yaml"))
    storage = f"sqlite:///{os.path.join(study_dir, 'study.db')}"
    study = optuna.load_study(study_name="stage2", storage=storage)
    complete = [t for t in study.get_trials(
        deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))]
    if not complete:
        raise RuntimeError("study 中无完成 trial,请先运行 search")
    top = sorted(complete, key=lambda t: t.value)[: args.top_k]
    seeds = [int(s) for s in args.seeds.split(",")]

    jobs = []
    names = {}
    for t in top:
        env_ov, ppo_ov = split_overrides(factors, t.params)
        name = f"cand{t.number:03d}"
        names[name] = t.params
        for s in seeds:
            jobs.append({
                "trial_id": f"{name}-s{s}",
                "stage": "confirm",
                "spec": _spec(s, args.num_evals, args.maxstep, env_ov, ppo_ov),
            })
    for s in seeds:
        jobs.append({
            "trial_id": f"confirm-baseline-s{s}",
            "stage": "confirm",
            "spec": _spec(s, args.num_evals, args.maxstep),
        })
    print(f"[confirm] 候选: {list(names)} + baseline,seeds={seeds},"
          f" maxstep={args.maxstep:g}")
    run_scheduler(study_dir, _gpus(args), static_jobs(jobs), args.poll_interval)

    candidates = {}
    for name in names:
        runs = [read_json(os.path.join(
            study_dir, "trials", f"{name}-s{s}", "result.json")) or {}
            for s in seeds]
        candidates[name] = runs
    baseline_runs = [read_json(os.path.join(
        study_dir, "trials", f"confirm-baseline-s{s}", "result.json")) or {}
        for s in seeds]
    verdict = write_stage3(study_dir, candidates, baseline_runs)
    print(f"[confirm] {verdict}(报告: {study_dir}/reports/stage3_confirm.md)")


# ---------------------------------------------------------------------------
# run2 子命令(docs/hyperparam_tuning_plan_run2.md 第 6 节)
# ---------------------------------------------------------------------------


def _base2_ref(study_dir: str) -> dict:
    ref = read_json(os.path.join(study_dir, "base2.json"))
    if ref is None:
        raise RuntimeError("未找到 base2.json,请先运行 tune.py base2(R0)")
    return ref


def cmd_base2(args) -> None:
    """R0:base2 建立与验证(cand010 增益移植回 baseline 网络,计划 6.R0)。"""
    from legwheel.tuning.report import write_run2_base2
    from legwheel.tuning.space import BASE2_ENV_OVERRIDES, BASE2_PPO_OVERRIDES

    study_dir = _study_dir(args)
    seeds = [int(s) for s in args.seeds.split(",")]
    jobs = [
        {
            "trial_id": f"base2-s{s}",
            "stage": "base2",
            "spec": dict(
                _spec(s, args.num_evals, args.maxstep,
                      dict(BASE2_ENV_OVERRIDES), dict(BASE2_PPO_OVERRIDES)),
                measure_deploy_action_rate=True,
            ),
        }
        for s in seeds
    ]
    run_scheduler(study_dir, _gpus(args), static_jobs(jobs), args.poll_interval)
    results = {}
    for s in seeds:
        r = read_json(os.path.join(study_dir, "trials", f"base2-s{s}", "result.json"))
        if r is None:
            raise RuntimeError(f"base2 seed {s} 无 result.json")
        results[s] = r
    out = write_run2_base2(study_dir, results)
    print(f"[base2] RMSE 均值={out['rmse_mean']:.4f}(逐 seed: {out['rmse_per_seed']})\n"
          f"        报告: {study_dir}/reports/run2_base2.md → gate-R0 人工确认后再跑 search2")


def cmd_search2(args) -> None:
    """R1:reward shaping 搜索(TPE + ASHA,主指标 + 分项硬约束,计划 6.R1)。"""
    import optuna

    from legwheel.tuning.report import write_run2_search
    from legwheel.tuning.space import (
        BASE2_PPO_OVERRIDES,
        RUN2_FACTORS,
    )

    study_dir = _study_dir(args)
    factors = RUN2_FACTORS
    dists = optuna_distributions(factors)
    print(f"[search2] 搜索因子数: {len(factors)}: "
          f"{sorted(f.name for f in factors)}")

    if args.dry_run:
        sampler = optuna.samplers.TPESampler(seed=0)
        tmp = optuna.create_study(sampler=sampler, direction="minimize")
        for _ in range(3):
            trial = tmp.ask(fixed_distributions=dists)
            env_ov, ppo_ov = split_overrides(factors, trial.params)
            print(f"--- sample: params={trial.params}\n"
                  f"    env={json.dumps(env_ov)}\n"
                  f"    ppo={json.dumps({**ppo_ov, **BASE2_PPO_OVERRIDES})}")
        return

    base2 = _base2_ref(study_dir)
    stage0 = read_json(os.path.join(study_dir, "stage0.json")) or {}
    energy_budget = stage0.get("energy_budget")
    if energy_budget is None:
        log.warning("无 stage0.json,能耗硬约束停用")
    constraints_ref = {"base2": base2, "energy_budget": energy_budget}

    sampler = optuna.samplers.TPESampler(
        n_startup_trials=args.n_startup, seed=0)
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=args.min_resource, max_resource=args.num_evals,
        reduction_factor=3)
    storage = f"sqlite:///{os.path.join(study_dir, 'study2.db')}"
    study = optuna.create_study(
        study_name="search2", storage=storage, sampler=sampler, pruner=pruner,
        direction="minimize", load_if_exists=True)

    terminal = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)

    def next_job():
        n_done = len(study.get_trials(deepcopy=False, states=terminal))
        if n_done >= args.n_trials:
            return None
        trial = study.ask(fixed_distributions=dists)
        env_ov, ppo_ov = split_overrides(factors, trial.params)
        # base2 的 4 个增益锚定:lr/updates 固定,kp/notsmoothing 由因子随行
        ppo_ov = {**ppo_ov, **BASE2_PPO_OVERRIDES}
        trial_id = f"search2-{trial.number:03d}"
        state = {"n_reported": 0}

        def poll(job, proc):
            rows = read_metrics_csv(os.path.join(
                job["spec"]["trial_dir"], "metrics.csv"))
            while state["n_reported"] < len(rows):
                state["n_reported"] += 1
                curve = rmse_curve(rows[: state["n_reported"]])
                if not curve:
                    continue
                trial.report(sum(curve) / len(curve), step=state["n_reported"])
                if trial.should_prune():
                    log.info("trial %d pruned at eval %d",
                             trial.number, state["n_reported"])
                    proc.terminate()
                    return

        def on_done(job, row):
            # gate-R1 决议:软约束——worker 落 objective_value(rmse + 罚),
            # 只有死亡/发散(无 RMSE)记 WORST
            if row["status"] == "complete" and row.get("rmse_tail3") is not None:
                study.tell(trial, row.get("objective_value") or row["rmse_tail3"])
            elif row["status"] == "pruned":
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            else:
                study.tell(trial, state=optuna.trial.TrialState.FAIL)

        return {
            "trial_id": trial_id,
            "stage": "search2",
            "spec": dict(
                _spec(0, args.num_evals, args.maxstep, env_ov, ppo_ov),
                measure_deploy_action_rate=True,
                constraints_ref=constraints_ref,
            ),
            "poll": poll,
            "on_done": on_done,
        }

    run_scheduler(study_dir, _gpus(args), next_job, args.poll_interval)

    trial_rows = []
    for t in study.get_trials(deepcopy=False, states=terminal):
        trial_id = f"search2-{t.number:03d}"
        r = collect_trial_row(study_dir, "search2", trial_id, 0)
        r["params"] = t.params
        terms = read_json(os.path.join(
            study_dir, "trials", trial_id, "reward_terms.json"))
        if terms:
            r["reward_terms"] = terms
        trial_rows.append(r)
    write_run2_search(study, study_dir, trial_rows, base2)
    print(f"[search2] 报告: {study_dir}/reports/run2_search.md; "
          f"gate-R1 核准后再运行 confirm2")


def cmd_confirm2(args) -> None:
    """R2:全量确认 top-k + base2 对照 × 3 seeds,帕累托判定(计划 6.R2)。"""
    import optuna

    from legwheel.tuning.report import write_run2_confirm
    from legwheel.tuning.space import (
        BASE2_ENV_OVERRIDES,
        BASE2_PPO_OVERRIDES,
        RUN2_FACTORS,
    )

    study_dir = _study_dir(args)
    base2 = _base2_ref(study_dir)
    storage = f"sqlite:///{os.path.join(study_dir, 'study2.db')}"
    study = optuna.load_study(study_name="search2", storage=storage)
    complete = [t for t in study.get_trials(
        deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))]
    if not complete:
        raise RuntimeError("study 中无完成 trial,请先运行 search2")
    # 按重算后的 objective(result.json,固定 σ 口径)排序;
    # study 内 t.value 是搜索期旧口径,仅作回退
    def _obj(t):
        r = read_json(os.path.join(
            study_dir, "trials", f"search2-{t.number:03d}", "result.json")) or {}
        return r.get("objective_value") or t.value or WORST
    if args.candidates:
        # gate-R1 人工指定候选(逗号分隔 trial 编号,如 "026,000")
        wanted = {int(x) for x in args.candidates.split(",")}
        top = [t for t in sorted(complete, key=_obj) if t.number in wanted]
        if len(top) != len(wanted):
            raise RuntimeError(
                f"指定候选 {sorted(wanted)} 中有人未完成: "
                f"实到 {[t.number for t in top]}")
    else:
        top = sorted(complete, key=_obj)[: args.top_k]
    seeds = [int(s) for s in args.seeds.split(",")]

    stage0 = read_json(os.path.join(study_dir, "stage0.json")) or {}
    constraints_ref = {"base2": base2, "energy_budget": stage0.get("energy_budget")}

    jobs = []
    names = {}
    for t in top:
        env_ov, ppo_ov = split_overrides(RUN2_FACTORS, t.params)
        ppo_ov = {**ppo_ov, **BASE2_PPO_OVERRIDES}
        name = f"cand{t.number:03d}"
        names[name] = t.params
        for s in seeds:
            jobs.append({
                "trial_id": f"{name}-s{s}",
                "stage": "confirm2",
                "spec": dict(
                    _spec(s, args.num_evals, args.maxstep, env_ov, ppo_ov),
                    measure_deploy_action_rate=True,
                    constraints_ref=constraints_ref,
                ),
            })
    for s in seeds:
        jobs.append({
            "trial_id": f"confirm2-base2-s{s}",
            "stage": "confirm2",
            "spec": dict(
                _spec(s, args.num_evals, args.maxstep,
                      dict(BASE2_ENV_OVERRIDES), dict(BASE2_PPO_OVERRIDES)),
                measure_deploy_action_rate=True,
            ),
        })
    print(f"[confirm2] 候选: {list(names)} + base2,seeds={seeds},"
          f" maxstep={args.maxstep:g}")
    run_scheduler(study_dir, _gpus(args), static_jobs(jobs), args.poll_interval)

    candidates = {}
    for name in names:
        candidates[name] = [read_json(os.path.join(
            study_dir, "trials", f"{name}-s{s}", "result.json")) or {}
            for s in seeds]
    base2_runs = [read_json(os.path.join(
        study_dir, "trials", f"confirm2-base2-s{s}", "result.json")) or {}
        for s in seeds]
    verdict = write_run2_confirm(study_dir, candidates, base2_runs)
    print(f"[confirm2] {verdict}(报告: {study_dir}/reports/run2_confirm.md)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _study_dir(args) -> str:
    path = os.path.join(_PROJECT_ROOT, "tuning", args.study)
    os.makedirs(path, exist_ok=True)
    return path


def _gpus(args) -> tuple:
    return tuple(int(g) for g in args.gpus.split(","))


def _baseline_summary(study_dir: str) -> dict:
    """阶段 0 baseline trials 的均值汇总(stage2 报告的参照)。"""
    import glob

    vals = []
    for path in glob.glob(os.path.join(study_dir, "trials", "baseline-s*", "result.json")):
        r = read_json(path)
        if r:
            vals.append(r)
    if not vals:
        return {}
    keys = ("rmse_tail3", "avg_episode_length_tail3", "energy_per_step_tail3",
            "episode_reward_tail3")
    return {k: sum(v[k] for v in vals if v.get(k) is not None)
            / max(1, sum(1 for v in vals if v.get(k) is not None))
            for k in keys}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study", default="default", help="study 名(tuning/<study>/)")
    ap.add_argument("--gpus", default="0,1", help="可用卡,逗号分隔(默认 0,1)")
    ap.add_argument("--poll-interval", type=float, default=10.0)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("baseline", help="阶段 0:噪声底估计")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--maxstep", type=float, default=1e8)
    p.add_argument("--num-evals", type=int, default=20)
    p.set_defaults(fn=cmd_baseline)

    p = sub.add_parser("screen", help="阶段 1:Plackett-Burman 筛选")
    p.add_argument("--maxstep", type=float, default=1e8)
    p.add_argument("--num-evals", type=int, default=20)
    p.add_argument("--top-k", type=int, default=10, help="建议进阶段二的因子数")
    p.add_argument("--dry-run", action="store_true", help="只打印设计矩阵与 override,不训练")
    p.set_defaults(fn=cmd_screen)

    p = sub.add_parser("search", help="阶段 2:TPE + ASHA")
    p.add_argument("--n-trials", type=int, default=40)
    p.add_argument("--n-startup", type=int, default=8, help="TPE 启动期随机采样数")
    p.add_argument("--min-resource", type=int, default=4, help="ASHA 首个 rung 的 eval 数")
    p.add_argument("--maxstep", type=float, default=1e8)
    p.add_argument("--num-evals", type=int, default=48, help="加密剪枝 rung(计划第 4 节)")
    p.add_argument("--dry-run", action="store_true", help="只采样打印 resolved 配置,不训练")
    p.set_defaults(fn=cmd_search)

    p = sub.add_parser("confirm", help="阶段 3:全量确认")
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--maxstep", type=float, default=4e8)
    p.add_argument("--num-evals", type=int, default=20)
    p.set_defaults(fn=cmd_confirm)

    p = sub.add_parser("base2", help="run2 R0:base2 建立与验证")
    p.add_argument("--seeds", default="0,1")
    p.add_argument("--maxstep", type=float, default=1e8)
    p.add_argument("--num-evals", type=int, default=20)
    p.set_defaults(fn=cmd_base2)

    p = sub.add_parser("search2", help="run2 R1:reward shaping 搜索(TPE+ASHA)")
    p.add_argument("--n-trials", type=int, default=25)
    p.add_argument("--n-startup", type=int, default=8)
    p.add_argument("--min-resource", type=int, default=4)
    p.add_argument("--maxstep", type=float, default=1e8)
    p.add_argument("--num-evals", type=int, default=48)
    p.add_argument("--dry-run", action="store_true", help="只采样打印 override,不训练")
    p.set_defaults(fn=cmd_search2)

    p = sub.add_parser("confirm2", help="run2 R2:全量确认 + 帕累托判定")
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--candidates", default="",
                   help="人工指定候选 trial 编号(逗号分隔,如 026,000;覆盖 top-k)")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--maxstep", type=float, default=4e8)
    p.add_argument("--num-evals", type=int, default=20)
    p.set_defaults(fn=cmd_confirm2)

    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )
    args.fn(args)


if __name__ == "__main__":
    main()
