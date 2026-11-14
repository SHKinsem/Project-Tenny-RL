"""阶段报告生成:stage0 噪声底 / stage1 PB 筛选 / stage2 TPE top-k / stage3 确认。

每阶段结束自动生成审查报告(计划 2.5 节人工核准门的审查材料);
自动异常标记只写入报告辅助人工判断,不自动淘汰。
"""

import json
import math
import os

from legwheel.tuning.metrics import fixed_score
from legwheel.tuning.space import ENERGY_BUDGET_RATIO, MIN_EPISODE_LENGTH, WORST


def _fmt(v, nd=4):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else str(v)


# ---------------------------------------------------------------------------
# 阶段 0:噪声底
# ---------------------------------------------------------------------------


def write_stage0(study_dir: str, seed_results: dict) -> dict:
    """seed → summary(result.json)。产出 stage0.json + reports/stage0_noise.md。"""
    rmses = {s: r["rmse_tail3"] for s, r in seed_results.items()
             if r.get("rmse_tail3") is not None}
    energies = [r["energy_per_step_tail3"] for r in seed_results.values()
                if r.get("energy_per_step_tail3") is not None]
    n = len(rmses)
    mean = sum(rmses.values()) / n
    # 样本标准差(n-1);n=4~5 时 σ 估计相对误差仍较大,阈值仅作分档指引
    sigma = (math.sqrt(sum((v - mean) ** 2 for v in rmses.values()) / (n - 1))
             if n > 1 else 0.0)
    energy_mean = sum(energies) / len(energies)
    out = {
        "baseline_rmse_per_seed": rmses,
        "baseline_rmse_mean": mean,
        "sigma_noise": sigma,
        "energy_per_step_mean": energy_mean,
        "energy_budget": energy_mean * ENERGY_BUDGET_RATIO,
        "n_seeds": n,
    }
    with open(os.path.join(study_dir, "stage0.json"), "w") as fp:
        json.dump(out, fp, indent=2)

    lines = [
        "# 阶段 0:噪声底估计(baseline ×%d seeds)" % n,
        "",
        "| seed | 尾部 3 eval RMSE | 存活(步) | 单位时间能耗 |",
        "|---|---|---|---|",
    ]
    for s, r in sorted(seed_results.items()):
        lines.append("| %s | %s | %s | %s |" % (
            s, _fmt(r.get("rmse_tail3")),
            _fmt(r.get("avg_episode_length_tail3"), 1),
            _fmt(r.get("energy_per_step_tail3")),
        ))
    lines += [
        "",
        f"- 主指标均值: **{_fmt(mean)}**,跨 seed 标准差 σ_noise = **{_fmt(sigma)}**",
        f"- 筛选用标准误: SE(E_i) = 2σ/√N;2·SE 为主效应分档阈值(计划第 3 节式 2)",
        f"- 能耗硬约束预算: 基线均值 {_fmt(energy_mean)} × {ENERGY_BUDGET_RATIO}"
        f" = **{_fmt(out['energy_budget'])}**(reward 代理量,非真实功耗)",
    ]
    os.makedirs(os.path.join(study_dir, "reports"), exist_ok=True)
    with open(os.path.join(study_dir, "reports", "stage0_noise.md"), "w") as fp:
        fp.write("\n".join(lines) + "\n")
    return out


# ---------------------------------------------------------------------------
# 阶段 1:PB 筛选
# ---------------------------------------------------------------------------


def write_stage1(study_dir: str, factors, effects: dict, ys: dict,
                 run_table: list, sigma_noise: float, n_runs: int,
                 top_k: int = 10) -> None:
    """产出 reports/stage1_screening.md + stage1_space.yaml(gate-1 审查材料)。

    ys: factor 无关,run_id → 主指标 y(尾部 3 eval RMSE 均值,死亡 run 记 WORST)。
    """
    from legwheel.tuning.space import save_search_space_yaml

    se = 2 * sigma_noise / math.sqrt(n_runs)
    threshold = 2 * se
    ranked = sorted(factors, key=lambda f: abs(effects[f.name]), reverse=True)
    searchable = ranked[:top_k]
    frozen = [f.name for f in ranked[top_k:]]
    indistinguishable = [f.name for f in factors
                         if abs(effects[f.name]) < threshold]

    lines = [
        "# 阶段 1:Plackett-Burman 主效应筛选(gate-1 审查材料)",
        "",
        f"- run 数 N = {n_runs},因子数 k = {len(factors)},y = 尾部 3 eval 归一化 RMSE"
        f"(死亡/发散 run 记 WORST={WORST})",
        f"- σ_noise = {_fmt(sigma_noise)},SE(E_i) = 2σ/√N = {_fmt(se)},"
        f"分档阈值 2·SE = {_fmt(threshold)}(同 seed=0 CRN,判据偏保守)",
        "",
        "## run 一览",
        "",
        "| run | y (RMSE) | 状态 |",
        "|---|---|---|",
    ]
    for row in run_table:
        lines.append("| %s | %s | %s |" % (row["run"], _fmt(row["y"]), row["status"]))
    lines += [
        "",
        "## 主效应排序(|E_i| 降序)",
        "",
        "| 排名 | 因子 | 层 | E_i | \\|E_i\\| vs 2·SE | 建议 |",
        "|---|---|---|---|---|---|",
    ]
    searchable_names = {f.name for f in searchable}
    for rank, f in enumerate(ranked, 1):
        e = effects[f.name]
        sig = "显著" if abs(e) >= threshold else "低于噪声底"
        suggestion = "进阶段二" if f.name in searchable_names else "冻结"
        lines.append("| %d | %s | %s | %+.4f | %s | %s |"
                     % (rank, f.name, f.layer, e, sig, suggestion))
    lines += [
        "",
        "## 建议冻结/搜索清单",
        "",
        f"- 进阶段二(top {top_k}): {', '.join(sorted(searchable_names))}",
        f"- 建议冻结: {', '.join(sorted(frozen))}",
        f"- 低于噪声底(gate-1 人工判断是否放回): {', '.join(sorted(indistinguishable))}",
        "",
        "→ 人工核准:编辑 `stage1_space.yaml`(放回误冻参数、收窄离谱区间)后放行阶段二。",
    ]
    os.makedirs(os.path.join(study_dir, "reports"), exist_ok=True)
    with open(os.path.join(study_dir, "reports", "stage1_screening.md"), "w") as fp:
        fp.write("\n".join(lines) + "\n")
    save_search_space_yaml(
        os.path.join(study_dir, "reports", "stage1_space.yaml"),
        searchable, frozen, indistinguishable)


# ---------------------------------------------------------------------------
# 阶段 2:TPE + ASHA
# ---------------------------------------------------------------------------


def anomaly_flags(trial_row: dict, baseline: dict) -> list:
    """自动异常标记(计划 2.5):只写入报告,不自动淘汰。"""
    flags = []
    terms = trial_row.get("reward_terms") or {}
    if terms:
        total = sum(abs(v) for v in terms.values())
        if total > 0:
            top = max(abs(v) for v in terms.values())
            if top / total > 0.5:
                top_name = max(terms, key=lambda k: abs(terms[k]))
                flags.append(f"分项畸变({top_name} 占 {top / total:.0%})")
    r = trial_row.get("episode_reward_tail3")
    l = trial_row.get("avg_episode_length_tail3")
    if (r is not None and l is not None
            and baseline.get("episode_reward_tail3") is not None
            and baseline.get("avg_episode_length_tail3") is not None):
        if (r > baseline["episode_reward_tail3"]
                and l < baseline["avg_episode_length_tail3"]):
            flags.append("疑似死亡刷分(return↑ 且 episode_length↓)")
    if trial_row.get("rmse_tail3") == WORST:
        flags.append("违反硬约束/死亡(记 WORST)")
    return flags


def write_stage2(study, study_dir: str, trial_rows: list, baseline: dict,
                 top_k: int = 5) -> None:
    """产出 reports/stage2_topk.md + 参数重要性/平行坐标/中间值图(gate-2 审查材料)。"""
    from legwheel.config import default_config

    rc = default_config().reward_config
    for row in trial_rows:
        if row.get("reward_terms"):
            row["fixed_score"] = fixed_score(row["reward_terms"], rc)

    complete = [r for r in trial_rows if r.get("rmse_tail3") is not None]
    ranked = sorted(complete, key=lambda r: r["rmse_tail3"])

    lines = [
        "# 阶段 2:TPE + ASHA 搜索结果(gate-2 审查材料)",
        "",
        f"- 完成 trial 数: {len(complete)},主目标:归一化跟踪 RMSE(越小越好)",
        f"- baseline RMSE: {_fmt(baseline.get('rmse_tail3'))},"
        f" S(θ)=固定权重重算分(sanity check,越大越好)",
        "",
        f"## Top {top_k}(按主指标)",
        "",
        "| 排名 | trial | RMSE | 存活 | 能耗/步 | return | S(θ) | 异常标记 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for rank, row in enumerate(ranked[:top_k], 1):
        flags = anomaly_flags(row, baseline)
        lines.append("| %d | %s | %s | %s | %s | %s | %s | %s |" % (
            rank, row["trial_id"], _fmt(row["rmse_tail3"]),
            _fmt(row.get("avg_episode_length_tail3"), 1),
            _fmt(row.get("energy_per_step_tail3")),
            _fmt(row.get("episode_reward_tail3"), 2),
            _fmt(row.get("fixed_score"), 2),
            "; ".join(flags) or "-",
        ))
    lines += ["", "## Top trial 参数", ""]
    for rank, row in enumerate(ranked[:top_k], 1):
        lines.append(f"### #{rank} trial {row['trial_id']}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(row.get("params", {}), indent=2, ensure_ascii=False))
        lines.append("```")
        terms = row.get("reward_terms") or {}
        if terms:
            pos = {k: v for k, v in terms.items() if k.startswith("tracking")
                   or k in ("pitch_cost", "roll_cost")}
            neg = {k: v for k, v in terms.items() if k not in pos}
            lines.append("")
            lines.append("- 正项: " + json.dumps(pos, ensure_ascii=False))
            lines.append("- 负项: " + json.dumps(neg, ensure_ascii=False))
        lines.append("")

    # 与 baseline 差异清单(top-1)
    if ranked:
        lines += ["## Top-1 与 baseline 的参数差异", "", "```json",
                  json.dumps(ranked[0].get("overrides", {}), indent=2,
                             ensure_ascii=False),
                  "```", ""]
    lines += [
        "→ 人工核准:排除可疑组合,确认进入全量确认的名单(默认 top-2)。",
    ]
    os.makedirs(os.path.join(study_dir, "reports"), exist_ok=True)
    with open(os.path.join(study_dir, "reports", "stage2_topk.md"), "w") as fp:
        fp.write("\n".join(lines) + "\n")

    # 可视化审查图(optuna matplotlib 后端,无头环境可用)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from optuna.visualization import matplotlib as oviz

        for name, fn in (
            ("param_importances", oviz.plot_param_importances),
            ("parallel_coordinate", oviz.plot_parallel_coordinate),
            ("intermediate_values", oviz.plot_intermediate_values),
        ):
            try:
                ax = fn(study)
                fig = ax.figure if hasattr(ax, "figure") else plt.gcf()
                fig.tight_layout()
                fig.savefig(os.path.join(
                    study_dir, "reports", f"stage2_{name}.png"), dpi=120)
                plt.close(fig)
            except Exception:  # noqa: BLE001 - 单图失败不阻塞报告
                pass
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# 阶段 3:全量确认
# ---------------------------------------------------------------------------


def write_stage3(study_dir: str, candidates: dict, baseline_runs: list) -> str:
    """candidates: name → [3 seeds 的 summary];baseline_runs: 同长度 baseline ×3。

    统计判定(计划第 5 节):Δ = mean(S_base) − mean(S_c) > 2·σ_pool,
    σ_pool = sqrt(σ_c²/3 + σ_base²/3),S=RMSE 越低越好。
    返回结论字符串(同时写 reports/stage3_confirm.md)。
    """
    base_rmses = [r["rmse_tail3"] for r in baseline_runs
                  if r.get("rmse_tail3") is not None]
    base_mean = sum(base_rmses) / len(base_rmses)
    base_var = sum((v - base_mean) ** 2 for v in base_rmses) / (len(base_rmses) - 1)

    lines = [
        "# 阶段 3:全量确认(4e8 步 × 3 seeds,gate-3 审查材料)",
        "",
        f"- baseline(同长度新跑对照组)RMSE: {_fmt(base_mean)} ± "
        f"{_fmt(math.sqrt(base_var))}(n={len(base_rmses)})",
        "",
        "| 候选 | RMSE 均值 | 标准差 | 存活 | 能耗/步 | Δ(base−cand) | 2·σ_pool | 判定 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    verdicts = []
    for name, runs in candidates.items():
        rmses = [r["rmse_tail3"] for r in runs if r.get("rmse_tail3") is not None]
        if not rmses:
            continue
        c_mean = sum(rmses) / len(rmses)
        c_var = (sum((v - c_mean) ** 2 for v in rmses) / (len(rmses) - 1)
                 if len(rmses) > 1 else 0.0)
        sigma_pool = math.sqrt(c_var / len(rmses) + base_var / len(base_rmses))
        delta = base_mean - c_mean
        passed = delta > 2 * sigma_pool
        alive = sum(r.get("avg_episode_length_tail3") or 0 for r in runs) / len(runs)
        energy = sum(r.get("energy_per_step_tail3") or 0 for r in runs) / len(runs)
        verdict = "通过(显著优于基线)" if passed else "未通过(差异在噪声内)"
        if alive < MIN_EPISODE_LENGTH:
            verdict += ";存活不达标"
        verdicts.append((name, c_mean, passed))
        lines.append("| %s | %s | %s | %s | %s | %+.4f | %.4f | %s |" % (
            name, _fmt(c_mean), _fmt(math.sqrt(c_var)), _fmt(alive, 1),
            _fmt(energy), delta, 2 * sigma_pool, verdict))
    lines += [
        "",
        "→ gate-3 人工签字后才进部署流程(batch_test/visualize 结果另附)。",
    ]
    os.makedirs(os.path.join(study_dir, "reports"), exist_ok=True)
    with open(os.path.join(study_dir, "reports", "stage3_confirm.md"), "w") as fp:
        fp.write("\n".join(lines) + "\n")
    best = min(verdicts, key=lambda v: v[1]) if verdicts else None
    if best and best[2]:
        return f"候选 {best[0]} 通过统计判定(Δ > 2·σ_pool),RMSE {_fmt(best[1])}"
    return "无候选通过统计判定(改善量在种子噪声内)"


# ---------------------------------------------------------------------------
# run2:base2 参照 / R1 搜索报告 / R2 帕累托确认(计划 5.3 / 6)
# ---------------------------------------------------------------------------

from legwheel.tuning.metrics import (  # noqa: E402
    RUN2_ACTION_RATE_DEPLOY_MAX,
    RUN2_ACTION_RATE_DEPLOY_MAX_PER_SEED,
    RUN2_KEY_TERM_MAX_DROP,
    RUN2_KEY_TERMS,
    RUN2_SINYAW_MIN_GAIN,
)

# 报告展示的达成率分项(全程 6 + 门控 2)
ATTAINMENT_TERMS = (
    "tracking_lin_vel", "tracking_ang_vel", "tracking_height_jnt",
    "tracking_height_action", "pitch_cost", "roll_cost",
    "tracking_sinyaw", "tracking_wheels_sinposition",
)


def _mean_attainment(runs: list) -> dict:
    """多个 run 的 summary  attainment 逐分项取均值。"""
    acc: dict[str, list] = {}
    for r in runs:
        for k, v in (r.get("attainment") or {}).items():
            acc.setdefault(k, []).append(v)
    return {k: sum(vs) / len(vs) for k, vs in acc.items()}


def write_run2_base2(study_dir: str, seed_results: dict) -> dict:
    """R0 产物:base2.json(达成率参照)+ reports/run2_base2.md(gate-R0 材料)。"""
    runs = list(seed_results.values())
    rmses = {s: r.get("rmse_tail3") for s, r in seed_results.items()}
    attainment = _mean_attainment(runs)
    ar = {s: r.get("action_rate_deploy_per_step")
          for s, r in seed_results.items()}
    out = {
        "rmse_per_seed": rmses,
        "rmse_mean": (sum(v for v in rmses.values() if v is not None)
                      / max(1, sum(1 for v in rmses.values() if v is not None))),
        "attainment": attainment,
        "action_rate_deploy_per_seed": ar,
        "n_seeds": len(runs),
    }
    with open(os.path.join(study_dir, "base2.json"), "w") as fp:
        json.dump(out, fp, indent=2)

    lines = [
        "# R0:base2 建立与验证(gate-R0 审查材料)",
        "",
        "base2 = cand010 的 4 个增益(lr/updates/kp/notsmoothing)移植回 "
        "baseline 网络 + 固件对齐映射(计划 6.R0)。",
        "",
        f"- RMSE 均值: **{_fmt(out['rmse_mean'])}**(逐 seed: "
        + ", ".join(f"s{s}={_fmt(v)}" for s, v in sorted(rmses.items())) + ")",
        "",
        "| seed | RMSE | 存活(步) | action_rate 部署口径/步 | 报表口径/步 |",
        "|---|---|---|---|---|",
    ]
    for s, r in sorted(seed_results.items()):
        sto = (r.get("per_step_stochastic") or {}).get("action_rate")
        lines.append("| %s | %s | %s | %s | %s |" % (
            s, _fmt(r.get("rmse_tail3")),
            _fmt(r.get("avg_episode_length_tail3"), 1),
            _fmt(r.get("action_rate_deploy_per_step")), _fmt(sto)))
    lines += ["", "## 达成率(尾部 3 eval 均值,seeds 平均)", "",
              "| 分项 | 达成率 |", "|---|---|---|"]
    for term in ATTAINMENT_TERMS:
        if term in attainment:
            lines.append(f"| {term} | {attainment[term]:.1%} |")
    lines += [
        "",
        "→ gate-R0:与 run1 baseline 1e8 对照(RMSE 0.7084±0.0472),"
        "预期 0.60~0.70;若反而更差,增益移植不成立,回退纯 baseline 起点。",
    ]
    os.makedirs(os.path.join(study_dir, "reports"), exist_ok=True)
    with open(os.path.join(study_dir, "reports", "run2_base2.md"), "w") as fp:
        fp.write("\n".join(lines) + "\n")
    return out


def write_run2_search(study, study_dir: str, trial_rows: list,
                      base2: dict, top_k: int = 5) -> None:
    """R1 报告:reports/run2_search.md —— 软约束目标 + 约束满足 + 达成率差值。

    gate-R1 决议:R1 搜索目标 = 真实 rmse + 软罚(objective_value);
    约束全过率仍按硬门槛口径统计(R2 判定用)。
    """
    complete = [r for r in trial_rows if r.get("rmse_tail3") is not None]
    ranked = sorted(complete,
                    key=lambda r: r.get("objective_value") or r["rmse_tail3"])
    n_ok = sum(1 for r in complete if not r.get("violated_constraints"))

    lines = [
        "# R1:reward shaping 搜索结果(gate-R1 审查材料)",
        "",
        f"- 完成 trial 数: {len(complete)},满足全部硬约束(硬门槛口径): **{n_ok}**",
        f"- base2 RMSE: {_fmt(base2.get('rmse_mean'))};"
        f"搜索目标 = 真实 RMSE + 软约束罚(gate-R1 决议),死亡/发散记 WORST={WORST}",
        "",
        f"## Top {top_k}(按搜索目标 objective)",
        "",
        "| 排名 | trial | objective | 真实 RMSE | 罚 | 存活 | 能耗/步 | ar 部署/步 | sinyaw 达成 | 违反约束 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for rank, row in enumerate(ranked[:top_k], 1):
        att = row.get("attainment") or {}
        lines.append("| %d | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            rank, row["trial_id"],
            _fmt(row.get("objective_value") or row["rmse_tail3"]),
            _fmt(row["rmse_tail3"]),
            _fmt(row.get("constraint_penalty") or 0.0),
            _fmt(row.get("avg_episode_length_tail3"), 1),
            _fmt(row.get("energy_per_step_tail3")),
            _fmt(row.get("action_rate_deploy_per_step")),
            f"{att['tracking_sinyaw']:.1%}" if "tracking_sinyaw" in att else "-",
            "; ".join(row.get("violated_constraints") or []) or "-",
        ))
    # 约束违反统计(哪些约束最常卡住)
    from collections import Counter

    viol_counter = Counter(
        c for r in complete for c in (r.get("violated_constraints") or []))
    if viol_counter:
        lines += ["", "## 约束违反统计", "", "| 约束 | 违反 trial 数 |",
                  "|---|---|---|"]
        for name, cnt in viol_counter.most_common():
            lines.append(f"| {name} | {cnt} |")
    lines += ["", "## Top trial 参数", ""]
    for rank, row in enumerate(ranked[:top_k], 1):
        lines.append(f"### #{rank} trial {row['trial_id']}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(row.get("params", {}), indent=2, ensure_ascii=False))
        lines.append("```")
        att = row.get("attainment") or {}
        base_att = base2.get("attainment") or {}
        if att and base_att:
            lines.append("")
            lines.append("| 分项 | 达成率 | vs base2 |")
            lines.append("|---|---|---|")
            for term in ATTAINMENT_TERMS:
                if term in att and term in base_att:
                    lines.append("| %s | %s | %+.1fpp |" % (
                        term, f"{att[term]:.1%}",
                        100 * (att[term] - base_att[term])))
        lines.append("")
    os.makedirs(os.path.join(study_dir, "reports"), exist_ok=True)
    with open(os.path.join(study_dir, "reports", "run2_search.md"), "w") as fp:
        fp.write("\n".join(lines) + "\n")


def write_run2_confirm(study_dir: str, candidates: dict,
                       base2_runs: list) -> str:
    """R2 报告:reports/run2_confirm.md —— 帕累托判定(计划 6.R2 五条全过才算赢)。

    candidates: name → [seeds 的 result.json summary];
    base2_runs: 同长度 base2 对照组。条件 5(batch_test)需人工另测填入。
    """
    base_rmses = [r["rmse_tail3"] for r in base2_runs
                  if r.get("rmse_tail3") is not None]
    base_mean = sum(base_rmses) / len(base_rmses)
    base_var = sum((v - base_mean) ** 2 for v in base_rmses) / (len(base_rmses) - 1)
    base_att = _mean_attainment(base2_runs)
    base_ar = [r.get("action_rate_deploy_per_step") for r in base2_runs]

    lines = [
        "# R2:全量确认与帕累托判定(gate-R2 审查材料)",
        "",
        f"- base2(同长度新跑对照组)RMSE: {_fmt(base_mean)} ± "
        f"{_fmt(math.sqrt(base_var))}(n={len(base_rmses)})",
        "",
        "通过条件(计划 6.R2,全部满足才算赢):",
        "1. RMSE ≤ base2 − 2·σ_pool",
        f"2. action_rate 部署口径均值 ≤ {RUN2_ACTION_RATE_DEPLOY_MAX} "
        f"且逐 seed ≤ {RUN2_ACTION_RATE_DEPLOY_MAX_PER_SEED}",
        f"3. sinyaw 达成率均值 ≥ base2 + {RUN2_SINYAW_MIN_GAIN:.0%}",
        f"4. 关键分项逐 seed 达成率退化 ≤ {RUN2_KEY_TERM_MAX_DROP:.0%}",
        "5. batch_test 成功率 ≥ base2(人工另测,本表不判)",
        "",
        "| 候选 | RMSE 均值±std | 条件1 | ar 部署/步 | 条件2 | sinyaw | 条件3 | 条件4 | 总判 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    verdicts = []
    for name, runs in candidates.items():
        rmses = [r["rmse_tail3"] for r in runs if r.get("rmse_tail3") is not None]
        if not rmses:
            continue
        c_mean = sum(rmses) / len(rmses)
        c_var = (sum((v - c_mean) ** 2 for v in rmses) / (len(rmses) - 1)
                 if len(rmses) > 1 else 0.0)
        sigma_pool = math.sqrt(c_var / len(rmses) + base_var / len(base_rmses))
        cond1 = base_mean - c_mean > 2 * sigma_pool
        ars = [r.get("action_rate_deploy_per_step") for r in runs]
        ar_vals = [v for v in ars if v is not None]
        ar_mean = sum(ar_vals) / len(ar_vals) if ar_vals else None
        cond2 = (ar_mean is not None and ar_mean <= RUN2_ACTION_RATE_DEPLOY_MAX
                 and all(v <= RUN2_ACTION_RATE_DEPLOY_MAX_PER_SEED for v in ar_vals))
        att = _mean_attainment(runs)
        syn, syn_base = att.get("tracking_sinyaw"), base_att.get("tracking_sinyaw")
        cond3 = (syn is not None and syn_base is not None
                 and syn >= syn_base + RUN2_SINYAW_MIN_GAIN)
        cond4 = True
        for r in runs:
            r_att = r.get("attainment") or {}
            for term in RUN2_KEY_TERMS:
                if term in r_att and term in base_att:
                    if r_att[term] < base_att[term] - RUN2_KEY_TERM_MAX_DROP:
                        cond4 = False
        passed = cond1 and cond2 and cond3 and cond4
        verdicts.append((name, c_mean, passed))
        yn = lambda b: "✓" if b else "✗"  # noqa: E731
        lines.append("| %s | %s±%s | %s | %s | %s | %s | %s | %s | %s |" % (
            name, _fmt(c_mean), _fmt(math.sqrt(c_var)), yn(cond1),
            _fmt(ar_mean), yn(cond2),
            f"{syn:.1%}" if syn is not None else "-", yn(cond3), yn(cond4),
            "**通过**" if passed else "未通过"))
    lines += [
        "",
        f"- base2 参照:ar 部署/步 = "
        + ", ".join(_fmt(v) for v in base_ar)
        + f";sinyaw 达成率 = {base_att.get('tracking_sinyaw', 0):.1%}",
        "",
        "→ gate-R2:帕累托判定 + 部署审查(docs/deploy_firmware_comparison.md 核对);"
        "条件 5 batch_test 结果另附。",
    ]
    os.makedirs(os.path.join(study_dir, "reports"), exist_ok=True)
    with open(os.path.join(study_dir, "reports", "run2_confirm.md"), "w") as fp:
        fp.write("\n".join(lines) + "\n")
    best = min((v for v in verdicts if v[2]), key=lambda v: v[1], default=None)
    if best:
        return f"候选 {best[0]} 通过帕累托判定(条件 1-4),RMSE {_fmt(best[1])}"
    return "无候选通过帕累托判定(条件 1-4)"
