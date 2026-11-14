"""主指标与汇总计算(计划第 2 节:归一化跟踪 RMSE 单目标 + 存活/能耗硬约束)。

数据来源:env 端直接累加的归一化 err²(legwheel/rewards.py 的 track/err2_*),
EvalWrapper 逐 episode 求和、acting 对 env 维取均值,因此
    单次 eval 的 RMSE = sqrt((err2_lin_sum + err2_ang_sum) / avg_episode_length)
"""

import csv
import math
import os

ERR2_LIN = "eval/episode_track/err2_lin_vel"
ERR2_ANG = "eval/episode_track/err2_ang_vel"
EP_LEN = "eval/avg_episode_length"
EPISODE_REWARD = "eval/episode_reward"
ENERGY = "eval/episode_reward/energy_cost"
REWARD_TERM_PREFIX = "eval/episode_reward/"
ACTIVE_SINPOS = "eval/episode_track/active_steps_sinpos"
ACTIVE_SINYAW = "eval/episode_track/active_steps_sinyaw"


def read_metrics_csv(path: str) -> list[dict]:
    """读取 worker 落盘的 metrics.csv → [{num_steps: float, key: value, ...}]。"""
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, newline="") as fp:
        for row in csv.DictReader(fp):
            parsed = {}
            for k, v in row.items():
                if v is None or v == "":
                    continue
                try:
                    parsed[k] = float(v)
                except ValueError:
                    pass
            rows.append(parsed)
    return rows


def eval_rmse(row: dict):
    """单次 eval 的归一化跟踪 RMSE;缺键时返回 None。"""
    if ERR2_LIN not in row or ERR2_ANG not in row or EP_LEN not in row:
        return None
    n = row[EP_LEN]
    if n <= 0:
        return None
    return math.sqrt((row[ERR2_LIN] + row[ERR2_ANG]) / n)


def rmse_curve(rows: list[dict]) -> list[float]:
    """每次 eval 的 RMSE 序列(跳过缺键的 eval)。"""
    return [r for r in (eval_rmse(row) for row in rows) if r is not None]


def tail3_rmse(rows: list[dict]):
    """主指标终值:尾部 3 个 eval 的 RMSE 均值;不足 3 个时用全部。"""
    curve = rmse_curve(rows)
    if not curve:
        return None
    tail = curve[-3:]
    return sum(tail) / len(tail)


def tail3_of(rows: list[dict], key: str):
    vals = [r[key] for r in rows if key in r]
    if not vals:
        return None
    tail = vals[-3:]
    return sum(tail) / len(tail)


def energy_per_step_tail3(rows: list[dict]):
    """单位时间能耗(reward 代理量,非真实电功率):energy_cost episode 和 / 步数。"""
    vals = [
        r[ENERGY] / r[EP_LEN]
        for r in rows
        if ENERGY in r and EP_LEN in r and r[EP_LEN] > 0
    ]
    if not vals:
        return None
    tail = vals[-3:]
    return sum(tail) / len(tail)


def reward_terms(row: dict) -> dict:
    """单次 eval 的各 reward 分项(brax 聚合键 eval/episode_reward/*,episode 和口径)。"""
    return {k[len(REWARD_TERM_PREFIX):]: v
            for k, v in row.items() if k.startswith(REWARD_TERM_PREFIX)}


def fixed_score(terms: dict, reward_cfg) -> float:
    """固定评估分 S(θ):用 baseline 权重对分项日志重算(计划 2.6 规则 1 的辅助指标)。

    与 rewards.py 的加权结构一致:
      vel = (tracking_lin_vel + tracking_ang_vel) * vel_scale
      posture = pitch_cost*pitch_scale + roll_cost*roll_scale
      zero_keep = (tracking_wheels_sinposition + tracking_sinyaw) * zero_keep_scale
      height = (tracking_height_jnt + tracking_height_action) * height_scale
      负项: energy_cost/action_rate/pitch_velocity_penalty 各自 × 其 scale
    """
    rc = reward_cfg
    s = 0.0
    s += (terms.get("tracking_lin_vel", 0.0)
          + terms.get("tracking_ang_vel", 0.0)) * rc.vel_scale
    s += terms.get("pitch_cost", 0.0) * rc.pitch_scale
    s += terms.get("roll_cost", 0.0) * rc.roll_scale
    s += (terms.get("tracking_wheels_sinposition", 0.0)
          + terms.get("tracking_sinyaw", 0.0)) * rc.zero_keep_scale
    s += (terms.get("tracking_height_jnt", 0.0)
          + terms.get("tracking_height_action", 0.0)) * rc.height_scale
    s += terms.get("energy_cost", 0.0) * rc.energy_scale
    s += terms.get("action_rate", 0.0) * rc.action_rate_scale
    s += terms.get("pitch_velocity_penalty", 0.0) * rc.pitch_velocity_penalty_scale
    return s


def summarize_rows(rows: list[dict], reward_cfg=None) -> dict:
    """由完整 eval 曲线汇总一个 run 的指标(完成态)。"""
    out = {
        "rmse_tail3": tail3_rmse(rows),
        "rmse_curve": rmse_curve(rows),
        "avg_episode_length_tail3": tail3_of(rows, EP_LEN),
        "energy_per_step_tail3": energy_per_step_tail3(rows),
        "episode_reward_tail3": tail3_of(rows, EPISODE_REWARD),
        "n_evals": len(rows),
    }
    if reward_cfg is not None and rows:
        out["fixed_score"] = fixed_score(reward_terms(rows[-1]), reward_cfg)
    return out


# ---------------------------------------------------------------------------
# run2:达成率 / per-step 口径与硬约束判定(计划 5.3 / 6.R1)
# ---------------------------------------------------------------------------

from legwheel.tuning.space import MIN_EPISODE_LENGTH  # noqa: E402

# 全程激活项(分母 = avg_episode_length)
FULL_TERMS = ("tracking_lin_vel", "tracking_ang_vel", "tracking_height_jnt",
              "tracking_height_action", "pitch_cost", "roll_cost")
# 门控项(分母 = 实测激活步数,plan 5.3 工程配套)
GATED_TERMS = {
    "tracking_sinyaw": ACTIVE_SINYAW,
    "tracking_wheels_sinposition": ACTIVE_SINPOS,
}
# 惩罚项(per-step 口径 = 分项和 / avg_episode_length)
PER_STEP_TERMS = ("action_rate", "pitch_velocity_penalty", "energy_cost")

# run2 硬约束阈值(计划 5.3;action_rate 门槛经 gate-R1 两轮决议放宽,
# 见 docs/hyperparam_tuning_plan_run2.md gate-R1 决议记录)
RUN2_ACTION_RATE_DEPLOY_MAX = 0.35  # 部署(deterministic)口径 per-step 均值门槛
RUN2_ACTION_RATE_DEPLOY_MAX_PER_SEED = 0.40  # R2 逐 seed 上限
RUN2_SINYAW_MIN_GAIN = 0.02         # sinyaw 达成率 ≥ base2 + 2pp
RUN2_KEY_TERM_MAX_DROP = 0.02       # 关键分项达成率退化 ≤ 2pp
RUN2_KEY_TERMS = ("tracking_lin_vel", "tracking_ang_vel", "pitch_cost", "roll_cost")


def attainment_rates(row: dict) -> dict:
    """单次 eval 的各分项达成率(0~1);门控项用实测激活步数做分母。"""
    terms = reward_terms(row)
    ep_len = row.get(EP_LEN)
    rates = {}
    for name in FULL_TERMS:
        if name in terms and ep_len and ep_len > 0:
            rates[name] = terms[name] / ep_len
    for name, active_key in GATED_TERMS.items():
        n_active = row.get(active_key)
        if name in terms and n_active and n_active > 0:
            rates[name] = terms[name] / n_active
    return rates


def per_step_values(row: dict) -> dict:
    """单次 eval 的惩罚项 per-step 物理量(分项和 / avg_episode_length)。"""
    terms = reward_terms(row)
    ep_len = row.get(EP_LEN)
    if not ep_len or ep_len <= 0:
        return {}
    return {name: terms[name] / ep_len for name in PER_STEP_TERMS if name in terms}


def tail3_attainment(rows: list[dict]) -> dict:
    """尾部 3 个 eval 的达成率均值(按分项)。"""
    tail = rows[-3:] if len(rows) >= 3 else rows
    acc: dict[str, list] = {}
    for row in tail:
        for k, v in attainment_rates(row).items():
            acc.setdefault(k, []).append(v)
    return {k: sum(vs) / len(vs) for k, vs in acc.items()}


def evaluate_run2_constraints(rows: list[dict], action_rate_deploy,
                              base2: dict, energy_budget,
                              attainment: dict = None) -> dict:
    """run2 硬约束判定(计划 6.R1):任一不满足则该 trial 记 WORST。

    base2: {"attainment": {term: rate}} 参照(R0 产物 base2.json);
    action_rate_deploy: worker 尾部确定性加测的 per-step 值(None 视为违反,
    保守处理并在报告中暴露测量失败);
    attainment: 固定 σ 实测达成率(deploy_evaluate 口径),缺省回退到
    训练 eval 的 tail3(注意:后者随 trial 的 σ 搜索值移动,不可比,仅供参照)。
    返回 {"checks": {name: {"value","threshold","ok"}}, "violated": [...]}。
    """
    checks: dict[str, dict] = {}

    def add(name, value, threshold, ok):
        checks[name] = {"value": value, "threshold": threshold, "ok": ok}

    alive = tail3_of(rows, EP_LEN)
    add("alive", alive, MIN_EPISODE_LENGTH,
        alive is not None and alive >= MIN_EPISODE_LENGTH)
    energy = energy_per_step_tail3(rows)
    if energy_budget is not None:
        add("energy", energy, energy_budget,
            energy is not None and energy <= energy_budget)
    add("action_rate_deploy", action_rate_deploy, RUN2_ACTION_RATE_DEPLOY_MAX,
        action_rate_deploy is not None
        and action_rate_deploy <= RUN2_ACTION_RATE_DEPLOY_MAX)

    rates = attainment if attainment is not None else tail3_attainment(rows)
    base_rates = (base2 or {}).get("attainment") or {}
    base_syn = base_rates.get("tracking_sinyaw")
    syn = rates.get("tracking_sinyaw")
    if base_syn is not None:
        add("sinyaw_gain", syn, base_syn + RUN2_SINYAW_MIN_GAIN,
            syn is not None and syn >= base_syn + RUN2_SINYAW_MIN_GAIN)
    for term in RUN2_KEY_TERMS:
        base_v = base_rates.get(term)
        if base_v is None:
            continue
        v = rates.get(term)
        add(f"no_regress[{term}]", v, base_v - RUN2_KEY_TERM_MAX_DROP,
            v is not None and v >= base_v - RUN2_KEY_TERM_MAX_DROP)

    violated = [name for name, c in checks.items() if c["ok"] is False]
    return {"checks": checks, "violated": violated, "attainment": rates}


# 软约束惩罚的归一化尺度(gate-R1 决议:R1 搜索用软约束,TPE 需要梯度;
# 每项"超出一个容差单位"记罚 1.0,与 RMSE 量级可比)
RUN2_PENALTY_SCALES = {
    "alive": 50.0,                        # 每差 50 步罚 1.0
    "energy": None,                       # 上限型:(value/threshold − 1)
    "action_rate_deploy": None,           # 上限型:同上;None 值罚 1.0
    "sinyaw_gain": RUN2_SINYAW_MIN_GAIN,  # 每差 2pp 罚 1.0
    "no_regress": RUN2_KEY_TERM_MAX_DROP,  # 每差 2pp 罚 1.0
}


def run2_soft_penalty(checks: dict) -> float:
    """由 constraint checks 计算软罚(gate-R1 决议,替换硬 WORST)。

    objective = rmse_tail3 + run2_soft_penalty(checks);只有死亡/发散
    (无 RMSE)才记 WORST。R2 最终判定仍用 evaluate_run2_constraints 的硬门槛。
    """
    penalty = 0.0
    for name, c in checks.items():
        if c["ok"] is not False:
            continue
        value, threshold = c["value"], c["threshold"]
        # NaN/None(测量失败或物理发散)按满罚一档保守处理
        if value is None or (isinstance(value, float) and math.isnan(value)):
            penalty += 1.0
        elif name == "energy" or name == "action_rate_deploy":
            penalty += max(0.0, value / threshold - 1.0)
        elif name == "alive":
            penalty += max(0.0, (threshold - value) / RUN2_PENALTY_SCALES["alive"])
        elif name == "sinyaw_gain":
            penalty += max(0.0, (threshold - value) / RUN2_PENALTY_SCALES["sinyaw_gain"])
        elif name.startswith("no_regress"):
            penalty += max(0.0, (threshold - value) / RUN2_PENALTY_SCALES["no_regress"])
        else:
            penalty += 1.0
    return penalty
