"""调参因子注册表、Plackett-Burman 筛选设计与阶段二搜索空间。

对应 docs/hyperparam_tuning_plan.md 第 0/1/3 节:
- A 层 = PPO 训练超参(target="ppo",经 curriculum_learning 的 ppo_overrides 接入);
- B 层 = 奖励权重/指令范围/速度环/动作映射(target="env",经 dotted-key env override 接入);
- C 层(域随机化/噪声/推力)与锁定层不进搜索。
- vel_scale=2.0 为锚定项,不进搜索(计划 2.6 规则 2);reward_scaling 归 A 层单独调。
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

# 死亡/发散 run 的哨兵主指标值(严格大于任何可达归一化 RMSE,见计划第 2 节)
WORST = 10.0
# 存活硬约束:eval/avg_episode_length 下限(episode_length=600)
MIN_EPISODE_LENGTH = 550
# 能耗硬约束:单位时间能耗 ≤ 基线 × ENERGY_BUDGET_RATIO(gate-1 核准值)
ENERGY_BUDGET_RATIO = 1.2


@dataclass(frozen=True)
class Factor:
    """一个可调因子。

    kind: "log"(对数均匀) / "float"(线性均匀) / "categorical"(离散)。
    low/high: PB 两水平,兼作阶段二连续因子的搜索上下界。
    choices: categorical 的全部取值;网络结构等复合取值编码为字符串(如 "128x64"),
             由 map_fn 解码为 override 值。
    map_fn: 因子值 → override dict;缺省为 {name: value}。
    """

    name: str
    layer: str  # "A" / "B"
    kind: str  # "log" / "float" / "categorical"
    baseline: Any
    low: Any
    high: Any
    choices: tuple = ()
    target: str = "env"  # "env" → env_overrides / "ppo" → ppo_overrides
    map_fn: Optional[Callable[[Any], dict]] = None

    def overrides(self, value: Any) -> dict:
        if self.map_fn is not None:
            return self.map_fn(value)
        return {self.name: value}


def _sym_range(key: str) -> Callable[[Any], dict]:
    """对称区间因子:值 x → {key: [-x, x]}。"""
    return lambda v: {key: [-float(v), float(v)]}


def _net_sizes(key: str) -> Callable[[Any], dict]:
    """网络结构因子:值 "128x64" → {key: [128, 64]}。"""
    return lambda v: {key: [int(x) for x in str(v).split("x")]}


# 29 个因子 → Plackett-Burman N = 4·⌈(29+1)/4⌉ = 32(计划第 3 节取整规则)
FACTORS: list[Factor] = [
    # ---- A 层:PPO 训练超参(PB 水平取对数尺度 ±0.5 dex 或与搜索范围一致) ----
    Factor("learning_rate", "A", "log", 1e-4, 3e-5, 3e-4, target="ppo"),
    Factor("entropy_cost", "A", "log", 1e-2, 3e-3, 3e-2, target="ppo"),
    Factor("half_life_s", "A", "log", 0.5, 0.2, 1.5, target="ppo"),
    Factor("reward_scaling", "A", "log", 0.1, 0.03, 0.3, target="ppo"),
    Factor("unroll_length", "A", "categorical", 75, 50, 100,
           choices=(50, 75, 100), target="ppo"),
    Factor("num_minibatches", "A", "categorical", 16, 8, 32,
           choices=(8, 16, 32), target="ppo"),
    Factor("num_updates_per_batch", "A", "categorical", 4, 2, 8,
           choices=(2, 4, 8), target="ppo"),
    Factor("policy_net", "A", "categorical", "64x64", "64x64", "128x128",
           choices=("64x64", "128x128", "128x64"), target="ppo",
           map_fn=_net_sizes("policy_hidden_layer_sizes")),
    Factor("value_net", "A", "categorical", "512x512x512x512", "512x512",
           "512x512x512x512", choices=("512x512", "512x512x512x512"),
           target="ppo", map_fn=_net_sizes("value_hidden_layer_sizes")),
    # ---- B 层:奖励权重(锚定 vel_scale 外)、指令范围、速度环、动作映射 ----
    # PB 水平 = 当前值 ±50%(计划第 3 节);搜索区间同。
    Factor("reward_config.height_scale", "B", "float", 2.0, 1.0, 3.0),
    Factor("reward_config.roll_scale", "B", "float", 4.0, 2.0, 6.0),
    Factor("reward_config.zero_keep_scale", "B", "float", 10.0, 5.0, 15.0),
    Factor("reward_config.energy_scale", "B", "float", -1.0, -1.5, -0.5),
    Factor("reward_config.action_rate_scale", "B", "float", -0.5, -0.75, -0.25),
    Factor("reward_config.pitch_velocity_penalty_scale", "B", "float",
           -0.5, -0.75, -0.25),
    Factor("reward_config.sigma", "B", "float", 0.5, 0.25, 0.75),
    Factor("reward_config.sigma_wheel", "B", "float", 0.5, 0.25, 0.75),
    Factor("vel_pid.kp", "B", "log", 2e-4, 1e-4, 3e-4),
    Factor("vel_pid.torque_limit", "B", "float", 0.0295, 0.01475, 0.04425),
    Factor("vel_pid.target_scale", "B", "float", 30.0, 15.0, 45.0),
    Factor("vel_pid.notsmoothing", "B", "float", 0.5, 0.25, 0.75),
    Factor("cmd_config.cmd_prob", "B", "float", 0.9, 0.45, 1.0),
    Factor("cmd_lin_vel_x_range", "B", "float", 0.2, 0.1, 0.3,
           map_fn=_sym_range("cmd_config.lin_vel_x")),
    Factor("cmd_ang_vel_yaw_range", "B", "float", 1.5, 0.75, 2.25,
           map_fn=_sym_range("cmd_config.ang_vel_yaw")),
    Factor("cmd_height_value_range", "B", "float", 0.5, 0.25, 0.75,
           map_fn=lambda v: {"cmd_config.height_value": [-float(v), 0.0]}),
    Factor("cmd_roll_value_range", "B", "float", 0.2, 0.1, 0.3,
           map_fn=_sym_range("cmd_config.roll_value")),
    Factor("action_scale.thigh", "B", "float", 0.25, 0.125, 0.375),
    Factor("action_scale.wheel", "B", "float", 0.0295, 0.01475, 0.04425),
    Factor("action_range_thigh_lo", "B", "float", -0.5, -0.75, -0.25,
           map_fn=lambda v: {"action_range.thigh": [float(v), 0.0]}),
]

_FACTOR_BY_NAME = {f.name: f for f in FACTORS}


def factor_by_name(name: str) -> Factor:
    return _FACTOR_BY_NAME[name]


def split_overrides(factors: list[Factor], values: dict) -> tuple[dict, dict]:
    """因子值 dict {factor_name: value} → (env_overrides, ppo_overrides)。"""
    env_ov: dict = {}
    ppo_ov: dict = {}
    for f in factors:
        if f.name not in values:
            continue
        target = ppo_ov if f.target == "ppo" else env_ov
        target.update(f.overrides(values[f.name]))
    return env_ov, ppo_ov


def pb_design(factors: list[Factor]) -> np.ndarray:
    """Plackett-Burman ±1 设计矩阵,N = 满足 N ≥ k+1 的最小 4 的倍数。"""
    from pyDOE3 import pbdesign

    return np.asarray(pbdesign(len(factors)))


def pb_row_values(factors: list[Factor], row: np.ndarray) -> dict:
    """PB 一行(±1)→ 各因子的低/高水平取值。"""
    return {f.name: (f.high if v > 0 else f.low) for f, v in zip(factors, row)}


def main_effects(factors: list[Factor], X: np.ndarray, y: np.ndarray) -> dict:
    """主效应 E_i = 高水平组均值 − 低水平组均值(计划第 3 节公式)。"""
    effects = {}
    for i, f in enumerate(factors):
        col = X[:, i]
        effects[f.name] = float((y[col > 0].mean() - y[col < 0].mean()))
    return effects


# ---------------------------------------------------------------------------
# 阶段二搜索空间(gate-1 产出的 stage1_space.yaml)
# ---------------------------------------------------------------------------


def save_search_space_yaml(
    path: str,
    searchable: list[Factor],
    frozen: list[str],
    indistinguishable: list[str],
) -> None:
    """写出 gate-1 可编辑的搜索空间 yaml(人工可放回误冻参数、收窄区间)。"""
    import yaml

    doc = {
        "# 说明": "gate-1 人工核准后可编辑;search 列表内为阶段二搜索因子",
        "frozen": frozen,
        "indistinguishable": indistinguishable,
        "search": [],
    }
    for f in searchable:
        entry = {"name": f.name}
        if f.kind == "categorical":
            entry["choices"] = [c if not isinstance(c, tuple) else list(c)
                                for c in f.choices]
        else:
            entry["low"] = f.low
            entry["high"] = f.high
        doc["search"].append(entry)
    with open(path, "w") as fp:
        yaml.safe_dump(doc, fp, allow_unicode=True, sort_keys=False)


def load_search_space_yaml(path: str) -> list[Factor]:
    """读取 gate-1 核准后的 yaml → 阶段二搜索因子列表(应用人工编辑的区间)。"""
    import dataclasses

    import yaml

    with open(path) as fp:
        doc = yaml.safe_load(fp)
    factors = []
    for entry in doc["search"]:
        base = factor_by_name(entry["name"])
        updates = {}
        if "choices" in entry:
            updates["choices"] = tuple(entry["choices"])
        if "low" in entry:
            updates["low"] = entry["low"]
        if "high" in entry:
            updates["high"] = entry["high"]
        factors.append(dataclasses.replace(base, **updates))
    return factors


def optuna_distributions(factors: list[Factor]) -> dict:
    """阶段二因子的 optuna 采样分布(供 study.ask 的 fixed_distributions)。"""
    import optuna

    dists = {}
    for f in factors:
        if f.kind == "log":
            dists[f.name] = optuna.distributions.FloatDistribution(
                float(f.low), float(f.high), log=True)
        elif f.kind == "float":
            dists[f.name] = optuna.distributions.FloatDistribution(
                float(f.low), float(f.high))
        else:
            dists[f.name] = optuna.distributions.CategoricalDistribution(f.choices)
    return dists


# ---------------------------------------------------------------------------
# run2 空间(docs/hyperparam_tuning_plan_run2.md 第 4 节原理初筛结论)
# ---------------------------------------------------------------------------

# base2 锚点:cand010 移植回 baseline 网络的 4 个增益(计划 6.R0);
# 动作映射/扭矩限幅取固件对齐值(= baseline 默认),不出现在 override 里。
BASE2_PPO_OVERRIDES = {
    "learning_rate": 0.0002994581357971835,
    "num_updates_per_batch": 8,
}
BASE2_ENV_OVERRIDES = {
    "vel_pid.kp": 0.00017026277433303862,
    "vel_pid.notsmoothing": 0.25082138932605313,
}


def _neg_scale(key: str) -> Callable[[Any], dict]:
    """负权重因子:搜索空间用绝对值(正,log 域可采样),override 时取负。"""
    return lambda v: {key: -float(v)}


# 9 个活 reward 权重 + vel_pid.kp/notsmoothing 随行(范围 = 计划 4.4/4.2 表)。
# base_scale/posture_scale 是代码中未被引用的死参数(run1 计划修订㊻已剔除),
# 不进空间;网络结构/动作映射/扭矩限幅按部署·物理·固件固化,根本不注册。
RUN2_FACTORS: list[Factor] = [
    # 跟踪权重(0.3×~3×)
    Factor("reward_config.vel_scale", "B", "log", 2.0, 0.6, 6.0),
    Factor("reward_config.height_scale", "B", "log", 2.0, 0.6, 6.0),
    Factor("reward_config.roll_scale", "B", "log", 4.0, 1.2, 12.0),
    # 零保持总闸(0.3×~3×)
    Factor("reward_config.zero_keep_scale", "B", "log", 10.0, 3.0, 30.0),
    # 惩罚项(因子值域为绝对值,正;baseline 也取绝对值,map_fn override 时取负)
    Factor("reward_config.action_rate_scale", "B", "log", 0.5, 0.05, 4.0,
           map_fn=_neg_scale("reward_config.action_rate_scale")),
    Factor("reward_config.energy_scale", "B", "log", 1.0, 0.1, 4.0,
           map_fn=_neg_scale("reward_config.energy_scale")),
    Factor("reward_config.pitch_velocity_penalty_scale", "B", "log", 0.5,
           0.05, 2.0, map_fn=_neg_scale("reward_config.pitch_velocity_penalty_scale")),
    # 形状:跟踪奖励容差带宽(0.5×~2×)
    Factor("reward_config.sigma", "B", "log", 0.5, 0.25, 1.0),
    Factor("reward_config.sigma_wheel", "B", "log", 0.5, 0.25, 1.0),
    # 控制增益随行(base2 值 ±30%)
    Factor("vel_pid.kp", "B", "log",
           BASE2_ENV_OVERRIDES["vel_pid.kp"],
           BASE2_ENV_OVERRIDES["vel_pid.kp"] * 0.7,
           BASE2_ENV_OVERRIDES["vel_pid.kp"] * 1.3),
    Factor("vel_pid.notsmoothing", "B", "log",
           BASE2_ENV_OVERRIDES["vel_pid.notsmoothing"],
           BASE2_ENV_OVERRIDES["vel_pid.notsmoothing"] * 0.7,
           BASE2_ENV_OVERRIDES["vel_pid.notsmoothing"] * 1.3),
]

_RUN2_FACTOR_BY_NAME = {f.name: f for f in RUN2_FACTORS}


def run2_factor_by_name(name: str) -> Factor:
    return _RUN2_FACTOR_BY_NAME[name]
