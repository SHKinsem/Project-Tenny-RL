"""trial 配置解析:baseline 深拷贝 + overrides 合并 → config_resolved.json。

worker 复用而不是绕开训练代码(6.5.2):解析走 get_ppo_config() 等真实函数,
保证"搜索时的训练"与"正式训练"是同一套逻辑。
"""

import json
import subprocess

from legwheel.config import apply_config_overrides, default_config
from legwheel.randomization import default_randomization_config
from legwheel.training.ppo import compute_discounting, get_ppo_config

# 与 curriculum_learning 内应用的 override 键保持一致(两处同步维护)
LOCAL_PPO_KEYS = (
    "half_life_s", "entropy_cost", "reward_scaling", "learning_rate",
    "policy_hidden_layer_sizes", "value_hidden_layer_sizes",
)


def git_info() -> dict:
    """git commit hash + dirty 标记(可追溯性元信息)。"""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        ).stdout.strip())
    except Exception:  # noqa: BLE001 - 非 git 环境下兜底
        commit, dirty = "unknown", True
    return {"git_commit": commit, "git_dirty": dirty}


def resolve_ppo_dict(env_cfg, ppo_overrides, num_evals, maxstep, seed) -> dict:
    """复现 curriculum_learning 的 PPO 参数解析(不训练),供 config_resolved.json。"""
    ppo_overrides = dict(ppo_overrides or {})
    local = {k: ppo_overrides.pop(k) for k in LOCAL_PPO_KEYS if k in ppo_overrides}
    ppo_params = get_ppo_config(
        env_cfg, num_evals=num_evals, maxstep_candidate=maxstep,
        save_checkpoint_path=None, ppo_overrides=ppo_overrides,
    )
    half_life_s = float(local.pop("half_life_s", 0.5))
    ppo_params.discounting = compute_discounting(env_cfg.ctrl_dt, half_life_s)
    ppo_params.learning_rate = float(local.pop("learning_rate", 1e-4))
    ppo_params.entropy_cost = float(local.pop("entropy_cost", 1e-2))
    ppo_params.reward_scaling = float(local.pop("reward_scaling", 0.1))
    ppo_params.network_factory.policy_hidden_layer_sizes = list(
        local.pop("policy_hidden_layer_sizes", [64, 64]))
    ppo_params.network_factory.value_hidden_layer_sizes = list(
        local.pop("value_hidden_layer_sizes", [512, 512, 512, 512]))
    if local:
        raise ValueError(f"未知 PPO override 键: {sorted(local)}")
    ppo_params.seed = seed
    resolved = ppo_params.to_dict()
    resolved["half_life_s"] = half_life_s
    return resolved


def build_resolved_config(
    env_overrides, ppo_overrides, rand_overrides, num_evals, maxstep, seed,
    meta=None,
):
    """返回 (resolved_dict, env_cfg, rand_cfg)。

    resolved_dict 为可 JSON 序列化的全量生效配置(env + PPO + randomization),
    任何 trial 可据此精确复跑;env_cfg/rand_cfg 供 worker 直接用于训练。
    """
    env_cfg = apply_config_overrides(default_config(), env_overrides or {})
    rand_cfg = default_randomization_config()
    rand_cfg.update(rand_overrides or {})
    resolved = {
        "env": env_cfg.to_dict(),
        "ppo": resolve_ppo_dict(env_cfg, ppo_overrides, num_evals, maxstep, seed),
        "randomization": {
            k: (list(v) if isinstance(v, (list, tuple)) else v)
            for k, v in rand_cfg.items()
        },
        "meta": meta or {},
    }
    return resolved, env_cfg, rand_cfg


def write_json(path: str, obj) -> None:
    with open(path, "w") as fp:
        json.dump(obj, fp, indent=2, ensure_ascii=False, default=str)
