"""部署口径实测:deterministic 策略 + 域随机化 + push + 噪声的在环评估。

口径与 tools/analyze_action_rate.py 一致(见 docs/hyperparam_tuning_plan_run2.md
5.2 口径修正):训练报表的 action_rate 是 stochastic 采样口径,77% 是探索 σ
噪声;部署真实出现的是 deterministic 均值动作在噪声+push 下的响应。

达成率口径修正(gate-R1 第三轮后发现):reward 达成率必须用**固定 σ**
(baseline 值 0.5)评估——σ 本身是搜索因子,用 trial 自己的 σ 算达成率
等于让被调参数移动评估球门(search2-026 用 σ=0.87 把达成率"评"高了)。
deploy_evaluate 的测量环境强制 sigma/sigma_wheel = baseline 值。

供 tuning worker 在 trial 尾部对最终 ckpt 加测(R1 软约束/R2 硬约束用),
以及 tools/reval_fixed_sigma.py 离线重估已完成 trial。
"""

import functools
import glob
import logging
import os

import jax
import numpy as np

log = logging.getLogger(__name__)

# 全程激活项(分母 = 总步数)/ 门控项(分母 = 实测激活步数)
FULL_TERMS = ("tracking_lin_vel", "tracking_ang_vel", "tracking_height_jnt",
              "tracking_height_action", "pitch_cost", "roll_cost")
GATED_TERMS = ("tracking_sinyaw", "tracking_wheels_sinposition")
ACTIVE_KEYS = {"tracking_sinyaw": "track/active_steps_sinyaw",
               "tracking_wheels_sinposition": "track/active_steps_sinpos"}


def latest_ckpt(ckpt_dir: str):
    """ckpt 目录下最新的数字编号子目录(brax 每次 eval 保存一档)。"""
    subs = [d for d in glob.glob(os.path.join(ckpt_dir, "[0-9]" * 12))
            if os.path.isdir(d)]
    if not subs:
        return None
    # orbax 要求绝对路径
    return os.path.abspath(max(subs, key=os.path.basename))


def deploy_evaluate(env_cfg, ckpt_path: str, num_envs: int = 1024,
                    seed: int = 42) -> dict:
    """对 ckpt 策略做确定性在环 rollout,返回部署口径指标。

    返回 {"action_rate": float, "attainment": {term: rate}, "avg_episode_length": float}
    - action_rate: mean(|Δa|) + mean(|Δ²a|),逐 env 只计首次 done 前;
    - attainment: 固定 σ(baseline 0.5)下的各 reward 分项达成率;
    测量环境 = trial env_cfg 但 sigma/sigma_wheel 强制为 baseline 值。
    """
    import copy

    from mujoco_playground import wrapper as playground_wrapper

    from legwheel import LegWheelRobot, default_config
    from legwheel.evaluation.runner import load_policy
    from legwheel.randomization import domain_randomize

    env_cfg = copy.deepcopy(env_cfg)
    env_cfg.push_config.enable = True
    # 固定 σ:达成率测量不随 trial 搜索值移动(见模块 docstring)
    base_rc = default_config().reward_config
    env_cfg.reward_config.sigma = base_rc.sigma
    env_cfg.reward_config.sigma_wheel = base_rc.sigma_wheel
    env = LegWheelRobot(env_cfg)
    rand_fn = functools.partial(
        domain_randomize,
        rng=jax.random.split(jax.random.PRNGKey(123), num_envs))
    env = playground_wrapper.BraxDomainRandomizationVmapWrapper(env, rand_fn)

    inference = jax.jit(load_policy(ckpt_path, deterministic=True))
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)

    @jax.jit
    def vmap_inference(obs, rngs):
        return jax.vmap(inference)(obs, rngs)

    T = int(env_cfg.episode_length)
    env_rngs = jax.random.split(jax.random.PRNGKey(seed), num_envs)
    states = reset(env_rngs)
    n_act = env.action_size
    acts = np.zeros((T, num_envs, n_act), dtype=np.float32)
    dones = np.zeros((T, num_envs), dtype=bool)
    term_sum = {k: 0.0 for k in FULL_TERMS + GATED_TERMS}
    active_sum = {k: 0.0 for k in GATED_TERMS}
    nan_count = 0
    act_rng = jax.random.PRNGKey(0)
    for t in range(T):
        act_rng, sub = jax.random.split(act_rng)
        actions, _ = vmap_inference(states.obs, jax.random.split(sub, num_envs))
        acts[t] = np.asarray(jax.device_get(actions))
        states = step(states, actions)
        dones[t] = np.asarray(jax.device_get(states.done)) > 0
        m = states.metrics
        for k in FULL_TERMS + GATED_TERMS:
            # 个别 env 物理发散时 reward 为 NaN;按彻底失败计 0 分并计数暴露
            v = np.asarray(jax.device_get(m[f"reward/{k}"]))
            nan_count += int(np.isnan(v).sum())
            term_sum[k] += float(np.nan_to_num(v, nan=0.0).sum())
        for k in GATED_TERMS:
            active_sum[k] += float(np.asarray(jax.device_get(m[ACTIVE_KEYS[k]])).sum())

    has_done = dones.any(axis=0)
    ep_len = np.where(has_done, np.argmax(dones, axis=0) + 1, T)
    c1_all, c2_all = [], []
    for e in range(num_envs):
        L = int(ep_len[e])
        if L < 3:
            continue
        a = acts[1:L, e]
        # 一阶差分 (L-2 步) 与二阶差分 (L-3 步) 长度差 1,分开统计
        c1_all.append(np.abs(a[1:] - a[:-1]).mean(axis=1))
        c2_all.append(np.abs(a[2:] - 2 * a[1:-1] + a[:-2]).mean(axis=1))
    c1 = np.concatenate(c1_all)
    c2 = np.concatenate(c2_all)
    rate = float(c1.mean() + c2.mean())

    total_steps = float(ep_len.sum())
    attainment = {k: term_sum[k] / total_steps for k in FULL_TERMS}
    for k in GATED_TERMS:
        if active_sum[k] > 0:
            attainment[k] = term_sum[k] / active_sum[k]
    log.info("deploy eval: action_rate=%.4f/步 avg_ep_len=%.0f nan_steps=%d (ckpt=%s, num_envs=%d)",
             rate, float(ep_len.mean()), nan_count, ckpt_path, num_envs)
    return {
        "action_rate": rate,
        "attainment": attainment,
        "avg_episode_length": float(ep_len.mean()),
        "nan_steps": nan_count,
    }


def deploy_action_rate(env_cfg, ckpt_path: str, num_envs: int = 1024,
                       seed: int = 42) -> float:
    """只要 action_rate 的便捷封装(口径同 deploy_evaluate)。"""
    return deploy_evaluate(env_cfg, ckpt_path, num_envs, seed)["action_rate"]
