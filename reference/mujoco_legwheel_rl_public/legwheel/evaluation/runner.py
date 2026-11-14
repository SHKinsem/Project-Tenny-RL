"""批量评估核心:环境/策略构建、并发 vmap 测试与统计。"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import jax
import jax.numpy as jp
from ml_collections import config_dict

try:
    from brax.training.agents.ppo import checkpoint as ppo_checkpoint
except ImportError:
    # 旧版 brax 无该模块(评估需训练机上的新版 brax,或用 legwheel.deployment.extract.load)
    ppo_checkpoint = None

from legwheel import LegWheelRobot, default_config
from legwheel.evaluation.failures import FailureRecord

logger = logging.getLogger(__name__)


def make_env(push: bool = True, overrides: Optional[Dict[str, Any]] = None) -> Tuple[LegWheelRobot, config_dict.ConfigDict]:
    """构建测试环境（默认与训练一致配置;overrides 用于按 trial 的训练配置评估）"""
    from legwheel.config import apply_config_overrides

    cfg = default_config()
    if overrides:
        apply_config_overrides(cfg, overrides)
    cfg.push_config.enable = push
    env = LegWheelRobot(cfg)
    return env, cfg


def load_policy(ckpt_path: str, deterministic: bool = True):
    """从 checkpoint 加载推理函数(需要带 ppo.checkpoint 的新版 brax)。

    deterministic=True: 部署口径(均值动作);False: 训练 eval 口径
    (brax ppo.train 的 deterministic_eval=False,采样动作)。

    brax 0.14.2 的 ppo_checkpoint.load_policy 在反序列化网络配置时可能抛
    KeyError: None(kernel initializer 名为 None);此时回退到手动重建:
    用 legwheel.deployment.extract.load 恢复参数,按参数形状推断 policy
    隐藏层,重建 inference 函数。
    """
    if ppo_checkpoint is None:
        raise ImportError(
            "当前 brax 版本缺少 brax.training.agents.ppo.checkpoint,无法加载 policy。"
            "请在训练机环境运行,或用 legwheel.deployment.extract.load 手动加载。"
        )
    try:
        return ppo_checkpoint.load_policy(ckpt_path, deterministic=deterministic)
    except Exception as exc:  # noqa: BLE001 - 见 docstring
        logger.warning("ppo_checkpoint.load_policy 失败(%r),回退到手动重建", exc)
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks

    from legwheel.deployment.extract import load as extract_load

    stats, policy_params, _ = extract_load(ckpt_path)
    env, _ = make_env(push=False)
    kernels = sorted(
        (k, tuple(v["kernel"].shape)) for k, v in policy_params["params"].items()
    )
    # 末层输出为分布参数(2*action_size),其余为隐藏层
    hidden_sizes = [shape[1] for _, shape in kernels[:-1]]
    ppo_net = ppo_networks.make_ppo_networks(
        env.observation_size,
        env.action_size,
        # 与训练一致:normalize_observations=True 时观测经 running statistics 归一化
        preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=hidden_sizes,
        value_hidden_layer_sizes=(1,),
        policy_obs_key="state",
        value_obs_key="privileged_state",
    )
    make_inference_fn = ppo_networks.make_inference_fn(ppo_net)
    return make_inference_fn((stats, policy_params), deterministic=deterministic)


# ──────────────────────────────────────────────────────────────
# 批量测试核心
# ──────────────────────────────────────────────────────────────
def batch_test(
    ckpt_path: str,
    num_envs: int = 4096,
    random_init: bool = True,
    push: bool = True,
    base_seed: int = 42,
    episode_multiplier: int = 1,
    overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[List[FailureRecord], Dict[str, Any]]:
    """
    并发 num_envs 个环境测试 policy。

    返回:
        failures: 失败记录列表
        stats: 统计字典 (成功率、平均 episode 长度等)
    """
    env, cfg = make_env(push=push, overrides=overrides)
    episode_length = cfg.episode_length * episode_multiplier

    logger.info("加载 checkpoint: %s", ckpt_path)
    jit_inference_fn = load_policy(ckpt_path)

    # ── 为每个环境生成独立的 RNG seed ──
    base_rng = jax.random.PRNGKey(base_seed)
    env_rngs = jax.random.split(base_rng, num_envs)  # (num_envs, 2)

    # ── vmap reset ──
    logger.info("正在重置 %d 个环境 (random_init=%s) ...", num_envs, random_init)
    vmap_reset = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))
    states = vmap_reset(env_rngs, random_init)

    # ── vmap step & inference ──
    vmap_step = jax.jit(jax.vmap(env.step))

    @jax.jit
    def vmap_inference(obs, rngs):
        return jax.vmap(jit_inference_fn)(obs, rngs)

    # ── 逐步仿真并收集轨迹信息 ──
    logger.info("开始仿真 %d 步 (episode_length=%d) ...", episode_length, episode_length)

    # 追踪每个环境的状态
    cumulative_rewards = jp.zeros(num_envs)
    done_flags = jp.zeros(num_envs, dtype=jp.bool_)  # 是否已终止
    fail_steps = jp.full(num_envs, episode_length, dtype=jp.int32)  # 默认=完整 episode
    # 记录最终的 sinpitch / sinroll / qpos / qvel (在 done 时刻)
    final_sinpitch = jp.zeros(num_envs)
    final_sinroll = jp.zeros(num_envs)
    final_qpos = jp.zeros((num_envs, env.mj_model.nq))
    final_qvel = jp.zeros((num_envs, env.mj_model.nv))

    # 保存初始 command
    initial_commands = jax.device_get(states.info["command"])  # (num_envs, 4)

    # 保存每个环境实际使用的 reset RNG key（用于精确重现）
    env_rngs_np = np.array(jax.device_get(env_rngs))  # (num_envs, 2)

    # 收集每步的动作（用于精确重现失败例子）
    all_actions_list: list[np.ndarray] = []  # 每个元素 shape (num_envs, action_size)

    act_rng = jax.random.PRNGKey(0)

    for step_i in range(episode_length):
        if step_i % 100 == 0:
            logger.info("  step %d / %d  (已失败: %d)", step_i, episode_length,
                        int(jax.device_get(done_flags.sum())))

        # 推理
        act_rng, sub_rng = jax.random.split(act_rng)
        act_rngs = jax.random.split(sub_rng, num_envs)

        obs_dict = states.obs
        actions, _ = vmap_inference(obs_dict, act_rngs)

        # 保存本步动作（从 device 拷贝到 CPU）
        all_actions_list.append(np.array(jax.device_get(actions)))

        # 环境 step
        next_states = vmap_step(states, actions)

        # 检测新的 done（本步首次终止的环境）
        cur_done = next_states.done.astype(jp.bool_)
        newly_done = cur_done & (~done_flags)

        # 记录失败时刻的信息
        fail_steps = jp.where(newly_done, step_i + 1, fail_steps)
        final_sinpitch = jp.where(newly_done, 
            jax.vmap(env._getSinPitch_newest)(next_states.data), final_sinpitch)
        final_sinroll = jp.where(newly_done,
            jax.vmap(env._getSinRoll_newest)(next_states.data), final_sinroll)
        final_qpos = jp.where(newly_done[:, None], next_states.data.qpos, final_qpos)
        final_qvel = jp.where(newly_done[:, None], next_states.data.qvel, final_qvel)

        # 累计奖励
        cumulative_rewards += next_states.reward * (~done_flags)

        # 更新 done 状态
        done_flags = done_flags | cur_done

        states = next_states

        # 提前结束：所有环境都已 done
        if jax.device_get(done_flags.all()):
            logger.info("  所有环境在 step %d 已终止", step_i + 1)
            break

    # ── 收集结果 ──
    logger.info("仿真完成，正在收集失败信息 ...")
    done_flags_np = np.array(jax.device_get(done_flags))
    fail_steps_np = np.array(jax.device_get(fail_steps))
    cumulative_rewards_np = np.array(jax.device_get(cumulative_rewards))
    sinpitch_np = np.array(jax.device_get(final_sinpitch))
    sinroll_np = np.array(jax.device_get(final_sinroll))
    qpos_np = np.array(jax.device_get(final_qpos))
    qvel_np = np.array(jax.device_get(final_qvel))
    commands_np = np.array(initial_commands)

    # 堆叠所有步的动作：shape (actual_steps, num_envs, action_size)
    all_actions_np = np.stack(all_actions_list, axis=0) if all_actions_list else np.empty((0, num_envs, 0))
    logger.info("动作缓冲区大小: %s (%.1f MB)", all_actions_np.shape,
                all_actions_np.nbytes / 1e6)

    failures: List[FailureRecord] = []
    for i in range(num_envs):
        if not done_flags_np[i]:
            continue  # 完整存活
        # 判断终止原因
        sp = sinpitch_np[i]
        sr = sinroll_np[i]
        has_nan = bool(np.isnan(qpos_np[i]).any() or np.isnan(qvel_np[i]).any())
        if has_nan:
            reason = "nan"
        elif sp < -0.6 or sp > 0.85:
            reason = "pitch"
        elif abs(sr) > 0.6:
            reason = "roll"
        else:
            reason = "unknown"

        fstep = int(fail_steps_np[i])
        failures.append(FailureRecord(
            env_idx=i,
            rng_seed=i,  # 仅用于显示，实际重现用 reset_rng
            command=commands_np[i],
            fail_step=fstep,
            episode_reward=float(cumulative_rewards_np[i]),
            sinpitch_at_fail=float(sp),
            sinroll_at_fail=float(sr),
            terminal_reason=reason,
            qpos_at_fail=qpos_np[i],
            qvel_at_fail=qvel_np[i],
            reset_rng=env_rngs_np[i],            # 实际 JAX PRNGKey
            actions=all_actions_np[:fstep, i],    # 截至失败的完整动作序列
        ))

    num_success = int((~done_flags_np).sum())
    num_fail = int(done_flags_np.sum())
    avg_reward = float(cumulative_rewards_np.mean())
    avg_fail_step = float(fail_steps_np[done_flags_np].mean()) if num_fail > 0 else episode_length

    stats = {
        "num_envs": num_envs,
        "episode_length": episode_length,
        "num_success": num_success,
        "num_fail": num_fail,
        "success_rate": num_success / num_envs,
        "avg_reward": avg_reward,
        "avg_reward_success": float(cumulative_rewards_np[~done_flags_np].mean()) if num_success > 0 else 0,
        "avg_reward_fail": float(cumulative_rewards_np[done_flags_np].mean()) if num_fail > 0 else 0,
        "avg_fail_step": avg_fail_step,
        "fail_reason_counts": {},
    }
    for f in failures:
        stats["fail_reason_counts"][f.terminal_reason] = \
            stats["fail_reason_counts"].get(f.terminal_reason, 0) + 1

    batch_context = {"env_rngs": env_rngs_np, "all_actions": all_actions_np}
    return failures, stats, batch_context


# ──────────────────────────────────────────────────────────────
# 失败统计 & 可视化
# ──────────────────────────────────────────────────────────────
def print_stats(stats: Dict[str, Any]) -> None:
    logger.info("=" * 60)
    logger.info("测试结果统计")
    logger.info("=" * 60)
    logger.info("  环境数: %d", stats["num_envs"])
    logger.info("  Episode 长度: %d", stats["episode_length"])
    logger.info("  成功数: %d  失败数: %d", stats["num_success"], stats["num_fail"])
    logger.info("  成功率: %.2f%%", stats["success_rate"] * 100)
    logger.info("  平均累计奖励: %.3f", stats["avg_reward"])
    logger.info("    成功 episode 平均奖励: %.3f", stats["avg_reward_success"])
    logger.info("    失败 episode 平均奖励: %.3f", stats["avg_reward_fail"])
    logger.info("  平均失败步数: %.1f", stats["avg_fail_step"])
    logger.info("  失败原因分布: %s", stats["fail_reason_counts"])
    logger.info("=" * 60)
