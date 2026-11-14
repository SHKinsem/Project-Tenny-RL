"""失败精确重现:bit-identical vmap 重放与单环境回放。"""

import os
import logging
from typing import List, Optional

import numpy as np
import jax
import jax.numpy as jp
from mujoco_playground._src import mjx_env

from legwheel.evaluation.failures import FailureRecord
from legwheel.evaluation.rendering import render_replay_video
from legwheel.evaluation.runner import load_policy, make_env

logger = logging.getLogger(__name__)


def replay_exact_vmap(
    failure: FailureRecord,
    batch_context_file: str,
    push: bool = True,
    render: bool = True,
    verbose: bool = True,
    save_video_path: Optional[str] = None,
) -> list:
    """
    使用全批量 vmap 精确重现失败 episode。

    通过重新执行与 batch_test 完全相同的 vmap 计算图（相同 batch size、
    相同 RNG、相同 action），确保浮点运算 bit-identical，从而精确复现失败。
    """
    logger.info("[精确模式] 加载批量上下文: %s", batch_context_file)
    ctx = np.load(batch_context_file)
    env_rngs_np = ctx["env_rngs"]       # (num_envs, 2)
    all_actions_np = ctx["all_actions"]  # (total_steps, num_envs, action_size)
    num_envs = env_rngs_np.shape[0]
    total_saved_steps = all_actions_np.shape[0]
    target_idx = failure.env_idx

    logger.info("[精确模式] batch_size=%d, 保存步数=%d, 目标 env_idx=%d",
                num_envs, total_saved_steps, target_idx)

    env, cfg = make_env(push=push)

    # ── 用完全相同的 vmap 执行路径 ──
    vmap_reset = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))
    vmap_step = jax.jit(jax.vmap(env.step))

    env_rngs = jp.array(env_rngs_np)
    states = vmap_reset(env_rngs, True)

    def extract_single(batched_states, idx):
        return jax.tree.map(lambda x: x[idx], batched_states)

    target_state = extract_single(states, target_idx)
    rollout = [target_state]
    step_actions_target = []   # 每步动作（用于视频 OSD）
    step_rewards_target = []   # 每步奖励（用于视频 OSD）

    max_steps = min(failure.fail_step + 10, total_saved_steps)

    state_pos = [
        ["cmd", 4], ["gyro", 3], ["sinyaw", 1],
        ["sinpitch", 1], ["sinroll", 1],
        ["jnts", 4], ["velocity", 4],
        ["linvel", 2], ["pid_hist", 4], ["pid_error", 2],
    ]
    if cfg.last_act_obs:
        state_pos.append(["last_act", 4])

    logger.info("=" * 60)
    logger.info("[精确重现] env_idx=%d, expected_fail_step=%d, reason=%s",
                target_idx, failure.fail_step, failure.terminal_reason)
    logger.info("command: %s", failure.command)
    logger.info("=" * 60)

    for step_i in range(max_steps):
        actions = jp.array(all_actions_np[step_i])  # (num_envs, action_size)
        states = vmap_step(states, actions)

        target_state = extract_single(states, target_idx)
        rollout.append(target_state)
        step_actions_target.append(np.array(all_actions_np[step_i, target_idx]))
        step_rewards_target.append(float(jax.device_get(target_state.reward)))

        if verbose and (step_i % 50 == 0 or step_i == failure.fail_step - 2
                        or step_i == failure.fail_step - 1):
            state_obs = jax.device_get(target_state.obs["state"])
            obs_str = f"step={step_i:4d}"
            for name, length in state_pos:
                vals = state_obs[:length]
                obs_str += f"  {name}={np.array2string(vals, precision=3, suppress_small=True)}"
                state_obs = state_obs[length:]
            sinpitch = float(jax.device_get(env._getSinPitch_newest(target_state.data)))
            sinroll = float(jax.device_get(env._getSinRoll_newest(target_state.data)))
            obs_str += f"  sinP={sinpitch:.4f}  sinR={sinroll:.4f}"
            push_vec = np.array(jax.device_get(target_state.info["push"]))
            push_norm = float(np.linalg.norm(push_vec))
            if push_norm > 1e-6:
                obs_str += f"  PUSH=({push_vec[0]:+.3f},{push_vec[1]:+.3f}) mag={push_norm:.3f}"
            logger.info(obs_str)

            metrics_str = "  metrics: "
            for k in target_state.metrics.keys():
                metrics_str += f"{k}={jax.device_get(target_state.metrics[k]):.3f} "
            logger.info(metrics_str)

        if jax.device_get(target_state.done) > 0:
            sinpitch = float(jax.device_get(env._getSinPitch_newest(target_state.data)))
            sinroll = float(jax.device_get(env._getSinRoll_newest(target_state.data)))
            actual_fail_step = step_i + 1
            logger.info("!! 精确重现终止于 step=%d  sinPitch=%.4f  sinRoll=%.4f",
                        actual_fail_step, sinpitch, sinroll)
            if actual_fail_step == failure.fail_step:
                logger.info("✓ 终止步数与记录完全一致 (step=%d)", failure.fail_step)
            else:
                logger.warning("✗ 终止步数不一致：记录=%d, 实际=%d（不应出现）",
                              failure.fail_step, actual_fail_step)
            break
    else:
        logger.warning("未在 %d 步内观察到终止（预期 step=%d）", max_steps, failure.fail_step)

    # ── 渲染 ──
    if render:
        logger.info("正在渲染精确重现视频 ...")
        render_replay_video(
            env=env,
            rollout=rollout,
            step_actions=step_actions_target,
            step_rewards=step_rewards_target,
            save_video_path=save_video_path,
            height=720,
            width=1280,
            rng_seed=failure.rng_seed,
        )

    return rollout


def replay_failure(
    ckpt_path: str,
    failure: FailureRecord,
    push: bool = True,
    render: bool = True,
    verbose: bool = True,
    save_video_path: Optional[str] = None,
    use_saved_actions: bool = True,
    batch_context_file: Optional[str] = None,
) -> List[mjx_env.State]:
    """
    重现一个失败 episode。

    优先使用精确 vmap 模式（需要 batch_context_file），保证 bit-identical。
    如果 batch_context_file 不可用，回退到 saved-action 模式（可能因浮点差异不完全一致）。

    Args:
        batch_context_file: 批量上下文 .npz 文件路径（由 batch_test 生成）
        use_saved_actions: True = 回放保存的动作序列；
                           False = 使用 policy 重新推理。
    """
    # ── 优先使用精确 vmap 重现 ──
    # 注意：精确 vmap 重现依赖“保存的动作序列”。当 use_saved_actions=False 时，
    # 用户明确希望使用 policy 重新推理，因此必须跳过精确重现分支。
    if use_saved_actions and batch_context_file and os.path.exists(batch_context_file):
        return replay_exact_vmap(
            failure=failure,
            batch_context_file=batch_context_file,
            push=push,
            render=render,
            verbose=verbose,
            save_video_path=save_video_path,
        )

    if not use_saved_actions:
        logger.info(
            "use_saved_actions=False：跳过精确 vmap 重现，使用单环境 policy 推理回放（可能因 vmap 浮点差异不完全一致）"
        )
    else:
        if batch_context_file:
            logger.warning(
                "批量上下文文件不可用: %s，回退到单环境回放模式（可能因 vmap 浮点差异不完全一致）",
                batch_context_file,
            )
        else:
            logger.warning(
                "未提供批量上下文文件，回退到单环境回放模式（可能因 vmap 浮点差异不完全一致）"
            )
    env, cfg = make_env(push=push)
    episode_length = cfg.episode_length

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    # ── 使用批量测试中保存的 **实际** reset RNG key ──
    has_saved_rng = hasattr(failure, 'reset_rng') and failure.reset_rng is not None
    if has_saved_rng:
        rng = jp.array(failure.reset_rng, dtype=jp.uint32)
        logger.info("使用保存的 reset_rng 进行重现")
    else:
        # 兼容旧的 failure 文件（没有 reset_rng 字段）
        rng = jax.random.PRNGKey(failure.rng_seed)
        logger.warning("⚠ 未找到 reset_rng，回退到 PRNGKey(%d)（可能无法精确重现）", failure.rng_seed)

    state = jit_reset(rng, True)

    # 如果没有保存的 reset_rng，才需要覆写 command（旧兼容路径）
    if not has_saved_rng:
        state = state.replace(
            info={**state.info, "command": jp.array(failure.command)}
        )

    rollout = [state]

    # ── 准备动作来源 ──
    has_saved_actions = (use_saved_actions
                        and hasattr(failure, 'actions')
                        and failure.actions is not None
                        and len(failure.actions) > 0)
    if has_saved_actions:
        saved_actions = jp.array(failure.actions)
        logger.info("使用保存的动作序列回放 (%d 步)", len(failure.actions))
        total_steps = len(failure.actions) + 10  # 多跑几步观察终止后状态
    else:
        jit_inference_fn = load_policy(ckpt_path)
        act_rng = jax.random.PRNGKey(0)
        # policy 重新推理回放：最多跑满 1 个 episode（即使未触发 done 也退出）
        total_steps = episode_length
        if use_saved_actions:
            logger.warning("⚠ 未找到保存的动作序列，回退到 policy 推理模式")

    state_pos = [
        ["cmd", 4], ["gyro", 3], ["sinyaw", 1],
        ["sinpitch", 1], ["sinroll", 1],
        ["jnts", 4], ["velocity", 4],
        ["linvel", 2], ["pid_hist", 4], ["pid_error", 2],
    ]
    if cfg.last_act_obs:
        state_pos.append(["last_act", 4])

    logger.info("=" * 60)
    logger.info("重现失败: env_idx=%d, expected_fail_step=%d, reason=%s",
                failure.env_idx, failure.fail_step, failure.terminal_reason)
    logger.info("command: %s", failure.command)
    logger.info("mode: %s", "保存动作回放" if has_saved_actions else "policy 重新推理")
    logger.info("=" * 60)

    step_actions_list = []   # 每步动作（用于视频 OSD）
    step_rewards_list = []   # 每步奖励（用于视频 OSD）

    for step_i in range(total_steps):
        # 获取动作
        if has_saved_actions:
            if step_i < len(failure.actions):
                ctrl = saved_actions[step_i]
            else:
                # 超出保存的动作序列，停止
                break
        else:
            act_rng, sub_rng = jax.random.split(act_rng)
            ctrl, _ = jit_inference_fn(state.obs, sub_rng)

        state = jit_step(state, ctrl)
        rollout.append(state)
        step_actions_list.append(np.array(jax.device_get(ctrl)))
        step_rewards_list.append(float(jax.device_get(state.reward)))

        if verbose and (step_i % 50 == 0 or step_i == failure.fail_step - 1
                        or step_i == failure.fail_step):
            state_obs = jax.device_get(state.obs["state"])
            obs_str = f"step={step_i:4d}"
            for name, length in state_pos:
                vals = state_obs[:length]
                obs_str += f"  {name}={np.array2string(vals, precision=3, suppress_small=True)}"
                state_obs = state_obs[length:]
            sinpitch = float(jax.device_get(env._getSinPitch_newest(state.data)))
            sinroll = float(jax.device_get(env._getSinRoll_newest(state.data)))
            obs_str += f"  sinP={sinpitch:.4f}  sinR={sinroll:.4f}"
            push_vec = np.array(jax.device_get(state.info["push"]))
            push_norm = float(np.linalg.norm(push_vec))
            if push_norm > 1e-6:
                obs_str += f"  PUSH=({push_vec[0]:+.3f},{push_vec[1]:+.3f}) mag={push_norm:.3f}"
            logger.info(obs_str)

            # 打印 metrics
            metrics_str = "  metrics: "
            for k in state.metrics.keys():
                metrics_str += f"{k}={jax.device_get(state.metrics[k]):.3f} "
            logger.info(metrics_str)

        if jax.device_get(state.done) > 0:
            sinpitch = float(jax.device_get(env._getSinPitch_newest(state.data)))
            sinroll = float(jax.device_get(env._getSinRoll_newest(state.data)))
            logger.info("!! 终止于 step=%d  sinPitch=%.4f  sinRoll=%.4f",
                        step_i + 1, sinpitch, sinroll)
            break

    # ── 渲染 ──
    if render:
        logger.info("正在渲染重现视频 ...")
        render_replay_video(
            env=env,
            rollout=rollout,
            step_actions=step_actions_list,
            step_rewards=step_rewards_list,
            save_video_path=save_video_path,
            height=720,
            width=1280,
            rng_seed=failure.rng_seed,
        )

    return rollout


def replay_top_failures(
    ckpt_path: str,
    failures: List[FailureRecord],
    n: int = 5,
    sort_by: str = "earliest",
    save_dir: str = "failures/videos",
    push: bool = True,
    batch_context_file: Optional[str] = None,
    use_saved_actions: bool = True,
):
    """
    重现排名靠前的 n 个失败例子。

    sort_by:
        'earliest'  - 最早失败的
        'reward'    - 累计奖励最低的
    """
    if not failures:
        logger.info("无失败记录可重现。")
        return

    if sort_by == "earliest":
        sorted_failures = sorted(failures, key=lambda x: x.fail_step)
    elif sort_by == "reward":
        sorted_failures = sorted(failures, key=lambda x: x.episode_reward)
    else:
        sorted_failures = failures

    os.makedirs(save_dir, exist_ok=True)

    for i, failure in enumerate(sorted_failures[:n]):
        logger.info("\n>>> 重现第 %d / %d 个失败 (env=%d, step=%d, reason=%s) <<<",
                    i + 1, n, failure.env_idx, failure.fail_step, failure.terminal_reason)
        video_path = os.path.join(save_dir, f"failure_{i:03d}_env{failure.env_idx}_step{failure.fail_step}.mp4")
        replay_failure(
            ckpt_path=ckpt_path,
            failure=failure,
            push=push,
            render=True,
            verbose=True,
            save_video_path=video_path,
            use_saved_actions=use_saved_actions,
            batch_context_file=batch_context_file,
        )
