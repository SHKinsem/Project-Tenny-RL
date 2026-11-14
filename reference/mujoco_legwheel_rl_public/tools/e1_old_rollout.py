#!/usr/bin/env python
"""E1 旧侧轨迹录制:旧代码(/tmp/lw_base)+ 旧策略(base2-s0)标准协议 rollout。

必须在旧 worktree 代码下运行(本脚本把 LW_BASE 插到 sys.path 最前):
  cd /tmp/lw_base && /path/to/.venv/bin/python \
      /path/to/legwheel_rl/tools/e1_old_rollout.py --out /tmp/e1_old_traj.npz

逐 step 录制原始量(命令、轮角 qL/qR、sinyaw、reward、done、在线旧式锚点与
hold 分项),供 tools/analyze_hold_causality.py 离线重算新旧两套 reward 公式。
"""
import argparse
import os
import sys

_LW_BASE = os.environ.get("LW_BASE", "/tmp/lw_base")
sys.path.insert(0, _LW_BASE)

import numpy as np  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=os.path.join(
        _LW_BASE, "tuning/run2/trials/base2-s0/ckpt/000115507200"))
    parser.add_argument("--out", default="/tmp/e1_old_traj.npz")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-init", type=int, default=1)
    args = parser.parse_args()

    import jax
    import jax.numpy as jp

    import legwheel
    assert legwheel.__file__.startswith(_LW_BASE), f"legwheel 来源错误: {legwheel.__file__}"
    from legwheel import LegWheelRobot, default_config
    from legwheel.config import apply_config_overrides
    from legwheel.evaluation.runner import load_policy

    cfg = default_config()
    # base2-s0 trial 训练配置(tuning/run2/trials/base2-s0/input.json)
    apply_config_overrides(cfg, {"vel_pid.kp": 0.00017, "vel_pid.notsmoothing": 0.25})
    env = LegWheelRobot(cfg)
    print(f"env from {legwheel.__file__}; ckpt={args.ckpt}", flush=True)

    inference = jax.jit(load_policy(args.ckpt, deterministic=True))
    num_envs, steps = args.num_envs, int(env._config.episode_length)

    vmap_reset = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))
    vmap_step = jax.jit(jax.vmap(env.step))
    vmap_infer = jax.jit(jax.vmap(inference))
    vmap_sinyaw = jax.jit(jax.vmap(env._getSinYaw_newest))
    q_w = [int(env._qpos_wanted_joints[1]), int(env._qpos_wanted_joints[3])]

    rng = jax.random.PRNGKey(args.seed)
    states = vmap_reset(jax.random.split(rng, num_envs), bool(args.random_init))
    act_rng = jax.random.PRNGKey(0)

    rec = {k: [] for k in
           ("cmd", "qL", "qR", "sinyaw", "reward", "done",
            "term_wsin", "term_sinyaw", "anch_w", "anch_y")}
    for i in range(steps):
        rec["cmd"].append(np.asarray(jax.device_get(states.info["command"])))
        act_rng, sub = jax.random.split(act_rng)
        acts, _ = vmap_infer(states.obs, jax.random.split(sub, num_envs))
        states = vmap_step(states, acts)
        rec["qL"].append(np.asarray(jax.device_get(states.data.qpos[:, q_w[0]])))
        rec["qR"].append(np.asarray(jax.device_get(states.data.qpos[:, q_w[1]])))
        rec["sinyaw"].append(np.asarray(jax.device_get(vmap_sinyaw(states.data))))
        rec["reward"].append(np.asarray(jax.device_get(states.reward)))
        rec["done"].append(np.asarray(jax.device_get(states.done)))
        rec["term_wsin"].append(np.asarray(jax.device_get(
            states.metrics["reward/tracking_wheels_sinposition"])))
        rec["term_sinyaw"].append(np.asarray(jax.device_get(
            states.metrics["reward/tracking_sinyaw"])))
        rec["anch_w"].append(np.asarray(jax.device_get(
            states.info["wheels_sinposition_start"])))
        rec["anch_y"].append(np.asarray(jax.device_get(states.info["sinyaw_start"])))
        if i % 100 == 0:
            print(f"  step {i}/{steps}", flush=True)

    arrs = {k: np.stack(v) for k, v in rec.items()}
    np.savez(args.out, **arrs,
             zero_keep_scale=float(env._config.reward_config.zero_keep_scale),
             sigma_wheel=float(env._config.reward_config.sigma_wheel),
             sigma=float(env._config.reward_config.sigma),
             cmd_scale_yaw=float(env._cmd_scales[3]),
             margin_vx=float(env._config.cmd_lin_vel_x_zero_margin),
             margin_yaw=float(env._config.cmd_ang_vel_zero_margin),
             wheel_sin_scale=float(env._config.wheel_sin_scale))
    print(f"saved {args.out}; shapes: reward {arrs['reward'].shape}")


if __name__ == "__main__":
    main()
