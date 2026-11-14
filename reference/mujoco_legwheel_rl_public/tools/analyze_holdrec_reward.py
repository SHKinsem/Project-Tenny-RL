#!/usr/bin/env python
"""holdrec-step2 奖励分项达成率实测(hold_position_heading 新方案失血点排查)。

口径:
  clean      干净口径:规范初始化(random=False)+noise=0+push=0+零指令(cmd_prob=0),
             deterministic,单环境(CPU)。期望接近满分,失血分项即嫌疑点。
  standard   标准口径(batch_test 同口径):默认配置、noise=1、
             push=on、随机初始化、随机指令,deterministic,vmap 批量(GPU)。
             对照 docs/hyperparam_tuning_plan_run2.md 5.1 表。
  trainproto 已不可用:hold_recovery 已从代码移除(见 docs/hold_recovery_training.md
             消亡史),训练协议口径无法复跑,仅保留 clean/standard。

用法:
  .venv/bin/python tools/analyze_holdrec_reward.py --mode clean
  .venv/bin/python tools/analyze_holdrec_reward.py --mode standard --num-envs 512
"""
import argparse
import os
import sys

CKPT = "ckpts/holdrec-step2/000403660800"

# 逐 step 记录的原始奖励分项(metrics key -> 显示名)
TERM_KEYS = [
    "reward/tracking_lin_vel",
    "reward/tracking_ang_vel",
    "reward/tracking_height_jnt",
    "reward/tracking_height_action",
    "reward/tracking_sinyaw",
    "reward/tracking_wheels_sinposition",
    "reward/pitch_cost",
    "reward/roll_cost",
    "reward/energy_cost",
    "reward/action_rate",
    "reward/pitch_velocity_penalty",
]
GATED = {  # 门控项 -> 激活步数 metrics key
    "reward/tracking_wheels_sinposition": "track/active_steps_sinpos",
    "reward/tracking_sinyaw": "track/active_steps_sinyaw",
}
# run2 计划 5.1 节 baseline(旧方案)实测水位:分项 episode 和 / 达成率
BASELINE = {
    "reward/tracking_lin_vel": (447.0, 0.745),
    "reward/tracking_ang_vel": (342.2, 0.570),
    "reward/tracking_height_jnt": (335.8, 0.560),
    "reward/tracking_height_action": (341.8, 0.570),
    "reward/tracking_sinyaw": (47.2, 0.850),
    "reward/tracking_wheels_sinposition": (30.9, 0.609),
    "reward/pitch_cost": (432.2, 0.720),
    "reward/roll_cost": (469.6, 0.783),
}


def weighted_terms(cfg, term_sums, ep_len):
    """把原始分项和折算成加权 reward 贡献(与 _get_reward 的 weights 一致)。"""
    rc = cfg.reward_config
    w = {
        "reward/tracking_lin_vel": rc.vel_scale,
        "reward/tracking_ang_vel": rc.vel_scale,
        "reward/tracking_height_jnt": rc.height_scale,
        "reward/tracking_height_action": rc.height_scale,
        "reward/tracking_sinyaw": rc.zero_keep_scale,
        "reward/tracking_wheels_sinposition": rc.zero_keep_scale,
        "reward/pitch_cost": rc.pitch_scale,
        "reward/roll_cost": rc.roll_scale,
        "reward/energy_cost": rc.energy_scale,
        "reward/action_rate": rc.action_rate_scale,
        "reward/pitch_velocity_penalty": rc.pitch_velocity_penalty_scale,
    }
    return {k: term_sums.get(k, 0.0) * w[k] for k in TERM_KEYS}


def print_term_table(tag, cfg, sums, act_sums, ep_lens, n_envs, reward_sum):
    """sums/act_sums: 全 batch 累计(只计存活步);ep_lens: 每 env 存活步数列表。"""
    import numpy as np

    mean_len = float(np.mean(ep_lens))
    print(f"\n==== {tag} ====")
    print(f"envs={n_envs}  平均存活 {mean_len:.1f}/600  "
          f"(存活率 {np.mean(np.array(ep_lens) >= 600) * 100:.1f}%)  "
          f"平均 return(加权) {reward_sum / n_envs:.1f}")
    header = f"{'分项':34s} {'ep和':>9s} {'激活步':>7s} {'达成率':>7s} {'per-step':>9s} {'加权贡献':>9s} {'baseline':>9s}"
    print(header)
    print("-" * len(header))
    term_means = {}
    for k in TERM_KEYS:
        s = sums[k] / n_envs
        term_means[k] = s
        if k in GATED:
            act = act_sums[GATED[k]] / n_envs
            ach = s / max(act, 1e-9)
            denom = act
        else:
            act = mean_len
            ach = s / max(mean_len, 1e-9)
            denom = mean_len
        per_step = s / max(mean_len, 1e-9)
        base = BASELINE.get(k)
        base_str = f"{base[1] * 100:.1f}%" if base else "-"
        mark = ""
        if base and ach < base[1] - 0.10:
            mark = "  <<< 失血"
        print(f"{k.replace('reward/', ''):34s} {s:9.1f} {denom:7.1f} "
              f"{ach * 100:6.1f}% {per_step:9.4f} "
              f"{weighted_terms(cfg, {k: s}, mean_len)[k]:9.1f} {base_str:>9s}{mark}")
    # 加权合计与实测 return 对账
    wsum = sum(weighted_terms(cfg, term_means, mean_len).values())
    print(f"加权分项合计 {wsum:.1f}  vs  实测 return {reward_sum / n_envs:.1f}")


def _term_reason(env, state):
    import jax.numpy as jp

    if bool(jp.isnan(state.data.qpos).any() | jp.isnan(state.data.qvel).any()):
        return "nan"
    sp = float(env._getSinPitch_newest(state.data))
    sr = float(env._getSinRoll_newest(state.data))
    if sp < -0.6 or sp > 0.85:
        return f"pitch({sp:+.3f})"
    if abs(sr) > 0.6:
        return f"roll({sr:+.3f})"
    return "unknown"


def run_clean(env, inference, seed=0, steps=600):
    import numpy as np
    import jax
    import jax.numpy as jp

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    rng = jax.random.PRNGKey(seed)
    state = jit_reset(rng, False)

    # 观测槽位索引(26 维):cmd(0-3) gyro(4-6) sinyaw(7) sinpitch(8) sinroll(9)
    # jnts(10-13: thighL,wheelL,thighR,wheelR) ...
    OBS_SINYAW, OBS_WL, OBS_WR = 7, 11, 13

    sums = {k: 0.0 for k in TERM_KEYS}
    act_sums = {v: 0.0 for v in GATED.values()}
    reward_sum = 0.0
    telem = []
    ep_len = steps
    for i in range(steps):
        rng, act_rng = jax.random.split(rng)
        act, _ = inference(state.obs, act_rng)
        state = jit_step(state, act)
        m = state.metrics
        for k in TERM_KEYS:
            sums[k] += float(m[k])
        for v in act_sums:
            act_sums[v] += float(m[v])
        reward_sum += float(state.reward)
        ds = float(env.get_hold_position(state.data) - state.info["position_start"])
        dsy = float(env._getSinYaw_newest(state.data) - state.info["sinyaw_start"])
        obs_state = np.asarray(state.obs["state"])
        telem.append(dict(
            t=i * 0.01, ds=ds, dsy=dsy,
            obs_ds_l=float(obs_state[OBS_WL]), obs_ds_r=float(obs_state[OBS_WR]),
            obs_dsy=float(obs_state[OBS_SINYAW]),
            lin_vel=float(env.get_local_linvel_x(state.data)),
            gyro_z=float(env.get_gyro_newest(state.data)[2]),
            sinpitch=float(env._getSinPitch_newest(state.data)),
            sinroll=float(env._getSinRoll_newest(state.data)),
            thigh_l=float(state.data.qpos[env._qpos_thigh_joints[0]]),
            thigh_r=float(state.data.qpos[env._qpos_thigh_joints[1]]),
            reward=float(state.reward),
            done=float(state.done),
        ))
        if bool(state.done):
            ep_len = i + 1
            print(f"!! 提前终止 @ step {i + 1}  原因 {_term_reason(env, state)}")
            break

    # obs/reward 一致性:噪声=0 时 obs 轮槽应与 Δs 完全相等,朝向槽与 Δψ 相等
    obs_ds_err = max(max(abs(t["obs_ds_l"] - t["ds"]), abs(t["obs_ds_r"] - t["ds"]))
                     for t in telem)
    obs_dsy_err = max(abs(t["obs_dsy"] - t["dsy"]) for t in telem)
    print(f"\n[obs/reward 一致性] max|obs轮槽-Δs|={obs_ds_err:.2e}  "
          f"max|obs朝向槽-Δψ|={obs_dsy_err:.2e}")

    print_term_table("干净口径 clean (random=False, noise=0, push=0, 零指令)",
                     env._config, sums, act_sums, [ep_len], 1, reward_sum)

    # 关键物理量时间线(首 1s / 1-3s / 3s 后)
    print("\n[物理量分段均值]  t(s)  |Δs|(rad) |Δψ|  |vx|(m/s) |wz|(rad/s) "
          "|sinpitch| |sinroll| reward/step")
    for lo, hi, name in [(0, 1.0, "0-1s"), (1.0, 3.0, "1-3s"), (3.0, 99, "3s+")]:
        seg = [t for t in telem if lo <= t["t"] < hi]
        if not seg:
            continue
        n = len(seg)
        print(f"  {name:5s} "
              f"{sum(abs(t['ds']) for t in seg) / n:8.4f} "
              f"{sum(abs(t['dsy']) for t in seg) / n:8.4f} "
              f"{sum(abs(t['lin_vel']) for t in seg) / n:8.4f} "
              f"{sum(abs(t['gyro_z']) for t in seg) / n:8.4f} "
              f"{sum(abs(t['sinpitch']) for t in seg) / n:8.4f} "
              f"{sum(abs(t['sinroll']) for t in seg) / n:8.4f} "
              f"{sum(t['reward'] for t in seg) / n:8.3f}")
    return telem


def run_batch(env, inference, num_envs, seed, deterministic, random_init=None):
    import numpy as np
    import jax
    import jax.numpy as jp

    steps = int(env._config.episode_length)
    vmap_reset = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))
    vmap_step = jax.jit(jax.vmap(env.step))
    vmap_infer = jax.jit(jax.vmap(inference))

    rng = jax.random.PRNGKey(seed)
    env_rngs = jax.random.split(rng, num_envs)
    states = vmap_reset(env_rngs, random_init)

    sums = {k: jp.zeros(num_envs) for k in TERM_KEYS}
    late_sums = {k: jp.zeros(num_envs) for k in TERM_KEYS}  # step>=100 稳态窗口
    act_sums = {v: jp.zeros(num_envs) for v in GATED.values()}
    late_act_sums = {v: jp.zeros(num_envs) for v in GATED.values()}
    reward_sum = jp.zeros(num_envs)
    alive = jp.ones(num_envs, dtype=jp.bool_)
    ep_len = jp.zeros(num_envs, dtype=jp.int32)
    late_len = jp.zeros(num_envs, dtype=jp.int32)
    act_rng = jax.random.PRNGKey(0)

    for i in range(steps):
        act_rng, sub = jax.random.split(act_rng)
        acts, _ = vmap_infer(states.obs, jax.random.split(sub, num_envs))
        states = vmap_step(states, acts)
        newly_alive = alive
        late = newly_alive & (i >= 100)
        for k in TERM_KEYS:
            sums[k] += states.metrics[k] * newly_alive
            late_sums[k] += states.metrics[k] * late
        for v in act_sums:
            act_sums[v] += states.metrics[v] * newly_alive
            late_act_sums[v] += states.metrics[v] * late
        reward_sum += states.reward * newly_alive
        ep_len += newly_alive.astype(jp.int32)
        late_len += late.astype(jp.int32)
        alive = alive & (~states.done.astype(jp.bool_))
        if i % 100 == 0:
            print(f"  step {i}/{steps} 存活 {int(jax.device_get(alive.sum()))}/{num_envs}",
                  flush=True)
        if bool(jax.device_get(alive.all())) is False and int(jax.device_get(alive.sum())) == 0:
            break

    sums = {k: float(jax.device_get(v.sum())) for k, v in sums.items()}
    late_sums = {k: float(jax.device_get(v.sum())) for k, v in late_sums.items()}
    act_sums = {k: float(jax.device_get(v.sum())) for k, v in act_sums.items()}
    late_act_sums = {k: float(jax.device_get(v.sum())) for k, v in late_act_sums.items()}
    ep_lens = np.asarray(jax.device_get(ep_len))
    late_lens = np.asarray(jax.device_get(late_len))
    tag = (f"标准口径 standard"
           f" ({'stochastic' if not deterministic else 'deterministic'}, "
           f"random_init={random_init}, "
           f"push={env._config.push_config.enable}, noise={env._config.noise_config.level})")
    print_term_table(tag, env._config, sums, act_sums, ep_lens, num_envs,
                     float(jax.device_get(reward_sum.sum())))
    print_term_table(tag + " [稳态窗口 step>=100]", env._config, late_sums,
                     late_act_sums, late_lens, num_envs, 0.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["clean", "standard", "trainproto"], required=True)
    parser.add_argument("--ckpt", default=CKPT)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stochastic", action="store_true",
                        help="采样动作(训练 eval 口径);默认 deterministic(部署口径)")
    args = parser.parse_args()

    if args.mode == "trainproto":
        raise SystemExit(
            "trainproto 口径不可用:hold_recovery 已从代码移除"
            "(见 docs/hold_recovery_training.md 消亡史),仅支持 clean/standard")

    if args.mode == "clean":
        # 单环境 CPU 即可(参考 tools/repro_stand_tall.py);--num-envs>1 时走 GPU 批量
        if args.num_envs <= 1:
            os.environ.setdefault("JAX_PLATFORMS", "cpu")
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
            os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.pop("MUJOCO_GL", None)

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    import jax  # noqa: E402

    from legwheel import LegWheelRobot, default_config  # noqa: E402
    from legwheel.evaluation.runner import load_policy  # noqa: E402

    cfg = default_config()
    if args.mode == "clean":
        cfg.noise_config.level = 0.0
        cfg.push_config.enable = False
        cfg.cmd_config.cmd_prob = 0.0  # sample_command 恒返回零指令
    env = LegWheelRobot(cfg)

    inference = jax.jit(load_policy(os.path.abspath(args.ckpt),
                                    deterministic=not args.stochastic))

    if args.mode == "clean" and args.num_envs <= 1:
        run_clean(env, inference)
    else:
        random_init = False if args.mode == "clean" else True
        run_batch(env, inference, args.num_envs, args.seed,
                  deterministic=not args.stochastic, random_init=random_init)


if __name__ == "__main__":
    main()
