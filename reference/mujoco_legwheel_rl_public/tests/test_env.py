"""环境单元测试。

运行方式（在项目根目录）：
    python tests/test_env.py
或安装了 pytest 时：
    pytest tests/test_env.py -v
"""

import os
import sys

# 未 pip install -e . 时，把项目根目录加入 sys.path 以找到 legwheel 包
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import jax
import jax.numpy as jp

from legwheel import LegWheelRobot, default_config


def _make_env() -> LegWheelRobot:
    return LegWheelRobot(default_config())


def test_sample_command_covers_negative_cmds():
    """负向指令（后退/反偏航）必须能被采样到，且死区外才保留。"""
    env = _make_env()
    cfg = env._config
    n = 5000
    keys = jax.random.split(jax.random.PRNGKey(0), n)
    cmds = jax.vmap(env.sample_command)(keys)
    cmds = jp.stack(cmds)

    lin, ang = cmds[:, 0], cmds[:, 1]
    # cmd_prob=0.9，其余 10% 为全零；非零指令应正负各半。
    # 修复前负值全部被清零，负向占比约为 0。
    assert (lin < -cfg.cmd_lin_vel_x_zero_margin).mean() > 0.2, (
        "lin_vel_x 负向指令采样比例过低，负向命令可能仍被清零"
    )
    assert (ang < -cfg.cmd_ang_vel_zero_margin).mean() > 0.2, (
        "ang_vel_yaw 负向指令采样比例过低，负向命令可能仍被清零"
    )
    assert (lin > cfg.cmd_lin_vel_x_zero_margin).mean() > 0.2
    assert (ang > cfg.cmd_ang_vel_zero_margin).mean() > 0.2
    # 非零值必须落在配置范围内
    nonzero_lin = jp.abs(lin[lin != 0.0])
    nonzero_ang = jp.abs(ang[ang != 0.0])
    assert jp.all(nonzero_lin <= cfg.cmd_config.lin_vel_x[1] + 1e-6)
    assert jp.all(nonzero_ang <= cfg.cmd_config.ang_vel_yaw[1] + 1e-6)


def test_obs_and_action_shapes():
    """state=26 维、privileged_state=55 维、action_size=4（与已部署策略一致的 26 维观测）。"""
    env = _make_env()
    state = env.reset(jax.random.PRNGKey(0))
    assert state.obs["state"].shape == (26,), state.obs["state"].shape
    assert state.obs["privileged_state"].shape == (55,), (
        state.obs["privileged_state"].shape
    )
    assert env.action_size == 4
    # 一步 step 冒烟测试
    state = env.step(state, jp.zeros(env.action_size))
    assert state.obs["state"].shape == (26,)
    assert jp.isfinite(state.obs["state"]).all()


def test_action_delay():
    """action_delay_steps>0 时动作延迟生效：缓冲区滚动、旧动作先出。"""
    cfg = default_config()
    cfg.action_delay_steps = 2
    env = LegWheelRobot(cfg)
    state = env.reset(jax.random.PRNGKey(0))
    act = jp.ones(env.action_size) * 0.5
    state = env.step(state, act)
    # 缓冲区最后一行是刚输出的动作，前面仍是初始 0（说明本步作用的是 2 步前的 0）
    assert jp.allclose(state.info["act_delay_buffer"][-1], 0.5)
    assert jp.allclose(state.info["act_delay_buffer"][0], 0.0)
    state = env.step(state, act)
    state = env.step(state, act)
    # 两步之后缓冲区应全部填满 0.5
    assert jp.allclose(state.info["act_delay_buffer"], 0.5)
    assert state.obs["state"].shape == (26,)
    assert jp.isfinite(state.obs["state"]).all()

    # delay=0 时关闭延迟，直接使用当前动作
    cfg0 = default_config()
    cfg0.action_delay_steps = 0
    env0 = LegWheelRobot(cfg0)
    s0 = env0.reset(jax.random.PRNGKey(0))
    s0 = env0.step(s0, jp.ones(env0.action_size) * 0.5)
    assert jp.isfinite(s0.obs["state"]).all()


def _standstill_step(env, state, position_start=None, sinyaw_start=None):
    """零指令、零动作走一步(不刷新位置/朝向原点)。"""
    info = dict(state.info)
    info["command"] = jp.zeros(4)
    if position_start is not None:
        info["position_start"] = position_start
    if sinyaw_start is not None:
        info["sinyaw_start"] = sinyaw_start
    state = state.replace(info=info)
    return env.step(state, jp.zeros(env.action_size))


def test_hold_error_clip_obs_not_anchor():
    """推理:锚点不动,只有 obs 里的 Δs 被 clip 到带宽;两轮槽相同。"""
    cfg = default_config()
    cfg.hold_error_clip.enable = True
    cfg.push_config.enable = False
    cfg.noise_config.level = 0.0
    cfg.cmd_config.cmd_prob = 0.0
    env = LegWheelRobot(cfg)
    state = env.reset(jax.random.PRNGKey(0))
    actual_s = env.get_hold_position(state.data)
    actual_y = env._getSinYaw_newest(state.data)
    far_s = actual_s - 8.0
    state = _standstill_step(
        env, state, position_start=far_s, sinyaw_start=actual_y)
    s_lim = env._hold_error_clip_wheel
    y_lim = env._hold_error_clip_sinyaw
    assert jp.allclose(state.info["position_start"], far_s, atol=1e-5)
    true_s = env.get_hold_position(state.data) - state.info["position_start"]
    assert jp.abs(true_s) > float(s_lim)
    obs = state.obs["state"]
    assert jp.abs(obs[11]) <= s_lim + 1e-5
    assert jp.abs(obs[13]) <= s_lim + 1e-5
    assert jp.allclose(obs[11], obs[13], atol=1e-5)
    assert jp.abs(obs[11]) > 1.0
    assert abs(float(s_lim) - 5.0) < 1e-6
    assert abs(float(y_lim) - 1.0) < 1e-6


def test_hold_error_clip_default_off_keeps_anchor():
    """训练默认关闭夹紧,人为拉开的位置原点在停车时保持不变。"""
    cfg = default_config()
    cfg.push_config.enable = False
    cfg.noise_config.level = 0.0
    cfg.cmd_config.cmd_prob = 0.0
    env = LegWheelRobot(cfg)
    assert cfg.hold_error_clip.enable is False
    state = env.reset(jax.random.PRNGKey(0))
    far_s = env.get_hold_position(state.data) - 5.0
    state = _standstill_step(env, state, position_start=far_s)
    assert jp.allclose(state.info["position_start"], far_s, atol=1e-5)


def test_prepare_inference_clips_before_policy():
    """推理前 prepare_inference 重算 obs 并 clip Δs,锚点不改。"""
    cfg = default_config()
    cfg.hold_error_clip.enable = True
    cfg.push_config.enable = False
    cfg.noise_config.level = 0.0
    env = LegWheelRobot(cfg)
    state = env.reset(jax.random.PRNGKey(0))
    far_s = env.get_hold_position(state.data) - 8.0
    info = dict(state.info)
    info["command"] = jp.zeros(4)
    info["position_start"] = far_s
    state = env.prepare_inference(state.replace(info=info))
    assert jp.allclose(state.info["position_start"], far_s, atol=1e-5)
    s_lim = env._hold_error_clip_wheel
    assert jp.abs(state.obs["state"][11]) <= s_lim + 1e-5
    assert jp.allclose(state.obs["state"][11], state.obs["state"][13], atol=1e-5)


def test_hold_position_reward_on_ds():
    """位置奖励只看 Δs;Δs=0 时为 1。"""
    env = _make_env()
    r = env._reward_tracking_wheels_sinposition(jp.array(0.0), jp.array(0.0))
    assert jp.allclose(r, 1.0)
    r_off = env._reward_tracking_wheels_sinposition(jp.array(1.0), jp.array(0.0))
    assert r_off < 0.2


def test_reset_default_is_qpos0():
    """默认 reset(rng) 用规范站姿 qpos0;随机姿态需显式传 random=True(viewer/batch_test)。"""
    env = _make_env()
    s0 = env.reset(jax.random.PRNGKey(1))
    assert jp.allclose(s0.data.qpos[3:7], env.mjx_model.qpos0[3:7])
    s1 = env.reset(jax.random.PRNGKey(1), True)
    assert not jp.allclose(s1.data.qpos[3:7], env.mjx_model.qpos0[3:7])


def test_yaw_frozen_on_vx():
    """只打 vx 不刷新朝向锚点(直行保向)。"""
    cfg = default_config()
    cfg.push_config.enable = False
    cfg.noise_config.level = 0.0
    cfg.cmd_config.cmd_prob = 0.0
    env = LegWheelRobot(cfg)
    state = env.reset(jax.random.PRNGKey(0))
    far_y = env._getSinYaw_newest(state.data) + 1.0
    info = dict(state.info)
    info["command"] = jp.array([0.2, 0.0, 0.0, 0.0])
    info["sinyaw_start"] = far_y
    state = env.step(state.replace(info=info), jp.zeros(env.action_size))
    assert jp.allclose(state.info["sinyaw_start"], far_y)


def test_position_frozen_on_yaw():
    """只打 yaw 不刷新位置原点。"""
    cfg = default_config()
    cfg.push_config.enable = False
    cfg.noise_config.level = 0.0
    cfg.cmd_config.cmd_prob = 0.0
    env = LegWheelRobot(cfg)
    state = env.reset(jax.random.PRNGKey(0))
    far_s = env.get_hold_position(state.data) + 3.0
    info = dict(state.info)
    info["command"] = jp.array([0.0, 0.5, 0.0, 0.0])
    info["position_start"] = far_s
    state = env.step(state.replace(info=info), jp.zeros(env.action_size))
    assert jp.allclose(state.info["position_start"], far_s)


def test_obs_wheel_slots_are_ds():
    """两轮观测槽都是 Δs,左右差为 0。"""
    cfg = default_config()
    cfg.push_config.enable = False
    cfg.noise_config.level = 0.0
    cfg.cmd_config.cmd_prob = 0.0
    env = LegWheelRobot(cfg)
    state = env.reset(jax.random.PRNGKey(0))
    far_s = env.get_hold_position(state.data) - 1.5
    state = _standstill_step(env, state, position_start=far_s)
    obs = state.obs["state"]
    assert jp.allclose(obs[11], obs[13], atol=1e-5)
    ds = env.get_hold_position(state.data) - state.info["position_start"]
    assert jp.allclose(obs[11], ds, atol=1e-4)


def test_hold_position_wheel_signs_are_same_forward():
    """左右轮同号增加才是前进共模:两侧 +1 → Δs=+1; 反向 ±1 → Δs=0。"""
    env = _make_env()
    state = env.reset(jax.random.PRNGKey(0))
    s0 = env.get_hold_position(state.data)
    q = state.data.qpos
    data_fwd = state.data.replace(qpos=q.at[9].add(1.0).at[12].add(1.0))
    assert jp.allclose(env.get_hold_position(data_fwd) - s0, 1.0, atol=1e-6)
    data_yaw = state.data.replace(qpos=q.at[9].add(1.0).at[12].add(-1.0))
    assert jp.allclose(env.get_hold_position(data_yaw) - s0, 0.0, atol=1e-6)


def test_imu_heading_change_only_grows_dyaw():
    """IMU 航向变化只增 Δψ(obs idx 7),共模位置误差 Δs(obs idx 11/13)不变。

    docs/hold_position_heading.md 6.5 第 4 条:停车保持下只改机身姿态
    (绕 z 转 δ),Δs 完全由轮转角决定,不应受 IMU 航向影响。
    注意要改 qpos[3:7] 四元数:env.step 的 forward 运动学由 qpos 重算 xmat,
    直接改 xmat 会被覆盖。
    """
    cfg = default_config()
    cfg.push_config.enable = False
    cfg.noise_config.level = 0.0
    cfg.cmd_config.cmd_prob = 0.0
    env = LegWheelRobot(cfg)
    state = env.reset(jax.random.PRNGKey(0))
    state = _standstill_step(env, state)  # 第 1 步:零指令停车,锚点冻结
    obs1 = state.obs["state"]

    # 第 2 步前只把机身四元数绕 z 轴转 δ(IMU 航向 +δ),轮转角/速度不动
    delta = 0.3
    qz = env.euler_to_quaternion(delta, 0.0, 0.0)  # wxyz

    def quat_mul(q1, q2):  # wxyz 约定
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return jp.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ])

    q_new = state.data.qpos.at[3:7].set(quat_mul(qz, state.data.qpos[3:7]))
    state = _standstill_step(env, state.replace(data=state.data.replace(qpos=q_new)))
    obs2 = state.obs["state"]

    # Δψ(obs idx 7)随之增大(sinyaw 映射 sin(ψ·sinyaw_scale)/sinyaw_scale);
    # Δs(obs idx 11/13,两轮同槽)不变
    dyaw_expected = jp.sin(delta * env._config.sinyaw_scale) / env._config.sinyaw_scale
    assert obs2[7] - obs1[7] > 0.1
    assert jp.allclose(obs2[7] - obs1[7], dyaw_expected, atol=5e-2)
    assert jp.allclose(obs2[11], obs1[11], atol=1e-3)
    assert jp.allclose(obs2[13], obs1[13], atol=1e-3)
    assert jp.allclose(obs2[11], obs2[13], atol=1e-5)


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {e}")
    sys.exit(1 if failures else 0)
