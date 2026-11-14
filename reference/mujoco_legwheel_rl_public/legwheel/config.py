"""环境与训练的默认配置。"""

import json

from ml_collections import config_dict

from legwheel.assets import XML_PATH


def default_config() -> config_dict.ConfigDict:
    # kp = 0.01
    sim_dt = 0.0025
    ctrl_dt = 0.010 #0.002 #0.006
    episode_time = 6.0  # seconds
    cmd_start_delay_s = 0.2  # seconds: delay before sampling the first command
    episode_length = int(round(episode_time / ctrl_dt))
    command_change_interval = int(round(3.0 / ctrl_dt))  # 3s (in env steps)
    return config_dict.create(
        _XML_PATH = XML_PATH,
        episode_time=episode_time,
        ctrl_dt=ctrl_dt,
        sim_dt=sim_dt,
        cmd_start_delay_s=cmd_start_delay_s,
        episode_length=episode_length,
        action_repeat=1,
        # 动作延迟步数（模拟控制链路延迟，1 步 = 10ms @100Hz 控制频率）。设为 0 关闭。
        # 与已部署策略 20260111-2109 的训练快照（82ba252 时代）一致：无动作延迟。
        action_delay_steps=0,
        obs_history_len=1,
        wheel_radius=0.0195,
        # 速度环 PID 配置（用于轮子速度->扭矩控制）
        # 以下参数与已部署策略 20260111-2109 的训练快照（82ba252 时代）一致：
        # 残差速度目标 + 0.5 低通 + delta torque 累加 P 控制。
        vel_pid=config_dict.create(
            enable=True,
            kp=0.0002,
            ki=0.0, #0.01,
            kd=0.0, #0.00005,
            iters=int(ctrl_dt/sim_dt),            # 每个环境 step 内 PID 控制次数
            torque_limit=0.0295, #0.04,   # 轮电机扭矩限幅（单位：N·m），需与模型匹配
            integral_limit=1.0,  #
            # True: 扭矩由(PI)直接生成并限幅；False: 采用“delta torque 累加”的方式（更像扭矩积分器）
            direct_torque=False,
            # True: 策略输出为“残差速度”，目标轮速 = 当前轮速 + 动作*target_scale，
            #       wheels_commands_history 记录残差（与固件 control.cpp 一致）；
            # False: 策略输出为绝对目标轮速，历史记录限幅后的目标。
            residual_target=True,
            # PID 模式下，轮目标（角速度）映射专用的缩放/偏置（与非 PID 的 action_scale/bias 分离）
            target_scale=30.0,  # 将动作 [-1,1] 映射到约 [-30,30] rad/s 残差
            target_bias=0.0,
            notsmoothing=0.5,      # 轮目标速度低通平滑系数（0-1），值越小越平滑
            obs_residual_history=True,  # 是否在观测中包含 previous_wheels_policyout 和 wheel_ctrl_vel_error
        ),
        cmd_lin_vel_x_zero_margin =0.000195, # 0.01*0.0195,
        cmd_ang_vel_zero_margin = 0.01,
        sinpitch_zero=True,
        wheel_sin_scale=0.1,
        wheel_sinpos_start=True,
        wheel_pos_nosin=False,
        sinyaw = True,
        sinyaw_scale = 0.5,
        # 推理/部署专用:锚点不动,只把观测里的 Δs / Δψ clip 到设定带宽。
        # 训练必须保持 enable=False(训练入口 curriculum_learning 有硬防护)。
        # 见 docs/hold_position_heading.md。
        hold_error_clip=config_dict.create(
            enable=False,
            wheel_sinpos=5.0,
            sinyaw=1.0,
        ),
        # 与已部署策略 20260111-2109 的训练快照一致：大腿增量动作、观测不含 last_act（26 维）。
        delta_thigh_action = True,
        last_act_obs = False,
        action_scale=config_dict.create(
            thigh=0.25, # ([-1,1] - 1)/4 = [-0.5, 0], 0.25
            wheel=0.0295, #0.04 #0.06 1.0 #1.24, 1.5
        ),
        action_bias=config_dict.create(
            thigh=-1.0, # -1.0
            wheel=0.0
        ),
        action_range=config_dict.create(
            thigh=[-0.5, 0.0],
        ),
        noise_config=config_dict.create(
          level=1.0,  # Set to 0.0 to disable noise.
          scales=config_dict.create(
              joint_pos=0.02,  # rad
              joint_vel=1.0,  # rad/s
              linvel=0.01, # m/s
              gyro=0.02,  # angvel.
              pitch=0.02, # pitch
              roll=0.02, # roll
          ),
        ),
        reward_config=config_dict.create(
            do_deconflict=False,
            height_scale=2.0,
            vel_scale=2.0,
            roll_scale=4.0,
            base_scale=1.0,      # 基础奖励系数
            pitch_scale=0.0,
            posture_scale=1.,   # 姿势相关基础系数
            zero_keep_scale=10.0, # 零保持奖励系数
            energy_scale=-1.0,  # 能耗系数
            action_rate_scale=-0.5, #-1.0, #-20.0, #-100.0, # 动作变化速率惩罚系数
            pitch_velocity_penalty_scale=-0.5, # 速度惩罚系数
            # 终止惩罚：用于“非正常死亡”(done=True，摔倒/数值异常)时给一个明显负奖励。
            # 注意：episode time limit 通常由 wrapper 截断(truncation)，不走这里的 done。
            termination_penalty_enable=False,
            termination_penalty_value=-10.0,
            sigma=0.5,            # 统一跟踪误差系数
            sigma_wheel=0.5,
            # roll-height 冲突消解(见 docs/hyperparam_tuning_plan_run2.md 5.1):
            # roll 靠 thigh 差动(split)实现,与 height(mean)共用 thigh 行程
            # [-0.5, 0];指令对 (h, φ) 在 |h| 接近端点时几何上不可同时满足。
            # True: height 参考值投影到给定 roll 指令下的可行区间
            #   h_ref = clip(h_cmd, lo+δ, hi-δ), δ = roll_thigh_coupling·|cmd_roll|
            # roll 拿足 split,height 在剩余可行域内做到最好,二者不再互相牺牲。
            height_feasible_projection=True,
            # 耦合系数:thigh 差动半幅 / |sinroll|,策略在环实测
            # (tools/analyze_roll_height_conflict.py,roll_height_conflict.png)
            roll_thigh_coupling=0.433,
        ),
        push_config=config_dict.create(
          enable=True, # Set to False to disable push.
          interval_range=[1.0, 5.0], # seconds
          magnitude_range=[0.0, 0.5], # m/s
          # 强推过采样(抗推力鲁棒性实验,docs/push_robustness_experiment_plan.md):
          # mag = lo + (hi-lo) * u^pow;pow<1 分布偏向高端(如 0.5 时 [0,0.5] 内
          # ≥0.4 的占比从 20% 升至 ~45%)。默认 1.0 = 原均匀分布,逐比特零变化。
          magnitude_pow=1.0,
          # burst 短间隔剧集:>0 时每个 episode 以该概率改用 burst_interval_range
          # 采样推力间隔(整集高频连推,练"未回稳再挨打")。默认 0.0 = 关闭,零变化。
          burst_prob=0.0,
          burst_interval_range=[0.3, 0.8], # seconds
        ),
        cmd_config=config_dict.create(
            command_change_interval=command_change_interval,
            cmd_prob=0.9, # Probability of doing command. 0.9
            lin_vel_x=[-0.2, 0.2],
            ang_vel_yaw=[-1.5, 1.5],
            height_value=[-0.5, 0.], # thigh joint range
            roll_value=[-0.2, 0.2],
            enable=config_dict.create(
                lin_vel_x=True,
                ang_vel_yaw=True,
                height_value=True,
                roll_value=True
            ),
        ),
        obs_normalize = config_dict.create(
            state_mean=None,
            state_std=None,
            privileged_state_mean=None,
            privileged_state_std=None
        )
    )


def parse_override_value(raw: str):
    """解析 "key=value" 形式 override 的 value 部分:优先按 JSON 解析
    (数字/列表/布尔/null),失败则按字符串处理。"""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw


def parse_override_pairs(pairs) -> dict:
    """把 ["a.b=1.0", "c=[1,2]"] 形式的参数列表解析为 {dotted_key: value}。"""
    overrides = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"override 须为 key=value 形式: {pair!r}")
        key, raw = pair.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"override 键为空: {pair!r}")
        overrides[key] = parse_override_value(raw.strip())
    return overrides


def _cast_like(existing, value, dotted_key: str):
    """把 override 值向已存在字段的类型对齐(ConfigDict 对类型不一致的赋值会抛错)。"""
    if isinstance(existing, bool):
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in ("true", "false"):
            return value.lower() == "true"
        raise TypeError(f"{dotted_key}: 期望 bool,得到 {value!r}")
    if isinstance(existing, int) and not isinstance(existing, bool):
        if isinstance(value, (int, float)) and float(value).is_integer():
            return int(value)
        raise TypeError(f"{dotted_key}: 期望 int,得到 {value!r}")
    if isinstance(existing, float):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        raise TypeError(f"{dotted_key}: 期望 float,得到 {value!r}")
    if isinstance(existing, str):
        return str(value)
    if isinstance(existing, (list, tuple)):
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"{dotted_key}: 期望 list,得到 {value!r}")
        return list(value)
    return value


def apply_config_overrides(
    cfg: config_dict.ConfigDict, overrides: dict
) -> config_dict.ConfigDict:
    """把 dotted-key override 写入 ConfigDict(原地修改并返回 cfg)。

    例: {"reward_config.vel_scale": 3.0, "vel_pid.kp": 0.0003}。
    中间路径必须已存在;叶子为新字段时按原样追加(调用方需自行保证键名正确)。
    """
    for dotted_key, value in (overrides or {}).items():
        parts = dotted_key.split(".")
        node = cfg
        for part in parts[:-1]:
            if part not in node:
                raise KeyError(f"未知配置路径: {dotted_key}")
            node = node[part]
        leaf = parts[-1]
        if leaf in node:
            value = _cast_like(node[leaf], value, dotted_key)
        setattr(node, leaf, value)
    return cfg
