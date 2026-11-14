"""LegWheelRobot MJX 环境:reset/step、观测、动作映射(含延迟)与传感器。"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from mujoco_playground._src import mjx_env

from legwheel.config import default_config
from legwheel.rewards import LegWheelRewardMixin


def add_noise(value: jax.Array, rng: jax.Array, noise_level: float, noise_scale: float) -> jax.Array:
    # 生成噪声并将其添加到输入值中
    noise = (2 * jax.random.uniform(rng, shape=value.shape) - 1) * noise_level * noise_scale
    return value + noise


class LegWheelRobot(LegWheelRewardMixin, mjx_env.MjxEnv):
    """自定义腿轮混合机器人环境"""
    
    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(config, config_overrides)
        _XML_PATH = self._config._XML_PATH
        self._xml_path = Path(_XML_PATH).as_posix()
        # 加载Mujoco模型：from_xml_path 使 include/mesh 相对 XML 所在目录解析，与 cwd 无关
        self._mj_model = mujoco.MjModel.from_xml_path(self._xml_path)
        self._mj_model.opt.timestep = config.sim_dt
        self._mjx_model = mjx.put_model(self._mj_model)
        self._post_init()
        
        # 奖励参数
        self.target_velocity = 0.01  # m/s
        self.energy_cost = 0.01

        self._set_normalize_params()

    def _set_normalize_params(self):
        self._cmd_ranges = {
            'lin_vel_x': jp.array(self._config.cmd_config.lin_vel_x),
            'ang_vel_yaw': jp.array(self._config.cmd_config.ang_vel_yaw),
            'height_value': jp.array(self._config.cmd_config.height_value),
            'roll_value': jp.array(self._config.cmd_config.roll_value)
        }

        self._cmd_biases = jp.array([
            (self._cmd_ranges['lin_vel_x'][0] + self._cmd_ranges['lin_vel_x'][1]) / 2,
            (self._cmd_ranges['ang_vel_yaw'][0] + self._cmd_ranges['ang_vel_yaw'][1]) / 2,
            (self._cmd_ranges['height_value'][0] + self._cmd_ranges['height_value'][1]) / 2,
            (self._cmd_ranges['roll_value'][0] + self._cmd_ranges['roll_value'][1]) / 2
        ])
        
        self._cmd_scales = jp.array([
            (self._cmd_ranges['lin_vel_x'][1] - self._cmd_ranges['lin_vel_x'][0]) / 2,
            (self._cmd_ranges['ang_vel_yaw'][1] - self._cmd_ranges['ang_vel_yaw'][0]) / 2,
            (self._cmd_ranges['height_value'][1] - self._cmd_ranges['height_value'][0]) / 2,
            (self._cmd_ranges['roll_value'][1] - self._cmd_ranges['roll_value'][0]) / 2
        ])

    def _normalize_command(self, raw_cmd):
        """动态归一化"""
        return (raw_cmd - self._cmd_biases) / (self._cmd_scales + 1e-6)

    def _denormalize_command(self, norm_cmd):
        """动态反归一化"""
        return norm_cmd * self._cmd_scales + self._cmd_biases

    def euler_to_quaternion(self, yaw, pitch, roll):
        """
        将欧拉角转换为四元数
        
        参数:
            yaw: 绕z轴旋转的角度
            pitch: 绕y轴旋转的角度
            roll: 绕x轴旋转的角度
        
        返回:
            quaternion: 形状为(4,)的数组，包含[w, x, y, z]四元数
        """
        # 计算四元数分量
        cr = jp.cos(roll * 0.5)
        sr = jp.sin(roll * 0.5)
        cp = jp.cos(pitch * 0.5)
        sp = jp.sin(pitch * 0.5)
        cy = jp.cos(yaw * 0.5)
        sy = jp.sin(yaw * 0.5)
        
        # ZYX顺序 (yaw -> pitch -> roll)
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return jp.array([w, x, y, z])

    def _post_init(self) -> None:
        # 初始化相关参数
        self._body_id = self._mj_model.body("root").id
        self._left_wheel_id = self._mj_model.body('wheel_left').id
        self._right_wheel_id = self._mj_model.body('wheel_right').id
        self._lowers = self._mj_model.jnt_range[1:, 0]
        self._uppers = self._mj_model.jnt_range[1:, 1]
        self._qpos_thigh_joints = jp.array([7, 10], dtype=jp.int32) # thigh joints
        self._qpos_wanted_joints = jp.array([7, 9, 10, 12], dtype=jp.int32)  # remove calf joints [8, 11]
        self._qvel_wanted_joints = jp.array([6, 8, 9, 11], dtype=jp.int32)  # remove calf joints [7, 10]
        self._hold_error_clip_wheel, self._hold_error_clip_sinyaw = (
            self._hold_error_clip_limits())

    def _hold_error_clip_limits(self):
        """推理夹紧带宽:Δs(共模轮角)与 Δψ。"""
        return (
            jp.asarray(float(self._config.hold_error_clip.wheel_sinpos)),
            jp.asarray(float(self._config.hold_error_clip.sinyaw)),
        )

    def get_hold_position(self, data: mjx.Data) -> jax.Array:
        """未饱和左右轮角共模,位置 hold 用。

        本 XML 左右轮铰链轴向在世界系同向(+Y),同号 q 表示同向滚动。
        前进时 q_L 与 q_R 一起增加,因此 s=(q_L+q_R)/2,不能写成 (q_L-q_R)/2。
        """
        q = data.qpos[self._qpos_wanted_joints]
        return 0.5 * (q[1] + q[3])

    def _clip_hold_obs_error(self, sinyaw: jax.Array, ds: jax.Array):
        """锚点不动,只把观测里的 Δψ / Δs clip 到训练上沿。"""
        if not self._config.hold_error_clip.enable:
            return sinyaw, ds
        sinyaw = jp.clip(
            sinyaw, -self._hold_error_clip_sinyaw, self._hold_error_clip_sinyaw)
        ds = jp.clip(ds, -self._hold_error_clip_wheel, self._hold_error_clip_wheel)
        return sinyaw, ds

    def prepare_inference(self, state: mjx_env.State) -> mjx_env.State:
        """推理前重算 obs(含当前 command);误差越界则 clip,锚点不改。"""
        if not self._config.hold_error_clip.enable:
            return state
        obs = self._get_obs(state.data, state.info)
        return state.replace(obs=obs)

    def _cmd_start_step(self) -> int:
        delay_s = float(self._config.cmd_start_delay_s)
        return int(round(delay_s / float(self._config.ctrl_dt)))

    def _hold_position_moving(self, command: jax.Array) -> jax.Array:
        """有前进指令时刷位置原点。"""
        return jp.abs(command[0]) > self._config.cmd_lin_vel_x_zero_margin

    def _hold_heading_moving(self, command: jax.Array) -> jax.Array:
        """有偏航指令时刷朝向原点。"""
        return jp.abs(command[1]) > self._config.cmd_ang_vel_zero_margin

    def get_sin_balance_angle(self, data) -> jax.Array:
        balance_angle = jp.where(self._config.sinpitch_zero, self.get_balance_angle(data), 0.0)
        sin_balance_zero = jp.sin(balance_angle)
        return sin_balance_zero
    
    def reset(self, rng: jax.Array, random: bool = False) -> mjx_env.State:
        rng, rng1, rng2 = jax.random.split(rng, 3)
        
        def random_qpos(rng: jax.Array):
            # 随机初始化位置
            qpos0 = self.mjx_model.qpos0
            # qpos0 = qpos0.at[:2].set(
            #     qpos0[:2]
            #     + jax.random.uniform(rng1, shape=(2,), minval=-8., maxval=8.0)
            # )

            # 随机初始化方向
            rng, key = jax.random.split(rng)
            ang_z = jax.random.uniform(key, shape=(1,), minval=-jp.pi, maxval=jp.pi)
            rng, key = jax.random.split(rng)
            ang_y = jax.random.uniform(key, shape=(1,), minval=-jp.pi/8, maxval=jp.pi/8) # -22~+22 from 0.09m height
            rng, key = jax.random.split(rng)
            ang_x = jax.random.uniform(key, shape=(1,), minval=-jp.pi/18, maxval=jp.pi/18) # -10~+10
            quat = self.euler_to_quaternion(ang_z[0], ang_y[0], ang_x[0])
            qpos0 = qpos0.at[3:7].set(quat)
            
            return qpos0
        qpos0 = jp.where(random, random_qpos(rng), self.mjx_model.qpos0)
        data = mjx_env.make_data(self.mjx_model, qpos=qpos0)


        sin_balance_zero = self.get_sin_balance_angle(data)


        position_start = self.get_hold_position(data)

        sinyaw_start = self._getSinYaw_newest(data)

        # 初始化metrics，根据reward_config中的配置
        metrics = {
            "reward/tracking_lin_vel": jp.zeros(()),
            "reward/tracking_wheels_sinposition": jp.zeros(()),
            "reward/tracking_sinyaw": jp.zeros(()),
            "reward/tracking_ang_vel": jp.zeros(()),
            "reward/pitch_cost": jp.zeros(()),
            "reward/roll_cost": jp.zeros(()),
            "reward/tracking_height_jnt": jp.zeros(()),
            "reward/tracking_height_action": jp.zeros(()),
            "reward/energy_cost": jp.zeros(()),
            "reward/action_rate": jp.zeros(()),
            "reward/pitch_velocity_penalty": jp.zeros(()),
            "reward/termination_penalty": jp.zeros(()),
            # 调参主指标用的归一化跟踪误差平方(见 rewards.py 与调参计划第 2 节);
            # 必须与 _get_reward 写入的键一致,否则 scan carry 结构不匹配
            "track/err2_lin_vel": jp.zeros(()),
            "track/err2_ang_vel": jp.zeros(()),
            # 门控项激活步数计数(与 rewards.py 写入键一致,scan carry 结构匹配)
            "track/active_steps_sinpos": jp.zeros(()),
            "track/active_steps_sinyaw": jp.zeros(()),
        }

        rng, cmd_rng = jax.random.split(rng)
        cmd = self.sample_command(cmd_rng)

        # Sample push interval.
        rng, push_rng = jax.random.split(rng)
        push_interval = jax.random.uniform(
            push_rng,
            minval=self._config.push_config.interval_range[0],
            maxval=self._config.push_config.interval_range[1],
        )
        push_interval_steps = jp.round(push_interval / self.dt).astype(jp.int32)
        info = {
            "rng": rng,
            "step": 0,
            # Episode 步数计数（用于区分：提前终止 vs 正常跑满 episode_length）
            "episode_step": jp.array(0, dtype=jp.int32),
            "command": cmd,
            "last_act": jp.zeros(self.mjx_model.nu),
            "last_last_act": jp.zeros(self.mjx_model.nu),
            "current_act": jp.zeros(self.mjx_model.nu),
            # Track previous wheel torque separately from last_act
            "last_wheel_torque": jp.zeros(2),
            "push": jp.array([0.0, 0.0]),
            "push_step": 0,
            "push_interval_steps": push_interval_steps,
            "sin_balance_zero": sin_balance_zero,
            "position_start": position_start,
            "sinyaw_start": sinyaw_start,
            # 速度环 PID 状态（按左右轮顺序）
            "pid_integral": jp.zeros(2),
            "pid_prev_error": jp.zeros(2),
            # 记录 RL 输出的目标轮角速度（用于奖励计算）
            "desired_wheel_angvel": jp.zeros(2),
            # store recent wheel policy commands (history length 2)
            "wheels_commands_history": jp.zeros((2, 2)),
            # last smoothed target wheel angular velocity
            "last_target_wheel_vel": jp.zeros(2),
            # 动作延迟缓冲：最近 action_delay_steps 步的策略输出，最旧的在第一行
            "act_delay_buffer": jp.zeros(
                (int(self._config.action_delay_steps), self.mjx_model.nu)
            ),
        }

        reward, done = jp.zeros(2)
        obs = self._get_obs(data, info)
        out = mjx_env.State(data, obs, reward, done, metrics, info)
        return out

    # ===== 速度环与工具函数 =====
    def _get_wheel_angvel(self, data: mjx.Data) -> jax.Array:
        """返回左右轮角速度[rad/s]，按关节角速度读取，顺序 [left, right]。"""
        joint_velocity = data.qvel[self._qvel_wanted_joints]
        wheel_vel_idx = jp.array([1, 3], dtype=jp.int32)
        wheel_angvel = joint_velocity[wheel_vel_idx]
        return wheel_angvel
  
    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        # 处理推力
        state.info["rng"], push1_rng, push2_rng = jax.random.split(state.info["rng"], 3)
        push_theta = jax.random.uniform(push1_rng, maxval=2 * jp.pi)
        push_magnitude = jax.random.uniform(
            push2_rng,
            minval=self._config.push_config.magnitude_range[0],
            maxval=self._config.push_config.magnitude_range[1],
        )
        push = jp.array([jp.cos(push_theta), jp.sin(push_theta)])
        push *= (
            jp.mod(state.info["push_step"] + 1, state.info["push_interval_steps"])
            == 0
        )
        push *= self._config.push_config.enable
        push *= push_magnitude
        qvel = state.data.qvel
        qvel = qvel.at[:2].set(push + qvel[:2])
        data = state.data.replace(qvel=qvel)
        state = state.replace(data=data)

        act_idx = jp.array([0, 1, 2, 3], dtype=jp.int32)
        raw_act = action[act_idx]
        state.info["current_act"] = raw_act

        # 动作延迟（模拟控制链路延迟）：作用于仿真的是 action_delay_steps 步前的策略输出。
        # 注意 last_act/current_act 等 obs 与 action_rate 惩罚仍记录策略原始输出 raw_act。
        if self._config.action_delay_steps > 0:
            delayed_act = state.info["act_delay_buffer"][0]
            state.info["act_delay_buffer"] = jp.vstack(
                [state.info["act_delay_buffer"][1:], action[None, :]]
            )
            action = delayed_act

        thigh_motorid = jp.array([0, 1], dtype=jp.int32)
        wheel_motorid = jp.array([2, 3], dtype=jp.int32)

        # 应用动作
        action_scale = jp.array([self._config.action_scale.thigh, self._config.action_scale.thigh,
                            self._config.action_scale.wheel, self._config.action_scale.wheel])
        action_bias = jp.array([self._config.action_bias.thigh, self._config.action_bias.thigh,
                            self._config.action_bias.wheel, self._config.action_bias.wheel])
        if self._config.delta_thigh_action:
            delta_action = action[thigh_motorid]
            thigh_pos_now = state.data.qpos[self._qpos_thigh_joints]
            motor_thigh_targets = thigh_pos_now + delta_action * action_scale[thigh_motorid]
            motor_thigh_targets = jp.clip(motor_thigh_targets, self._config.action_range.thigh[0], self._config.action_range.thigh[1])
        else:
            # 非增量模式下：前两位为期望大腿位置，后两位为期望轮角速度
            scaled = (action + action_bias) * action_scale
            motor_thigh_targets = scaled[thigh_motorid]
            motor_thigh_targets = jp.clip(motor_thigh_targets, self._config.action_range.thigh[0], self._config.action_range.thigh[1])

        # wheel 目标（角速度）统一逻辑：
        # - vel_pid.enable=True：动作 [-1,1] -> 轮角速度指令 [rad/s]
        #   （residual_target=True 时在下方作为残差叠加到当前轮速；否则作为绝对目标）
        # - vel_pid.enable=False：沿用原先的 action_bias/scale 直接下发到仿真
        if self._config.vel_pid.enable:
            desired_wheel_angvel = (action[wheel_motorid] + self._config.vel_pid.target_bias) * self._config.vel_pid.target_scale
        else:
            desired_wheel_angvel = (action[wheel_motorid] + action_bias[wheel_motorid]) * action_scale[wheel_motorid]

        # 基于目标轮速 -> 平滑 -> P 控制扭矩
        if self._config.vel_pid.enable:
            iters = int(self._config.vel_pid.iters)
            iters = max(1, iters)

            residual_target = getattr(self._config.vel_pid, "residual_target", False)
            if residual_target:
                # 残差模式（与已部署策略 20260111-2109 的训练语义一致）：
                # 策略输出 -> 速度残差；目标轮速 = step 开始时的当前轮速 + 残差（硬限幅）；
                # wheels_commands_history 记录残差（对应观测 previous_wheels_policyout，
                # 与固件 control.cpp 中 wheels_commands_history 的语义一致）。
                desired_residual = action[wheel_motorid] * self._config.vel_pid.target_scale
                state.info["wheels_commands_history"] = jp.vstack(
                    [state.info["wheels_commands_history"][1:], desired_residual[None, :]]
                )
                current_wheel_angvel = self._get_wheel_angvel(state.data)
                target_vel = jp.clip(
                    current_wheel_angvel + desired_residual,
                    -self._config.vel_pid.target_scale,
                    self._config.vel_pid.target_scale,
                )
            else:
                # wheel 绝对目标角速度（硬限幅以防 target_bias/scale 组合越界）
                target_vel = jp.clip(
                    desired_wheel_angvel,
                    -self._config.vel_pid.target_scale,
                    self._config.vel_pid.target_scale,
                )

                # append to wheels commands history (keep last 2)
                state.info["wheels_commands_history"] = jp.vstack(
                    [state.info["wheels_commands_history"][1:], target_vel[None, :]]
                )

            # 低通平滑（重要）
            last_tgt = state.info.get("last_target_wheel_vel", jp.zeros_like(target_vel))
            target_vel_smoothed = self._config.vel_pid.notsmoothing * target_vel + (1 - self._config.vel_pid.notsmoothing) * last_tgt
            state.info["last_target_wheel_vel"] = target_vel_smoothed

            # 记录用于日志/奖励的期望速度（这里记为 smoothed 目标）
            state.info["desired_wheel_angvel"] = target_vel_smoothed

            # 内循环：在每次子步中用 P 控制把速度跟踪到目标速度（产生扭矩）
            kp = self._config.vel_pid.kp
            ki = self._config.vel_pid.ki
            torque_limit = self._config.vel_pid.torque_limit
            integral_limit = self._config.vel_pid.integral_limit
            dt_each = self._config.sim_dt
            direct_torque = getattr(self._config.vel_pid, "direct_torque", False)

            def body_fun(i, carry):
                data_c, last_ctrl_c, integral_c = carry
                current_wheel_angvel_c = self._get_wheel_angvel(data_c)
                # 速度误差
                err = target_vel_smoothed - current_wheel_angvel_c

                # 积分项（抗稳态误差），并做反积分饱和
                integral_n = jp.clip(integral_c + err * dt_each, -integral_limit, integral_limit)

                if direct_torque:
                    # 直接生成扭矩（PI）：tau = kp*err + ki*integral
                    torque_cmd = kp * err + ki * integral_n
                    new_torque = jp.clip(torque_cmd, -torque_limit, torque_limit)
                else:
                    # 兼容旧逻辑：delta torque 累加（相当于对扭矩做积分器）
                    delta_torque = kp * err
                    last_torque = last_ctrl_c[-2:]
                    new_torque = jp.clip(last_torque + delta_torque, -torque_limit, torque_limit)

                ctrl = jp.concatenate([motor_thigh_targets, new_torque])
                data_n = mjx_env.step(self.mjx_model, data_c, ctrl, 1)
                return (data_n, ctrl, integral_n)

            # use dedicated last_wheel_torque instead of last_act for previous torque
            prev_torque = state.info.get("last_wheel_torque", jp.zeros(2))
            pid_integral0 = state.info.get("pid_integral", jp.zeros(2))
            carry0 = (state.data, jp.concatenate([motor_thigh_targets, prev_torque]), pid_integral0)
            data, ctrl, pid_integral = jax.lax.fori_loop(0, iters, body_fun, carry0)
            state.info["pid_integral"] = pid_integral
        else:
            # 不启用 PID：保持旧逻辑，直接把轮的“目标角速度”传给仿真
            ctrl = jp.concatenate([motor_thigh_targets, desired_wheel_angvel])
            data = mjx_env.step(self.mjx_model, state.data, ctrl, self.n_substeps)
            # 非 PID 模式下，同样保存期望角速度（虽本需求仅要求 PID 模式）
            state.info["desired_wheel_angvel"] = desired_wheel_angvel

        # 计算奖励
        rewards = self._get_reward(data, ctrl, state.info, state.metrics)
        reward = sum(rewards.values())

        # 更新状态信息
        state.info["last_last_act"] = state.info["last_act"]
        state.info["last_act"] = raw_act
        state.info["last_wheel_torque"] = ctrl[2:]

        state.info["push"] = push
        state.info["push_step"] += 1
        state.info["step"] += 1
        state.info["rng"], cmd_rng = jax.random.split(state.info["rng"])
        cmd_start_step = self._cmd_start_step()
        state.info["command"] = jp.where(
            (state.info["step"] == cmd_start_step) & (self._config.cmd_config.cmd_prob > 0.0),
            self.sample_command(cmd_rng),
            state.info["command"],
        )

        # 位置原点只随 vx 刷;朝向原点只随 yaw 刷。
        pos_moving = self._hold_position_moving(state.info["command"])
        yaw_moving = self._hold_heading_moving(state.info["command"])
        state.info["position_start"] = jp.where(
            pos_moving, self.get_hold_position(data),
            state.info["position_start"])
        state.info["sinyaw_start"] = jp.where(
            yaw_moving, self._getSinYaw_newest(data), state.info["sinyaw_start"])

        sin_balance_zero = self.get_sin_balance_angle(data)
        state.info["sin_balance_zero"] = sin_balance_zero

        # 更新 episode 步数（每调用一次 step +1）
        state.info["episode_step"] = state.info.get("episode_step", jp.array(0, dtype=jp.int32)) + 1

        # 判断终止条件
        done = self._is_termination(data)

        # 非正常终止惩罚：正常跑满 episode_length 不惩罚；提前终止(done=True 且 step < episode_length)才惩罚。
        termination_penalty = jp.zeros(())
        if self._config.reward_config.termination_penalty_enable:
            early_termination = done & (state.info["episode_step"] < jp.array(self._config.episode_length, dtype=jp.int32))
            termination_penalty = jp.where(
                early_termination,
                jp.array(self._config.reward_config.termination_penalty_value),
                jp.zeros(()),
            )
            reward = reward + termination_penalty
            state.metrics["reward/termination_penalty"] = termination_penalty

        # 重置步数计数器
        state.info["step"] = jp.where(
            done | (state.info["step"] > self._config.cmd_config.command_change_interval),
            0,
            state.info["step"],
        )

        # 若提前终止，重置 episode_step（防止某些 wrapper 不立即 reset 时计数继续累积）
        state.info["episode_step"] = jp.where(done, jp.array(0, dtype=jp.int32), state.info["episode_step"])

        done = done.astype(float)
        # 获取观察值
        obs = self._get_obs(data, state.info)

        return mjx_env.State(data, obs, reward, done, state.metrics, state.info)
    
    def _is_termination(self, data: mjx.Data) -> jax.Array:
        # 判断是否终止
        done = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()

        sinpitch = self._getSinPitch_newest(data)
        # isfall = jp.abs(sinpitch) > 0.6
        isfall = (sinpitch < -0.6) | (sinpitch > 0.85)
        sinroll = self._getSinRoll_newest(data)
        isfall2 = jp.abs(sinroll) > 0.6
        done = done | isfall | isfall2

        return done

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        # 获取观察值
        info["rng"], noise_rng = jax.random.split(info["rng"])
        gyro = self.get_gyro_newest(data)
        noisy_gyro = add_noise(gyro, noise_rng, self._config.noise_config.level, self._config.noise_config.scales.gyro)

        accelerometer = self.get_accelerometer_newest(data)

        info["rng"], noise_rng = jax.random.split(info["rng"])
        sinyaw = self._getSinYaw_newest(data)
        sinyaw_diff = sinyaw - info["sinyaw_start"]
        sinyaw = jp.where(self._config.sinyaw, sinyaw_diff, sinyaw)
        ds = self.get_hold_position(data) - info["position_start"]
        ds = jp.where(self._config.wheel_sinpos_start, ds, self.get_hold_position(data))
        sinyaw, ds = self._clip_hold_obs_error(sinyaw, ds)
        wheels_sinposition = jp.stack([ds, ds])
        noisy_sinyaw = add_noise(sinyaw, noise_rng, self._config.noise_config.level, self._config.noise_config.scales.pitch)

        info["rng"], noise_rng = jax.random.split(info["rng"])
        sinpitch = self._getSinPitch_newest(data) - info["sin_balance_zero"]
        noisy_sinpitch = add_noise(sinpitch, noise_rng, self._config.noise_config.level, self._config.noise_config.scales.pitch)

        info["rng"], noise_rng = jax.random.split(info["rng"])
        sinroll = self._getSinRoll_newest(data) - jp.sin(info["command"][3])
        noisy_sinroll = add_noise(sinroll, noise_rng, self._config.noise_config.level, self._config.noise_config.scales.roll)

        qpos_jnts = data.qpos[self._qpos_wanted_joints]
        wheel_pos_idx = jp.array([1, 3], dtype=jp.int32)
        qpos_jnts = qpos_jnts.at[wheel_pos_idx].set(wheels_sinposition)
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_jnts = add_noise(qpos_jnts, noise_rng, self._config.noise_config.level, self._config.noise_config.scales.joint_pos)

        joint_velocity = data.qvel[self._qvel_wanted_joints]
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_velocity = add_noise(joint_velocity, noise_rng, self._config.noise_config.level, self._config.noise_config.scales.joint_vel)
        thigh_vel_idx = jp.array([0, 2], dtype=jp.int32)
        # 真机无法稳定测量大腿速度：该维改喂零均值纯噪声（而非恒零），让策略学会忽略它；
        # 同时归一化 std 退化为噪声尺度而非 ~1e-6，避免固件喂入真实速度时被归一化放大到分布外
        info["rng"], noise_rng = jax.random.split(info["rng"])
        thigh_vel_noise = add_noise(jp.zeros(2), noise_rng, self._config.noise_config.level, self._config.noise_config.scales.joint_vel)
        noisy_velocity = noisy_velocity.at[thigh_vel_idx].set(thigh_vel_noise)

        wheel_vel_idx = jp.array([1, 3], dtype=jp.int32)
        wheel_lin_vel = joint_velocity[wheel_vel_idx] * self._config.wheel_radius
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_wheel_linvel = add_noise(wheel_lin_vel, noise_rng, self._config.noise_config.level, self._config.noise_config.scales.linvel)

        lin_vel = self.get_local_linvel_x(data)

        # previous wheel policy actions (history length 2, flattened)
        previous_wheels_policyout = info.get("wheels_commands_history", jp.zeros((2, 2))).reshape(-1)

        # joint velocity error = last smoothed target wheel vel - current wheel joint velocity
        wheel_ctrl_vel_error = info.get("last_target_wheel_vel", jp.zeros(2)) - joint_velocity[wheel_vel_idx]
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_wheel_ctrl_vel_error = add_noise(wheel_ctrl_vel_error, noise_rng, self._config.noise_config.level, self._config.noise_config.scales.joint_vel)

        state_list = [info["command"], noisy_gyro, noisy_sinyaw, noisy_sinpitch, noisy_sinroll,
                      noisy_jnts, noisy_velocity, noisy_wheel_linvel, previous_wheels_policyout, noisy_wheel_ctrl_vel_error]
    
        if self._config.last_act_obs:
            state_list.append(info["last_act"])
        
        state = jp.hstack(state_list)

        if self._config.obs_history_len>1:
            if "obs_history" not in info:
                info["obs_history"] = jp.tile(jp.zeros_like(state), (self._config.obs_history_len, 1))
            info["obs_history"] = jp.vstack([info["obs_history"][1:], state[None, :]])
            state_history = info["obs_history"].reshape(-1)
        else:
            state_history = state

        priviledged_state_list = [state_history, gyro, accelerometer, sinyaw, sinpitch, sinroll, self.get_upvector(data), qpos_jnts, joint_velocity, wheel_lin_vel,
                                               lin_vel, previous_wheels_policyout, wheel_ctrl_vel_error]

        if self._config.last_act_obs:
            priviledged_state_list.append(info["last_act"])
        priviledged_state = jp.hstack(priviledged_state_list)
        
        if self._config.obs_normalize.state_mean is not None:
            state_history = (state_history - self._config.obs_normalize.state_mean) / (self._config.obs_normalize.state_std + 1e-6)
            privileged_state = (privileged_state - self._config.obs_normalize.privileged_state_mean) / (self._config.obs_normalize.privileged_state_std + 1e-6)

        return {"state": state_history, "privileged_state": priviledged_state}

    def get_wheels_sinposition(self, data: mjx.Data) -> jax.Array:
        qpos_jnts = data.qpos[self._qpos_wanted_joints]
        wheel_idx = jp.array([1, 3], dtype=jp.int32)
        positions = qpos_jnts[wheel_idx] * self._config.wheel_sin_scale
        positions = jp.where(self._config.wheel_pos_nosin, positions, jp.sin(positions))
        positions = positions / self._config.wheel_sin_scale
        return positions

    def get_upvector(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(self.mj_model, data, 'root_upvector')

    def get_gravity(self, data: mjx.Data) -> jax.Array:
        imu_site_id = self.mj_model.site("root").id
        return data.site_xmat[imu_site_id].T @ jp.array([0, 0, -1])

    def get_global_linvel(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(
            self.mj_model, data, 'root_linvel'
        )

    def get_global_angvel(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(
            self.mj_model, data, 'root_angvel'
        )

    def get_local_linvel_x(self, data: mjx.Data) -> jax.Array:
        lenvel_x = mjx_env.get_sensor_data(self.mj_model, data, 'root_local_linvel')[0]
        return lenvel_x
    
    def get_local_wheel_linvel_x(self, data: mjx.Data) -> jax.Array:
        left_vel_x = mjx_env.get_sensor_data(self.mj_model, data, 'wheel_left_site_linvel')[0]
        right_vel_x = mjx_env.get_sensor_data(self.mj_model, data, 'wheel_right_site_linvel')[0]
        lenvel_x = jp.array([left_vel_x, right_vel_x])
        return lenvel_x

    def get_accelerometer_newest(self, data: mjx.Data) -> jax.Array:
        newval = mjx_env.get_sensor_data(self.mj_model, data, 'root_accel')
        return newval
    
    def get_gyro_newest(self, data: mjx.Data) -> jax.Array:
        nowval = mjx_env.get_sensor_data(self.mj_model, data, 'root_gyro')
        return nowval

    def angle_between_vectors(self, v1, v2):
        norm_v1 = jp.linalg.norm(v1)
        norm_v2 = jp.linalg.norm(v2)
        dot_product = jp.dot(v1, v2)
        cos_theta = dot_product / (norm_v1 * norm_v2 + 1e-8)  # 避免除零错误
        theta = jp.arccos(jp.clip(cos_theta, -1.0, 1.0))  # 弧度制
        return theta

    def get_balance_angle(self, data: mjx.Data) -> jax.Array:
        com = data.subtree_com[self._body_id]
        left_wheel_pos = data.xpos[self._left_wheel_id]
        right_wheel_pos = data.xpos[self._right_wheel_id]
        midpoint_wheel = (left_wheel_pos + right_wheel_pos) / 2
        vector_com_to_midpoint = midpoint_wheel - com

        xmat_head = data.xmat[self._body_id]
        local_x_axis = xmat_head[:, 0]

        angle_rad = self.angle_between_vectors(vector_com_to_midpoint, local_x_axis)
        balance_angle = jp.pi/2 - angle_rad

        return balance_angle
        
    def sample_command(self, rng: jax.Array) -> jax.Array:
        # 采样命令
        rng1, rng2, rng3, rng4, rng5, rng6, rng7 = jax.random.split(rng, 7)

        lin_vel_x = jax.random.uniform(
            rng1, minval=self._config.cmd_config.lin_vel_x[0], maxval=self._config.cmd_config.lin_vel_x[1]
        )

        ang_vel_yaw = jax.random.uniform(
            rng3,
            minval=self._config.cmd_config.ang_vel_yaw[0],
            maxval=self._config.cmd_config.ang_vel_yaw[1],
        )

        height_value = jax.random.uniform(
            rng5, minval=self._config.cmd_config.height_value[0], maxval=self._config.cmd_config.height_value[1]
        )

        roll_value = jax.random.uniform(
            rng6, minval=self._config.cmd_config.roll_value[0], maxval=self._config.cmd_config.roll_value[1]
        )

        zero_states = jp.array([0.0, 0.0, 0.0, 0.0])
        rnd_states = jp.array([lin_vel_x, ang_vel_yaw, height_value, roll_value])
        # 死区判断必须用 abs：负向指令（后退/反偏航）同样有效，只在幅值过小时清零
        tmp_onevalue = jp.where(self._config.cmd_config.enable.lin_vel_x, lin_vel_x, 0.0)
        tmp_onevalue = jp.where(jp.abs(tmp_onevalue) > self._config.cmd_lin_vel_x_zero_margin, tmp_onevalue, 0.0)
        rnd_states = rnd_states.at[0].set(tmp_onevalue)
        tmp_onevalue = jp.where(self._config.cmd_config.enable.ang_vel_yaw, ang_vel_yaw, 0.0)
        tmp_onevalue = jp.where(jp.abs(tmp_onevalue) > self._config.cmd_ang_vel_zero_margin, tmp_onevalue, 0.0)
        rnd_states = rnd_states.at[1].set(tmp_onevalue)
        tmp_onevalue = jp.where(self._config.cmd_config.enable.height_value, height_value, 0.0)
        rnd_states = rnd_states.at[2].set(tmp_onevalue)
        tmp_onevalue = jp.where(self._config.cmd_config.enable.roll_value, roll_value, 0.0)
        rnd_states = rnd_states.at[3].set(tmp_onevalue)

        notdo_cmd = jax.random.bernoulli(rng4, p=1.0-self._config.cmd_config.cmd_prob)
        raw_cmd = jp.where(
            notdo_cmd,
            zero_states,
            rnd_states,
        )

        # return self._normalize_command(raw_cmd)
        return raw_cmd

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self.mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model
