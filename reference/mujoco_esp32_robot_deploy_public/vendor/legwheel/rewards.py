"""奖励函数集合(mixin)。

依赖宿主类(LegWheelRobot)提供:self._config、self._cmd_scales
以及各传感器 getter(get_local_linvel_x / get_gyro_newest / ...)。
方法体从原 legwheel_env.py 原样搬移。
"""

from typing import Any

import jax
import jax.numpy as jp
from mujoco import mjx


class LegWheelRewardMixin:


    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        metrics: dict[str, Any],
    ) -> jax.Array:
        # actual_cmd = self._denormalize_command(info["command"])
        actual_cmd = info["command"]

        # 计算基础跟踪奖励
        tracking_lin_vel = self._reward_tracking_lin_vel_x(
            actual_cmd[0], self.get_local_linvel_x(data)
        )

        # tracking_wheels_sinposition = jp.zeros(())
        tracking_wheels_sinposition = self._reward_tracking_wheels_sinposition(
            self.get_hold_position(data), info["position_start"]
        )
        # 位置项:仅 vx 在死区内;朝向项:仅 yaw 在死区内
        gate_sinpos = jp.abs(actual_cmd[0]) <= self._config.cmd_lin_vel_x_zero_margin
        gate_sinyaw = jp.abs(actual_cmd[1]) <= self._config.cmd_ang_vel_zero_margin
        tracking_wheels_sinposition = jp.where(
            jp.logical_not(gate_sinpos), 0.0, tracking_wheels_sinposition)

        tracking_sinyaw = self._reward_tracking_sinyaw(
            self._getSinYaw_newest(data), info["sinyaw_start"]
        )
        tracking_sinyaw = jp.where(jp.logical_not(gate_sinyaw), 0.0, tracking_sinyaw)

        tracking_ang_vel = self._reward_tracking_ang_vel(
            actual_cmd[1], self.get_gyro_newest(data)[2]
        )
        
        # 计算姿势奖励
        pitch_cost = self._reward_getSinPitch(data, info['sin_balance_zero'])
        roll_cost = self._reward_getSinRoll(actual_cmd[3], data)
        
        # 计算高度关节跟踪奖励
        # roll-height 冲突消解:height 参考值投影到当前 roll 指令下的可行区间
        # (h_ref = clip(h_cmd, lo+δ, hi-δ), δ = coupling·|cmd_roll|),
        # roll 占用的 split 不再以牺牲 height 跟踪为代价(反之亦然)。
        height_cmd = actual_cmd[2]
        if self._config.reward_config.height_feasible_projection:
            delta_roll = (self._config.reward_config.roll_thigh_coupling
                          * jp.abs(actual_cmd[3]))
            h_lo = self._config.action_range.thigh[0] + delta_roll
            h_hi = self._config.action_range.thigh[1] - delta_roll
            height_ref = jp.clip(height_cmd, h_lo, h_hi)
        else:
            height_ref = height_cmd
        tracking_height_jnt = self._reward_tracking_height_jnt(height_ref,
                                                               data.qpos[self._qpos_thigh_joints],
                                                               True)

        tracking_height_action = self._reward_tracking_height_action(height_ref, action[:2])

        # 计算能耗
        qvel_idx = self._qvel_wanted_joints
        energy_cost_thigh_L = jp.abs(data.qvel[qvel_idx[0]])
        energy_cost_thigh_R = jp.abs(data.qvel[qvel_idx[2]])
        energy_cost_action_wheel_L = jp.abs(action[2])
        energy_cost_action_wheel_R = jp.abs(action[3])
        energy_cost = (energy_cost_thigh_L + energy_cost_thigh_R + energy_cost_action_wheel_L + energy_cost_action_wheel_R)/4
        
        # 计算动作变化率惩罚
        action_rate = self._cost_action_rate(
            info["current_act"], info["last_act"], info["last_last_act"]
        )

        # calculate pitch velocity penalty
        gyro = self.get_gyro_newest(data)
        pitch_velocity_penalty = jp.abs(gyro[1])

        deconflict_lin_ang = 1.0
        deconflict_roll_height = 1.0
        if self._config.reward_config.do_deconflict:
            # deconflict between tracking_lin_vel and tracking_ang_vel
            deconflict_lin_ang = self._get_deconflict_factor(
                self._diff_tracking_lin_vel_x(actual_cmd[0], self.get_local_linvel_x(data)), 
                self._diff_tracking_ang_vel(actual_cmd[1], self.get_gyro_newest(data)[2])
            )

            # deconflict between roll and height joint
            deconflict_roll_height = self._get_deconflict_factor(
                self._diff_tracking_roll(actual_cmd[3], data), 
                self._diff_tracking_height_jnt(actual_cmd[2], data.qpos[self._qpos_thigh_joints])
            )

        metrics["reward/tracking_lin_vel"] = tracking_lin_vel
        metrics["reward/tracking_wheels_sinposition"] = tracking_wheels_sinposition
        metrics["reward/tracking_sinyaw"] = tracking_sinyaw
        metrics["reward/tracking_ang_vel"] = tracking_ang_vel
        metrics["reward/pitch_cost"] = pitch_cost
        metrics["reward/roll_cost"] = roll_cost
        metrics["reward/tracking_height_jnt"] = tracking_height_jnt
        metrics["reward/energy_cost"] = energy_cost
        metrics["reward/action_rate"] = action_rate
        metrics["reward/pitch_velocity_penalty"] = pitch_velocity_penalty
        metrics["reward/tracking_height_action"] = tracking_height_action

        # 调参主指标:归一化跟踪误差平方(见 docs/hyperparam_tuning_plan.md 第 2 节)。
        # 归一化常数固定为基线指令量程 v_max=0.2 m/s、ω_max=1.5 rad/s,
        # 不随 trial 的 cmd 范围变化,保证跨 trial 可比。
        # EvalWrapper 对 metrics 逐 episode 求和,RMSE = sqrt(mean(err²和)/avg_episode_length)。
        lin_vel_err = self._diff_tracking_lin_vel_x(actual_cmd[0], self.get_local_linvel_x(data))
        ang_vel_err = self._diff_tracking_ang_vel(actual_cmd[1], self.get_gyro_newest(data)[2])
        metrics["track/err2_lin_vel"] = (lin_vel_err / 0.2) ** 2
        metrics["track/err2_ang_vel"] = (ang_vel_err / 1.5) ** 2

        # 门控项达成率的实测分母(run2 计划 5.3 工程配套):
        # 本步是否处于零指令窗口(激活即计 1,EvalWrapper 逐 episode 求和)
        metrics["track/active_steps_sinpos"] = jp.where(gate_sinpos, 1.0, 0.0)
        metrics["track/active_steps_sinyaw"] = jp.where(gate_sinyaw, 1.0, 0.0)

        # 修改后的动态权重分配部分
        height_scale = self._config.reward_config.height_scale
        vel_scale = self._config.reward_config.vel_scale
        pitch_scale = self._config.reward_config.pitch_scale
        zero_keep_scale = self._config.reward_config.zero_keep_scale
        roll_scale = self._config.reward_config.roll_scale

        # 姿势奖励（最高优先级）
        posture = pitch_cost * pitch_scale
        reward_num = 1
        if self._config.cmd_config.enable.roll_value:
            posture += roll_cost * roll_scale
            reward_num += 1

        zero_keep = 0.0
        if self._config.wheel_sinpos_start:
            zero_keep += tracking_wheels_sinposition
            reward_num += 1
        if self._config.sinyaw:
            zero_keep += tracking_sinyaw
            reward_num += 1
        zero_keep *= zero_keep_scale

        # 速度跟踪奖励（次优先级）
        vel_tracking = 0.0
        reward_num_vel = 0
        if self._config.cmd_config.enable.lin_vel_x:
            vel_tracking += tracking_lin_vel
            reward_num_vel += 1
        if self._config.cmd_config.enable.ang_vel_yaw:
            vel_tracking += tracking_ang_vel
            reward_num_vel += 1
            
        vel_tracking *= vel_scale #base_scale # * deconflict_lin_ang * (1 + posture/reward_num)
        

        # 高度奖励（最低优先级）
        height_reward = 0.0
        if self._config.cmd_config.enable.height_value:
            height_reward += tracking_height_jnt + tracking_height_action
        height_reward *= height_scale #base_scale # * deconflict_roll_height * (1 + vel_tracking/reward_num_vel)

        # 能耗相关惩罚项
        energy_scale = self._config.reward_config.energy_scale
        action_rate_scale = self._config.reward_config.action_rate_scale
        energy_cost_reward = energy_cost * energy_scale
        action_rate_reward = action_rate * action_rate_scale
        pitch_velocity_penalty_reward = pitch_velocity_penalty * self._config.reward_config.pitch_velocity_penalty_scale
        
        rewards = {
            "combined_tracking": vel_tracking,
            "posture": posture,
            "zero_keep": zero_keep,
            "height_reward": height_reward,
            "energy_cost": energy_cost_reward,
            "action_penalty": action_rate_reward,
            "pitch_velocity_penalty": pitch_velocity_penalty_reward,
        }

        return rewards

    def _get_deconflict_factor(
        self,
        one: jax.Array,
        two: jax.Array,
    ) -> jax.Array:
        one_error = jp.abs(one)
        two_error = jp.abs(two)
        sigma = self._config.reward_config.sigma
        conflict_factor = jp.exp(-(one_error * two_error)/sigma)
        return conflict_factor
    
    # Tracking rewards.
    def _diff_tracking_lin_vel_x(
        self,
        commands: jax.Array,
        local_vel_x: jax.Array,
    ) -> jax.Array:
        """计算线性速度跟踪误差"""
        return local_vel_x - commands

    def _diff_tracking_ang_vel(
        self,
        commands: jax.Array,
        ang_vel: jax.Array,
    ) -> jax.Array:
        """计算角速度跟踪误差"""
        return ang_vel - commands

    def _reward_tracking_lin_vel_x(
        self,
        commands: jax.Array,
        local_vel_x: jax.Array,
    ) -> jax.Array:
        lin_vel_error = jp.abs(self._diff_tracking_lin_vel_x(commands, local_vel_x))
        # lin_vel_error = lin_vel_error / self._config.wheel_radius  # convert m/s to rad/s
        sigma = self._config.reward_config.sigma * self._cmd_scales[0]
        # sigma = jp.where(commands<0.03, sigma/10, sigma)
        return jp.exp(-lin_vel_error / sigma)
    
    def _reward_tracking_wheels_sinposition(
        self,
        hold_position: jax.Array,
        position_start: jax.Array,
    ) -> jax.Array:
        """位置 hold:|Δs|,s 为未饱和左右轮角共模。"""
        diff_s = hold_position - position_start
        sigma = self._config.reward_config.sigma_wheel
        return jp.exp(-jp.abs(diff_s) / sigma)

    def _reward_tracking_sinyaw(
        self,
        sinyaw: jax.Array,
        sinyaw_start: jax.Array,
    ) -> jax.Array:
        diff_sinyaw = sinyaw - sinyaw_start
        sigma = self._config.reward_config.sigma * self._cmd_scales[3]
        return jp.exp(-jp.abs(diff_sinyaw) / sigma)

    def _reward_tracking_ang_vel(
        self,
        commands: jax.Array,
        ang_vel: jax.Array,
    ) -> jax.Array:
        ang_vel_error = jp.abs(self._diff_tracking_ang_vel(commands, ang_vel))
        sigma = self._config.reward_config.sigma * self._cmd_scales[1]
        return jp.exp(-ang_vel_error / sigma)

    def _reward_tracking_height_jnt(
        self,
        commands: jax.Array,
        thigh_jnt: jax.Array,
        use_both_thighs: bool = True,
    ) -> jax.Array:
        diff_height = self._diff_tracking_height_jnt(commands, thigh_jnt, use_both_thighs)
        sigma = self._config.reward_config.sigma * self._cmd_scales[2]
        return jp.exp(-jp.abs(diff_height) / sigma)

    def _reward_tracking_height_action(
        self,
        commands: jax.Array,
        action: jax.Array,
    ) -> jax.Array:
        diff_action = jp.mean(action - commands)
        sigma = self._config.reward_config.sigma * self._cmd_scales[2]
        return jp.exp(-jp.abs(diff_action) / sigma)

    def _diff_tracking_height_jnt(
        self,
        commands: jax.Array,
        thigh_jnt: jax.Array,
        use_both_thighs: bool = True,
    ) -> jax.Array:
        diff = jp.where(
            use_both_thighs,
            jp.mean(thigh_jnt - commands),
            jp.min(thigh_jnt) - commands,
        )
        return diff

    def _cost_action_rate(
        self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array
    ) -> jax.Array:
        c1 = jp.mean(jp.abs(act - last_act))
        c2 = jp.mean(jp.abs(act + last_last_act - 2 * last_act))
        return c1 + c2
    
    def _reward_getSinPitch(self, data, sin_balance_zero) -> jax.Array:
        sinpitch = self._getSinPitch_newest(data)
        sigma = self._config.reward_config.sigma * self._cmd_scales[3]
        reward = jp.exp(-jp.abs(sinpitch - sin_balance_zero)/sigma)
        return reward
    
    def _getSinPitch_newest(self, data, newest: bool=False) -> jax.Array:
        R = data.xmat[self._body_id]
        sinpitch = -R[2, 0]
        return sinpitch
    
    def _reward_getSinRoll(self, cmd, data) -> jax.Array:
        sinroll = self._getSinRoll_newest(data)
        sigma = self._config.reward_config.sigma * self._cmd_scales[3]
        reward = jp.exp(-jp.abs(sinroll - cmd)/sigma)
        return reward
    
    def _getSinRoll_newest(self, data) -> jax.Array:
        R = data.xmat[self._body_id]
        sinpitch = self._getSinPitch_newest(data)
        cospitch = jp.sqrt(1 - sinpitch**2)
        sinroll = R[2, 1] / cospitch
        return sinroll
    
    def _getSinYaw_newest(self, data) -> jax.Array:
        R = data.xmat[self._body_id]
        yaw = jp.arctan2(R[1, 0], R[0, 0])
        sinyaw_scale = self._config.sinyaw_scale
        yaw_scale = yaw * sinyaw_scale
        sin_yaw = jp.sin(yaw_scale) / sinyaw_scale
        return sin_yaw

    def _diff_tracking_roll(self, commands, data):
        sinroll = self._getSinRoll_newest(data)
        return  sinroll - commands
