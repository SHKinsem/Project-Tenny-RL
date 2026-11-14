"""按命令用例评估 checkpoint:速度/姿态/高度/roll 跟踪曲线 + 演示视频。"""

import collections
import os
from datetime import datetime

import jax
import jax.numpy as jp
import mujoco
import mediapy as media
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    from brax.training.agents.ppo import checkpoint
except ImportError:
    # 旧版 brax 无该模块(评估需训练机上的新版 brax)
    checkpoint = None

from legwheel import LegWheelRobot, default_config


class CommandEvaluator:
    """控制指令评测器，负责环境初始化和策略加载"""
    def __init__(self, ckpt_path, ctrl_dt=None, sim_dt=None,
                 wheel_sin_scale=None, wheel_sinpos_start = False, wheel_pos_nosin = False,
                 push=False, ckpt_path_norm=None):
        self.ckpt_path_norm = ckpt_path_norm
        self.env = self._init_env(ctrl_dt, sim_dt, wheel_sin_scale, wheel_sinpos_start, wheel_pos_nosin, push)
        self.jit_reset = jax.jit(self.env.reset)
        self.jit_step = jax.jit(self.env.step)
        self.jit_get_local_linvel_x = jax.jit(self.env.get_local_linvel_x)
        self.params = checkpoint.load(ckpt_path)
        self.jit_inference_fn = checkpoint.load_policy(ckpt_path)
        self.dt = self.env.dt

    def _init_env(self, ctrl_dt=None, sim_dt=None,
                  wheel_sin_scale=None, wheel_sinpos_start=False, wheel_pos_nosin = False, push=False):
        """初始化测试环境配置"""
        cfg = default_config()
        if ctrl_dt is not None:
            cfg.ctrl_dt = ctrl_dt
        if sim_dt is not None:
            cfg.sim_dt = sim_dt
        if wheel_sin_scale is not None:
            cfg.wheel_sin_scale = wheel_sin_scale
        cfg.push_config.enable = push
        cfg.wheel_sinpos_start  = wheel_sinpos_start
        cfg.wheel_pos_nosin = wheel_pos_nosin

        if self.ckpt_path_norm is not None: # normalize
            params = checkpoint.load(self.ckpt_path_norm)
            state_mean = params[0].mean['state']
            state_std = params[0].std['state']
            cfg.obs_normalize.state_mean = state_mean
            cfg.obs_normalize.state_std = state_std
            privileged_state_mean = params[0].mean['privileged_state']
            privileged_state_std = params[0].std['privileged_state']
            cfg.obs_normalize.privileged_state_mean = privileged_state_mean
            cfg.obs_normalize.privileged_state_std = privileged_state_std

        env = LegWheelRobot(cfg)
        return env


class TestRunner:
    """测试运行器，包含完整的测试逻辑"""
    def __init__(self, evaluator, writefile=False):
        self.writefile = writefile
        self.evaluator = evaluator
        self.test_cases = []
        self.state_pos = [
            ["cmd", 4], ["gyro", 3], ["sinyaw", 1],
            ["sinpitch", 1], ["sinroll", 1],
            ["jnts", 4], ["velocity", 4],
            ["linvel", 2], ['pid_hist', 4], ['pid_error', 2], ["last_act", 4]
        ]
        self.index_map = self._build_index_map()

    def add_test_case(self, name, command):
        """添加测试用例
        Args:
            name: 测试名称 (e.g. "Forward_0.5mps")
            command: 控制指令数组 [lin_vel, ang_vel, height, roll]
        """
        self.test_cases.append((name, command))

    def _build_index_map(self):
        """构建state观测向量的索引映射"""
        index_map = {}
        current_idx = 0
        for name, length in self.state_pos:
            clean_name = name.strip().replace('\n', '')  # 清理字段名中的特殊字符
            index_map[clean_name] = (current_idx, current_idx + length)
            current_idx += length
        return index_map


    def save_state_obs(self, state_obs, filename="state_obs_output.txt"):
        """保存状态观测值到文件，保留6位小数"""
        with open(filename, "a") as file:
            obs_list = state_obs.tolist()
            obs_str = ",".join(f"{x:.6f}" for x in obs_list)
            file.write(obs_str + "\n")  # 每行保存一个 step 的数据

    def _run_test(self, command, duration=5.0):
        """执行单个测试"""
        data = []
        rollout = []
        rng = jax.random.PRNGKey(2211)
        state = self.evaluator.jit_reset(rng, True)

        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.evaluator.env.mj_model, self.camera)
        self.camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        root_body_id = mujoco.mj_name2id(
            self.evaluator.env.mj_model,
            mujoco.mjtObj.mjOBJ_BODY,
            'root'
        )
        self.camera.trackbodyid = root_body_id
        self.camera.azimuth = 30
        self.camera.elevation = -20
        self.camera.distance = 0.8 #df0.3
        self.scene_option = mujoco.MjvOption()

        # 覆盖初始命令
        # 注意：state 是 JAX pytree/不可变结构，直接修改 state.info 在 JIT 下不可靠。
        # 这里用 replace 确保 command 真正进入后续的 jit_step。
        state = state.replace(info={**state.info, "command": jp.array(command)})
        index_map = self.index_map
        first_write = True
        for step in range(int(duration / self.evaluator.dt)):
            act_rng, rng = jax.random.split(rng)
            ctrl, _ = self.evaluator.jit_inference_fn(state.obs, act_rng)
            state = self.evaluator.jit_step(state, ctrl)
            linvel_x = self.evaluator.jit_get_local_linvel_x(state.data)
            rollout.append(state)  # 记录状态用于渲染
            # 记录观测数据
            state_obs = state.obs['state']
            if self.evaluator.ckpt_path_norm is not None: # denormalize
                state_obs = state.obs['state'] * self.evaluator.env._config.obs_normalize.state_std + self.evaluator.env._config.obs_normalize.state_mean
            obs_data = {
                'time': step * self.evaluator.dt,
                'command': state.info["command"],
                'lin_vel': linvel_x,
                'ang_vel': state_obs[index_map['gyro'][0]+2],  # 取yaw轴角速度
                'height_left': state_obs[index_map['jnts'][0]],   # 取第一个thigh关节高度
                'height_right': state_obs[index_map['jnts'][0]+2],   # 取第二个thigh关节高度
                'wheel_left': state_obs[index_map['jnts'][0]+1],
                'wheel_right': state_obs[index_map['jnts'][0]+3],
                'roll': state_obs[index_map['sinroll'][0]],  # 滚转角正弦值
                'ctrl': state.data.ctrl,
                'sinpitch': state_obs[index_map['sinpitch'][0]],
                # 车轮角速度与扭矩
                'wheel_angvel_left': state_obs[index_map['velocity'][0]+1],
                'wheel_angvel_right': state_obs[index_map['velocity'][0]+3],
                'torque_wheel_left': state.data.ctrl[2],
                'torque_wheel_right': state.data.ctrl[3],
                'gyro_x': state_obs[index_map['gyro'][0]],
                'gyro_y': state_obs[index_map['gyro'][0]+1],
                'gyro_z': state_obs[index_map['gyro'][0]+2]
            }
            if 'sinyaw' in index_map:
                obs_data['sinyaw'] = state_obs[index_map['sinyaw'][0]]
            if self.writefile:
                if first_write:
                    write_type = "w"
                    first_write = False
                else:
                    write_type = "a"
                simtimestamp = step * self.evaluator.dt * 1000
                ctrl = state.data.ctrl.tolist()
                with open("state_obs_output.txt", write_type) as file:
                    obslist = [simtimestamp,0]+state_obs.tolist()+ctrl+[0]*8
                    formatted = [f"{x:.6f}" for x in obslist]
                    obs_str = ",".join(formatted)  # 使用逗号分隔，保留6位小数
                    file.write(obs_str + "\n")  # 每行保存一个 step 的数据
            data.append(obs_data)
        return pd.DataFrame(data), rollout

    def _analyze(self, df, folder, test_name, rollout):
        render_every = 10
        frames = self.evaluator.env.render(
            rollout[::render_every],
            height=720,
            width=1280,
            camera=self.camera,
            scene_option=self.scene_option
        )
        fps = 1/self.evaluator.dt/render_every
        media.write_video(f"{folder}/{test_name}_demo.mp4", frames, fps=fps)

        imagename = f"{folder}/{test_name}_performance.jpg"
        print(test_name)
        """数据分析与可视化"""
        fig, axs = plt.subplots(9, 1, figsize=(12, 40.5))

        # 速度跟踪
        axs_0 = axs[0]
        axs_0.plot(df['time'], df['command'].apply(lambda x: x[0]), 'r--', label='Target LinVel')
        axs_0.plot(df['time'], df['lin_vel'], 'b-', label='Actual LinVel')
        axs_0.set_title(f'{test_name}\nLinear Velocity Tracking')

        # 姿态跟踪
        axs_1 = axs[1]
        axs_1.plot(df['time'], df['command'].apply(lambda x: x[1]), 'r--', label='Target AngVel')
        axs_1.plot(df['time'], df['ang_vel'], 'g-', label='Actual AngVel')
        axs_1.set_title('Angular Velocity Tracking')

        ctrl = np.stack(df['ctrl'].values)
        ctrl_thigh_left = ctrl[:, 0]
        ctrl_thigh_right = ctrl[:, 1]

        # 高度跟踪
        axs_2 = axs[2]
        axs_2.plot(df['time'], df['command'].apply(lambda x: x[2]), 'r--', label='Target Height')
        axs_2.plot(df['time'], df['height_left'], 'm-', label='Actual Height Left')
        axs_2.plot(df['time'], df['height_right'], 'y-', label='Actual Height Right')
        axs_2.plot(df['time'], ctrl_thigh_left, 'b-', label='Ctrl Thigh Left')
        axs_2.plot(df['time'], ctrl_thigh_right, 'g-', label='Ctrl Thigh Right')
        axs_2.set_title('Joint Height Tracking')

        # 滚转控制
        axs_3 = axs[3]
        axs_3.plot(df['time'], df['command'].apply(lambda x: x[3]), 'r--', label='Target Roll')
        axs_3.plot(df['time'], df['roll'], 'c-', label='Actual Roll')
        axs_3.set_title('Body Roll Tracking')

        # pitch and yaw（yaw 放到右侧 Y 轴）
        axs_4 = axs[4]
        ax4_r = axs_4.twinx()

        # 左轴：pitch 相关
        axs_4.plot(df['time'], df['sinpitch'], 'm-', label='Actual Pitch')
        if 'pitch_zero' in df.columns:
            axs_4.plot(df['time'], df['pitch_zero'], 'r--', label='Pitch Zero')
        axs_4.set_ylabel('Pitch (sin)')

        # 右轴：yaw 相关
        if 'sinyaw' in df.columns:
            ax4_r.plot(df['time'], df['sinyaw'], 'c-', label='Actual SinYaw')
        if 'sinyaw_start' in df.columns:
            ax4_r.plot(df['time'], df['sinyaw_start'], 'g--', label='SinYaw Start')
        ax4_r.set_ylabel('Yaw (sin)')

        # 单独设置左右轴图例位置，避免遮挡
        ax4_r.legend(loc='lower right')

        axs_4.set_title('Body SinPitch / SinYaw')

        # 车轮位置
        axs_5 = axs[5]
        axs_5.plot(df['time'], df['wheel_left'], 'r-', label='wheel_left position')
        axs_5.plot(df['time'], df['wheel_right'], 'g-', label='wheel_right position')
        if 'wheels_sinposition_start_left' in df.columns:
            axs_5.plot(df['time'], df['wheels_sinposition_start_left'], 'r--', label='Wheel Left SinPos Start')
        if 'wheels_sinposition_start_right' in df.columns:
            axs_5.plot(df['time'], df['wheels_sinposition_start_right'], 'g--', label='Wheel Right SinPos Start')
        axs_5.set_title('Wheel Position')
        # 车轮动作
        ctrl_wheel_left = ctrl[:, 2]
        ctrl_wheel_right = ctrl[:, 3]
        ctrl_rate_wheel_left = np.mean(np.abs(ctrl_wheel_left[1:] - ctrl_wheel_left[:-1]))
        ctrl_rate_wheel_right = np.mean(np.abs(ctrl_wheel_right[1:] - ctrl_wheel_right[:-1]))
        axs_6 = axs[6]
        axs_6.plot(df['time'], ctrl_wheel_left, 'r-', label='Ctrl Wheel Left')
        axs_6.plot(df['time'], ctrl_wheel_right, 'g-', label='Ctrl Wheel Right')
        axs_6.set_title(f'Ctrl Wheel \
            \nCtrl Rate: Left:{ctrl_rate_wheel_left:.5f}, Right:{ctrl_rate_wheel_right:.5f} \
            \nLast One Value: Left: {ctrl_wheel_left[-1]:.5f} Right: {ctrl_wheel_right[-1]:.5f}')

        # 陀螺仪三轴角速度
        for axis in ['gyro_x', 'gyro_y', 'gyro_z']:
            if axis in df.columns:
                df[axis] = df[axis].apply(lambda v: float(v) if not np.isscalar(v) else v)

        # 计算并打印各轴的均值和标准差
        gyro_mean = df[['gyro_x', 'gyro_y', 'gyro_z']].mean()
        gyro_std = df[['gyro_x', 'gyro_y', 'gyro_z']].std()

        axs_7 = axs[7]
        axs_7.plot(
            df['time'], df['gyro_x'], 'r-',
            label=f"Gyro X (μ={gyro_mean['gyro_x']:.3f}, σ={gyro_std['gyro_x']:.3f})"
        )
        axs_7.plot(
            df['time'], df['gyro_y'], 'g-',
            label=f"Gyro Y (μ={gyro_mean['gyro_y']:.3f}, σ={gyro_std['gyro_y']:.3f})"
        )
        axs_7.plot(
            df['time'], df['gyro_z'], 'b-',
            label=f"Gyro Z (μ={gyro_mean['gyro_z']:.3f}, σ={gyro_std['gyro_z']:.3f})"
        )
        axs_7.set_title('Gyro Angular Velocity (x/y/z)')
        axs_7.set_ylabel('Angular Velocity')

        # 扭矩与角速度随时间（双Y轴）
        ax_l = axs[8]
        ax_r = ax_l.twinx()
        ax_l.plot(df['time'], df['torque_wheel_left'], 'r-', label='Torque Left')
        ax_l.plot(df['time'], df['torque_wheel_right'], 'g-', label='Torque Right')
        ax_l.set_xlabel('Time (s)')
        ax_l.set_ylabel('Wheel Torque (command)')
        ax_l.set_title('Wheel Torque and Angular Velocity vs Time')
        ax_r.plot(df['time'], df['wheel_angvel_left'], 'r--', label='AngVel Left')
        ax_r.plot(df['time'], df['wheel_angvel_right'], 'g--', label='AngVel Right')
        ax_r.set_ylabel('Wheel Angular Velocity')
        # 单独为右轴添加图例，避免被下面的循环覆盖
        ax_r.legend(loc='lower right')

        for ax in axs:
            ax.legend()
            ax.grid(True)
        plt.tight_layout()
        plt.savefig(imagename)
        plt.close()

    def run_all(self, duration=5.0):
        """执行全部测试用例"""
        results = {}
        timestr = datetime.now().strftime("%Y%m%d-%H%M")
        save_dir = f"test/{timestr}"
        os.makedirs(save_dir, exist_ok=True)  # 自动创建目录
        for name, cmd in self.test_cases:
            df, rollout = self._run_test(cmd, duration=duration)
            self._analyze(df, save_dir, name, rollout)
            results[name] = df
        return results


def load_params(ckpt_folders, info_dict):
    abs_path = os.getcwd()
    all_data = []

    for ckpt_folder in ckpt_folders:
        ckpt_path = os.path.join(abs_path, ckpt_folder)
        params = checkpoint.load(ckpt_path)

        state_mean = params[0].mean['state']
        state_std = params[0].std['state']
        state_len = sum(dim for dim in info_dict.values())

        # 提取最后25个元素
        state_mean = state_mean[-state_len:]
        state_std = state_std[-state_len:]

        # 构建特征名称列表和对应的长度
        feature_names = list(info_dict.keys())
        feature_lengths = list(info_dict.values())

        # 创建有序字典
        state_dict = collections.OrderedDict()
        current_idx = 0
        for i in range(len(feature_names)):
            feature_name = feature_names[i]
            feature_length = feature_lengths[i]
            state_dict[feature_name] = {
                'mean': state_mean[current_idx:current_idx + feature_length],
                'std': state_std[current_idx:current_idx + feature_length]
            }
            current_idx += feature_length

        # 对 mean 和 std 进行四舍五入到三位小数
        for key in state_dict:
            state_dict[key]['mean'] = np.around(state_dict[key]['mean'], 3)
            state_dict[key]['std'] = np.around(state_dict[key]['std'], 3)

        # 使用 pandas 美化输出
        df = pd.DataFrame({
            'Feature': feature_names,
            'Mean': [state_dict[name]['mean'] for name in feature_names],
            'Std': [state_dict[name]['std'] for name in feature_names]
        })

        # 插入一行显示检查点文件名
        header_row = pd.DataFrame({'Checkpoint': [f"Checkpoint: {ckpt_folder}"], 'Feature': [None], 'Mean': [None], 'Std': [None]})
        all_data.append(header_row)
        all_data.append(df)

        # 插入一个空行
        empty_row = pd.DataFrame({'Checkpoint': [None], 'Feature': [None], 'Mean': [None], 'Std': [None]})
        all_data.append(empty_row)

    # 合并所有检查点的数据
    final_df = pd.concat(all_data, ignore_index=True)

    # 调整列顺序，将 'Checkpoint' 列移到最前面
    final_df = final_df[['Checkpoint', 'Feature', 'Mean', 'Std']]

    print(final_df)

    # 保存为 CSV 文件
    csv_file_path = os.path.join(abs_path, 'state_normalization.csv')
    final_df.to_csv(csv_file_path, index=False, na_rep='')

    # 打印原始信息
    print("state -> normalization -> mean:", state_mean, len(state_mean))
    print("state -> normalization std:", state_std, len(state_std))


def evaluate(
    ckpt_path: str,
    push: bool = True,
    duration: float = 6.0,
    writefile: bool = True,
    ckpt_path_norm=None,
):
    """初始化评测系统并运行默认测试用例集。"""
    evaluator = CommandEvaluator(ckpt_path, ctrl_dt = 0.010, sim_dt = 0.0025,
                                 wheel_sin_scale=0.1, wheel_sinpos_start = True, wheel_pos_nosin = False,
                                 push=push, ckpt_path_norm=ckpt_path_norm)
    runner = TestRunner(evaluator, writefile=writefile)

    # 添加测试场景
    runner.add_test_case("FailTest", [0.08, 0.0, -0.36, -0.09])

    # 执行测试并生成结果
    return runner.run_all(duration=duration)


def params_analyze():
    info_dict = collections.OrderedDict([
        ("info[\"command\"]", 4),
        ("noisy_gyro", 3),
        ("noisy_sinpitch", 1),
        ("noisy_sinroll", 1),
        ("noisy_jnts", 4),
        ("noisy_velocity", 4),
        ("noisy_linvel", 2),
        ("info[\"last_act\"]", 4)
    ])
    print(info_dict)
    ckpt_folders = [
        'ckpts/20250506-0923/000020889600',
        'ckpts/20250612-2107/000403660800',
    ]
    load_params(ckpt_folders, info_dict)
