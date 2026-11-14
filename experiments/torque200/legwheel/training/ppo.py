"""PPO 训练:超参配置、训练循环、课程学习与训练后推理渲染。"""

import functools
import inspect
import logging
import os
import time
from datetime import datetime
from typing import Optional

import numpy as np
import jax
import mediapy as media
import mujoco
from ml_collections import config_dict
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import wrapper

from legwheel import LegWheelRobot, default_config
from legwheel.randomization import domain_randomize
from legwheel.training.progress import make_progress_fn
from legwheel.utils import elapsed_timer, log_gpu_memory, save_cpu_checkpoint

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "False")


def _patch_brax_eval_episode_length_std() -> None:
    """给 brax Evaluator 补充 eval/std_episode_length 指标。

    brax 0.12.1 的 run_evaluation 对 episode_steps 只上报 np.mean
    (acting.py),新版 brax 会额外输出 std;progress 曲线标题依赖该 key,
    缺失时显示 nan。这里复制原实现并补上 np.std;若 brax 已原生支持则跳过。
    """
    from brax.training import acting

    if "std_episode_length" in inspect.getsource(acting.Evaluator.run_evaluation):
        return

    def run_evaluation(self, policy_params, training_metrics, aggregate_episodes=True):
        self._key, unroll_key = jax.random.split(self._key)

        t = time.time()
        eval_state = self._generate_eval_unroll(policy_params, unroll_key)
        eval_metrics = eval_state.info["eval_metrics"]
        eval_metrics.active_episodes.block_until_ready()
        epoch_eval_time = time.time() - t
        metrics = {}
        for fn in [np.mean, np.std]:
            suffix = "_std" if fn is np.std else ""
            metrics.update(
                {
                    f"eval/episode_{name}{suffix}": (
                        fn(value) if aggregate_episodes else value
                    )
                    for name, value in eval_metrics.episode_metrics.items()
                }
            )
        metrics["eval/avg_episode_length"] = np.mean(eval_metrics.episode_steps)
        metrics["eval/std_episode_length"] = np.std(eval_metrics.episode_steps)
        metrics["eval/epoch_eval_time"] = epoch_eval_time
        metrics["eval/sps"] = self._steps_per_unroll / epoch_eval_time
        self._eval_walltime = self._eval_walltime + epoch_eval_time
        return {
            "eval/walltime": self._eval_walltime,
            **training_metrics,
            **metrics,
        }

    acting.Evaluator.run_evaluation = run_evaluation


_patch_brax_eval_episode_length_std()


def compute_discounting(ctrl_dt: float, half_life_s: float) -> float:
    """由时间半衰期换算每个 ctrl step 的折扣因子:discounting = 0.5^(ctrl_dt / half_life_s)。

    折扣因子 discounting(γ) 是"每个环境 step"的折扣。
    这里用"时间半衰期"来定义折扣:half_life_s 秒之后,未来回报权重衰减到 0.5。
    提为纯函数供 curriculum_learning 与 tune.py 复用同一换算,避免两处公式漂移。
    """
    if half_life_s <= 0:
        raise ValueError("half_life_s must be > 0")
    return float(np.exp(np.log(0.5) * (ctrl_dt / half_life_s)))


def get_ppo_config(
    env_cfg=None,
    num_evals=20,
    maxstep_candidate=2e8,
    save_checkpoint_path: Optional[str] = None,
    ppo_overrides: Optional[dict] = None,
) -> config_dict.ConfigDict:
    """
    这是brax ppo train中有关num_evals, num_timesteps的代码：
    num_resets_per_eval默认为。
      # The number of environment steps executed for every training step.
      env_step_per_training_step = (
          batch_size * unroll_length * num_minibatches * action_repeat
      )
      num_evals_after_init = max(num_evals - 1, 1)
      # The number of training_step calls per training_epoch call.
      # equals to ceil(num_timesteps / (num_evals * env_step_per_training_step *
      #                                 num_resets_per_eval))
      num_training_steps_per_epoch = np.ceil(
          num_timesteps
          / (
              num_evals_after_init
              * env_step_per_training_step
              * max(num_resets_per_eval, 1)
          )
      ).astype(int)
    """
    if env_cfg is None:
        env_cfg = default_config()
    # 结构性超参(影响 num_envs/num_timesteps 计算)从 override 提前取出;
    # 其余键在 rl_config 建成后按顶层字段覆盖(仅限已存在字段,防拼写错误静默生效)。
    ppo_overrides = dict(ppo_overrides or {})
    batch_size = int(ppo_overrides.pop("batch_size", 512))
    num_minibatches = int(ppo_overrides.pop("num_minibatches", 16))
    episode_length = env_cfg.episode_length
    unroll_length = int(ppo_overrides.pop("unroll_length", 150))
    num_updates_per_batch = int(ppo_overrides.pop("num_updates_per_batch", 4))
    num_envs = batch_size * num_minibatches
    num_timesteps = unroll_length * batch_size * num_minibatches * num_evals
    num_timesteps = max(num_timesteps, int(maxstep_candidate))
    if save_checkpoint_path is None:
        save_checkpoint_path = os.path.join(
            os.getcwd(), "ckpts", datetime.now().strftime("%Y%m%d-%H%M")
        )
    # orbax 要求 checkpoint 路径为绝对路径,相对路径会在保存时抛
    # "Checkpoint path should be absolute"
    save_checkpoint_path = os.path.abspath(save_checkpoint_path)
    rl_config = config_dict.create(
        seed=0,
        num_timesteps=num_timesteps,
        num_evals=num_evals,
        reward_scaling=1.0,
        episode_length=episode_length,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=unroll_length,
        num_minibatches=num_minibatches,
        num_updates_per_batch=num_updates_per_batch,
        discounting=0.99,
        learning_rate=3e-4,
        entropy_cost=1e-2,
        num_envs=num_envs,
        batch_size=batch_size,
        max_grad_norm=1.0,
        network_factory=config_dict.create(
            policy_hidden_layer_sizes=(128, 128),
            value_hidden_layer_sizes=(128, 128),
            policy_obs_key="state",
            value_obs_key="privileged_state",
        ),
        save_checkpoint_path=save_checkpoint_path,
    )
    # 其余 override 键按顶层字段覆盖(仅允许已存在字段,防止拼写错误静默生效)
    for key, value in ppo_overrides.items():
        if key not in rl_config:
            raise KeyError(f"未知 PPO 配置键: {key}")
        setattr(rl_config, key, value)
    return rl_config


def train_ppo(params=None, env=None, env_cfg=None, ppo_params=None, iter_now=0,
              progress_fn=None, randomization_fn=None):
    # 初始化环境
    if env is None:
        env = LegWheelRobot()
    if env_cfg is None:
        env_cfg = default_config()
    times = [datetime.now()]
    # progress_fn / randomization_fn 为可选注入点(调参用),None 时走原逻辑,行为零变化
    progress = (
        progress_fn if progress_fn is not None
        else make_progress_fn(env_cfg, ppo_params, iter_now)
    )

    randomizer = (
        randomization_fn if randomization_fn is not None else domain_randomize
    )
    if ppo_params is None:
        ppo_params = get_ppo_config()
    ppo_training_params = dict(ppo_params)
    network_factory = ppo_networks.make_ppo_networks
    if "network_factory" in ppo_params:
        del ppo_training_params["network_factory"]
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks,
            **ppo_params.network_factory,
        )
    # 只传递当前 brax 版本支持的参数(如 save_checkpoint_path 仅新版 brax 支持)
    supported = inspect.signature(ppo.train).parameters
    dropped = [k for k in ppo_training_params if k not in supported]
    for k in dropped:
        if k == "save_checkpoint_path":
            logging.warning(
                "当前 brax 版本的 ppo.train 不支持 save_checkpoint_path,跳过自动保存"
            )
        del ppo_training_params[k]
    train_fn = functools.partial(
        ppo.train,
        **dict(ppo_training_params),
        network_factory=network_factory,
        randomization_fn=randomizer,
        progress_fn=progress,
    )

    # restore_params 仅新版 brax 支持;旧版用 restore_checkpoint_path 恢复
    restore_kwargs = {}
    if params is not None:
        if "restore_params" in supported:
            restore_kwargs["restore_params"] = params
        else:
            raise ValueError(
                "当前 brax 版本不支持 restore_params,请使用 restore_checkpoint_path"
            )

    make_inference_fn, params, metrics = train_fn(
        environment=env,
        eval_env=LegWheelRobot(config=env_cfg),
        wrap_env_fn=wrapper.wrap_for_brax_training,
        **restore_kwargs,
    )
    print(f"time to jit+train: {datetime.now() - times[0]}")
    log_gpu_memory(f"train done, iter {iter_now}")

    return make_inference_fn, params, metrics


def curriculum_config(
    iter_now=0, num_evals=20, maxstep_candidate=2e8, save_checkpoint_path=None,
    env_cfg=None, ppo_overrides=None,
):
    if env_cfg is None:
        env_cfg = default_config()
    ppo_params = get_ppo_config(
        env_cfg,
        num_evals=num_evals,
        maxstep_candidate=maxstep_candidate,
        save_checkpoint_path=save_checkpoint_path,
        ppo_overrides=ppo_overrides,
    )
    ppo_params.seed = iter_now
    return env_cfg, ppo_params


def curriculum_learning(
    iter_num: int = 1,
    num_evals: int = 10,
    maxstep_candidate: float = 4e8,
    learning_rate: float = 1e-4,
    restore_checkpoint_path: Optional[str] = None,
    save_checkpoint_path: Optional[str] = None,
    env_cfg: Optional[config_dict.ConfigDict] = None,
    ppo_overrides: Optional[dict] = None,
    seed: Optional[int] = None,
    progress_fn=None,
    randomization_fn=None,
):
    # ppo_overrides 支持的键:
    #   结构性(传给 get_ppo_config): batch_size/num_minibatches/unroll_length/
    #     num_updates_per_batch,以及 rl_config 已存在的顶层字段;
    #   本节内应用: half_life_s(换算 discounting)、learning_rate、entropy_cost、
    #     reward_scaling、policy_hidden_layer_sizes、value_hidden_layer_sizes。
    ppo_overrides = dict(ppo_overrides or {})
    local_keys = (
        "half_life_s", "entropy_cost", "reward_scaling", "learning_rate",
        "policy_hidden_layer_sizes", "value_hidden_layer_sizes",
    )
    local_overrides = {k: ppo_overrides.pop(k) for k in local_keys if k in ppo_overrides}

    env_cfg, ppo_params = curriculum_config(
        num_evals=num_evals,
        maxstep_candidate=maxstep_candidate,
        save_checkpoint_path=save_checkpoint_path,
        env_cfg=env_cfg,
        ppo_overrides=ppo_overrides,
    )
    # 硬防护:hold_error_clip 是推理/部署专用开关(viewer.py 与下方 inference()
    # 是合法打开点);训练误开会在 _get_obs 内静默 clip 观测(训练推理共用该通路),
    # 因此训练入口直接拒绝,见 docs/hold_position_heading.md。
    if env_cfg.hold_error_clip.enable:
        raise ValueError(
            "hold_error_clip.enable=True 仅用于推理/部署(viewer、inference),"
            "训练必须保持 enable=False(default_config 默认);"
            "误开会静默 clip 观测 Δs/Δψ,见 docs/hold_position_heading.md")
    if restore_checkpoint_path:
        # 与 save_checkpoint_path 一致,统一转绝对路径供 orbax 使用
        ppo_params.restore_checkpoint_path = os.path.abspath(restore_checkpoint_path)

    ctrl_dt = env_cfg.ctrl_dt
    half_life_s = float(local_overrides.pop("half_life_s", 0.5)) #env_cfg.episode_time / 2.0
    discounting = compute_discounting(ctrl_dt, half_life_s)
    logging.info(
        "sim_dt: %f, ctrl_dt: %f, half_life_s: %f, discounting: %f",
        env_cfg.sim_dt,
        ctrl_dt,
        half_life_s,
        discounting,
    )
    log_gpu_memory("startup")

    ppo_params.learning_rate = float(local_overrides.pop("learning_rate", learning_rate))
    ppo_params.normalize_observations = True
    ppo_params.reward_scaling = float(local_overrides.pop("reward_scaling", 0.1))
    ppo_params.entropy_cost = float(local_overrides.pop("entropy_cost", 1e-2))
    ppo_params.discounting = discounting
    ppo_params.network_factory.policy_hidden_layer_sizes = list(
        local_overrides.pop("policy_hidden_layer_sizes", [64, 64]))
    ppo_params.network_factory.value_hidden_layer_sizes = list(
        local_overrides.pop("value_hidden_layer_sizes", [512, 512, 512, 512]))
    if local_overrides:
        raise ValueError(f"未知 PPO override 键: {sorted(local_overrides)}")
    if seed is not None:
        ppo_params.seed = seed

    make_inference_fn, params, metrics = None, None, None
    with elapsed_timer("curriculum_learning"):
        for iter_now in range(1, iter_num + 1):
            logging.info(
                "====================curriculum_learning iter_now: %d====================",
                iter_now,
            )
            env = LegWheelRobot(env_cfg)
            logging.info(env_cfg)
            logging.info(ppo_params)
            make_inference_fn, params, metrics = train_ppo(
                params=params,
                env=env,
                env_cfg=env_cfg,
                ppo_params=ppo_params,
                iter_now=iter_now,
                progress_fn=progress_fn,
                randomization_fn=randomization_fn,
            )
            # 恢复只在第一次迭代生效,后续迭代直接沿用上轮参数
            ppo_params.restore_checkpoint_path = None
            if iter_now != iter_num:
                ppo_params.save_checkpoint_path = os.path.join(
                    os.getcwd(),
                    "ckpts",
                    datetime.now().strftime("%Y%m%d-%H%M"),
                )

    # Save a CPU version of the final checkpoint
    if hasattr(ppo_params, "save_checkpoint_path") and ppo_params.save_checkpoint_path:
        save_cpu_checkpoint(params, ppo_params.save_checkpoint_path)

    return make_inference_fn, params, metrics, ppo_params


def inference(make_inference_fn, params, ppo_params, n_episodes: int = 1):
    """训练后推理:rollout + 跟踪相机渲染,视频保存到 checkpoint 目录。"""
    with elapsed_timer("inference"):
        env_cfg = default_config()
        env_cfg.hold_error_clip.enable = True
        env = LegWheelRobot(env_cfg)

        jit_reset = jax.jit(env.reset)
        jit_step = jax.jit(env.step)
        jit_prepare = jax.jit(env.prepare_inference)
        jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))
        rng = jax.random.PRNGKey(1)
        rollout = []
        state_pos = [
            ["cmd", 4],
            ["gyro", 3],
            ["sinyaw", 1],
            ["sinpitch", 1],
            ["sinroll", 1],
            ["jnts", 4],
            ["velocity", 4],
            ["linvel", 2],
            ["previous_torque", 4],
        ]

        for _ in range(n_episodes):
            state = jit_reset(rng, True)
            rollout.append(state)
            for i in range(env_cfg.episode_length * 4):
                act_rng, rng = jax.random.split(rng)
                state = jit_prepare(state)
                ctrl, _ = jit_inference_fn(state.obs, act_rng)
                state_obs = jax.device_get(state.obs["state"])
                state_str = f"obs_len:{len(state_obs)}"
                for pos in state_pos:
                    state_str += f" {pos[0]}:{state_obs[:pos[1]]},"
                    state_obs = state_obs[pos[1] :]
                state_str += f"ctrl:{ctrl}"
                metrics_str = ""
                for k in state.metrics.keys():
                    metrics_str += f"{k}: {jax.device_get(state.metrics[k]):.3f}, "
                state = jit_step(state, ctrl)
                rollout.append(state)

        camera_tracking = True
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(env.mj_model, camera)
        if camera_tracking:
            camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            root_body_id = mujoco.mj_name2id(
                env.mj_model, mujoco.mjtObj.mjOBJ_BODY, "root"
            )
            camera.trackbodyid = root_body_id
        camera.lookat = [0.0, 0.0, 0.0]
        camera.azimuth = 30
        camera.elevation = -20
        camera.distance = 0.8
        render_every = 10
        scene_option = mujoco.MjvOption()
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        scene_option.frame = mujoco.mjtFrame.mjFRAME_WORLD
        frames = env.render(
            rollout[::render_every],
            height=1080,
            width=1920,
            camera=camera,
            scene_option=scene_option,
        )
        fps_value = 1.0 / env.dt / render_every
        print(f"Rendering at {fps_value} FPS")
        media.show_video(frames, fps=fps_value)
        save_dir = None
        if hasattr(ppo_params, "save_checkpoint_path"):
            save_dir = ppo_params.save_checkpoint_path
            print(f"save video to {save_dir}")
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            video_path = os.path.join(
                save_dir, f"inference_video_{datetime.now().strftime('%Y%m%d-%H%M%S')}.mp4"
            )
            media.write_video(video_path, frames, fps=fps_value)
            print(f"Saved inference video to {video_path}")
