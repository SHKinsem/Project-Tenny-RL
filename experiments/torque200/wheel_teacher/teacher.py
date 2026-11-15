"""Native MuJoCo author-teacher sampling, with measured actuator-torque labels.

Only the original four-substep wheel speed loop is transcribed here; observation
packing and author weights use the original deployment modules. No per-step JAX.
"""
from pathlib import Path
import importlib
import math
import sys
import time
import numpy as np
from wheel_teacher.contract import (original_path, teacher_policy_path, metadata,
                                    WHEEL_TORQUE_LIMIT, CONTROL_HZ, PHYSICS_HZ)


class CollectionCancelled(RuntimeError):
    pass


def _imports(original_root):
    root = original_path(original_root)
    source, deployment = root / "mujoco_legwheel_rl_public", root / "mujoco_esp32_robot_deploy_public"
    for path in (deployment, source):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    modules = {}
    for name in ("legwheel.config", "host.policy.rl_policy", "host.controller.obs",
                 "host.controller.hold_obs", "host.controller.sim_frame"):
        module = importlib.import_module(name)
        expected = source if name.startswith("legwheel") else deployment
        if not Path(module.__file__).resolve().is_relative_to(expected.resolve()):
            raise RuntimeError(f"Wrong source imported for {name}")
        modules[name] = module
    return modules


def student_observation(teacher_observation, torque_history, wheel_velocity):
    obs = np.asarray(teacher_observation, dtype=np.float32)
    history = np.asarray(torque_history, dtype=np.float32)
    velocity = np.asarray(wheel_velocity, dtype=np.float32)
    if obs.shape != (26,) or history.shape != (2, 2) or velocity.shape != (2,):
        raise ValueError("Student observation dimensions do not match the contract")
    result = np.concatenate((obs[:20], history.reshape(-1), velocity))
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite student observation")
    return result


def torque_label(teacher_action, actual_wheel_torques):
    action = np.asarray(teacher_action, dtype=np.float64)
    actual = np.asarray(actual_wheel_torques, dtype=np.float64)
    if action.shape != (4,) or actual.shape != (4, 2):
        raise ValueError("Expected four actor outputs and four measured wheel-torque substeps")
    if not np.isfinite(action).all() or not np.isfinite(actual).all():
        raise FloatingPointError("Nonfinite teacher sample")
    if np.max(np.abs(actual)) > WHEEL_TORQUE_LIMIT + 1e-8 or np.max(np.abs(action)) > 1 + 1e-6:
        raise ValueError("Teacher torque/action is outside the student contract; labels are not clipped")
    return np.concatenate((action[:2], actual.mean(axis=0) / WHEEL_TORQUE_LIMIT)).astype(np.float32)


class AuthorTeacher:
    def __init__(self, original_root=None):
        import mujoco
        modules = _imports(original_root)
        self.mujoco = mujoco
        self.cfg = modules["legwheel.config"].default_config()
        cfg = self.cfg
        if (cfg.ctrl_dt != .01 or cfg.sim_dt != .0025 or not cfg.delta_thigh_action or
                not cfg.vel_pid.enable or cfg.action_scale.thigh != .25 or
                cfg.vel_pid.torque_limit != WHEEL_TORQUE_LIMIT or not cfg.vel_pid.residual_target or
                cfg.vel_pid.direct_torque or cfg.vel_pid.ki != 0 or cfg.vel_pid.kd != 0 or
                cfg.action_delay_steps != 0 or cfg.obs_history_len != 1 or cfg.last_act_obs):
            raise ValueError("Author controller changed; native teacher must be revalidated explicitly")
        from mujoco_playground._src.dm_control_suite import common
        self.model = mujoco.MjModel.from_xml_path(str(cfg._XML_PATH), assets=common.get_assets())
        self.model.opt.timestep = cfg.sim_dt
        self.data = mujoco.MjData(self.model)
        self.policy = modules["host.policy.rl_policy"].RL_Policy(path=teacher_policy_path(original_root))
        self.pack = modules["host.controller.obs"].ObsPack()
        self.pack.cfg.vel_pid_target_scale = cfg.vel_pid.target_scale
        self.pack.cfg.vel_pid_smoothing = cfg.vel_pid.notsmoothing
        self.pack.cfg.wheel_radius = cfg.wheel_radius
        self.hold = modules["host.controller.hold_obs"].HoldObs(reset_on_fallen=False)
        self.hold.hold_error_clip_enable = False
        self.hold.yaw_sin_scale = cfg.sinyaw_scale
        self.Frame = modules["host.controller.sim_frame"].SimFrame
        self.q = [self.model.jnt_qposadr[self.model.joint(f"{part}_{side}_joint").id]
                  for side in ("left", "right") for part in ("thigh", "wheel")]
        self.v = [self.model.jnt_dofadr[self.model.joint(f"{part}_{side}_joint").id]
                  for side in ("left", "right") for part in ("thigh", "wheel")]
        self.wheel_v = np.array([self.v[1], self.v[3]])
        self.body = self.model.body("root").id
        self.wheel_bodies = [self.model.body(f"wheel_{side}").id for side in ("left", "right")]
        self._validate_actuators()
        self.reset()

    def _validate_actuators(self):
        if self.model.nu != 4:
            raise ValueError("Expected four original actuators")
        names = ("thigh_left_pos", "thigh_right_pos", "wheel_left_motor", "wheel_right_motor")
        for index, name in enumerate(names):
            if self.model.actuator(name).id != index or self.model.actuator_gear[index, 0] != 1:
                raise ValueError("Original actuator order/gear changed")
        for index in (2, 3):
            if (self.model.actuator_gainprm[index, 0] != 1 or
                    np.any(self.model.actuator_biasprm[index] != 0) or
                    self.model.actuator_ctrlrange[index, 0] > -WHEEL_TORQUE_LIMIT or
                    self.model.actuator_ctrlrange[index, 1] < WHEEL_TORQUE_LIMIT):
                raise ValueError("Wheel motor is not a unit-gain direct-torque actuator")

    def sensor(self, name):
        item = self.model.sensor(name)
        return np.array(self.data.sensordata[item.adr[0]:item.adr[0] + item.dim[0]], copy=True)

    def frame(self):
        d = self.data
        # Original _getSin* and deployment from_mjx both read the derived xmat.
        # Keep that physics phase rather than mixing in a newer qpos quaternion.
        matrix = d.xmat[self.body].reshape(3, 3)
        pitch = math.asin(float(np.clip(-matrix[2, 0], -1, 1)))
        roll = math.asin(float(np.clip(matrix[2, 1] / max(math.cos(pitch), 1e-6), -1, 1)))
        yaw = math.atan2(matrix[1, 0], matrix[0, 0])
        delta = d.xpos[self.wheel_bodies].mean(axis=0) - d.subtree_com[self.body]
        axis = d.xmat[self.body].reshape(3, 3)[:, 0]
        cosine = np.dot(delta, axis) / (np.linalg.norm(delta) * np.linalg.norm(axis) + 1e-8)
        balance = math.pi / 2 - math.acos(float(np.clip(cosine, -1, 1)))
        q, v = d.qpos[self.q], d.qvel[self.v]
        return self.Frame(q_l=q[1], q_r=q[3], yaw_rad=yaw, pitch_rad=pitch, roll_rad=roll,
            gyro=self.sensor("root_gyro").tolist(), thigh_l=q[0], thigh_r=q[2],
            thigh_vel_l=v[0], thigh_vel_r=v[2], wheel_vel_l=v[1], wheel_vel_r=v[3],
            pitch_zero_rad=balance)

    def reset(self, command=None, *, qpos=None, qvel=None):
        self.mujoco.mj_resetData(self.model, self.data)
        if qpos is not None:
            self.data.qpos[:] = qpos
        if qvel is not None:
            self.data.qvel[:] = qvel
        self.mujoco.mj_forward(self.model, self.data)
        self.command = np.asarray([0., 0., 0., 0.] if command is None else command, dtype=float)
        if self.command.shape != (4,) or not np.isfinite(self.command).all():
            raise ValueError("Invalid teacher command")
        self.pack.reset()
        self.hold.hold_anchors_inited = False
        self.policy.clean_history()
        self.last_torque = np.zeros(2)
        self.torque_history = np.zeros((2, 2), dtype=np.float32)
        self.steps = 0
        return self.observation()

    def observation(self):
        frame = self.frame()
        hold = self.hold.step(self.command, frame)
        return np.asarray(self.pack.pack(self.command, frame, hold), dtype=np.float32)

    def student_observation(self):
        return student_observation(self.observation(), self.torque_history, self.data.qvel[self.wheel_v])

    def privileged_observation(self, obs):
        frame = self.frame()
        state = np.asarray(obs, dtype=np.float32)
        joints = self.data.qpos[self.q].copy()
        joints[[1, 3]] = state[[11, 13]]
        velocity = self.data.qvel[self.v]
        values = [state, self.sensor("root_gyro"), self.sensor("root_accel"), state[7:10],
                  self.sensor("root_upvector"), joints, velocity,
                  velocity[[1, 3]] * self.cfg.wheel_radius, self.sensor("root_local_linvel")[:1],
                  self.torque_history.reshape(-1), velocity[[1, 3]]]
        result = np.concatenate(values).astype(np.float32)
        if result.shape != (55,) or not np.isfinite(result).all():
            raise ValueError("Invalid privileged observation")
        return result

    def action(self):
        return self.policy.run(self.observation())

    def step(self, action, *, student=False):
        action = np.asarray(action, dtype=float)
        if action.shape != (4,) or not np.isfinite(action).all() or np.max(np.abs(action)) > 1 + 1e-6:
            raise ValueError("Invalid bounded policy action")
        action = np.clip(action, -1, 1)
        thigh = np.clip(self.data.qpos[np.array(self.q)[[0, 2]]] + .25 * action[:2], -.5, 0.)
        if not student:
            self.pack.note_policy_action(action, self.frame())
        samples = []
        for _ in range(4):
            if student:
                torque = WHEEL_TORQUE_LIMIT * action[2:]
            else:
                error = np.asarray(self.pack.last_target_wheel_vel) - self.data.qvel[self.wheel_v]
                torque = np.clip(self.last_torque + self.cfg.vel_pid.kp * error,
                                 -WHEEL_TORQUE_LIMIT, WHEEL_TORQUE_LIMIT)
            self.data.ctrl[:] = np.concatenate((thigh, torque))
            self.mujoco.mj_step(self.model, self.data)
            self.last_torque = torque.copy()
            samples.append(self.data.qfrc_actuator[self.wheel_v].copy())
        actual = np.asarray(samples)
        label = torque_label(action, actual)
        self.torque_history = np.vstack((self.torque_history[1:], label[2:][None, :]))
        self.steps += 1
        frame = self.frame()
        fallen = (math.sin(frame.pitch_rad) < -.6 or math.sin(frame.pitch_rad) > .85 or
                  abs(math.sin(frame.roll_rad)) > .6)
        finite = np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()
        return dict(done=bool(fallen or not finite), actual_wheel_torques=actual,
                    label=label, episode_seconds=self.steps / CONTROL_HZ)


def collect_dataset(original_root, *, episodes=8, steps_per_episode=600, seed=0,
                    on_progress=None, cancelled=None):
    if episodes < 2 or steps_per_episode < 1 or steps_per_episode > 600:
        raise ValueError("Use at least two episodes and 1..600 steps per episode")
    teacher = AuthorTeacher(original_root)
    rng = np.random.default_rng(seed)
    obs, priv, actions, episode_ids, lengths, variation = [], [], [], [], [], []
    peak = np.zeros(2)
    started = time.monotonic()
    for episode in range(episodes):
        if cancelled and cancelled():
            raise CollectionCancelled("Teacher collection cancelled")
        # Include stationary author baseline; other episodes are fixed small commands.
        command = np.zeros(4) if episode % 3 == 0 else np.array([
            rng.uniform(-.08, .08), rng.uniform(-.4, .4), rng.uniform(-.35, 0.), rng.uniform(-.08, .08)])
        teacher.reset(command)
        for step in range(steps_per_episode):
            if cancelled and cancelled():
                raise CollectionCancelled("Teacher collection cancelled")
            observation = teacher.student_observation()
            privileged = teacher.privileged_observation(observation)
            action = teacher.action()
            result = teacher.step(action)
            obs.append(observation)
            priv.append(privileged)
            actions.append(result["label"])
            episode_ids.append(episode)
            actual = result["actual_wheel_torques"]
            peak = np.maximum(peak, np.max(np.abs(actual), axis=0))
            variation.append(np.ptp(actual, axis=0))
            if result["done"]:
                break
        lengths.append((step + 1) / CONTROL_HZ)
        if on_progress:
            on_progress(dict(stage="teacher_collection", episode=episode + 1, episodes=episodes,
                             samples=len(obs), elapsed_seconds=time.monotonic() - started))
    contract = metadata(original_root)
    report = dict(teacher_source=contract["teacher_sha256"], teacher_checkpoint=contract["teacher_checkpoint"],
        sample_count=len(obs), episodes=episodes, episode_seconds=lengths, control_hz=100, physics_hz=400,
        wheel_torque_limits_nm=[WHEEL_TORQUE_LIMIT] * 2, peak_actual_wheel_torque_nm=peak.tolist(),
        max_within_cycle_torque_variation_nm=np.max(variation, axis=0).tolist(),
        invalid_samples=0, out_of_range_samples=0, observation_noise=False, external_pushes=False,
        command_schedule="fixed bounded command per episode; stationary every third episode",
        labels="author leg actor outputs; mean of four actual qfrc_actuator wheel torques / limit",
        elapsed_seconds=time.monotonic() - started)
    return dict(observations=np.asarray(obs, dtype=np.float32),
                privileged_observations=np.asarray(priv, dtype=np.float32),
                actions=np.asarray(actions, dtype=np.float32), episode_ids=np.asarray(episode_ids, dtype=np.int32),
                report=report)


def evaluate_student(original_root, policy_callable, *, episodes=2, steps_per_episode=100,
                     seed=12345, cancelled=None):
    teacher, student = AuthorTeacher(original_root), AuthorTeacher(original_root)
    rng = np.random.default_rng(seed)
    teacher_lengths, student_lengths, errors = [], [], []
    for episode in range(episodes):
        command = np.zeros(4) if episode == 0 else np.array([rng.uniform(-.05, .05), 0., -.15, 0.])
        teacher.reset(command)
        initial_q, initial_v = teacher.data.qpos.copy(), teacher.data.qvel.copy()
        student.reset(command, qpos=initial_q, qvel=initial_v)
        for step in range(steps_per_episode):
            if cancelled and cancelled():
                raise CollectionCancelled("Student rollout cancelled")
            predicted = np.asarray(policy_callable(teacher.student_observation()))
            result = teacher.step(teacher.action())
            errors.append(predicted[2:] - result["label"][2:])
            if result["done"]:
                break
        teacher_lengths.append((step + 1) / CONTROL_HZ)
        for step in range(steps_per_episode):
            if cancelled and cancelled():
                raise CollectionCancelled("Student rollout cancelled")
            result = student.step(policy_callable(student.student_observation()), student=True)
            if result["done"]:
                break
        student_lengths.append((step + 1) / CONTROL_HZ)
    return dict(teacher_episode_seconds=teacher_lengths, student_episode_seconds=student_lengths,
        teacher_state_wheel_torque_nrmse=float(np.sqrt(np.mean(np.square(errors)))),
        nrmse_definition="RMS error / wheel torque limit on identical held-out teacher states",
        closed_loop_comparison="independent trajectories from identical initial states and commands; not pointwise aligned",
        evaluated_seconds_per_episode=steps_per_episode / CONTROL_HZ,
        stable_balance_validated=False, seed=seed)


def verify_native_reference(original_root, *, seed=0):
    """One CPU MJX step against native sampling, before trusting a new teacher path.

    This is intentionally separate from the per-step collector and may compile once.
    It verifies the original controller/observation contract, not learned stability.
    """
    teacher = AuthorTeacher(original_root)
    import jax
    import jax.numpy as jp
    from mujoco import mjx
    from legwheel import LegWheelRobot, default_config
    with jax.default_device(jax.devices("cpu")[0]):
        cfg = default_config()
        cfg.noise_config.level = 0.
        cfg.push_config.enable = False
        cfg.cmd_config.cmd_prob = 0.
        env = LegWheelRobot(config=cfg)
        state = env.reset(jax.random.PRNGKey(seed))
        state = state.replace(data=mjx.put_data(env.mj_model, teacher.data))
        state.info["command"] = jp.asarray(teacher.command)
        state.info["position_start"] = env.get_hold_position(state.data)
        state.info["sinyaw_start"] = env._getSinYaw_newest(state.data)
        state.info["sin_balance_zero"] = env.get_sin_balance_angle(state.data)
        state = state.replace(obs=env._get_obs(state.data, state.info))
        initial_error = float(np.max(np.abs(np.asarray(state.obs["state"]) - teacher.observation())))
        action = teacher.action()
        expected = jax.jit(env.step)(state, jp.asarray(action))
        jax.block_until_ready(expected)
        teacher.step(action)
        errors = dict(initial_observation_max_abs_error=initial_error,
            qpos_max_abs_error=float(np.max(np.abs(np.asarray(expected.data.qpos) - teacher.data.qpos))),
            qvel_max_abs_error=float(np.max(np.abs(np.asarray(expected.data.qvel) - teacher.data.qvel))),
            ctrl_max_abs_error=float(np.max(np.abs(np.asarray(expected.data.ctrl) - teacher.data.ctrl))),
            final_observation_max_abs_error=float(np.max(np.abs(np.asarray(expected.obs["state"]) - teacher.observation()))))
    tolerances = dict(initial_observation_max_abs_error=1e-4, qpos_max_abs_error=1e-4,
                      qvel_max_abs_error=.02, ctrl_max_abs_error=1e-5,
                      final_observation_max_abs_error=.02)
    if any(not math.isfinite(value) or value > tolerances[key] for key, value in errors.items()):
        raise AssertionError(f"Native teacher diverged from original CPU MJX: {errors}")
    return dict(passed=True, backend="cpu", scope="one original control cycle; no stability claim",
                errors=errors, tolerances=tolerances)
