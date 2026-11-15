"""Reachable 100 Hz force oracle on real motor states, with isolated author imports."""
import argparse
import json
import math
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import tempfile
import threading
import time
import numpy as np
from hybrid.core import (ROOT, XML, TORQUE_LIMITS, HIP_RANGE, read_geometry, scan_geometry,
                         map_action, MAX_EQ_POSITION, MAX_LENGTH_VELOCITY, MIN_JACOBIAN_RATIO)
from force_teacher.contract import metadata


class CollectionCancelled(RuntimeError):
    pass


def force_label(actual_torque, jacobian, force_scales):
    """Fit ONE held axial-force action to four measured motor-torque substeps.

    No clipping: caller discards unreachable cycles and reports the actual coverage.
    """
    torque, jac, scales = map(lambda v: np.asarray(v, dtype=float), (actual_torque, jacobian, force_scales))
    if torque.shape != (4, 4) or jac.shape != (4, 2) or scales.shape != (2,):
        raise ValueError("Expected four substeps of four torques and two Jacobians")
    if not all(np.isfinite(v).all() for v in (torque, jac, scales)) or np.any(scales <= 0) or np.any(np.abs(jac) < 1e-8):
        raise ValueError("Invalid or degenerate measured force sample")
    gains = jac * scales
    raw_force = torque[:, :2] / jac
    instantaneous_action = raw_force / scales
    leg_action = np.sum(gains * torque[:, :2], axis=0) / np.sum(gains ** 2, axis=0)
    action = np.concatenate((leg_action, torque[:, 2:].mean(axis=0) / np.asarray(TORQUE_LIMITS[2:])))
    reconstructed = np.column_stack((gains * leg_action, np.tile(torque[:, 2:].mean(axis=0), (4, 1))))
    reachable = bool(np.all(np.abs(torque) <= np.asarray(TORQUE_LIMITS) + 1e-8) and
                     np.all(np.abs(instantaneous_action) <= 1 + 1e-8) and np.all(np.abs(action) <= 1))
    return dict(action=action.astype(np.float32), reachable=reachable,
                instantaneous_force_n=raw_force, instantaneous_action=instantaneous_action,
                reconstructed_torque_nm=reconstructed, torque_error_nm=reconstructed - torque)


def _state_geometry(teacher, legs, bounds):
    d, model = teacher.data, teacher.model
    jac, valid = [], True
    for index, (side, leg) in enumerate(zip(("left", "right"), legs)):
        h, k = model.joint(f"thigh_{side}_joint"), model.joint(f"calf_{side}_joint")
        qh, qk = float(d.qpos[h.qposadr[0]]), float(d.qpos[k.qposadr[0]])
        vh, vk = float(d.qvel[h.dofadr[0]]), float(d.qvel[k.dofadr[0]])
        length, j, partial = leg.evaluate(qh, qk)
        valid = valid and (HIP_RANGE[0] <= qh <= HIP_RANGE[1] and
            abs(qk - leg.coupled_knee(qh)) <= MAX_EQ_POSITION and
            abs(partial * (vk - leg.slope(qh) * vh)) <= MAX_LENGTH_VELOCITY and
            abs(j) / bounds[index] >= MIN_JACOBIAN_RATIO)
        jac.append(j)
    valid = valid and np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()
    return np.asarray(jac), bool(valid)


def _observations(teacher, previous_mean_torque):
    # Original author pack's first 20 fields match except for its deliberately
    # hidden leg velocities. The torque student observes both actual hip velocities.
    first = teacher.observation()[:20].copy()
    velocities = teacher.data.qvel[teacher.v]
    first[[14, 16]] = velocities[[0, 2]]
    last = np.asarray(previous_mean_torque) / np.asarray(TORQUE_LIMITS)
    obs = np.concatenate((first, last)).astype(np.float32)
    joints = teacher.data.qpos[teacher.q].copy()
    joints[[1, 3]] = first[[11, 13]]
    privileged = np.concatenate((obs, teacher.sensor("root_gyro"), teacher.sensor("root_accel"),
        first[7:10], teacher.sensor("root_upvector"), joints, velocities,
        velocities[[1, 3]] * teacher.cfg.wheel_radius, teacher.sensor("root_local_linvel")[:1], last)).astype(np.float32)
    if obs.shape != (24,) or privileged.shape != (51,) or not np.isfinite(privileged).all():
        raise ValueError("Invalid force student observations")
    return obs, privileged


class _RecordingBackend:
    def __init__(self, teacher, legs, bounds):
        self.backend, self.teacher, self.legs, self.bounds = teacher.mujoco, teacher, legs, bounds
        self.jacobians, self.torques, self.valid = [], [], []

    def __getattr__(self, key):
        return getattr(self.backend, key)

    def clear(self):
        self.jacobians.clear()
        self.torques.clear()
        self.valid.clear()

    def mj_step(self, model, data):
        jac, valid = _state_geometry(self.teacher, self.legs, self.bounds)
        self.jacobians.append(jac)
        self.valid.append(valid)
        self.backend.mj_step(model, data)
        indices = np.asarray(self.teacher.v)[[0, 2, 1, 3]]
        self.torques.append(data.qfrc_actuator[indices].copy())


def projected_action(torque, jacobian, force_scales):
    """Explicit reachable-oracle projection, not an equivalent servo label."""
    torque, jac, scales = (np.asarray(v, dtype=float) for v in (torque, jacobian, force_scales))
    if torque.shape != (4,) or jac.shape != (2,) or scales.shape != (2,):
        raise ValueError("Invalid projected teacher dimensions")
    if not all(np.isfinite(v).all() for v in (torque, jac, scales)) or np.any(scales <= 0) or np.any(np.abs(jac) < 1e-8):
        raise ValueError("Invalid projected teacher geometry")
    raw = np.r_[torque[:2] / (jac * scales), torque[2:] / np.asarray(TORQUE_LIMITS[2:])]
    action = np.clip(raw, -1., 1.).astype(np.float32)
    projected = map_action(action, jac, scales, "virtual_leg_force")
    return dict(action=action, raw_action=raw, projected=np.abs(raw) > 1.,
                requested_torque_nm=torque, projected_torque_nm=projected,
                projection_error_nm=np.asarray(projected) - torque)


def termination_details(student):
    state, reasons, residuals = student.state, [], []
    for i, (side, leg) in enumerate(zip(("left", "right"), student.legs)):
        h, k = state.model.joint(f"thigh_{side}_joint"), state.model.joint(f"calf_{side}_joint")
        qh, qk = (float(state.data.qpos[j.qposadr[0]]) for j in (h, k))
        vh, vk = (float(state.data.qvel[j.dofadr[0]]) for j in (h, k))
        _, jac, partial = leg.evaluate(qh, qk)
        eq, vel = qk - leg.coupled_knee(qh), partial * (vk - leg.slope(qh) * vh)
        ratio = abs(jac) / student.bounds[i]
        for bad, label in ((not HIP_RANGE[0] <= qh <= HIP_RANGE[1], "HIP_RANGE"),
                           (abs(eq) > MAX_EQ_POSITION, "EQ_POSITION"),
                           (abs(vel) > MAX_LENGTH_VELOCITY, "EQ_VELOCITY"),
                           (ratio < MIN_JACOBIAN_RATIO, "DEADPOINT")):
            if bad: reasons.append(f"{side}:{label}")
        residuals.append(dict(hip_rad=qh, equality_position_rad=eq, equality_velocity_mps=vel, jacobian_ratio=ratio))
    frame = state.frame()
    pitch, roll = math.sin(frame.pitch_rad), math.sin(frame.roll_rad)
    if pitch < -.6 or pitch > .85: reasons.append("PITCH")
    if abs(roll) > .6: reasons.append("ROLL")
    if not np.isfinite(state.data.qpos).all() or not np.isfinite(state.data.qvel).all(): reasons.append("NONFINITE")
    return dict(reasons=reasons, legs=residuals, pitch_sin=pitch, roll_sin=roll)


class NativeStudent:
    """Native motor model with the exact calibrated 400 Hz force mapping."""
    def __init__(self, original_root):
        from wheel_teacher.teacher import AuthorTeacher
        self.state = AuthorTeacher(original_root)
        from mujoco_playground._src.dm_control_suite import common
        state = self.state
        state.model = state.mujoco.MjModel.from_xml_path(str(XML), assets=common.get_assets())
        state.model.opt.timestep = .0025
        state.data = state.mujoco.MjData(state.model)
        for i in range(4):
            if state.model.actuator_gainprm[i, 0] != 1 or np.any(state.model.actuator_biasprm[i] != 0):
                raise ValueError("Native force student cannot use position servo actuators")
        self.legs, scan = read_geometry(), scan_geometry(read_geometry())
        self.scales = [r["force_scale_n"] for r in scan]
        self.bounds = [r["jmax_bound"] for r in scan]
        self.reset()

    def reset(self, command=None, qpos=None, qvel=None):
        self.state.reset(command, qpos=qpos, qvel=qvel)
        self.previous = np.zeros(4)
        self.steps = 0
        return self.observation()

    def observation(self):
        return _observations(self.state, self.previous)[0]

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (4,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1 + 1e-6):
            raise ValueError("Invalid force student action")
        actual, sent, jacobians = [], [], []
        invalid = False
        reasons = []
        state = self.state
        for _ in range(4):
            jac, valid = _state_geometry(state, self.legs, self.bounds)
            invalid |= not valid
            if not valid:
                reasons.extend(termination_details(self)["reasons"])
            tau = map_action(np.zeros(4) if invalid else action, jac, self.scales, "virtual_leg_force")
            state.data.ctrl[:] = tau
            state.mujoco.mj_step(state.model, state.data)
            sent.append(tau)
            actual.append(state.data.qfrc_actuator[np.asarray(state.v)[[0, 2, 1, 3]]].copy())
            jacobians.append(jac)
        self.previous = np.mean(sent, axis=0)
        self.steps += 1
        frame = state.frame()
        _, valid_after = _state_geometry(state, self.legs, self.bounds)
        done = invalid or not valid_after or math.sin(frame.pitch_rad) < -.6 or math.sin(frame.pitch_rad) > .85 or abs(math.sin(frame.roll_rad)) > .6
        termination = termination_details(self)
        termination["reasons"] = sorted(set(reasons + termination["reasons"]))
        return dict(done=bool(done), actual_torques=np.asarray(actual), sent_torques=np.asarray(sent),
                    jacobians=np.asarray(jacobians), episode_seconds=self.steps / 100, termination=termination)


class ReachableTeacher:
    """Private author memory queries real student states; only bounded F/tau executes."""
    def __init__(self, original_root):
        from wheel_teacher.teacher import AuthorTeacher
        self.teacher = AuthorTeacher(original_root)

    def reset(self, command, student):
        self.teacher.reset(command, qpos=student.state.data.qpos.copy(), qvel=student.state.data.qvel.copy())

    def action(self, student):
        teacher, state = self.teacher, student.state
        teacher.data.qpos[:] = state.data.qpos
        teacher.data.qvel[:] = state.data.qvel
        teacher.data.qacc_warmstart[:] = state.data.qacc_warmstart
        teacher.data.time = state.data.time
        teacher.mujoco.mj_forward(teacher.model, teacher.data)
        actor = teacher.action()
        target = np.clip(teacher.data.qpos[np.asarray(teacher.q)[[0, 2]]] + .25 * actor[:2], -.5, 0.)
        teacher.pack.note_policy_action(actor, teacher.frame())
        wheel = np.clip(teacher.last_torque + teacher.cfg.vel_pid.kp * (
            np.asarray(teacher.pack.last_target_wheel_vel) - state.data.qvel[state.wheel_v]), -.0295, .0295)
        teacher.data.ctrl[:] = np.r_[target, wheel]
        teacher.mujoco.mj_forward(teacher.model, teacher.data)
        torque = teacher.data.qfrc_actuator[np.asarray(teacher.v)[[0, 2, 1, 3]]].copy()
        jac, valid = _state_geometry(state, student.legs, student.bounds)
        if not valid: raise ValueError("Oracle queried outside valid student geometry")
        return projected_action(torque, jac, student.scales)

    def executed(self, action):
        # During DAgger, controller torque memory follows what was actually sent.
        self.teacher.last_torque = np.asarray(action)[2:] * np.asarray(TORQUE_LIMITS[2:])


def _commands(episode):
    return np.array(([0., 0., 0., 0.], [.04, 0., -.15, 0.], [-.04, 0., -.15, 0.])[episode % 3])


def _collect_native(original_root, episodes, steps_per_episode, seed, progress, policy=None, student_probability=0.):
    student, oracle = NativeStudent(original_root), ReachableTeacher(original_root)
    rng = np.random.default_rng(seed)
    observations, privileged, actions, ids, errors, projected, raw = [], [], [], [], [], [], []
    lengths, episode_reports, mixed = [], [], 0
    for episode in range(episodes):
        command = np.zeros(4)  # Match standing_v2's fixed command sampler.
        student.reset(command)
        # Keep nominal gate cases; remaining episodes cover small initial tilt/rate.
        if episode in (1, 2):
            pitch = .01 if episode == 1 else -.01
            qpos = student.state.data.qpos.copy()
            qpos[3:7] = [math.cos(pitch/2), 0., math.sin(pitch/2), 0.]
            student.reset(command, qpos, student.state.data.qvel.copy())
        elif episode >= 3:
            data = student.state.data
            roll, pitch = rng.uniform(-.012, .012), rng.uniform(-.02, .02)
            data.qpos[3:7] = [math.cos(roll/2)*math.cos(pitch/2), math.sin(roll/2)*math.cos(pitch/2),
                              math.cos(roll/2)*math.sin(pitch/2), -math.sin(roll/2)*math.sin(pitch/2)]
            data.qvel[3:6] = rng.uniform(-.03, .03, 3)
            student.reset(command, data.qpos.copy(), data.qvel.copy())
        oracle.reset(command, student)
        initial = dict(qpos=student.state.data.qpos.tolist(), qvel=student.state.data.qvel.tolist())
        for step in range(steps_per_episode):
            obs, priv = _observations(student.state, student.previous)
            label = oracle.action(student)
            execute = label["action"]
            if policy is not None and rng.random() < student_probability:
                execute = np.asarray(policy(obs), dtype=float)
                mixed += 1
            observations.append(obs); privileged.append(priv); actions.append(label["action"]); ids.append(episode)
            errors.append(label["projection_error_nm"]); projected.append(label["projected"]); raw.append(label["requested_torque_nm"])
            result = student.step(execute)
            oracle.executed(execute)
            if result["done"]: break
        lengths.append((step + 1) / 100)
        episode_reports.append(dict(command=command.tolist(), seconds=lengths[-1], initial=initial, termination=result["termination"]))
        progress(dict(stage="teacher_collection", episode=episode+1, episodes=episodes,
                      samples=len(actions), sampled_cycles=len(actions), discarded_cycles=0))
    contract = metadata(original_root)
    report = dict(teacher_source=contract["teacher_sha256"], teacher_checkpoint=contract["teacher_checkpoint"],
        teacher_schema=contract["teacher_schema"], force_scales_n=list(student.scales), torque_limits_nm=list(TORQUE_LIMITS),
        sampled_cycles=len(actions), retained_cycles=len(actions), retention_fraction=1.,
        rejected_current_cycles=0, rejected_prior_history_cycles=0,
        projected_cycle_fraction=np.mean(projected, axis=0).tolist(),
        oracle_unreachable_cycle_fraction=float(np.mean(np.any(projected, axis=1))),
        projection_torque_rmse_nm=np.sqrt(np.mean(np.square(errors), axis=0)).tolist(),
        peak_requested_oracle_torque_nm=np.max(np.abs(raw), axis=0).tolist(),
        episode_seconds=lengths, episodes=episode_reports, invalid_samples=0, out_of_range_samples=0,
        mixed_student_cycles=mixed, student_probability=student_probability,
        observation_history="previous actually sent motor torque mean / limits; no private author target memory",
        coverage="true motor student trajectories, nominal starts and small initial tilt/angular-rate perturbations",
        label_definition=contract["teacher_label_semantics"], projection_policy=contract["unreachable_label_policy"],
        teacher_information=contract["teacher_information"], observation_noise=False, external_pushes=False)
    return dict(observations=np.asarray(observations, dtype=np.float32).reshape(-1,24),
                privileged_observations=np.asarray(privileged,dtype=np.float32).reshape(-1,51),
                actions=np.asarray(actions,dtype=np.float32).reshape(-1,4),episode_ids=np.asarray(ids,dtype=np.int32), report=report)


def _evaluate_native(original_root, policy, episodes, steps_per_episode, seed, evaluation_profile="standing_gate"):
    student, oracle = NativeStudent(original_root), ReachableTeacher(original_root)
    teacher_lengths, student_lengths, errors, reports = [], [], [], []
    for episode in range(episodes):
        command = np.zeros(4) if evaluation_profile == "standing_gate" else _commands(episode)
        student.reset(command)
        if evaluation_profile == "standing_gate":
            pitch = (0., .01, -.01)[episode % 3]
            qpos = student.state.data.qpos.copy()
            qpos[3:7] = [math.cos(pitch/2), 0., math.sin(pitch/2), 0.]
            student.reset(command, qpos, student.state.data.qvel.copy())
        initial_qpos, initial_qvel = student.state.data.qpos.copy(), student.state.data.qvel.copy()
        oracle.reset(command, student)
        for step in range(steps_per_episode):
            label = oracle.action(student)
            errors.append(np.asarray(policy(student.observation())) - label['action'])
            result = student.step(label['action'])
            oracle.executed(label['action'])
            if result['done']: break
        teacher_lengths.append((step + 1) / 100)
        teacher_termination = result['termination']
        student.reset(command, initial_qpos, initial_qvel)
        for step in range(steps_per_episode):
            result = student.step(policy(student.observation()))
            if result['done']: break
        student_lengths.append((step + 1) / 100)
        reports.append(dict(command=command.tolist(), seconds=student_lengths[-1],
            termination=result['termination'], oracle_seconds=teacher_lengths[-1],
            oracle_termination=teacher_termination, initial_qpos=initial_qpos.tolist(), initial_qvel=initial_qvel.tolist()))
    passed = all(r['seconds'] == steps_per_episode / 100 and not r['termination']['reasons'] for r in reports)
    return dict(teacher_episode_seconds=teacher_lengths, student_episode_seconds=student_lengths,
        teacher_reachable_state_action_nrmse=float(np.sqrt(np.mean(np.square(errors)))),
        reachable_comparison_samples=len(errors), evaluated_seconds_per_episode=steps_per_episode/100,
        episodes=reports, gate_passed=bool(passed and episodes >= 3 and steps_per_episode == 600), evaluation_profile=evaluation_profile,
        closed_loop_comparison='reachable oracle and student on true motor model; identical nominal starts; 100Hz F/wheel torque held',
        teacher_information='oracle uses author 26D/private history; student only 24D',
        stable_balance_validated=bool(passed and episodes >= 3 and steps_per_episode == 600), seed=seed)


def _worker(operation, original_root, episodes, steps_per_episode, seed, policy=None, on_progress=None, cancelled=None, student_probability=0., evaluation_profile="standing_gate"):
    with tempfile.TemporaryDirectory(prefix="tenny-force-teacher-") as directory:
        output = Path(directory) / "result.npz"
        env = os.environ.copy()
        env.update(JAX_PLATFORMS="cpu", XLA_PYTHON_CLIENT_PREALLOCATE="false", PYTHONUNBUFFERED="1")
        argv = [sys.executable, "-u", "-m", "force_teacher.teacher", operation, "--original-root", str(original_root),
                "--output", str(output), "--episodes", str(episodes), "--steps", str(steps_per_episode), "--seed", str(seed)]
        if operation == "collect":
            argv += ["--student-probability", str(student_probability)]
        if operation == "evaluate":
            argv += ["--evaluation-profile", evaluation_profile]
        process = subprocess.Popen(argv, cwd=ROOT, env=env, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf8",
            start_new_session=os.name != "nt")
        messages = queue.Queue()
        def read():
            for line in process.stdout:
                messages.put(line)
            messages.put(None)
        thread = threading.Thread(target=read, daemon=True)
        thread.start()
        tail = []
        try:
            while True:
                if cancelled and cancelled():
                    raise CollectionCancelled("Force teacher subprocess cancelled")
                try:
                    line = messages.get(timeout=.1)
                except queue.Empty:
                    continue
                if line is None:
                    break
                tail.append(line.rstrip())
                tail = tail[-30:]
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if event.get("type") == "policy":
                    if policy is None:
                        raise ValueError("Unexpected policy request")
                    action = np.asarray(policy(np.asarray(event["observation"], dtype=np.float32)))
                    process.stdin.write(json.dumps(action.tolist(), allow_nan=False) + "\n")
                    process.stdin.flush()
                elif event.get("type") == "progress" and on_progress:
                    on_progress(event["data"])
            code = process.wait(timeout=10)
            if code:
                raise RuntimeError("Force teacher subprocess failed: " + "\n".join(tail))
            with np.load(output, allow_pickle=False) as saved:
                result = {key: saved[key].copy() for key in saved.files if key != "report"}
                result["report"] = json.loads(str(saved["report"]))
            return result if operation in ("collect", "probe") else result["report"]
        finally:
            if process.poll() is None:
                if os.name == "nt":
                    process.terminate()
                else:
                    os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            process.stdin.close()
            process.stdout.close()
            thread.join(timeout=1)


def collect_dataset(original_root, *, episodes=8, steps_per_episode=600, seed=0, on_progress=None, cancelled=None,
                    policy_callable=None, student_probability=0.):
    if not 0 <= student_probability <= 1 or (student_probability > 0 and policy_callable is None):
        raise ValueError("Student mixing requires a callable and probability in [0,1]")
    return _worker("collect", original_root, episodes, steps_per_episode, seed, policy=policy_callable,
                   on_progress=on_progress, cancelled=cancelled, student_probability=student_probability)


def evaluate_student(original_root, policy_callable, *, episodes=3, steps_per_episode=600, seed=12345, cancelled=None,
                     evaluation_profile="standing_gate"):
    if evaluation_profile not in ("standing_gate", "small_commands"):
        raise ValueError("Unknown force evaluation profile")
    return _worker("evaluate", original_root, episodes, steps_per_episode, seed, policy=policy_callable,
                   cancelled=cancelled, evaluation_profile=evaluation_profile)


def verify_native_reference(original_root):
    """One CPU MJX motor/observation cycle against an isolated native student."""
    probe = _worker("probe", original_root, 1, 1, 0)
    import jax
    import jax.numpy as jp
    import mujoco
    from mujoco import mjx
    from force_teacher.env import StudentEnv, student_config
    with jax.default_device(jax.devices("cpu")[0]):
        cfg = student_config()
        cfg.noise_config.level = 0.
        cfg.cmd_config.cmd_prob = 0.
        env = StudentEnv(cfg)
        state = env.reset(jax.random.PRNGKey(0))
        data = mujoco.MjData(env.mj_model)
        data.qpos[:], data.qvel[:] = probe["initial_qpos"], probe["initial_qvel"]
        mujoco.mj_forward(env.mj_model, data)
        state = state.replace(data=mjx.put_data(env.mj_model, data))
        state.info["command"] = jp.zeros(4)
        state.info["position_start"] = env.get_hold_position(state.data)
        state.info["sinyaw_start"] = env._getSinYaw_newest(state.data)
        state.info["sin_balance_zero"] = env.get_sin_balance_angle(state.data)
        state = state.replace(obs=env._get_obs(state.data, state.info))
        before = float(np.max(np.abs(np.asarray(state.obs["state"]) - probe["initial_obs"])))
        result = jax.jit(env.step)(state, jp.asarray(probe["action"]))
        jax.block_until_ready(result)
        errors = dict(initial_observation=before,
            qpos=float(np.max(np.abs(np.asarray(result.data.qpos) - probe["qpos"]))),
            qvel=float(np.max(np.abs(np.asarray(result.data.qvel) - probe["qvel"]))),
            sent_substep_torque=float(np.max(np.abs(np.asarray(result.info["hybrid_command_substeps"]) - probe["sent"]))),
            actual_substep_torque=float(np.max(np.abs(np.asarray(result.info["hybrid_actual_substeps"]) - probe["actual"]))),
            final_observation=float(np.max(np.abs(np.asarray(result.obs["state"]) - probe["obs"]))))
        if state.obs["state"].shape != (24,) or state.obs["privileged_state"].shape != (51,):
            raise AssertionError("Force student observation dimensions changed")
    tolerances = dict(initial_observation=1e-4, qpos=1e-4, qvel=.02,
                      sent_substep_torque=1e-5, actual_substep_torque=1e-5, final_observation=.02)
    if any(not math.isfinite(value) or value > tolerances[key] for key, value in errors.items()):
        raise AssertionError(f"Native force student differs from CPU MJX: {errors}")
    return dict(passed=True, backend="cpu", actor_dim=24, critic_dim=51,
                errors=errors, tolerances=tolerances, source="four real motors; 400 Hz J*F mapping")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("collect", "evaluate", "probe"))
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--student-probability", type=float, default=0.)
    parser.add_argument("--evaluation-profile", choices=("standing_gate", "small_commands"), default="standing_gate")
    args = parser.parse_args()
    if not 1 <= args.episodes <= 64 or not 1 <= args.steps <= 600:
        parser.error("Invalid bounded sampling size")
    if not 0 <= args.student_probability <= 1:
        parser.error("Invalid mixing probability")
    def progress(event):
        print(json.dumps(dict(type="progress", data=event)), flush=True)
    def policy(obs):
        print(json.dumps(dict(type="policy", observation=obs.tolist())), flush=True)
        return np.asarray(json.loads(sys.stdin.readline()), dtype=np.float32)
    if args.operation == "probe":
        student = NativeStudent(args.original_root)
        data = dict(initial_qpos=student.state.data.qpos.copy(), initial_qvel=student.state.data.qvel.copy(),
                    initial_obs=student.observation(), action=np.array([.2, -.15, .1, -.1]))
        result = student.step(data["action"])
        data.update(qpos=student.state.data.qpos.copy(), qvel=student.state.data.qvel.copy(),
                    obs=student.observation(), sent=result["sent_torques"], actual=result["actual_torques"],
                    report=dict(done=result["done"], motor_model=True))
    elif args.operation == "collect":
        data = _collect_native(args.original_root, args.episodes, args.steps, args.seed, progress,
                               policy if args.student_probability > 0 else None, args.student_probability)
    else:
        data = dict(report=_evaluate_native(args.original_root, policy, args.episodes, args.steps, args.seed, args.evaluation_profile))
    np.savez(args.output, **{k: v for k, v in data.items() if k != "report"},
             report=json.dumps(data["report"], allow_nan=False))


if __name__ == "__main__":
    main()
