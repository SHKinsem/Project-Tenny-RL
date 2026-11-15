"""Bounded native-task checks against independent Python MuJoCo C API loops.

Run with the built native_eval package on PYTHONPATH. Supply an already exported
descriptor and the original reset-state NPZ. This script never imports a training
runtime, JAX, or a policy artifact; its constant actions are exactly representable
in float32. The force reference derives length Jacobians with mj_jac from the
compiled model, independently of the task's reduced LegGeometry implementation.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import mujoco
import numpy as np

from native_eval import Evaluator, load_descriptor


MODES = ("virtual_leg_force", "direct_torque", "matched_envelope")
ACTION = np.asarray([1 / 32, -1 / 64, 1 / 64, -1 / 128], dtype=np.float32)
FORCE_ATOL = 2e-11
FORCE_RTOL = 2e-11


def constant_policy():
    return {
        "type": "mlp", "mean": [0.0] * 24, "std": [1.0] * 24,
        "layers": [{"weights": np.zeros((4, 24)).tolist(),
                    "bias": ACTION.tolist(), "activation": "identity"}],
        "action_size": 4, "output_activation": "identity",
    }


def load_initial(path, nq, nv):
    """Preserve supplied reset anchors; the physical reference needs qpos/qvel."""
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    for name, width in (("qpos", nq), ("qvel", nv)):
        value = arrays.get(name)
        if value is None or value.ndim != 2 or value.shape[1] != width:
            raise ValueError(f"initial {name} must be [world,{width}]")
        if not np.isfinite(value).all():
            raise ValueError(f"initial {name} must be finite")
    count = arrays["qpos"].shape[0]
    if arrays["qvel"].shape[0] != count or count < 3:
        raise ValueError("initial states must provide at least three matching worlds")
    result = []
    for world in range(count):
        initial = {name: arrays[name][world].tolist() for name in ("qpos", "qvel")}
        extra = {}
        for name in ("position_start", "sinyaw_start", "sin_balance_zero"):
            if name in arrays:
                values = arrays[name]
                if values.shape not in ((count,), (count, 1)) or not np.isfinite(values).all():
                    raise ValueError(f"initial {name} must contain one finite value per world")
                extra[name] = [float(np.asarray(values[world]).reshape(-1)[0])]
        if extra:
            initial["extra"] = extra
        if "command" in arrays:
            command = arrays["command"]
            if command.shape == (4,):
                command = command.copy()
            elif command.shape == (count, 4):
                command = command[world]
            else:
                raise ValueError("initial command must have shape [4] or [world,4]")
            if not np.isfinite(command).all():
                raise ValueError("initial command must be finite")
            initial["command"] = command.tolist()
        result.append(initial)
    return result


class CapiReference:
    """Independent FK/Jacobians plus mj_step; no native task calls or callbacks."""

    def __init__(self, descriptor):
        self.cfg = descriptor["task"]["config"]
        self.model = mujoco.MjModel.from_xml_path(descriptor["xml_path"])
        if descriptor.get("physics_dt", 0):
            self.model.opt.timestep = descriptor["physics_dt"]
        self.data = mujoco.MjData(self.model)
        self.kinematic = mujoco.MjData(self.model)
        self.hips = self.ids("hip_joints", mujoco.mjtObj.mjOBJ_JOINT)
        self.knees = self.ids("knee_joints", mujoco.mjtObj.mjOBJ_JOINT)
        self.wheels = self.ids("wheel_joints", mujoco.mjtObj.mjOBJ_JOINT)
        self.wheel_bodies = self.ids("wheel_bodies", mujoco.mjtObj.mjOBJ_BODY)
        self.actuators = self.ids("actuators", mujoco.mjtObj.mjOBJ_ACTUATOR)
        self.root = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.cfg["root_body"])
        self.hip_q = self.model.jnt_qposadr[self.hips]
        self.knee_q = self.model.jnt_qposadr[self.knees]
        self.hip_v = self.model.jnt_dofadr[self.hips]
        self.knee_v = self.model.jnt_dofadr[self.knees]
        self.actual_v = self.model.jnt_dofadr[[*self.hips, *self.wheels]]
        self.hip_refs = self.model.qpos0[self.hip_q]
        self.knee_refs = self.model.qpos0[self.knee_q]
        self.poly = []
        for hip, knee in zip(self.hips, self.knees):
            matches = np.flatnonzero(
                (self.model.eq_type == mujoco.mjtEq.mjEQ_JOINT)
                & (self.model.eq_obj1id == knee) & (self.model.eq_obj2id == hip)
            )
            if matches.size != 1:
                raise ValueError("reference requires one polynomial knee/hip equality per leg")
            self.poly.append(self.model.eq_data[matches[0], :5].copy())
        self.limits = np.asarray(self.cfg["torque_scales"], dtype=np.float64)
        self.force_scales = np.asarray(self.cfg["force_scales"], dtype=np.float64)
        self.jmax = np.asarray(self.cfg["jmax"], dtype=np.float64)
        self.hip_range = self.cfg["hip_range"]
        self.jac_wheel = np.empty((3, self.model.nv))
        self.jac_hip = np.empty((3, self.model.nv))

    def ids(self, key, kind):
        ids = [mujoco.mj_name2id(self.model, kind, name) for name in self.cfg[key]]
        if min(ids) < 0:
            raise ValueError("reference could not resolve " + key)
        return np.asarray(ids, dtype=np.int32)

    def geometry(self):
        d, scratch, model = self.data, self.kinematic, self.model
        if not (np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()):
            return np.zeros(2), False
        # Do not alter the physics world's derived-stage data or warm starts.
        scratch.qpos[:] = d.qpos
        mujoco.mj_kinematics(model, scratch)
        mujoco.mj_comPos(model, scratch)
        jacobians = []
        valid = True
        for side, (hip, knee, wheel_body) in enumerate(zip(self.hips, self.knees, self.wheel_bodies)):
            hip_anchor = scratch.xanchor[hip]
            displacement = scratch.xpos[wheel_body] - hip_anchor
            length = np.linalg.norm(displacement)
            mujoco.mj_jacBody(model, scratch, self.jac_wheel, None, int(wheel_body))
            mujoco.mj_jac(model, scratch, self.jac_hip, None, hip_anchor,
                          int(model.jnt_bodyid[hip]))
            # Spatial length derivative, including the independently measured
            # hip partial (analytically zero in this supported geometry).
            dl = (displacement @ (self.jac_wheel - self.jac_hip)) / (length + 1e-15)
            x = d.qpos[self.hip_q[side]] - self.hip_refs[side]
            polynomial = self.poly[side]
            slope = np.polynomial.polynomial.polyval(
                x, polynomial[1:] * np.arange(1, 5)
            )
            coupled_knee = self.knee_refs[side] + np.polynomial.polynomial.polyval(x, polynomial)
            partial = dl[self.knee_v[side]]
            jacobian = dl[self.hip_v[side]] + slope * partial
            position_residual = d.qpos[self.knee_q[side]] - coupled_knee
            velocity_residual = partial * (d.qvel[self.knee_v[side]] - slope * d.qvel[self.hip_v[side]])
            qh = d.qpos[self.hip_q[side]]
            valid = valid and bool(
                np.isfinite([length, jacobian, velocity_residual]).all()
                and self.hip_range[0] <= qh <= self.hip_range[1]
                and abs(position_residual) <= self.cfg.get("equality_position_limit", .01)
                and abs(velocity_residual) <= self.cfg.get("length_velocity_limit", .05)
                and abs(jacobian) / self.jmax[side] >= self.cfg.get("minimum_jacobian_ratio", .05)
            )
            jacobians.append(jacobian)
        return np.asarray(jacobians), valid

    def reset(self, initial):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = initial["qpos"]
        self.data.qvel[:] = initial["qvel"]
        mujoco.mj_forward(self.model, self.data)

    def step(self, mode, substeps):
        invalid = False
        for _ in range(substeps):
            jacobian, valid = self.geometry()
            invalid = invalid or not valid
            gains = self.limits.copy()
            if mode != "direct_torque":
                gains[:2] = (abs(jacobian) if mode == "matched_envelope" else jacobian) * self.force_scales
            torque = np.zeros(4) if invalid else np.clip(gains * ACTION, -self.limits, self.limits)
            self.data.ctrl[self.actuators] = torque
            mujoco.mj_step(self.model, self.data)
            invalid = invalid or not np.isfinite(self.data.qfrc_actuator[self.actual_v]).all()
        _, valid_after = self.geometry()
        r = self.data.xmat[self.root].reshape(3, 3)
        pitch = -r[2, 0]
        roll = r[2, 1] / np.sqrt(1 - pitch * pitch)
        return bool(invalid or not valid_after
                    or pitch < self.cfg.get("pitch_min", -.6)
                    or pitch > self.cfg.get("pitch_max", .85)
                    or abs(roll) > self.cfg.get("roll_limit", .6))

    def trajectory(self, initial, mode, substeps, steps):
        records = {name: [] for name in ("qpos", "qvel", "ctrl", "done")}
        self.reset(initial)
        for _ in range(steps):
            done = self.step(mode, substeps)
            records["qpos"].append(self.data.qpos.copy())
            records["qvel"].append(self.data.qvel.copy())
            records["ctrl"].append(self.data.ctrl.copy())
            records["done"].append(done)
            if done:
                self.reset(initial)
        return {name: np.asarray(values) for name, values in records.items()}


def reset_initial_state(descriptor, initial):
    """Make an explicit airborne tilted state, with freshly computed anchors."""
    model = mujoco.MjModel.from_xml_path(descriptor["xml_path"])
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, descriptor["task"]["config"]["root_body"])
    first_joint = model.body_jntadr[root]
    joints = range(first_joint, first_joint + model.body_jntnum[root])
    free_joints = [joint for joint in joints if model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE]
    if len(free_joints) != 1:
        raise ValueError("tilted-reset check requires one free joint on the configured root body")
    address = int(model.jnt_qposadr[free_joints[0]])
    tilted = deepcopy(initial)
    qpos = np.asarray(tilted["qpos"], dtype=np.float64).copy()
    # Raising the explicit initial state keeps the wheel/ground contact impulse
    # out of this falling-angle test; the normal force/geometry checks stay on.
    qpos[address + 2] += 1.0
    roll = 1.0
    qpos[address + 3:address + 7] = [np.cos(roll / 2), np.sin(roll / 2), 0., 0.]
    tilted["qpos"] = qpos.tolist()
    tilted["qvel"] = np.zeros(model.nv).tolist()
    # The archive's balance/hold anchors belonged to its unmodified state.
    tilted.pop("extra", None)
    return tilted


def validate_resets(descriptor, initial):
    """Compare reset cycles to independently constructed fresh native runs.

    Reward is intentionally an internal consistency check here, not a second
    reward oracle. Nonzero constant torques make uncleared actual-torque rate
    history visible in the repeated first-step reward.
    """
    fields = ("observation", "action", "qpos", "qvel", "ctrl", "reward", "done")
    tilted = reset_initial_state(descriptor, initial[0])
    reference = CapiReference(descriptor)
    control_dt = 4 * reference.model.opt.timestep
    cases, failures = [], []
    for mode in MODES:
        candidate = dict(descriptor)
        candidate["task"] = {"type": "legwheel", "config": dict(descriptor["task"]["config"], action_mode=mode)}
        candidate["policy"] = constant_policy()
        for kind, selected_initial, episode_steps, rollout_steps, period in (
            ("early_termination", tilted, 100, 4, 1),
            ("time_limit", initial[0], 2, 6, 2),
        ):
            # The expected period comes from another evaluator and another
            # evaluate call, rather than a prefix of the result under test.
            expected = Evaluator(candidate).evaluate(
                [selected_initial], num_envs=1, steps=period, episode_steps=episode_steps,
                control_dt=control_dt, seed=0, trace_worlds=[0],
            )
            result = Evaluator(candidate).evaluate(
                [selected_initial], num_envs=1, steps=rollout_steps, episode_steps=episode_steps,
                control_dt=control_dt, seed=0, trace_worlds=[0],
            )
            first, trace = expected["traces"][0], result["traces"][0]
            case = {"mode": mode, "kind": kind, "substeps": 4, "num_envs": 1,
                    "steps": rollout_steps, "episode_steps": episode_steps,
                    "expected_period_steps": period,
                    "resets": int(result["resets"][0]),
                    "first_episode_length": int(result["lengths"][0]),
                    "first_episode_return": float(result["returns"][0]),
                    "first_episode_reason": result["reasons"][0],
                    "all_fields_bit_exact": True, "compared_fields": list(fields),
                    "passed": True}
            label = f"{mode}/reset={kind}"
            try:
                assert result["control_steps"] == rollout_steps
                assert result["physics_steps"] == 4 * rollout_steps
                assert not result["cancelled"] and not expected["cancelled"]
                assert trace["world"] == 0 and trace["steps"] == rollout_steps
                assert np.any(first["ctrl"] != 0), "reset check needs nonzero actual motor commands"
                for field in fields:
                    tiled = np.concatenate([first[field]] * (rollout_steps // period), axis=0)
                    case["all_fields_bit_exact"] &= bool(np.array_equal(trace[field], tiled))
                    np.testing.assert_array_equal(trace[field], tiled, err_msg=field)
                assert result["lengths"].tolist() == [period]
                assert result["resets"].tolist() == [rollout_steps // period]
                # Evaluator returns account for the first episode only.
                np.testing.assert_array_equal(result["returns"], expected["returns"])
                assert result["returns"][0] == float(first["reward"].sum())
                if kind == "early_termination":
                    assert result["terminated"].tolist() == [True]
                    assert result["truncated"].tolist() == [False]
                    assert result["reasons"][0] in ("pitch", "roll")
                    np.testing.assert_array_equal(trace["done"], np.ones(rollout_steps, dtype=bool))
                else:
                    assert result["terminated"].tolist() == [False]
                    assert result["truncated"].tolist() == [True]
                    assert result["reasons"] == ["time_limit"]
                    np.testing.assert_array_equal(trace["done"], [False, True] * 3)
            except AssertionError as exc:
                case["passed"] = False
                failures.append(label + ": " + str(exc).strip().replace("\n", " ")[:300])
            cases.append(case)
    return cases, failures


def validate(descriptor, initial, steps):
    reference = CapiReference(descriptor)
    cases, failures = [], []
    for mode in MODES:
        candidate = dict(descriptor)
        candidate["task"] = {"type": "legwheel", "config": dict(descriptor["task"]["config"], action_mode=mode)}
        candidate["policy"] = constant_policy()
        engine = Evaluator(candidate)
        for substeps in (1, 3, 4):
            for num_envs in (1, 3):
                result = engine.evaluate(
                    initial[:num_envs], num_envs=num_envs, steps=steps,
                    episode_steps=steps + 1, control_dt=reference.model.opt.timestep * substeps,
                    seed=0, trace_worlds=list(range(num_envs)),
                )
                case = {"mode": mode, "substeps": substeps, "num_envs": num_envs,
                        "steps": steps, "max_abs_error": {"qpos": 0., "qvel": 0., "ctrl": 0.},
                        "all_physics_values_bit_exact": True, "resets": result["resets"].tolist(),
                        "passed": True}
                label = f"{mode}/substeps={substeps}/worlds={num_envs}"
                try:
                    assert result["control_steps"] == steps * num_envs
                    assert result["physics_steps"] == steps * substeps * num_envs
                    assert not result["cancelled"]
                    assert len(result["traces"]) == num_envs
                    for world, trace in enumerate(result["traces"]):
                        expected = reference.trajectory(initial[world], mode, substeps, steps)
                        assert trace["world"] == world and trace["steps"] == steps
                        np.testing.assert_array_equal(trace["action"], np.tile(ACTION, (steps, 1)))
                        for field in ("qpos", "qvel", "ctrl"):
                            observed = trace[field]
                            assert observed.shape == expected[field].shape
                            assert np.isfinite(observed).all()
                            error = float(np.max(np.abs(observed - expected[field])))
                            case["max_abs_error"][field] = max(case["max_abs_error"][field], error)
                            case["all_physics_values_bit_exact"] &= bool(np.array_equal(observed, expected[field]))
                        for field in ("qpos", "qvel", "ctrl"):
                            observed = trace[field]
                            if mode == "direct_torque":
                                np.testing.assert_array_equal(observed, expected[field], err_msg=field)
                            else:
                                np.testing.assert_allclose(observed, expected[field], atol=FORCE_ATOL,
                                                           rtol=FORCE_RTOL, err_msg=field)
                        np.testing.assert_array_equal(trace["done"], expected["done"])
                        first_done = np.flatnonzero(expected["done"])
                        length = int(first_done[0] + 1) if first_done.size else steps
                        assert result["lengths"][world] == length
                        assert bool(result["terminated"][world]) == bool(first_done.size)
                        assert not result["truncated"][world]
                        assert result["resets"][world] == int(expected["done"].sum())
                except AssertionError as exc:
                    case["passed"] = False
                    failures.append(label + ": " + str(exc).strip().replace("\n", " ")[:300])
                cases.append(case)
    reset_cases, reset_failures = validate_resets(descriptor, initial)
    failures.extend(reset_failures)
    return {"schema_version": 1, "passed": not failures, "cases": cases,
            "reset_cases": reset_cases, "failures": failures,
            "constant_action": ACTION.tolist(),
            "reference": "Python MuJoCo mj_step; independent spatial mj_jac length Jacobian and model equality polynomial",
            "tolerances": {"direct_torque": "bit exact qpos/qvel/ctrl", "force_atol": FORCE_ATOL,
                           "force_rtol": FORCE_RTOL},
            "scope": ["all three action modes", "substeps 1/3/4", "world counts 1/3",
                      "physics traces", "constant policy actions", "termination/reset/workload",
                      "fall termination and time-limit reset of observation/action/actual-torque history"],
            "reset_reference": "Separate fresh native one/two-step runs; internal reward-history consistency only",
            "not_covered": ["reward equality", "learned policy behavior", "long-horizon stability"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptor", type=Path, required=True)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=12, help="control steps per case; bounded to 8..16")
    parser.add_argument("--output", type=Path, help="optional JSON report destination")
    args = parser.parse_args()
    if not 8 <= args.steps <= 16:
        parser.error("--steps must be between 8 and 16")
    descriptor = load_descriptor(args.descriptor)
    if descriptor["task"]["type"] != "legwheel":
        parser.error("descriptor must contain a legwheel task")
    model = mujoco.MjModel.from_xml_path(descriptor["xml_path"])
    initial = load_initial(args.initial, model.nq, model.nv)
    report = validate(descriptor, initial, args.steps)
    encoded = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(encoded + "\n", encoding="utf8")
    print(encoded, flush=True)
    assert report["passed"], "legwheel mode reference comparison failed: " + "; ".join(report["failures"])


if __name__ == "__main__":
    main()
