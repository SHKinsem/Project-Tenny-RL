"""Three comparable opt-in 100/200 Hz modes on the existing torque200 environment.

Original torque200/default checkpoints remain separate. This family changes the
previous-command/effort semantics for ALL three modes together, not only force.
"""
from pathlib import Path

import jax
import jax.numpy as jp
from mujoco_playground._src import mjx_env

from legwheel.config import default_config
from legwheel.env import LegWheelRobot
from hybrid.core import (HIP_RANGE, MODES, TORQUE_LIMITS, read_geometry,
                         scan_geometry, metadata, XML, MAX_EQ_POSITION, MAX_LENGTH_VELOCITY,
                         MIN_JACOBIAN_RATIO)
from hybrid.timing import configure_timing, validate_timing, DEFAULT_CONTROL_HZ


def hybrid_config(control_hz=DEFAULT_CONTROL_HZ, action_delay_ms=0.0):
    cfg = default_config()
    cfg.push_config.enable = False  # First stage: flat ground, no added push curriculum.
    return configure_timing(cfg, control_hz, action_delay_ms)


class HybridLegWheelRobot(LegWheelRobot):
    def __init__(self, mode="virtual_leg_force", config=None):
        if mode not in MODES:
            raise ValueError(mode)
        self.mode = mode
        self.legs = read_geometry()
        self.scan = scan_geometry(self.legs)
        self.force_scales = jp.asarray([r["force_scale_n"] for r in self.scan])
        self.jmax = jp.asarray([r["jmax_bound"] for r in self.scan])
        cfg = config if config is not None else hybrid_config()
        timing = validate_timing(cfg)
        self.control_hz = timing.control_hz
        if tuple(cfg.torque_control.scales) != TORQUE_LIMITS:
            raise ValueError("Changed torque limits require explicit recalibration")
        if Path(cfg._XML_PATH).resolve() != XML.resolve():
            raise ValueError("Hybrid geometry must use the calibrated source XML")
        super().__init__(cfg)
        if self.n_substeps != timing.n_substeps or self.dt != timing.ctrl_dt:
            raise ValueError("Simulator timing differs from the hybrid timing contract")
        # All actuator/dof orders are checked again by validate.py on compiled model.
        self.hip_q = jp.asarray([self.mj_model.jnt_qposadr[self.mj_model.joint(f"thigh_{s}_joint").id]
                                 for s in ("left", "right")])
        self.knee_q = jp.asarray([self.mj_model.jnt_qposadr[self.mj_model.joint(f"calf_{s}_joint").id]
                                  for s in ("left", "right")])
        self.hip_v = jp.asarray([self.mj_model.jnt_dofadr[self.mj_model.joint(f"thigh_{s}_joint").id]
                                 for s in ("left", "right")])
        self.knee_v = jp.asarray([self.mj_model.jnt_dofadr[self.mj_model.joint(f"calf_{s}_joint").id]
                                  for s in ("left", "right")])
        self.act_v = jp.asarray([self.mj_model.jnt_dofadr[self.mj_model.joint(f"{part}_{s}_joint").id]
                                 for part in ("thigh", "wheel") for s in ("left", "right")])
        self.policy_metadata = metadata(mode, self.scan, cfg.control_hz, cfg.action_delay_ms)

    def geometry(self, data):
        qh, qk = data.qpos[self.hip_q], data.qpos[self.knee_q]
        results = [leg.evaluate(qh[i], qk[i], jp) for i, leg in enumerate(self.legs)]
        length, jac, partial = (jp.asarray([r[i] for r in results]) for i in range(3))
        slope = jp.asarray([leg.slope(qh[i]) for i, leg in enumerate(self.legs)])
        residual = qk-jp.asarray([leg.coupled_knee(qh[i]) for i, leg in enumerate(self.legs)])
        residual_v = data.qvel[self.knee_v]-slope*data.qvel[self.hip_v]
        return length, jac, residual, residual_v, partial*residual_v

    def torque_command(self, action, jac):
        a = jp.clip(action, -1.0, 1.0)
        gain = (self._torque_scales[:2] if self.mode == "direct_torque" else
                (jp.abs(jac) if self.mode == "matched_envelope" else jac)*self.force_scales)
        raw = jp.concatenate((gain*a[:2], self._torque_scales[2:]*a[2:]))
        return jp.clip(raw, -self._torque_scales, self._torque_scales), raw

    def reset(self, rng, random=False):
        state = super().reset(rng, random)
        state.info.update(
            hybrid_mean_actual_torque=jp.zeros(4), hybrid_effort=jp.zeros(()),
            hybrid_current_actual=jp.zeros(4), hybrid_last_actual=jp.zeros(4),
            hybrid_last_last_actual=jp.zeros(4), hybrid_action_rate=jp.zeros(()),
            hybrid_command_substeps=jp.zeros((self.n_substeps, 4)), hybrid_actual_substeps=jp.zeros((self.n_substeps, 4)),
            hybrid_jacobian=jp.zeros(2), hybrid_invalid=jp.zeros((), dtype=bool),
            hybrid_bad_request=jp.zeros((), dtype=bool),
        )
        for key in ("command", "position_start", "sinyaw_start", "sin_balance_zero"):
            state.info["hybrid_initial_"+key] = state.info[key]
        state.metrics.update({"hybrid/"+k: jp.zeros(()) for k in (
            "saturation_fraction", "deadpoint_fraction", "out_of_range", "invalid_state",
            "equality_position_max_rad", "equality_velocity_max_rad_s",
            "length_velocity_residual_max_m_s", "min_jacobian_ratio", "actual_effort")})
        return state

    def _step_control(self, data, action, info, metrics):
        # Incoming action has already traversed the shared policy-delay queue.
        # Recompute at EVERY physics substep; no stale data.xpos after mjx.step.
        def substep(carry, _):
            d, previous, previous_previous, latched = carry
            length, jac, rpos, rvel, rlength = self.geometry(d)
            qh = d.qpos[self.hip_q]
            outside = jp.any((qh < HIP_RANGE[0]) | (qh > HIP_RANGE[1]))
            finite = (jp.all(jp.isfinite(d.qpos)) & jp.all(jp.isfinite(d.qvel)) &
                      jp.all(jp.isfinite(action)) & jp.all(jp.isfinite(jac)) &
                      jp.all(jp.isfinite(length)) & jp.all(jp.isfinite(rlength)))
            dead = jp.any(jp.abs(jac)/self.jmax < MIN_JACOBIAN_RATIO)
            residual_bad = jp.any(jp.abs(rpos) > MAX_EQ_POSITION) | jp.any(jp.abs(rlength) > MAX_LENGTH_VELOCITY)
            latched = latched | ~finite | outside | dead | residual_bad
            safe_a = jp.where(~latched, action, jp.zeros_like(action))
            tau, raw = self.torque_command(safe_a, jp.where(jp.isfinite(jac), jac, 0.0))
            # Checked gear=1, stateless fixed-gain motors; no hidden inner loop.
            new = mjx_env.step(self.mjx_model, d, tau, 1)
            actual = new.qfrc_actuator[self.act_v]
            normalized = actual/self._torque_scales
            # Substep time-integrated first/second difference of ACTUAL torque.
            ratio = self._config.reward_config.reference_dt/self._config.sim_dt
            rate = (jp.mean(jp.abs(normalized-previous))*ratio +
                    jp.mean(jp.abs(normalized+previous_previous-2*previous))*ratio**2)
            latched = latched | ~jp.all(jp.isfinite(actual))
            diagnostics = jp.asarray([
                jp.mean(jp.abs(raw) > self._torque_scales),
                jp.mean(jp.abs(jac)/self.jmax < MIN_JACOBIAN_RATIO), outside.astype(float), latched.astype(float),
                jp.max(jp.abs(rpos)), jp.max(jp.abs(rvel)), jp.max(jp.abs(rlength)),
                jp.min(jp.abs(jac)/self.jmax)])
            return (new, normalized, previous, latched), (tau, actual, jac, diagnostics, rate)

        initial = (data, info["hybrid_last_actual"], info["hybrid_last_last_actual"], info["hybrid_bad_request"])
        (data, last, last_last, invalid), (sent, actual, jac, diag, rates) = jax.lax.scan(
            substep, initial, None, length=self.n_substeps)
        sent_mean = jp.mean(sent, axis=0)
        actual_norm = actual/self._torque_scales
        actual_mean_norm = jp.mean(actual_norm, axis=0)
        info["current_act"] = sent_mean/self._torque_scales
        info["last_torque_action"] = info["current_act"]
        info["applied_torque"] = sent_mean
        info["hybrid_command_substeps"] = sent
        info["hybrid_actual_substeps"] = actual
        info["hybrid_mean_actual_torque"] = jp.mean(actual, axis=0)
        info["hybrid_effort"] = jp.mean(jp.square(actual_norm))
        info["hybrid_current_actual"] = actual_mean_norm
        info["hybrid_action_rate"] = jp.mean(rates)
        info["hybrid_last_last_actual"] = last_last
        info["hybrid_last_actual"] = last
        info["hybrid_jacobian"] = jp.mean(jac, axis=0)
        info["hybrid_invalid"] = invalid
        names = ("saturation_fraction", "deadpoint_fraction", "out_of_range", "invalid_state",
                 "equality_position_max_rad", "equality_velocity_max_rad_s",
                 "length_velocity_residual_max_m_s", "min_jacobian_ratio")
        for i, name in enumerate(names):
            metrics["hybrid/"+name] = jp.min(diag[:, i]) if i == 7 else jp.max(diag[:, i])
        metrics["hybrid/actual_effort"] = info["hybrid_effort"]
        return data, info["current_act"], sent_mean

    def _get_reward(self, data, action, info, metrics):
        result = super()._get_reward(data, action, info, metrics)
        # Rewards penalize actual actuator torque, NOT Fv, requested ctrl or
        # total generalized forces containing gravity/contact/equality terms.
        metrics["reward/energy_cost"] = info["hybrid_effort"]
        metrics["reward/action_rate"] = info["hybrid_action_rate"]
        result["energy_cost"] = info["hybrid_effort"]*self._config.reward_config.energy_scale
        result["action_penalty"] = info["hybrid_action_rate"]*self._config.reward_config.action_rate_scale
        return result

    def _is_termination(self, data):
        qh = data.qpos[self.hip_q]
        length, jac, rpos, _, rlength = self.geometry(data)
        return (super()._is_termination(data) | ~jp.all(jp.isfinite(data.qpos)) |
                ~jp.all(jp.isfinite(data.qvel)) | ~jp.all(jp.isfinite(length)) |
                ~jp.all(jp.isfinite(jac)) | ~jp.all(jp.isfinite(rlength)) |
                jp.any(qh < HIP_RANGE[0]) | jp.any(qh > HIP_RANGE[1]) |
                jp.any(jp.abs(rpos) > MAX_EQ_POSITION) |
                jp.any(jp.abs(rlength) > MAX_LENGTH_VELOCITY) |
                jp.any(jp.abs(jac)/self.jmax < MIN_JACOBIAN_RATIO))

    def step(self, state, action):
        # Brax auto-reset restores initial physics/observation, not arbitrary
        # environment info. AutoResetWrapper resets `steps` before calling us.
        if "steps" in state.info:
            fresh = state.info["steps"] == 0
            for key in ("hybrid_last_actual", "hybrid_last_last_actual", "hybrid_current_actual",
                        "hybrid_mean_actual_torque", "hybrid_command_substeps", "hybrid_actual_substeps",
                        "current_act", "last_torque_action", "applied_torque", "last_act", "last_last_act",
                        "act_delay_buffer", "episode_step", "step", "push_step", "push"):
                state.info[key] = jp.where(fresh, jp.zeros_like(state.info[key]), state.info[key])
            for key in ("command", "position_start", "sinyaw_start", "sin_balance_zero"):
                state.info[key] = jp.where(fresh, state.info["hybrid_initial_"+key], state.info[key])
        state.info["hybrid_bad_request"] = ~jp.all(jp.isfinite(action))
        action = jp.where(state.info["hybrid_bad_request"], jp.zeros_like(action), action)
        result = super().step(state, action)
        return result.replace(done=jp.maximum(result.done, result.info["hybrid_invalid"].astype(float)))
