"""Opt-in actuator capability; baseline reward and observation torque units stay fixed."""
import jax
import jax.numpy as jp
from mujoco import mjx
from mujoco_playground._src import mjx_env

from experiments.reward_ablation.legwheel_env import RewardExperimentEnv
from hybrid.core import (HIP_RANGE, MAX_EQ_POSITION, MAX_LENGTH_VELOCITY,
                         MIN_JACOBIAN_RATIO)
from .contract import actuator_contract


class TorqueLimitEnv(RewardExperimentEnv):
    def __init__(self, arm, config=None, *, root_body='root', leg_limit_nm=2.):
        if arm!='baseline':
            raise ValueError('This comparison keeps the original reward')
        super().__init__(arm,config,root_body=root_body)
        contract=actuator_contract(leg_limit_nm,[row['jmax_bound'] for row in self.scan])
        self.actuator_contract=contract
        # _torque_scales remains the original normalization for all inherited
        # reward, rate history, and previous-torque observations.
        self._execution_limits=self._torque_scales.at[:2].set(float(leg_limit_nm))
        self.force_scales=jp.asarray(contract['force_scales_n'])
        if leg_limit_nm!=1.:
            self._mj_model.actuator_ctrlrange[:2]=[[-leg_limit_nm,leg_limit_nm]]*2
            self._mj_model.actuator_forcerange[:2]=[[-leg_limit_nm,leg_limit_nm]]*2
            self._mjx_model=mjx.put_model(self._mj_model)
        self.policy_metadata.update(experimental_only=True,actuator_contract=contract,
            torque_limits_nm=contract['torque_limits_nm'],force_scales_n=contract['force_scales_n'],
            torque_normalization_scales=contract['torque_normalization_scales'])

    def torque_command(self, action, jac):
        a=jp.clip(action,-1.,1.)
        gain=jac*self.force_scales
        raw=jp.concatenate((gain*a[:2],self._execution_limits[2:]*a[2:]))
        return jp.clip(raw,-self._execution_limits,self._execution_limits),raw

    def _step_control(self, data, action, info, metrics):
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
            new = mjx_env.step(self.mjx_model, d, tau, 1)
            actual = new.qfrc_actuator[self.act_v]
            normalized = actual/self._torque_scales
            ratio = self._config.reward_config.reference_dt/self._config.sim_dt
            rate = (jp.mean(jp.abs(normalized-previous))*ratio +
                    jp.mean(jp.abs(normalized+previous_previous-2*previous))*ratio**2)
            latched = latched | ~jp.all(jp.isfinite(actual))
            diagnostics = jp.asarray([
                jp.mean(jp.abs(raw) > self._execution_limits),
                jp.mean(jp.abs(jac)/self.jmax < MIN_JACOBIAN_RATIO), outside.astype(float), latched.astype(float),
                jp.max(jp.abs(rpos)), jp.max(jp.abs(rvel)), jp.max(jp.abs(rlength)),
                jp.min(jp.abs(jac)/self.jmax)])
            return (new, normalized, previous, latched), (tau, actual, jac, diagnostics, rate)

        initial = (data, info['hybrid_last_actual'], info['hybrid_last_last_actual'], info['hybrid_bad_request'])
        (data, last, last_last, invalid), (sent, actual, jac, diag, rates) = jax.lax.scan(
            substep, initial, None, length=self.n_substeps)
        sent_mean = jp.mean(sent, axis=0)
        actual_norm = actual/self._torque_scales
        actual_mean_norm = jp.mean(actual_norm, axis=0)
        info['current_act'] = sent_mean/self._torque_scales
        info['last_torque_action'] = info['current_act']
        info['applied_torque'] = sent_mean
        info['hybrid_command_substeps'] = sent
        info['hybrid_actual_substeps'] = actual
        info['hybrid_mean_actual_torque'] = jp.mean(actual, axis=0)
        info['hybrid_effort'] = jp.mean(jp.square(actual_norm))
        info['hybrid_current_actual'] = actual_mean_norm
        info['hybrid_action_rate'] = jp.mean(rates)
        info['hybrid_last_last_actual'] = last_last
        info['hybrid_last_actual'] = last
        info['hybrid_jacobian'] = jp.mean(jac, axis=0)
        info['hybrid_invalid'] = invalid
        names = ('saturation_fraction', 'deadpoint_fraction', 'out_of_range', 'invalid_state',
                 'equality_position_max_rad', 'equality_velocity_max_rad_s',
                 'length_velocity_residual_max_m_s', 'min_jacobian_ratio')
        for i, name in enumerate(names):
            metrics['hybrid/'+name] = jp.min(diag[:, i]) if i == 7 else jp.max(diag[:, i])
        metrics['hybrid/actual_effort'] = info['hybrid_effort']
        return data, info['current_act'], sent_mean
