"""Same policy request scale, separate calibrated execution capacities."""
import jax.numpy as jp
from experiments.torque_limit_ablation.env import TorqueLimitEnv
from .contract import aligned_contract


class AlignedLimitEnv(TorqueLimitEnv):
    def __init__(self,arm,config=None,*,root_body='root',leg_limit_nm=1.):
        super().__init__(arm,config,root_body=root_body,leg_limit_nm=leg_limit_nm)
        self.actuator_contract=aligned_contract(leg_limit_nm,[row['jmax_bound'] for row in self.scan])
        self._policy_force_scales=jp.asarray(self.actuator_contract['policy_leg_force_scales'])
        self.policy_metadata.update(actuator_contract=self.actuator_contract,
            policy_leg_force_scales=self.actuator_contract['policy_leg_force_scales'],
            initialization_alignment='frozen_BC_mean_anchor_v1')

    def torque_command(self,action,jac):
        a=jp.clip(action,-1.,1.)
        requested_force=a[:2]*self._policy_force_scales
        force=jp.clip(requested_force,-self.force_scales,self.force_scales)
        raw=jp.concatenate((jac*force,self._execution_limits[2:]*a[2:]))
        return jp.clip(raw,-self._execution_limits,self._execution_limits),raw
