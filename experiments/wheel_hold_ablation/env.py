"""Change only the weighted wheel-position contribution after original rewards."""
import math
from numbers import Real

from experiments.reward_ablation.legwheel_env import RewardExperimentEnv
from . import WHEEL_HOLD


class WheelHoldEnv(RewardExperimentEnv):
    def __init__(self, config=None, *, root_body='root', wheel_position_weight=1.):
        if (isinstance(wheel_position_weight,bool) or not isinstance(wheel_position_weight,Real)
                or not math.isfinite(wheel_position_weight) or not 0<=wheel_position_weight<=10):
            raise ValueError('wheel_position_weight must be finite and between zero and ten')
        self.wheel_position_weight=float(wheel_position_weight)
        super().__init__('position',config,root_body=root_body)
        if self._config.reward_config.zero_keep_scale!=WHEEL_HOLD['original_weight']:
            raise ValueError('Need original shared wheel/yaw reward weight ten')
        if not self._config.wheel_sinpos_start or not self._config.sinyaw:
            raise ValueError('Need the unchanged enabled wheel-position and yaw hold rewards')

    def _get_reward(self, data, action, info, metrics):
        rewards=super()._get_reward(data,action,info,metrics)
        # The inherited raw metric already contains the original command gate.
        # Its native/JAX sensor stage and reference are preserved.  Shared
        # zero_keep_scale also weights yaw and must never be changed globally.
        if self.wheel_position_weight==WHEEL_HOLD['original_weight']:
            return rewards
        value=dict(rewards)
        value['zero_keep']=value['zero_keep']+(self.wheel_position_weight-WHEEL_HOLD['original_weight'])*metrics[WHEEL_HOLD['metric']]
        # Inherited step applies the same dt/reference_dt once to the full sum.
        return value
