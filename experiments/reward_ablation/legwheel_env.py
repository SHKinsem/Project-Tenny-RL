"""Reward-only standing adapter over the unchanged deployed student dynamics."""
import jax.numpy as jp
import mujoco

from force_teacher.env import StudentEnv
from .rewards import coefficients, extra_rewards


class RewardExperimentEnv(StudentEnv):
    def __init__(self, arm, config=None, *, root_body='root'):
        self.reward_coefficients = coefficients(arm)
        super().__init__(config=config)
        body = self.mj_model.body(root_body).id
        joint = int(self.mj_model.body_jntadr[body])
        if self.mj_model.body_jntnum[body] != 1 or self.mj_model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError('Reward experiment needs a free-root legwheel body')
        self._experiment_root_qpos = int(self.mj_model.jnt_qposadr[joint])
        if self._config.cmd_config.cmd_prob != 0 or self._config.push_config.enable or self._config.noise_config.level != 0:
            raise ValueError('Matched reward experiment requires the fixed nominal standing task')

    def reset(self, rng, random=False):
        state = super().reset(rng, random)
        q = self._experiment_root_qpos
        state.info['ablation_xy_origin'] = state.data.qpos[q:q+2]
        state.info['ablation_pitch_reference_rad'] = jp.arcsin(jp.clip(state.info['sin_balance_zero'], -1., 1.))
        # These constants intentionally persist when Brax restores this exact
        # initial physics state. They never follow the moving body or dynamic
        # sin_balance_zero that the inherited observation controller may update.
        state.metrics.update({'reward/ablation_posture': jp.zeros(()),
                              'reward/ablation_position': jp.zeros(())})
        return state

    def _get_reward(self, data, action, info, metrics):
        rewards = super()._get_reward(data, action, info, metrics)
        q = self._experiment_root_qpos
        w, x, y, z = data.qpos[q+3:q+7]
        pitch = jp.arcsin(jp.clip(2*(w*y-z*x), -1., 1.))
        additions = extra_rewards(pitch-info['ablation_pitch_reference_rad'],
                                  data.qpos[q:q+2]-info['ablation_xy_origin'],
                                  self.reward_coefficients, jp)
        for key, value in additions.items():
            metrics['reward/'+key] = value
        return dict(rewards, **additions)
