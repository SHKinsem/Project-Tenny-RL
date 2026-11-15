"""Student retains original position-actuated legs, replacing only wheel control."""
from pathlib import Path
import jax.numpy as jp
from legwheel.env import LegWheelRobot
from legwheel.config import default_config
import legwheel.env as original_env
from wheel_teacher.contract import WHEEL_TORQUE_LIMIT


def student_config():
    cfg = default_config()
    cfg.vel_pid.enable = False
    cfg.action_scale.wheel = WHEEL_TORQUE_LIMIT
    cfg.action_bias.wheel = 0.0
    cfg.action_delay_steps = 0
    cfg.obs_history_len = 1
    cfg.last_act_obs = False
    return cfg


class StudentEnv(LegWheelRobot):
    def __init__(self, config=None):
        # A 24D all-torque LegWheelRobot has the same class name: fail closed.
        if "mujoco_legwheel_rl_public" not in Path(original_env.__file__).resolve().parts:
            raise RuntimeError("wheel_torque_bc must import the original legwheel source")
        cfg = student_config() if config is None else config
        if (cfg.ctrl_dt != .01 or cfg.sim_dt != .0025 or cfg.vel_pid.enable or
                not cfg.delta_thigh_action or cfg.action_scale.thigh != .25 or
                cfg.action_scale.wheel != WHEEL_TORQUE_LIMIT or cfg.action_bias.wheel != 0 or
                list(cfg.action_range.thigh) != [-.5, 0.] or
                cfg.action_delay_steps != 0 or cfg.obs_history_len != 1 or cfg.last_act_obs):
            raise ValueError("Student control contract differs from wheel_torque_bc")
        super().__init__(cfg)
        if self.mj_model.nu != 4 or self.n_substeps != 4:
            raise ValueError("Expected original four-actuator, 100/400 Hz model")

    def reset(self, rng, random=False):
        state = super().reset(rng, random)
        state.info["student_torque_history"] = jp.zeros((2, 2))
        return state

    def _get_obs(self, data, info):
        result = super()._get_obs(data, info)
        history = info.get("student_torque_history", jp.zeros((2, 2))).reshape(-1)
        # Reuse already-noised measured wheel velocity, never teacher target/error.
        wheels = result["state"][jp.array([15, 17])]
        state = result["state"].at[20:26].set(jp.concatenate((history, wheels)))
        privileged = result["privileged_state"].at[:26].set(state)
        privileged = privileged.at[-6:].set(jp.concatenate((history, self._get_wheel_angvel(data))))
        return {"state": state, "privileged_state": privileged}

    def step(self, state, action):
        if "steps" in state.info:
            fresh = state.info["steps"] == 0
            for key in ("student_torque_history", "last_act", "last_last_act", "current_act",
                        "last_wheel_torque", "pid_integral", "last_target_wheel_vel", "wheels_commands_history"):
                if key in state.info:
                    state.info[key] = jp.where(fresh, jp.zeros_like(state.info[key]), state.info[key])
        finite = jp.all(jp.isfinite(action))
        action = jp.clip(jp.where(finite, action, jp.zeros_like(action)), -1., 1.)
        state.info["student_torque_history"] = jp.vstack(
            [state.info["student_torque_history"][1:], action[2:][None, :]])
        result = super().step(state, action)
        return result.replace(done=jp.maximum(result.done, (~finite).astype(float)))
