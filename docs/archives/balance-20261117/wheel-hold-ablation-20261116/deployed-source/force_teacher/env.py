"""True motor-actuated 100 Hz student, reusing the calibrated hybrid environment."""
from hybrid.env import HybridLegWheelRobot, hybrid_config


def student_config():
    """Initial standing stage only; later command/robustness stages are not automatic.

    Keep command enables and reward weights intact: cmd_prob=0 also makes the
    command sampled during reset zero, rather than only disabling later changes.
    The force PPO adapter separately disables dynamics randomization.
    """
    cfg = hybrid_config(100)
    cfg.cmd_config.cmd_prob = 0.0
    cfg.noise_config.level = 0.0
    cfg.push_config.enable = False
    return cfg


class StudentEnv(HybridLegWheelRobot):
    def __init__(self, config=None):
        cfg = student_config() if config is None else config
        if cfg.control_hz != 100:
            raise ValueError("leg_force_bc is fixed at 100 Hz")
        super().__init__("virtual_leg_force", cfg)
        # Reject servo XMLs even if their joint order matches the motor model.
        for index in range(4):
            if (self.mj_model.actuator_gainprm[index, 0] != 1 or
                    any(self.mj_model.actuator_biasprm[index] != 0) or
                    self.mj_model.actuator_gear[index, 0] != 1):
                raise ValueError("leg_force_bc requires four unit-gain motor actuators, never leg position servos")
