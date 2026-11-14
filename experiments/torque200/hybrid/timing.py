"""Dependency-free timing contract for the opt-in hybrid family only.

100/200 Hz are accepted experiment configurations, not stability certificates.
Legacy reference and torque200 entry points are deliberately not changed.
"""
from dataclasses import dataclass
import math

SUPPORTED_CONTROL_HZ = (100, 200)
DEFAULT_CONTROL_HZ = 200
PHYSICS_HZ = 400
TIMING_SCHEMA = "hybrid-timing-v1"
EPISODE_SECONDS = 6.0
COMMAND_PERIOD_SECONDS = 3.0
COMMAND_START_SECONDS = 0.2
REWARD_REFERENCE_DT = 0.010
DISCOUNT_HALF_LIFE_SECONDS = 0.5
BASELINE_CTRL_DT = 0.005
BASELINE_GAE_LAMBDA = 0.95


@dataclass(frozen=True)
class Timing:
    control_hz: int = DEFAULT_CONTROL_HZ

    def __post_init__(self):
        if type(self.control_hz) is not int or self.control_hz not in SUPPORTED_CONTROL_HZ:
            raise ValueError(f"control_hz must be one of {SUPPORTED_CONTROL_HZ}; other rates are not enabled")
        if PHYSICS_HZ % self.control_hz:
            raise ValueError("Control period must contain an integer number of 400 Hz physics steps")

    @property
    def ctrl_dt(self):
        return 1.0 / self.control_hz

    @property
    def sim_dt(self):
        return 1.0 / PHYSICS_HZ

    @property
    def n_substeps(self):
        return PHYSICS_HZ // self.control_hz

    def steps(self, seconds):
        """No silent rounding of a physical duration to a policy-step count."""
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Duration must be finite and nonnegative")
        count = seconds * self.control_hz
        if not math.isclose(count, round(count), rel_tol=0, abs_tol=1e-9):
            raise ValueError("Duration must be an exact multiple of the control period")
        return int(round(count))

    def contract(self, action_delay_ms=0.0):
        if isinstance(action_delay_ms, bool) or not isinstance(action_delay_ms, (int, float)):
            raise ValueError("action_delay_ms must be a number")
        delay_steps = self.steps(action_delay_ms / 1000.0)
        return dict(timing_schema=TIMING_SCHEMA, control_hz=self.control_hz,
                    physics_hz=PHYSICS_HZ, ctrl_dt=self.ctrl_dt, sim_dt=self.sim_dt,
                    n_substeps=self.n_substeps, episode_seconds=EPISODE_SECONDS,
                    episode_length=self.steps(EPISODE_SECONDS),
                    command_period_seconds=COMMAND_PERIOD_SECONDS,
                    command_period_steps=self.steps(COMMAND_PERIOD_SECONDS),
                    command_start_seconds=COMMAND_START_SECONDS,
                    command_start_steps=self.steps(COMMAND_START_SECONDS),
                    action_repeat=1, action_delay_ms=float(action_delay_ms),
                    action_delay_steps=delay_steps, reward_reference_dt=REWARD_REFERENCE_DT,
                    discount_half_life_seconds=DISCOUNT_HALF_LIFE_SECONDS,
                    discounting=0.5 ** (self.ctrl_dt / DISCOUNT_HALF_LIFE_SECONDS),
                    gae_lambda=BASELINE_GAE_LAMBDA ** (self.ctrl_dt / BASELINE_CTRL_DT))


def configure_timing(cfg, control_hz=DEFAULT_CONTROL_HZ, action_delay_ms=0.0):
    """Recompute derived fields together, without modifying the legacy factory."""
    timing = Timing(control_hz)
    values = timing.contract(action_delay_ms)
    cfg.control_hz = control_hz
    cfg.ctrl_dt, cfg.sim_dt = timing.ctrl_dt, timing.sim_dt
    cfg.episode_time, cfg.episode_length = EPISODE_SECONDS, values["episode_length"]
    cfg.cmd_start_delay_s = COMMAND_START_SECONDS
    # The inherited step resets when counter > threshold. Subtract one so the
    # repeat cycle is exactly 3 s, rather than 3 s plus one policy step.
    cfg.cmd_config.command_change_interval = values["command_period_steps"] - 1
    cfg.action_repeat = 1
    cfg.action_delay_ms = float(action_delay_ms)
    cfg.action_delay_steps = values["action_delay_steps"]
    cfg.reward_config.reference_dt = REWARD_REFERENCE_DT
    return cfg


def validate_timing(cfg):
    """Reject partial ctrl_dt edits and inconsistent persisted configuration."""
    timing = Timing(cfg.control_hz)
    values = timing.contract(cfg.action_delay_ms)
    pairs = ((cfg.ctrl_dt, timing.ctrl_dt), (cfg.sim_dt, timing.sim_dt),
             (cfg.episode_time, EPISODE_SECONDS), (cfg.episode_length, values["episode_length"]),
             (cfg.cmd_start_delay_s, COMMAND_START_SECONDS),
             (cfg.cmd_config.command_change_interval, values["command_period_steps"] - 1),
             (cfg.action_repeat, 1), (cfg.action_delay_steps, values["action_delay_steps"]),
             (cfg.reward_config.reference_dt, REWARD_REFERENCE_DT))
    if any(actual != expected for actual, expected in pairs):
        raise ValueError("Inconsistent hybrid timing configuration; use hybrid_config(control_hz=...)")
    return timing
