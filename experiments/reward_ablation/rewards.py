"""Independent, bounded additions to the unchanged standing reward.

Inputs are pitch error in radians and world-plane displacement from the reset
in metres. These terms have no positive survival constant. The posture arm adds
only direct pitch feedback; the pre-existing roll and wheel-angle terms remain
the caller's responsibility. No array or training framework is imported here.
"""
from dataclasses import dataclass
import math
from types import MappingProxyType


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(name + ' must be a finite number')
    return float(value)


@dataclass(frozen=True)
class RewardCoefficients:
    posture_weight: float = 0.0
    position_weight: float = 0.0
    pitch_scale_rad: float = math.radians(10.0)
    position_scale_m: float = 0.05

    def __post_init__(self):
        for name in ('posture_weight', 'position_weight', 'pitch_scale_rad', 'position_scale_m'):
            object.__setattr__(self, name, _number(getattr(self, name), name))
        if self.posture_weight < 0 or self.position_weight < 0:
            raise ValueError('Reward weights must be nonnegative')
        if not 0 < self.pitch_scale_rad <= math.pi or self.position_scale_m <= 0:
            raise ValueError('Scales must be positive; pitch scale cannot exceed pi radians')


ARMS = MappingProxyType({
    'baseline': RewardCoefficients(),
    'posture': RewardCoefficients(posture_weight=4.0),
    'position': RewardCoefficients(position_weight=10.0),
    'both': RewardCoefficients(posture_weight=4.0, position_weight=10.0),
})


def coefficients(arm_or_coeffs):
    if isinstance(arm_or_coeffs, RewardCoefficients):
        return arm_or_coeffs
    if isinstance(arm_or_coeffs, str) and arm_or_coeffs in ARMS:
        return ARMS[arm_or_coeffs]
    raise ValueError('Expected a RewardCoefficients value or one of: ' + ', '.join(ARMS))


def scalar_rewards(pitch_error_rad, displacement_xy, arm_or_coeffs):
    """Standard-library reference for a single state; displacements are deltas."""
    cfg = coefficients(arm_or_coeffs)
    pitch = _number(pitch_error_rad, 'pitch_error_rad')
    if not isinstance(displacement_xy, (list, tuple)) or len(displacement_xy) != 2:
        raise ValueError('displacement_xy must contain two metre-valued coordinates')
    xy = [_number(value, 'displacement_xy') for value in displacement_xy]
    # hypot avoids overflow in x*x + y*y for otherwise finite coordinates.
    pitch_ratio = pitch / cfg.pitch_scale_rad
    position_ratio = math.hypot(*xy) / cfg.position_scale_m
    return {
        'ablation_posture': cfg.posture_weight * math.expm1(-pitch_ratio * pitch_ratio),
        'ablation_position': cfg.position_weight * math.expm1(-position_ratio * position_ratio),
    }


def extra_rewards(pitch_error_rad, displacement_xy, arm_or_coeffs, xp):
    """Array-backend implementation, preserving JAX tracers and batch axes.

    ``xp`` provides asarray, sum and expm1 (for example jax.numpy). The last
    displacement axis must have width two; leading axes broadcast with pitch.
    State finiteness is checked by the environment, not converted to Python
    booleans inside a traced reward calculation.
    """
    cfg = coefficients(arm_or_coeffs)
    pitch = xp.asarray(pitch_error_rad)
    xy = xp.asarray(displacement_xy)
    if not xy.shape or xy.shape[-1] != 2:
        raise ValueError('displacement_xy needs a final axis of length two')
    pitch_sq = (pitch / cfg.pitch_scale_rad) ** 2
    position_sq = xp.sum((xy / cfg.position_scale_m) ** 2, axis=-1)
    return {
        'ablation_posture': cfg.posture_weight * xp.expm1(-pitch_sq),
        'ablation_position': cfg.position_weight * xp.expm1(-position_sq),
    }
