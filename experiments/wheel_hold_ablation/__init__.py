"""Reward-only wheel-position comparison from the identical mature initializer."""
from types import MappingProxyType


WHEEL_HOLD = MappingProxyType(dict(original_weight=10., training_weight=1.,
    metric='reward/tracking_wheels_sinposition'))
