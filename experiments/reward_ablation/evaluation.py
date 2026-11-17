"""Common physical outcome measurements for reward-ablation policies.

These scores do not use a policy's training reward. They count only recorded
first-episode intervals and never award standing time after failure or reset.
"""
import math

from experiments.balance_diagnostics.metrics import analyze_trace


def _seconds(value, label, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(label + ' must be finite seconds')
    if value < 0 or (positive and value == 0):
        raise ValueError(label + (' must be positive' if positive else ' must be nonnegative'))
    return float(value)


def _interval_time(times, in_band, start, end):
    total = longest = continuous = coverage = 0.0
    for index, (left, right) in enumerate(zip(times, times[1:])):
        # Do not invent a state at the settling/horizon boundary.
        if left < start or right > end:
            continuous = 0.0
            continue
        duration = right - left
        coverage += duration
        if in_band[index] and in_band[index + 1]:
            total += duration
            continuous += duration
            longest = max(longest, continuous)
        else:
            continuous = 0.0
    return total, longest, coverage


def standing_metrics(trace, *, settling_seconds=1.0, horizon_seconds=30.0):
    """Measure joint pitch/position/speed bands using actual sample endpoints.

    Position is the world's XY displacement from this trace's reset. Pitch is
    relative to its saved equilibrium reference. A task failure exactly at the
    horizon does not count as reaching the horizon; a time-limit truncation does.
    """
    analysis = analyze_trace(trace, windows_s=())
    settling = _seconds(settling_seconds, 'settling_seconds')
    horizon = _seconds(horizon_seconds, 'horizon_seconds', positive=True)
    if settling > horizon:
        raise ValueError('settling_seconds must not exceed horizon_seconds')
    times = trace['t_s']
    origin = trace['base_position_m'][0]
    reference = trace['metadata']['reference_pitch_rad']
    pitch_limit = math.radians(10.0)
    in_band = [
        abs(pitch - reference) <= pitch_limit
        and math.hypot(position[0] - origin[0], position[1] - origin[1]) <= 0.05
        and math.hypot(velocity[0], velocity[1]) <= 0.1
        for pitch, position, velocity in zip(
            trace['pitch_rad'], trace['base_position_m'], trace['base_velocity_m_s'])
    ]
    total, longest, coverage = _interval_time(times, in_band, settling, horizon)
    all_total, all_longest, _ = _interval_time(times, in_band, 0.0, horizon)
    episode_seconds = float(times[-1])
    at_horizon = math.isclose(episode_seconds, horizon, rel_tol=0, abs_tol=1e-9)
    reached = ((episode_seconds > horizon and not at_horizon)
               or (at_horizon and analysis['termination']['truncated']))
    return {
        'schema_version': 1,
        'in_band_seconds_after_settling': total,
        'longest_in_band_seconds_after_settling': longest,
        'in_band_seconds_including_settling': all_total,
        'longest_in_band_seconds_including_settling': all_longest,
        'observed_interval_seconds_after_settling': coverage,
        'survival_seconds': min(episode_seconds, horizon),
        'observed_episode_seconds': episode_seconds,
        'reached_horizon': bool(reached),
        'settling_seconds': settling,
        'horizon_seconds': horizon,
        'band': {
            'pitch_error_max_deg': 10.0,
            'horizontal_displacement_max_m': 0.05,
            'horizontal_speed_max_m_s': 0.1,
            'combination': 'All three inclusive limits must hold at both interval endpoints.',
            'position_reference': 'The supplied trace reset world XY position.',
            'pitch_reference': 'The supplied trace metadata.reference_pitch_rad.',
        },
        'sampling': (
            'Sum complete recorded intervals with both endpoints in band, left endpoint at/after settling, '
            'and right endpoint at/before horizon. No interpolation or time after first termination. '
            'Intervals straddling a window boundary are omitted.'),
        'scope': (
            'Common physical criteria for a complete first episode, independent of the training reward. '
            'These declared bands are experiment criteria, not established hardware acceptance limits. '
            'Compare standing duration together with survival and horizon status; an early death with '
            'little displacement is not evidence of better balance.'),
    }
