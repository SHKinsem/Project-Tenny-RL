"""Pure interval-weighted actuator occupancy; requests and outputs stay distinct."""
import math


def summarize_actuator_intervals(times, requested_nm, command_nm, actual_nm,
                                normalized_action, limits_nm, *, start_s=0., end_s=None):
    """Each row is weighted by [times[i], times[i+1]).

    ``requested_nm`` is the finite torque after the safety guard but before
    torque clipping. ``normalized_action`` is the policy output before its
    [-1, 1] clip. Windows use exact overlap duration, without inventing states.
    Near-limit output is not evidence that a request was clipped.
    The caller must identify whether each row is a held value, a physical
    substep measurement, or a representative final-substep sample. Weighting
    final-substep samples does not recover unobserved intermediate values.
    """
    times, limits = list(times), list(limits_nm)
    if len(times) < 2 or times[0] != 0 or not all(math.isfinite(v) for v in times):
        raise ValueError('Need finite interval boundaries beginning at zero')
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError('Interval boundaries must strictly increase')
    if not limits or any(not math.isfinite(v) or v <= 0 for v in limits):
        raise ValueError('Need positive finite actuator limits')
    count, width = len(times)-1, len(limits)
    matrices = [list(value) for value in (requested_nm, command_nm, actual_nm, normalized_action)]
    for matrix in matrices:
        if len(matrix) != count or any(len(row) != width for row in matrix):
            raise ValueError('All interval matrices must have the same actuator shape')
        if any(not math.isfinite(v) for row in matrix for v in row):
            raise ValueError('Actuator values must be finite')
    if end_s is None:
        end_s = times[-1]
    if not (math.isfinite(start_s) and math.isfinite(end_s) and 0 <= start_s <= end_s <= times[-1]+1e-9):
        raise ValueError('Window must lie inside the recorded intervals')
    weights = [max(0., min(right, end_s)-max(left, start_s))
               for left, right in zip(times, times[1:])]
    duration = math.fsum(weights)

    def occupancy(mask):
        total = math.fsum(weight for weight, flag in zip(weights, mask) if flag)
        run = longest = 0.
        for weight, flag in zip(weights, mask):
            run = run+weight if weight > 0 and flag else 0.
            longest = max(longest, run)
        return dict(seconds=total, fraction=total/duration if duration else None,
                    longest_continuous_seconds=longest)

    rows = []
    for axis, limit in enumerate(limits):
        raw, sent, actual, action = ([row[axis] for row in matrix] for matrix in matrices)
        active = [i for i, weight in enumerate(weights) if weight > 0]
        peak = lambda values: max((abs(values[i]) for i in active), default=None)
        rms = math.sqrt(math.fsum(weights[i]*actual[i]**2 for i in active)/duration) if duration else None
        rows.append(dict(actuator_index=axis, limit_nm=limit, max_abs_requested_nm=peak(raw),
            max_abs_command_nm=peak(sent), max_abs_actual_nm=peak(actual), actual_rms_nm=rms,
            max_abs_normalized_action=peak(action),
            request_clipped=occupancy([abs(v)>limit+1e-12 for v in raw]),
            command_changed_by_clip=occupancy([abs(a-b)>1e-12 for a,b in zip(raw,sent)]),
            actual_near_limit={str(level):occupancy([abs(v)>=level*limit for v in actual])
                               for level in (.9,.95,.99)},
            action_near_bound={str(level):occupancy([abs(v)>=level for v in action])
                               for level in (.9,.95,.99)},
            action_outside_bound=occupancy([abs(v)>1.+1e-12 for v in action])))
    return dict(schema_version=1, start_s=start_s, end_s=end_s, observed_seconds=duration,
                sampling='Interval-weighted input samples; overlap-clipped windows. Physical coverage is supplied by the caller.',
                actuators=rows)
