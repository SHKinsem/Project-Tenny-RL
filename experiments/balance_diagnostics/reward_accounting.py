"""Pure accounting for existing native legwheel reward metrics, without simulation."""


def weighted_components(metrics, config, control_dt):
    """Convert preweight episode metric sums into contributions to native return.

    Metric names containing ``cost`` may be positive tracking scores. This adapter
    follows the recorded task's signed coefficients; PPO reward_scaling is separate.
    """
    mapping = {
        'tracking_lin_vel': 'vel_scale',
        'tracking_ang_vel': 'vel_scale',
        'tracking_wheels_sinposition': 'zero_keep_scale',
        'tracking_sinyaw': 'zero_keep_scale',
        'roll_cost': 'roll_scale',
        'tracking_height_jnt': 'height_scale',
        'pitch_cost': 'pitch_scale',
        'energy_cost': 'energy_scale',
        'action_rate': 'action_rate_scale',
        'pitch_velocity_penalty': 'pitch_velocity_penalty_scale',
    }
    ratio = float(control_dt) / float(config['reference_dt'])
    if ratio <= 0:
        raise ValueError('Positive control and reference intervals are required')
    result = {name: float(metrics['reward/' + name]) * float(config[field]) * ratio
              for name, field in mapping.items()}
    # Native termination metric already carries its enabled signed penalty.
    result['termination_penalty'] = float(metrics['reward/termination_penalty']) * ratio
    return result


def verify_accounting(metrics, config, control_dt, *, tolerance=1e-6):
    components = weighted_components(metrics, config, control_dt)
    error = sum(components.values()) - float(metrics['reward'])
    if abs(error) > tolerance:
        raise ValueError(f'Reward accounting mismatch: {error}')
    return dict(components=components, reconstructed_return=sum(components.values()),
                native_return=float(metrics['reward']), error=error)
