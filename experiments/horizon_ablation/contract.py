"""Only change the outer training episode limit of a frozen baseline manifest."""
from copy import deepcopy


def horizon_manifest(baseline, seconds):
    if type(seconds) is not int or seconds not in (6, 30):
        raise ValueError('This comparison supports 6 or 30 seconds')
    if baseline['ppo']['episode_length'] != 600 or baseline['env_config']['ctrl_dt'] != .01:
        raise ValueError('Require the frozen 100 Hz, 600-step baseline')
    if baseline['env_config']['reward_config']['termination_penalty_enable']:
        raise ValueError('An internal episode-length-dependent reward would confound this comparison')
    result = deepcopy(baseline)
    result['ppo']['episode_length'] = seconds * 100
    result['kind'] = 'training_horizon_ablation_v1'
    result['training_episode_seconds'] = seconds
    result['order_by_seed'] = {str(seed): ['baseline'] for seed in result['seeds']}
    assert_only_horizon_changed(baseline['ppo'], result['ppo'])
    return result


def assert_only_horizon_changed(short, long):
    if set(short) != set(long):
        raise ValueError('PPO fields differ beyond episode_length')
    changed = {key for key in short if short[key] != long[key]}
    if changed - {'episode_length'}:
        raise ValueError('PPO settings differ beyond episode_length: '+str(sorted(changed)))
    if short['episode_length'] != 600 or long['episode_length'] not in (600, 3000):
        raise ValueError('Expected 600 and 600/3000 training steps')
