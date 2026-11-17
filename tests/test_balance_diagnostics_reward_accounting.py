import unittest

from experiments.balance_diagnostics.reward_accounting import verify_accounting


class RewardAccountingTests(unittest.TestCase):
    def test_tracking_scores_and_signed_costs_reconstruct_return(self):
        cfg = dict(vel_scale=2, zero_keep_scale=10, roll_scale=4, height_scale=2,
                   pitch_scale=0, energy_scale=-1, action_rate_scale=-.5,
                   pitch_velocity_penalty_scale=-.5, reference_dt=.01)
        metrics = {'reward/' + name: value for name, value in dict(
            tracking_lin_vel=1, tracking_ang_vel=2, tracking_wheels_sinposition=3,
            tracking_sinyaw=4, roll_cost=5, tracking_height_jnt=6, pitch_cost=1000,
            energy_cost=7, action_rate=8, pitch_velocity_penalty=9,
            termination_penalty=0).items()}
        metrics['reward'] = 92.5
        result = verify_accounting(metrics, cfg, .01)
        self.assertEqual(result['components']['pitch_cost'], 0)
        self.assertEqual(result['components']['roll_cost'], 20)
        self.assertEqual(result['components']['energy_cost'], -7)
        metrics['reward'] = 93
        with self.assertRaisesRegex(ValueError, 'accounting mismatch'):
            verify_accounting(metrics, cfg, .01)


if __name__ == '__main__':
    unittest.main()
