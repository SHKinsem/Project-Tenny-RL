"""Host-only cache/partition/cancellation tests; no simulator or GPU training."""
from contextlib import nullcontext
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/torque200"))
from wheel_teacher import pretrain as bc


def dataset():
    rng = np.random.default_rng(2)
    return dict(observations=rng.normal(size=(32, 26)).astype(np.float32),
                privileged_observations=rng.normal(size=(32, 55)).astype(np.float32),
                actions=rng.uniform(-.5, .5, size=(32, 4)).astype(np.float32),
                episode_ids=np.repeat(np.arange(8, dtype=np.int32), 4), report={"source": "test"})


def environment():
    noise = SimpleNamespace(level=1., scales=dict(gyro=.02, pitch=.02, roll=.02,
                                                 joint_pos=.02, joint_vel=1., linvel=.01))
    commands = SimpleNamespace(lin_vel_x=[-.2, .2], ang_vel_yaw=[-1.5, 1.5],
                               height_value=[-.5, 0], roll_value=[-.2, .2])
    return SimpleNamespace(_config=SimpleNamespace(noise_config=noise, cmd_config=commands), action_size=4)


class DataTests(unittest.TestCase):
    def test_episode_split_never_leaks_rows_from_the_same_episode(self):
        ids = dataset()["episode_ids"]
        fit, heldout, report = bc.split_episodes(ids, 0)
        self.assertFalse(set(ids[fit]) & set(ids[heldout]))
        self.assertEqual(len(fit) + len(heldout), len(ids))
        self.assertEqual(len(report['train_episodes']), 6)
        self.assertEqual(len(report['heldout_episodes']), 2)

    def test_invalid_labels_are_rejected_without_clipping(self):
        for value in (1.00001, -1.00001, np.nan, np.inf):
            data = dataset()
            data['actions'][0, 2] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                bc.validate_dataset(data, bc.PreparationConfig())

    def test_real_privileged_observations_and_episode_count_required(self):
        data = dataset()
        data['privileged_observations'] = data['observations']
        with self.assertRaises(ValueError):
            bc.validate_dataset(data, bc.PreparationConfig())
        data = dataset()
        data['episode_ids'][:] = 0
        with self.assertRaises(ValueError):
            bc.validate_dataset(data, bc.PreparationConfig())

    def test_noise_prior_matches_original_sensor_order_and_student_suffix(self):
        profile = bc.noise_profile(environment())
        prior = np.asarray(profile['actor_variance_prior'])
        self.assertAlmostEqual(prior[14], 1/3)
        self.assertAlmostEqual(prior[24], 1/3)
        self.assertAlmostEqual(prior[4], .02**2/3)
        self.assertTrue(np.all(prior[20:24] == 0))
        np.testing.assert_allclose(prior[:4], np.square([.4, 3., .5, .4]) / 12.)
        np.testing.assert_array_equal(profile['critic_variance_prior'][:26], prior)
        self.assertTrue(np.all(np.asarray(profile['critic_variance_prior'])[26:] == 0))

    def test_constant_commands_receive_configured_variance_prior(self):
        profile = bc.noise_profile(environment())
        # A zero-variance training column retains finite scale when PPO samples
        # its configured commands; the same prior enters summed_variance/count.
        count = 128
        prior = np.asarray(profile['actor_variance_prior'])[:4]
        summed_variance = np.zeros(4) + count * prior
        np.testing.assert_allclose(np.sqrt(summed_variance/count), np.array([.4, 3., .5, .4]) / np.sqrt(12.))
        self.assertGreater(float(np.sqrt(prior).min()), .1)

    def test_configuration_has_bounded_work(self):
        for kw in ({'episodes': 1}, {'updates': 99999}, {'budget_seconds': 301},
                   {'exploration_std': 0}, {'learning_rate': float('nan')}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                bc.PreparationConfig(**kw)

    def test_numpy_actor_uses_mean_head_not_exploration_head(self):
        layers = {}
        for i, shape in enumerate(((26, 64), (64, 64), (64, 8))):
            layers[f'hidden_{i}'] = dict(kernel=np.zeros(shape, np.float32), bias=np.zeros(shape[1], np.float32))
        layers['hidden_2']['bias'] = np.array([.1, -.2, .3, -.4, 9, 9, 9, 9], np.float32)
        params = (SimpleNamespace(mean={'state': np.zeros(26)}, std={'state': np.ones(26)}), {'params': layers}, {})
        np.testing.assert_allclose(bc.numpy_actor(params)(np.zeros(26)), np.tanh([.1, -.2, .3, -.4]), atol=1e-7)


class StageTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run = self.root/'run'; self.run.mkdir()
        self.args = SimpleNamespace(seed=0, restore=None, torque_root=self.root, run_dir=self.run, original_root=self.root)
        self.monitor = Mock()
        self.teacher = SimpleNamespace(collect_dataset=Mock(side_effect=lambda *a, **k: dataset()),
                                       evaluate_student=Mock(return_value={'student_episode_seconds': [.5],
                                                                           'stable_balance_validated': False}),
                                       CollectionCancelled=type('CollectionCancelled', (RuntimeError,), {}))
        self.params = object()
        self.fit = Mock(return_value=(self.params, b'finite-test-params',
                                     {'completed': True, 'parameter_tree_sha256': 'tree'}))
        for target, value in (('_identities', lambda *a: ({'data': 'same'}, {'bc': 'same'})),
                              ('_locked', lambda *a: nullcontext()), ('_fit', self.fit),
                              ('_load_params', lambda *a: self.params), ('parameter_digest', lambda *a: 'tree'),
                              ('numpy_actor', lambda *a: lambda obs: np.zeros(4))):
            handle = patch.object(bc, target, value)
            handle.start(); self.addCleanup(handle.stop)
        handle = patch.dict(sys.modules, {'wheel_teacher.teacher': self.teacher})
        handle.start(); self.addCleanup(handle.stop)

    def prepare(self):
        return bc.prepare(self.args, self.monitor, environment(), SimpleNamespace())

    def test_data_and_bc_success_are_reused_and_initial_hash_persisted(self):
        first = self.prepare()
        self.assertFalse(first['report']['data_cache_hit'])
        self.assertFalse(first['report']['bc_cache_hit'])
        second = self.prepare()
        self.assertTrue(second['report']['data_cache_hit'])
        self.assertTrue(second['report']['bc_cache_hit'])
        self.assertEqual(self.teacher.collect_dataset.call_count, 1)
        self.assertEqual(self.fit.call_count, 1)
        self.assertEqual(first['report']['initial_params_sha256'], second['report']['initial_params_sha256'])
        self.assertEqual(second['report']['parameter_tree_sha256'], 'tree')

    def test_failed_bc_preserves_data_and_does_not_publish_bc(self):
        self.fit.side_effect = RuntimeError('fit failed')
        with self.assertRaisesRegex(RuntimeError, 'fit failed'):
            self.prepare()
        self.assertEqual(len(list(self.root.rglob('data-success.json'))), 1)
        self.assertEqual(len(list(self.root.rglob('bc-success.json'))), 0)
        self.fit.side_effect = None
        result = self.prepare()
        self.assertTrue(result['report']['data_cache_hit'])
        self.assertFalse(result['report']['bc_cache_hit'])
        self.assertEqual(self.teacher.collect_dataset.call_count, 1)

    def test_corrupt_bc_bytes_refit_without_collecting_data(self):
        result = self.prepare()
        Path(result['report']['initial_params_path']).write_bytes(b'damaged')
        result = self.prepare()
        self.assertTrue(result['report']['data_cache_hit'])
        self.assertFalse(result['report']['bc_cache_hit'])
        self.assertEqual(self.teacher.collect_dataset.call_count, 1)
        self.assertEqual(self.fit.call_count, 2)

    def test_replay_cancellation_preserves_completed_bc_and_becomes_console_stop(self):
        class ConsoleStop(Exception):
            pass
        def cancel(*a, **kw):
            self.monitor.check_stop.side_effect = ConsoleStop('user_stop')
            raise self.teacher.CollectionCancelled('cancelled')
        self.teacher.evaluate_student.side_effect = cancel
        with self.assertRaises(ConsoleStop):
            self.prepare()
        self.assertEqual(len(list(self.root.rglob('bc-success.json'))), 1)
        self.monitor.check_stop.side_effect = None
        self.teacher.evaluate_student.side_effect = None
        self.assertTrue(self.prepare()['report']['bc_cache_hit'])
        self.assertEqual(self.fit.call_count, 1)

    def test_checkpoint_resume_skips_teacher_and_bc(self):
        self.args.restore = self.root/'saved-checkpoint'
        result = self.prepare()
        self.assertIsNone(result['params'])
        self.assertTrue(result['report']['skipped'])
        self.teacher.collect_dataset.assert_not_called()
        self.fit.assert_not_called()


if __name__ == '__main__':
    unittest.main()
