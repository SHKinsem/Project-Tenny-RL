"""Host-only cache/partition/cancellation tests; no simulator or GPU training."""
from contextlib import contextmanager, nullcontext
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/torque200"))
from force_teacher import pretrain as bc


def dataset():
    rng = np.random.default_rng(2)
    return dict(observations=rng.normal(size=(32, 24)).astype(np.float32),
                privileged_observations=rng.normal(size=(32, 51)).astype(np.float32),
                actions=rng.uniform(-.5, .5, size=(32, 4)).astype(np.float32),
                episode_ids=np.repeat(np.arange(8, dtype=np.int32), 4), report={"source": "test"})


def environment():
    noise = SimpleNamespace(level=1., scales=dict(gyro=.02, pitch=.02, roll=.02,
                                                 joint_pos=.02, joint_vel=1., linvel=.01))
    commands = SimpleNamespace(lin_vel_x=[-.2, .2], ang_vel_yaw=[-1.5, 1.5],
                               height_value=[-.5, 0], roll_value=[-.2, .2])
    return SimpleNamespace(_config=SimpleNamespace(noise_config=noise, cmd_config=commands), action_size=4, force_scales=np.array([8.87, 8.87]))


class DataTests(unittest.TestCase):
    def test_experiment_values_drive_bounded_preparation_and_uniform_sampling(self):
        from force_teacher.options import normalize_options
        options=normalize_options(dict(bc_episodes=4,bc_updates=13,bc_batch_size=12,bc_learning_rate=.002,
            bc_exploration_std=.02,preparation_budget_seconds=45,seed=17,dagger_rounds=8,
            balanced_sampling=False,wheel_loss_weight=16,fixed_normalization=False,wheel_head='residual',wheel_residual_scale=.1))
        cfg=bc.preparation_config(options)
        self.assertEqual((cfg.episodes,cfg.updates,cfg.batch_size,cfg.seed,cfg.dagger_rounds),(4,13,12,17,8))
        self.assertEqual((cfg.learning_rate,cfg.exploration_std,cfg.budget_seconds,cfg.wheel_loss_weight),(.002,.02,45,16))
        self.assertFalse(cfg.fixed_normalization)
        weights=bc.training_sampling_probabilities(np.array([0]*100+[1]*2),np.arange(102),False)
        np.testing.assert_array_equal(weights,np.full(102,1/102))

    def test_bc_source_change_keeps_teacher_dataset_identity(self):
        source = {'revision': 'before'}
        contract = SimpleNamespace(
            original_path=lambda root: Path(root),
            metadata=lambda root, options=None: dict(mode='leg_force_bc', control_hz=100,
                                      source_fingerprint=source['revision'],
                                      force_source_fingerprint=source['revision']))
        with tempfile.TemporaryDirectory() as root, patch.dict(sys.modules, {'force_teacher.contract': contract}), \
                patch.object(bc.importlib.metadata, 'version', return_value='test-version'), \
                patch.object(bc, '_file_hash', side_effect=lambda path:
                             source['revision'] if Path(path).name == 'pretrain.py' else 'unchanged'):
            args = SimpleNamespace(original_root=root)
            config = SimpleNamespace(network_factory={}, normalize_observations=True)
            data_before, fit_before = bc._identities(args, bc.PreparationConfig(), config)
            source['revision'] = 'after'
            data_after, fit_after = bc._identities(args, bc.PreparationConfig(), config)
        self.assertEqual(data_before, data_after)
        self.assertNotEqual(fit_before, fit_after)
        self.assertEqual(data_after['contract'], {'mode': 'leg_force_bc', 'control_hz': 100})

    def test_episode_split_never_leaks_rows_from_the_same_episode(self):
        ids = dataset()["episode_ids"]
        fit, heldout, report = bc.split_episodes(ids, 0)
        self.assertFalse(set(ids[fit]) & set(ids[heldout]))
        self.assertEqual(len(fit) + len(heldout), len(ids))
        self.assertEqual(len(report['train_episodes']), 6)
        self.assertEqual(len(report['heldout_episodes']), 2)

    def test_startup_and_short_dagger_episodes_are_not_diluted(self):
        ids = np.array([0]*600 + [1]*30 + [2]*600)
        train = np.arange(630)  # Episode 2 is held out.
        weights = bc.training_sampling_probabilities(ids, train)
        self.assertAlmostEqual(weights[:50].sum(), .25)
        self.assertAlmostEqual(weights[50:600].sum(), .25)
        self.assertAlmostEqual(weights[600:].sum(), .5)
        self.assertEqual(len(weights), len(train))
        self.assertAlmostEqual(weights.sum(), 1.)

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

    def test_missing_unreachable_episodes_allowed_but_coverage_required(self):
        data = dataset()
        # Strict reachability filtering may remove complete teacher episodes.
        data['episode_ids'] = np.repeat(np.array([0, 2, 5, 7], np.int32), 8)
        arrays = bc.validate_dataset(data, bc.PreparationConfig())
        fit, heldout, _ = bc.split_episodes(arrays['episode_ids'])
        self.assertFalse(set(arrays['episode_ids'][fit]) & set(arrays['episode_ids'][heldout]))
        data['episode_ids'] = np.repeat(np.array([0, 2], np.int32), 16)
        with self.assertRaises(bc.PreparationDataInsufficient):
            bc.validate_dataset(data, bc.PreparationConfig())
        data = dataset()
        for name in ('observations', 'privileged_observations', 'actions', 'episode_ids'):
            data[name] = data[name][:31]
        with self.assertRaises(bc.PreparationDataInsufficient):
            bc.validate_dataset(data, bc.PreparationConfig())

    def test_wheel_mode_dataset_cannot_enter_force_bc(self):
        data = dataset()
        data['observations'] = np.zeros((32, 26), np.float32)
        data['privileged_observations'] = np.zeros((32, 55), np.float32)
        with self.assertRaises(ValueError):
            bc.validate_dataset(data, bc.PreparationConfig())

    def test_noise_prior_matches_original_sensor_order_and_student_suffix(self):
        profile = bc.noise_profile(environment())
        prior = np.asarray(profile['actor_variance_prior'])
        self.assertAlmostEqual(prior[14], 1/3)
        self.assertEqual(len(prior), 24)
        self.assertAlmostEqual(prior[4], .02**2/3)
        self.assertTrue(np.all(prior[20:24] == 0))
        np.testing.assert_allclose(prior[:4], np.square([.4, 3., .5, .4]) / 12.)
        np.testing.assert_array_equal(profile['critic_variance_prior'][:24], prior)
        self.assertTrue(np.all(np.asarray(profile['critic_variance_prior'])[24:] == 0))

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
        for i, shape in enumerate(((24, 64), (64, 64), (64, 8))):
            layers[f'hidden_{i}'] = dict(kernel=np.zeros(shape, np.float32), bias=np.zeros(shape[1], np.float32))
        layers['hidden_2']['bias'] = np.array([.1, -.2, .3, -.4, 9, 9, 9, 9], np.float32)
        params = (SimpleNamespace(mean={'state': np.zeros(24)}, std={'state': np.ones(24)}), {'params': layers}, {})
        np.testing.assert_allclose(bc.numpy_actor(params)(np.zeros(24)), np.tanh([.1, -.2, .3, -.4]), atol=1e-7)
        np.testing.assert_allclose(bc.numpy_actor(params, {'wheel_head':'residual'})(np.zeros(24)),
                                   np.tanh([.1, -.2, .015, -.02]), atol=1e-7)


class StageTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run = self.root/'run'; self.run.mkdir()
        self.args = SimpleNamespace(seed=0, restore=None, torque_root=self.root, run_dir=self.run, original_root=self.root)
        self.monitor = Mock()
        self.teacher = SimpleNamespace(collect_dataset=Mock(side_effect=lambda *a, **k: dataset()),
                                       evaluate_student=Mock(return_value={'student_episode_seconds': [6., 6., 6.],
                                           'gate_passed': True, 'episodes': [{'termination': {'reasons': []}}] * 3,
                                           'stable_balance_validated': False}),
                                       CollectionCancelled=type('CollectionCancelled', (RuntimeError,), {}))
        self.params = object()
        self.fit = Mock(return_value=(self.params, b'finite-test-params',
                                     {'completed': True, 'parameter_tree_sha256': 'tree'}))
        self.identity_builder = bc._identities
        for target, value in (('_identities', lambda *a: ({'data': 'same'}, {'bc': 'same'})),
                              ('_locked', lambda *a: nullcontext()), ('_fit', self.fit),
                              ('_load_params', lambda *a: self.params), ('parameter_digest', lambda *a: 'tree'),
                              ('numpy_actor', lambda *a: lambda obs: np.zeros(4))):
            handle = patch.object(bc, target, value)
            handle.start(); self.addCleanup(handle.stop)
        handle = patch.dict(sys.modules, {'force_teacher.teacher': self.teacher})
        handle.start(); self.addCleanup(handle.stop)

    def prepare(self):
        return bc.prepare(self.args, self.monitor, environment(),
                          SimpleNamespace(network_factory={}, normalize_observations=True))

    @contextmanager
    def real_stage_identities(self):
        def metadata(root, options):
            return dict(mode='leg_force_bc', source_fingerprint='same', force_source_fingerprint='same',
                experiment_config=options, experiment_config_sha256=bc._hash_bytes(bc._canonical(options)),
                training_profile='standing' if options['standing_only'] else 'commands',
                evaluation_policy=options['deterministic_eval'], normalization_policy=options['fixed_normalization'],
                policy_head=options['wheel_head'], wheel_residual_scale=options['wheel_residual_scale'],
                previous_torque_clip=1e-5)
        contract = SimpleNamespace(metadata=metadata, original_path=lambda root: Path(root))
        with patch.object(bc, '_identities', self.identity_builder), \
                patch.dict(sys.modules, {'force_teacher.contract': contract}), \
                patch.object(bc.importlib.metadata, 'version', return_value='test-version'):
            yield

    def failed_gate(self):
        return dict(gate_passed=False, student_episode_seconds=[.2] * 3,
                    episodes=[{'termination': {'reasons': ['fallen']}}] * 3)

    def dagger_receipts(self):
        records = [json.loads(path.read_text(encoding='utf8')) for path in
                   self.root.glob('artifacts/force-teacher-pretrain/data/*/data-success.json')]
        return [record['identity'] for record in records if record['identity']['dagger_round'] == 1]

    def test_real_bc_edit_reuses_collection_but_refits_and_replays(self):
        with self.real_stage_identities():
            self.prepare()
            self.args.experiment_config = {'bc_updates': 512, 'bc_learning_rate': .0005}
            changed = self.prepare()['report']
            self.assertTrue(changed['data_cache_hit'])
            self.assertFalse(changed['bc_cache_hit'])
            self.assertFalse(changed['replay_cache_hit'])
            self.assertTrue(self.prepare()['report']['bc_cache_hit'])
        self.assertEqual(self.teacher.collect_dataset.call_count, 1)
        self.assertEqual(self.fit.call_count, 2)
        self.assertEqual(self.teacher.evaluate_student.call_count, 2)

    def test_real_ppo_and_budget_edits_reuse_all_preparation_stages(self):
        with self.real_stage_identities():
            self.prepare()
            for overrides in ({'ppo_learning_rate': .0001}, {'entropy_cost': .002},
                              {'deterministic_eval': False}, {'preparation_budget_seconds': 240.},
                              {'bc_gate_enabled': False, 'dagger_rounds': 4}):
                with self.subTest(overrides=overrides):
                    self.args.experiment_config = overrides
                    report = self.prepare()['report']
                    self.assertTrue(report['data_cache_hit'])
                    self.assertTrue(report['bc_cache_hit'])
                    self.assertTrue(report['replay_cache_hit'])
        self.assertEqual(self.teacher.collect_dataset.call_count, 1)
        self.assertEqual(self.fit.call_count, 1)
        self.assertEqual(self.teacher.evaluate_student.call_count, 1)

    def test_real_dagger_recollects_when_parent_policy_changes(self):
        passed = self.teacher.evaluate_student.return_value
        self.teacher.evaluate_student.side_effect = [self.failed_gate(), passed, self.failed_gate(), passed]
        with self.real_stage_identities():
            self.prepare()
            self.args.experiment_config = {'bc_updates': 512}
            self.fit.return_value = (self.params, b'changed-parent-policy',
                                     {'completed': True, 'parameter_tree_sha256': 'tree'})
            report = self.prepare()['report']
            self.assertEqual(report['dagger_round'], 1)
            self.assertFalse(report['data_cache_hit'])
            self.assertTrue(self.prepare()['report']['data_cache_hit'])
        self.assertEqual(self.teacher.collect_dataset.call_count, 3)
        parents = self.dagger_receipts()
        self.assertEqual(len({item['parent_params_sha256'] for item in parents}), 2)
        self.assertEqual(len({item['parent_dataset_sha256'] for item in parents}), 1)

    def test_real_dagger_recollects_when_parent_data_changes_with_same_policy(self):
        passed = self.teacher.evaluate_student.return_value
        self.teacher.evaluate_student.side_effect = [self.failed_gate(), passed, self.failed_gate(), passed]
        with self.real_stage_identities():
            first = self.prepare()['report']
            data_root = self.root / 'artifacts/force-teacher-pretrain/data' / first['attempts'][0]['data_key']
            dataset_path = data_root / 'dataset.npz'
            with np.load(dataset_path, allow_pickle=False) as source:
                arrays = {key: source[key] for key in source.files}
            arrays['observations'][0, 0] += 1
            bc._save_dataset(dataset_path, arrays)
            receipt_path = data_root / 'data-success.json'
            receipt = json.loads(receipt_path.read_text(encoding='utf8'))
            receipt['dataset_sha256'] = bc._file_hash(dataset_path)
            bc._json(receipt_path, receipt)
            self.assertFalse(self.prepare()['report']['data_cache_hit'])
        self.assertEqual(self.teacher.collect_dataset.call_count, 3)
        parents = self.dagger_receipts()
        self.assertEqual(len({item['parent_params_sha256'] for item in parents}), 1)
        self.assertEqual(len({item['parent_dataset_sha256'] for item in parents}), 2)

    def test_real_collection_and_source_changes_still_invalidate_data(self):
        with self.real_stage_identities():
            self.prepare()
            self.args.experiment_config = {'initial_state_perturbation': False}
            self.assertFalse(self.prepare()['report']['data_cache_hit'])
            for filename in ('options.py', 'contract.py'):
                original_hash = bc._file_hash
                def changed_hash(path):
                    digest = original_hash(path)
                    return bc._hash_bytes((digest + 'changed').encode()) if Path(path).name == filename else digest
                with self.subTest(filename=filename), patch.object(bc, '_file_hash', side_effect=changed_hash):
                    self.assertFalse(self.prepare()['report']['data_cache_hit'])
        self.assertEqual(self.teacher.collect_dataset.call_count, 4)

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
        self.assertTrue(second['report']['replay_cache_hit'])
        self.assertEqual(self.teacher.evaluate_student.call_count, 1)

    def test_corrupt_replay_only_replays_and_keeps_fitted_parameters(self):
        self.prepare()
        report_path = next(self.root.glob('artifacts/force-teacher-pretrain/replay/*/report.json'))
        report_path.write_text('{"tampered":true}', encoding='utf8')
        result = self.prepare()
        self.assertTrue(result['report']['data_cache_hit'])
        self.assertTrue(result['report']['bc_cache_hit'])
        self.assertFalse(result['report']['replay_cache_hit'])
        self.assertEqual(self.teacher.collect_dataset.call_count, 1)
        self.assertEqual(self.fit.call_count, 1)
        self.assertEqual(self.teacher.evaluate_student.call_count, 2)

    def test_replay_setting_change_does_not_refit_or_recollect(self):
        self.prepare()
        self.args.pretrain_config = bc.PreparationConfig(replay_episodes=4)
        self.teacher.evaluate_student.return_value = dict(gate_passed=True,
            student_episode_seconds=[6.] * 4, episodes=[{'termination': {'reasons': []}}] * 4)
        result = self.prepare()
        self.assertTrue(result['report']['data_cache_hit'])
        self.assertTrue(result['report']['bc_cache_hit'])
        self.assertFalse(result['report']['replay_cache_hit'])
        self.assertEqual(self.fit.call_count, 1)
        self.assertEqual(self.teacher.evaluate_student.call_count, 2)

    def test_failed_physical_gate_never_returns_ppo_parameters(self):
        self.args.pretrain_config = bc.PreparationConfig(dagger_rounds=0)
        self.teacher.evaluate_student.return_value = dict(gate_passed=False,
            student_episode_seconds=[.3, .4, .2], episodes=[{'termination': {'reasons': ['fallen']}}] * 3)
        with self.assertRaises(bc.PreparationGateFailed):
            self.prepare()
        import json
        report = json.loads((self.run / 'pretraining-report.json').read_text(encoding='utf8'))
        self.assertFalse(report['completed'])
        self.assertFalse(report['bc_gate']['passed'])
        self.assertEqual(len(report['attempts']), 1)

    def test_disabled_gate_keeps_failed_replay_and_returns_parameters_without_dagger(self):
        self.args.experiment_config={'bc_gate_enabled':False,'seed':23,'bc_updates':11,'balanced_sampling':False}
        self.teacher.evaluate_student.return_value=dict(gate_passed=False,
            student_episode_seconds=[.2,.3,.4],episodes=[{'termination':{'reasons':['PITCH']}}]*3)
        result=self.prepare()
        self.assertTrue(result['report']['completed'])
        self.assertFalse(result['report']['bc_gate']['passed'])
        self.assertFalse(result['report']['bc_gate']['enforced'])
        self.assertEqual(self.teacher.evaluate_student.call_count,1)
        self.assertEqual(self.teacher.collect_dataset.call_count,1)
        self.assertEqual(self.fit.call_args.args[1].updates,11)
        self.assertEqual(self.fit.call_args.args[1].seed,23)
        self.assertFalse(self.fit.call_args.args[1].balanced_sampling)
        self.assertEqual(self.teacher.collect_dataset.call_args.kwargs['experiment_config']['seed'],23)

    def test_reference_dagger_rejected_before_any_collection(self):
        self.args.experiment_config={'teacher_source':'reference','dagger_rounds':1}
        with self.assertRaises(ValueError):self.prepare()
        self.teacher.collect_dataset.assert_not_called()

    def test_failed_gate_triggers_bounded_dagger_with_distinct_episode_ids(self):
        passed = self.teacher.evaluate_student.return_value
        self.teacher.evaluate_student.side_effect = [dict(gate_passed=False,
            student_episode_seconds=[.2] * 3, episodes=[{'termination': {'reasons': ['fallen']}}] * 3), passed]
        result = self.prepare()
        self.assertTrue(result['report']['bc_gate']['passed'])
        self.assertEqual(result['report']['dagger_round'], 1)
        self.assertEqual(self.teacher.collect_dataset.call_count, 2)
        self.assertEqual(self.fit.call_count, 2)
        second = self.fit.call_args.args[0]
        self.assertEqual(len(np.unique(second['episode_ids'])), 16)
        self.assertEqual(len(second['episode_ids']), 64)
        self.assertIsNotNone(self.teacher.collect_dataset.call_args.kwargs['policy_callable'])
        self.assertEqual(self.teacher.collect_dataset.call_args.kwargs['student_probability'], 1.)

    def test_insufficient_reachable_coverage_preserves_honest_failure_report(self):
        data = dataset()
        data['episode_ids'][:] = 0
        data['report'] = {'reachability': {'rejected_fraction': .9}}
        self.teacher.collect_dataset.side_effect = lambda *a, **kw: data
        with self.assertRaises(bc.PreparationDataInsufficient):
            self.prepare()
        import json
        report = json.loads((self.run / 'pretraining-report.json').read_text(encoding='utf8'))
        self.assertFalse(report['completed'])
        self.assertEqual(report['teacher'], data['report'])
        self.assertFalse(list(self.root.rglob('data-success.json')))
        self.assertFalse(list(self.root.rglob('bc-success.json')))
        self.fit.assert_not_called()

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


class BootstrapTests(unittest.TestCase):
    def test_bootstrap_network_tuples_match_json_receipt_lists(self):
        import json
        config = SimpleNamespace(network_factory=dict(policy_hidden_layer_sizes=(64,64),
            value_hidden_layer_sizes=(512,512,512,512), policy_obs_key='state',value_obs_key='privileged_state'))
        env = SimpleNamespace(_config=SimpleNamespace(to_dict=lambda: {'ctrl_dt':.01}))
        with patch.dict(sys.modules, {'force_teacher.contract': SimpleNamespace(metadata=lambda root, options=None: {'source':'same'})}), \
             patch.object(bc.importlib.metadata, 'version', return_value='fixture'):
            identity = bc._bootstrap_identity(SimpleNamespace(original_root=Path('.')),env,config)
            self.assertEqual(identity, json.loads(json.dumps(identity)))
            config.network_factory['policy_hidden_layer_sizes']=[64,64]
            config.network_factory['value_hidden_layer_sizes']=[512,512,512,512]
            self.assertEqual(identity,bc._bootstrap_identity(SimpleNamespace(original_root=Path('.')),env,config))

    def test_only_complete_current_hash_bound_receipts_are_reusable(self):
        import json
        from force_teacher.options import normalize_options
        identity={'source':'current','experiment_config':normalize_options()}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for name in bc.BOOTSTRAP_FILES:
                (directory/name).write_bytes(b'fixture')
            bc._save_dataset(directory/'dataset.npz', {key:value for key,value in dataset().items() if key!='report'})
            record = dict(schema=bc.BOOTSTRAP_SCHEMA,completed=True,identity=identity,
                bc_gate={'passed':True},files={name:bc._file_hash(directory/name) for name in bc.BOOTSTRAP_FILES},
                calibration={},parameter_tree_sha256='tree')
            def save():
                (directory/'verified.json').write_text(json.dumps(record),encoding='utf8')
            save()
            with patch.object(bc,'_load_params',return_value='params'), patch.object(bc,'parameter_digest',return_value='tree'):
                self.assertEqual(bc._read_bootstrap(directory,identity,None)[0],'params')
                with self.assertRaises(ValueError):
                    bc._read_bootstrap(directory,{'source':'changed'},None)
                removed=record['files'].pop('validation.json');save()
                with self.assertRaises(ValueError):
                    bc._read_bootstrap(directory,identity,None)
                record['files']['validation.json']=removed;save()
                (directory/'params.msgpack').write_bytes(b'corrupt')
                with self.assertRaises(ValueError):
                    bc._read_bootstrap(directory,identity,None)


if __name__ == '__main__':
    unittest.main()
