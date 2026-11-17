import csv
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.reward_ablation.report import ARMS, STATES, paired_summary, write_ablation_report


def bundle_fixture():
    cases = []
    for identity in ['initial']+[f'{arm}-seed-{seed}' for arm in ARMS for seed in (0, 1, 2)]:
        if identity == 'initial':
            effective = 2.
        else:
            arm, _, seed = identity.partition('-seed-')
            effective = {'baseline': 1., 'posture': 1.5, 'position': .5, 'both': 2.5}[arm]+.1*int(seed)
        for state in STATES:
            trace = {'t_s': [0., 1., 2., 4.], 'done': [False, False, False, True],
                'base_position_m': [[0., 0., .1], [0., 0., .1], [.01, 0., .1], [.02, 0., .1]],
                'base_velocity_m_s': [[0., 0., 0.]]*4, 'pitch_rad': [0.]*4,
                'metadata': {'reference_pitch_rad': 0.},
                'termination': {'terminated': True, 'truncated': False, 'reason': 'fixture'}}
            cases.append({'policy_id': identity, 'label': '<script>label</script>', 'initial_state_id': state,
                'trace': trace, 'analysis': {'windows': {'full_episode': {'observed_seconds': 4.,
                    'max_horizontal_displacement_m': .02, 'pitch_error_rms_deg': 0., 'horizontal_speed_rms_m_s': 0.}}},
                'standing': {'in_band_seconds_after_settling': effective,
                    'longest_in_band_seconds_after_settling': effective, 'survival_seconds': 4., 'reached_horizon': False}})
    statuses = [{'phase': 'completed', 'steps': 2097152, 'arm': arm, 'seed': seed,
                 'initial_array_digest': 'same-initializer'} for arm in ARMS for seed in (0, 1, 2)]
    return {'schema_version': 1, 'kind': 'balance_diagnostics', 'horizon_s': 30., 'cases': cases,
            'experiment_manifest': {'seeds': [0, 1, 2], 'steps_per_run': 2097152}, 'run_statuses': statuses}


class RewardAblationReportTests(unittest.TestCase):
    def test_pairing_uses_training_seeds_and_keeps_bc_reference(self):
        summary, rows, _ = paired_summary(bundle_fixture())
        self.assertEqual(len(rows), 65)
        self.assertEqual(len(summary['arms']['posture']['paired_results']), 3)
        self.assertAlmostEqual(summary['arms']['posture']['paired_delta_in_band_seconds']['median'], .5)
        self.assertAlmostEqual(summary['initial']['primary']['mean_in_band_seconds'], 2.)
        self.assertIn('减缓退化', summary['arms']['posture']['interpretation'])
        self.assertIn('同时高于', summary['arms']['both']['interpretation'])

    def test_missing_or_duplicate_condition_fails(self):
        bundle = bundle_fixture()
        bundle['cases'].pop()
        with self.assertRaises(ValueError):
            paired_summary(bundle)
        bundle = bundle_fixture()
        bundle['cases'].append(deepcopy(bundle['cases'][0]))
        with self.assertRaises(ValueError):
            paired_summary(bundle)

    def test_incomplete_or_unequal_step_matrix_fails(self):
        for change in ({'phase': 'failed'}, {'steps': 1048576}, {'initial_array_digest': 'different'}):
            bundle = bundle_fixture()
            bundle['run_statuses'][0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                paired_summary(bundle)

    def test_full_cases_without_statuses_are_labelled_unverified(self):
        bundle = bundle_fixture()
        del bundle['run_statuses']
        summary, _, _ = paired_summary(bundle)
        self.assertFalse(summary['matrix']['statuses_verified'])

    def test_first_done_rule_and_standing_duration_are_validated(self):
        bundle = bundle_fixture()
        bundle['cases'][0]['trace']['done'][1] = True
        with self.assertRaises(ValueError):
            paired_summary(bundle)
        bundle = bundle_fixture()
        bundle['cases'][0]['standing']['in_band_seconds_after_settling'] = 4.
        with self.assertRaises(ValueError):
            paired_summary(bundle)

    def test_offline_report_escapes_labels_and_exports_complete_rows(self):
        with tempfile.TemporaryDirectory() as directory, patch('experiments.reward_ablation.report._plot'):
            paths = write_ablation_report(bundle_fixture(), directory)
            html = Path(paths['report_html']).read_text(encoding='utf-8')
            self.assertNotIn('<script>', html)
            self.assertIn('&lt;script&gt;', html)
            self.assertIn('comparison-detail.png', html)
            self.assertIn('早期前10秒（early first 10 s）', html)
            self.assertIn('href="results.json"', html)
            self.assertNotIn('https://', html)
            with Path(paths['summary_csv']).open(encoding='utf-8', newline='') as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 65)
            summary = json.loads(Path(paths['paired_summary_json']).read_text(encoding='utf-8'))
            self.assertTrue(summary['matrix']['statuses_verified'])
