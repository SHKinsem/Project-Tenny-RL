from copy import deepcopy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.horizon_ablation.report import CONDITIONS, STATES, paired_summary, write_horizon_report


def fixture():
    cases = []
    values = {'initial': 2.7, 'h6': [1., 2., 2.5], 'h30': [1.2, 1.8, 2.6]}
    for identity in ['initial']+[f'{condition}-seed-{seed}' for condition in CONDITIONS for seed in (0, 1, 2)]:
        condition, _, seed = identity.partition('-seed-')
        effective = values['initial'] if identity == 'initial' else values[condition][int(seed)]
        for state in STATES:
            trace = dict(t_s=[0., 1., 2., 4.], done=[False, False, False, True],
                base_position_m=[[0., 0., .1], [0., 0., .1], [.01, 0., .1], [.02, 0., .1]],
                base_velocity_m_s=[[0., 0., 0.]]*4, pitch_rad=[0.]*4,
                metadata={'reference_pitch_rad': 0.},
                termination={'terminated': True, 'truncated': False, 'reason': 'fixture'})
            cases.append(dict(policy_id=identity, label='<script>label</script>', initial_state_id=state,
                trace=trace, analysis={'windows': {'full_episode': {'observed_seconds': 4.,
                    'max_horizontal_displacement_m': .02, 'pitch_error_rms_deg': 0., 'horizontal_speed_rms_m_s': 0.}}},
                standing=dict(in_band_seconds_after_settling=effective,
                    longest_in_band_seconds_after_settling=effective, survival_seconds=4., reached_horizon=False)))
    statuses = [dict(condition=condition, seed=seed, phase='completed', steps=2097152,
                    initial_array_digest='same-start', reused=condition == 'h6')
                for condition in CONDITIONS for seed in (0, 1, 2)]
    return dict(schema_version=1, kind='balance_diagnostics', horizon_s=30., cases=cases,
                experiment_manifest={'seeds': [0, 1, 2], 'steps_per_run': 2097152}, run_statuses=statuses)


class HorizonReportTests(unittest.TestCase):
    def test_pairs_seeds_before_computing_median_and_keeps_bc(self):
        summary, rows, _ = paired_summary(fixture())
        self.assertEqual(len(rows), 35)
        self.assertAlmostEqual(summary['paired_delta_in_band_seconds']['median'], .1)
        self.assertAlmostEqual(summary['conditions']['h30']['primary_in_band_seconds']['median'], 1.8)
        self.assertAlmostEqual(summary['conditions']['h6']['primary_in_band_seconds']['median'], 2.)
        self.assertAlmostEqual(summary['initial']['primary']['mean_in_band_seconds'], 2.7)
        self.assertIn('减缓退化', summary['interpretation'])
        self.assertEqual(summary['improved_seed_counts']['cumulative'], 2)

    def test_missing_duplicate_and_wrong_identity_are_rejected(self):
        for mutation in ('missing', 'duplicate', 'wrong'):
            bundle = fixture()
            if mutation == 'missing':
                bundle['cases'].pop()
            elif mutation == 'duplicate':
                bundle['cases'].append(deepcopy(bundle['cases'][0]))
            else:
                bundle['cases'][0]['policy_id'] = 'baseline-seed-0'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                paired_summary(bundle)

    def test_statuses_reject_budget_initializer_and_reuse_mismatch(self):
        for change in ({'phase': 'failed'}, {'steps': 1}, {'initial_array_digest': 'other'}, {'reused': False}):
            bundle = fixture()
            bundle['run_statuses'][0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                paired_summary(bundle)

    def test_early_failure_with_small_displacement_does_not_rank_better(self):
        bundle = fixture()
        for case in bundle['cases']:
            if case['policy_id'] != 'h30-seed-1':
                continue
            case['trace']['t_s'] = [0., .5, 1., 2.]
            case['analysis']['windows']['full_episode'].update(observed_seconds=2., max_horizontal_displacement_m=.001)
            case['standing'].update(in_band_seconds_after_settling=.5,
                longest_in_band_seconds_after_settling=.5, survival_seconds=2.)
        summary, _, _ = paired_summary(bundle)
        self.assertAlmostEqual(summary['paired_results'][1]['delta_in_band_seconds'], -1.5)

    def test_trace_after_first_done_and_false_horizon_success_fail(self):
        bundle = fixture()
        bundle['cases'][0]['trace']['done'][1] = True
        with self.assertRaises(ValueError):
            paired_summary(bundle)
        bundle = fixture()
        bundle['cases'][0]['standing']['reached_horizon'] = True
        with self.assertRaises(ValueError):
            paired_summary(bundle)

    def test_no_statuses_are_not_claimed_verified(self):
        bundle = fixture()
        del bundle['run_statuses']
        summary, _, _ = paired_summary(bundle)
        self.assertFalse(summary['matrix']['statuses_verified'])

    def test_offline_report_escapes_labels_and_exports_all_cases(self):
        with tempfile.TemporaryDirectory() as directory, patch('experiments.horizon_ablation.report._plot'):
            paths = write_horizon_report(fixture(), directory)
            content = Path(paths['report_html']).read_text(encoding='utf-8')
            self.assertNotIn('<script>', content)
            self.assertIn('&lt;script&gt;', content)
            self.assertIn('复用上一轮', content)
            self.assertIn('不能据此证明或排除控制频率瓶颈', content)
            self.assertIn('comparison-detail.png', content)
            self.assertNotIn('https://', content)
            with Path(paths['summary_csv']).open(encoding='utf-8', newline='') as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 35)
            summary = json.loads(Path(paths['paired_summary_json']).read_text(encoding='utf-8'))
            self.assertTrue(summary['matrix']['statuses_verified'])
