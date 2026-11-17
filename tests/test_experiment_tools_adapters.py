import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from experiments.experiment_tools import load_study, select_snapshot, summarize
from experiments.experiment_tools.adapters import FORMATS, METRICS, PARTITIONS, WINDOWS
from experiments.experiment_tools.__main__ import main
from experiments.experiment_tools.io import load_json, save_json


def legacy(kind):
    baseline, treatment = FORMATS[kind]
    data = dict(schema_version=1, kind=kind, conditions={}, case_rows=[], progress_curves={})
    for condition in (baseline, treatment):
        policy = condition + '-seed-0'
        data['conditions'][condition] = dict(seed_results=[dict(seed=0, policy_id=policy)])
        windows = {window: dict(complete_set=True, **{m: 0.01 for m in METRICS}) for window in WINDOWS}
        data['progress_curves'][policy] = [dict(steps=10, primary=dict(windows=windows))]
    for policy in ['initial', baseline+'-seed-0', treatment+'-seed-0']:
        for case in sum(PARTITIONS.values(), []):
            row = dict(policy_id=policy, condition=policy.rsplit('-seed-', 1)[0], seed=0,
                       initial_state_id=case, reached_30s=True, reached_120s=True)
            row.update({w+'_'+m: 0.01 for w in WINDOWS for m in METRICS})
            data['case_rows'].append(row)
    return data


class AdapterTests(unittest.TestCase):
    def test_two_legacy_formats_and_early_termination(self):
        for kind in FORMATS:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                data = legacy(kind)
                baseline = FORMATS[kind][0]
                row = next(r for r in data['case_rows'] if r['condition'] == baseline)
                row['reached_120s'] = False
                path = Path(tmp)/'summary.json'
                save_json(path, data)
                summary = summarize(load_study(path))
                metrics = summary['condition_summaries'][baseline]['primary']['metrics']
                self.assertIsNone(metrics['first_120s_rms_xy_displacement_m']['median'])
                self.assertEqual(metrics['first_30s_rms_xy_displacement_m']['count'], 1)
                self.assertEqual(metrics['first_120s_within_1cm_seconds_after_settling']['count'], 1)

    def test_recorded_full_parameter_evidence_is_bound_to_descriptor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_json(root/'comparison/summary.json', legacy('wheel_hold_ablation_summary'))
            selected = root/'selected-snapshot'
            save_json(selected/'descriptor.json', {'kind': 'test_policy'})
            digest = hashlib.sha256((selected/'descriptor.json').read_bytes()).hexdigest()
            save_json(selected/'summary.json', dict(selected=dict(seed=0, steps=10), descriptor_sha256=digest,
                checkpoint='/remote/parameters', checkpoint_sha256='a'*64, parameter_digest='parameter-tree'))
            save_json(selected/'verification.json', dict(phase='verified', all_parameter_leaves_finite=True,
                checkpoint_to_saved_policy_action_max_abs_error=0, checkpoint_sha256='a'*64,
                complete_parameter_digest='parameter-tree'))
            study = load_study(root)
            result = select_snapshot(study, condition='weak_wheel', metric='first_30s_rms_xy_displacement_m',
                                     require_full_parameters=True)
            self.assertEqual(result['selected']['artifacts']['capability'], 'full_parameters_verified_at_run')
            self.assertTrue(result['selected']['artifacts']['live_validation_required'])
            (selected/'descriptor.json').write_text('{}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'differs'):
                load_study(root)

    def test_descriptor_only_is_never_a_resumable_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_json(root/'comparison/summary.json', legacy('mature_balance_finetune_summary'))
            selected = root/'selected-stage'
            save_json(selected/'descriptor.json', {'kind': 'test_policy'})
            digest = hashlib.sha256((selected/'descriptor.json').read_bytes()).hexdigest()
            save_json(selected/'summary.json', dict(selected=dict(policy_id='high_lr-seed-0', steps=10)))
            save_json(selected/'verification.json', dict(phase='verified', descriptor_sha256=digest))
            study = load_study(root)
            result = select_snapshot(study, condition='high_lr', metric='first_30s_rms_xy_displacement_m')
            self.assertEqual(result['selected']['artifacts']['capability'], 'descriptor_only')
            with self.assertRaisesRegex(ValueError, 'No eligible'):
                select_snapshot(study, condition='high_lr', metric='first_30s_rms_xy_displacement_m',
                                require_full_parameters=True)

    def test_selection_excludes_small_early_failure_and_other_partition(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'summary.json'
            save_json(path, legacy('wheel_hold_ablation_summary'))
            study = load_study(path)
            original = study['snapshots'][1]
            failed = copy.deepcopy(original)
            failed['id'] = 'failed'; failed['step'] = 11
            metric = 'first_30s_rms_xy_displacement_m'
            failed['values'][metric] = 0.00001; failed['eligible'][metric] = False
            heldout = copy.deepcopy(original)
            heldout['id'] = 'heldout'; heldout['partition'] = 'heldout'
            heldout['values'][metric] = 0.000001
            study['snapshots'].extend([failed, heldout])
            result = select_snapshot(study, condition='weak_wheel', metric=metric)
            self.assertEqual(result['selected']['id'], original['id'])
            self.assertEqual(result['excluded'], [dict(id='failed', reason='ineligible_or_missing')])
            study['snapshots'][1]['seed'] = '0'
            with self.assertRaisesRegex(ValueError, 'seed'):
                select_snapshot(study, condition='weak_wheel', metric=metric)

    def test_json_rejects_duplicate_fields_and_nonfinite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'bad.json'
            for text in ['{"seed":0,"seed":1}', '{"value":NaN}', '{"value":1e999}']:
                path.write_text(text, encoding='utf-8')
                with self.assertRaises(ValueError):
                    load_json(path)

    def test_invalid_output_does_not_create_partial_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'output.json'
            with self.assertRaises(ValueError):
                save_json(path, {'value': float('nan')})
            self.assertFalse(path.exists())

    def test_cli_compare_and_exclusive_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, out = Path(tmp)/'summary.json', Path(tmp)/'report.json'
            save_json(path, legacy('wheel_hold_ablation_summary'))
            self.assertEqual(main(['compare', str(path), '--output', str(out)]), 0)
            self.assertEqual(load_json(out)['counts']['records'], 10)
            original = out.read_bytes()
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(['compare', str(path), '--output', str(out)]), 2)
                self.assertEqual(main(['compare', str(path), '--output', str(path)]), 2)
            self.assertEqual(out.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
