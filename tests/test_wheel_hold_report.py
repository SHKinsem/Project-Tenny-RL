from copy import deepcopy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.mature_balance_finetune.report import measure_case as mature_measure
from experiments.wheel_hold_ablation.report import STATES, measure_case, summarize, write_report


def trace(seconds=120,displacement=.02,*,terminated=False):
    times=[float(i) for i in range(seconds+1)]
    return dict(schema_version=1,t_s=times,
        base_position_m=[[0.,0.,.1]]+[[displacement,0.,.1] for _ in times[1:]],
        base_velocity_m_s=[[0.,0.,0.] for _ in times],pitch_rad=[0. for _ in times],
        reward=[0.]+[1. for _ in times[1:]],done=[False]*(len(times)-1)+[True],
        metadata=dict(reference_pitch_rad=0.),
        termination=dict(terminated=terminated,truncated=not terminated,
                         reason='pitch' if terminated else 'time_limit',time_s=float(seconds)))


def measurement(value,condition):
    result=measure_case(value)
    if condition=='weak_wheel':
        common=result.pop('objective_return')
        raw=value['t_s'][-1]
        result.update(common_xy_objective_return=common,native_wheel_position_raw_score_sum=raw,
            wheel_position_adjustment_return=-9*raw,training_objective_return=common-9*raw)
    return result


def fixture():
    policies=[dict(id='initial',source=dict(condition='initial'))]
    initial=trace()
    cases=[dict(policy_id='initial',initial_state_id=state,trace=deepcopy(initial),measurement=measure_case(initial)) for state in STATES]
    statuses=[]
    for condition,weight,base in (('original_wheel',10.,.016),('weak_wheel',1.,.012)):
        for seed in range(3):
            identity=f'{condition}-seed-{seed}';reused=condition=='original_wheel'
            policies.append(dict(id=identity,source=dict(condition=condition,seed=seed,
                learning_rate=3e-5,wheel_position_weight=weight,reused=reused)))
            for state in STATES:
                value=trace(displacement=base+seed*.001)
                cases.append(dict(policy_id=identity,initial_state_id=state,trace=value,
                                  measurement=measurement(value,condition)))
            statuses.append(dict(condition=condition,seed=seed,phase='completed',steps=2097152,
                reused=reused,initial_array_digest='same-mature-policy',elapsed_seconds=(100. if reused else 80.)+seed,
                source_run='historical-mature-xy' if reused else 'new-weak-wheel'))
    return dict(kind='wheel_hold_ablation_v1',policies=policies,cases=cases,run_statuses=statuses,
        experiment_manifest=dict(steps_per_run=2097152,model_origin=dict(run='mature',checkpoint='2801664')))


class WheelHoldReportTests(unittest.TestCase):
    def test_physical_measurement_is_the_existing_function(self):
        self.assertIs(measure_case,mature_measure)

    def test_pairs_weights_with_same_lr_and_discloses_reused_budget(self):
        summary=summarize(fixture())
        self.assertEqual(set(summary['conditions']),{'original_wheel','weak_wheel'})
        self.assertTrue(all(value['learning_rate']==3e-5 for value in summary['conditions'].values()))
        self.assertEqual(summary['conditions']['original_wheel']['wheel_position_weight'],10.)
        self.assertEqual(summary['conditions']['weak_wheel']['wheel_position_weight'],1.)
        receipt=summary['training_statuses']
        self.assertEqual(receipt['reused_runs'],3);self.assertEqual(receipt['new_runs'],3)
        self.assertEqual(receipt['new_environment_steps'],6291456)
        self.assertEqual(receipt['reused_run_elapsed_seconds_sum'],303.)
        self.assertAlmostEqual(summary['paired_stats']['primary']['first_120s']['last_10s_mean_xy_displacement_m']['median'],-.004)

    def test_common_evaluation_is_not_weak_training_objective(self):
        summary=summarize(fixture())
        original=next(row for row in summary['case_rows'] if row['condition']=='original_wheel')
        weak=next(row for row in summary['case_rows'] if row['condition']=='weak_wheel')
        self.assertEqual(original['training_objective_return'],original['common_xy_objective_return'])
        self.assertEqual(original['wheel_position_adjustment_return'],0.)
        self.assertAlmostEqual(weak['training_objective_return'],weak['common_xy_objective_return']-1080.)
        self.assertNotIn('objective_return',weak)
        self.assertEqual(summary['objective_accounting']['weak_cases_with_training_objective'],15)

    def test_source_rejects_fake_lr_groups_wrong_weights_and_reuse(self):
        for field,value in (('learning_rate',3e-6),('wheel_position_weight',1.),('reused',False)):
            bundle=fixture();bundle['policies'][1]['source'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):summarize(bundle)
        bundle=fixture();bundle['kind']='mature_balance_finetune_v1'
        with self.assertRaises(ValueError):summarize(bundle)

    def test_wrong_training_objective_or_native_wheel_sum_is_rejected(self):
        for field,value in (('training_objective_return',0.),('native_wheel_position_raw_score_sum',1.)):
            bundle=fixture();case=next(c for c in bundle['cases'] if c['policy_id']=='weak_wheel-seed-0')
            case['measurement'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):summarize(bundle)

    def test_missing_weak_target_remains_unknown_instead_of_using_legacy_alias(self):
        bundle=fixture();case=next(c for c in bundle['cases'] if c['policy_id']=='weak_wheel-seed-0')
        case['measurement']=measure_case(case['trace'])
        summary=summarize(bundle)
        row=next(r for r in summary['case_rows'] if r['policy_id']==case['policy_id'] and r['initial_state_id']==case['initial_state_id'])
        self.assertIsNone(row['training_objective_return'])
        self.assertIsNotNone(row['common_xy_objective_return'])
        self.assertEqual(summary['objective_accounting']['weak_cases_with_training_objective'],14)

    def test_early_failure_small_drift_does_not_win(self):
        bundle=fixture();case=next(c for c in bundle['cases'] if c['policy_id']=='weak_wheel-seed-0')
        case['trace']=trace(2,0.,terminated=True);case['measurement']=measurement(case['trace'],'weak_wheel')
        summary=summarize(bundle)
        self.assertEqual(len(summary['failures']),1)
        self.assertIsNone(summary['paired_results'][0]['primary']['first_120s']['rms_xy_displacement_m'])
        self.assertEqual(summary['paired_stats']['primary']['first_120s']['rms_xy_displacement_m']['count'],2)

    def test_statuses_reject_old_runs_mislabeled_new_or_unequal_initialization(self):
        for field,value in (('reused',False),('steps',1),('initial_array_digest','other')):
            bundle=fixture();bundle['run_statuses'][0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):summarize(bundle)

    def test_progress_accepts_legacy_baseline_and_explicit_weak_objective(self):
        bundle=fixture();records={}
        for condition in ('original_wheel','weak_wheel'):
            values=[dict(initial_state_id=state,**measurement(trace(30,.015),condition)) for state in STATES]
            records[condition+'-seed-0']=[dict(steps=0,case_metrics=values),dict(steps=2097152,case_metrics=values)]
        bundle['progress_records']=records
        summary=summarize(bundle)
        original=summary['progress_curves']['original_wheel-seed-0'][0]
        weak=summary['progress_curves']['weak_wheel-seed-0'][0]
        self.assertEqual(original['mean_common_xy_objective_return'],original['mean_training_objective_return'])
        self.assertAlmostEqual(weak['mean_training_objective_return'],weak['mean_common_xy_objective_return']-270.)
        self.assertNotIn('mean_objective_return',weak)

    def test_report_describes_weight_comparison_and_exports_35_cases(self):
        with tempfile.TemporaryDirectory() as directory,patch('experiments.wheel_hold_ablation.report._plot'):
            paths=write_report(fixture(),directory)
            content=Path(paths['report_html']).read_text(encoding='utf-8')
            self.assertIn('权重10',content);self.assertIn('权重1',content)
            self.assertIn('新增6,291,456步',content);self.assertIn('303.000秒',content)
            self.assertIn('common_xy_objective_return',content);self.assertIn('training_objective_return',content)
            self.assertNotIn('3e-6',content);self.assertNotIn('低学习率',content)
            self.assertIn('不能用早倒前的小位移算作改善',content)
            with Path(paths['summary_csv']).open(encoding='utf-8',newline='') as stream:
                self.assertEqual(len(list(csv.DictReader(stream))),35)
            result=json.loads(Path(paths['summary_json']).read_text(encoding='utf-8'))
            self.assertEqual(result['training_statuses']['reused_runs'],3)
