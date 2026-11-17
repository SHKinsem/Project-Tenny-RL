from copy import deepcopy
import csv
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.mature_balance_finetune.report import STATES, measure_case, summarize, write_report


def trace(seconds=120,displacement=.02,*,terminated=False):
    times=[float(i) for i in range(seconds+1)]
    return dict(schema_version=1,t_s=times,
        base_position_m=[[0.,0.,.1]]+[[displacement,0.,.1] for _ in times[1:]],
        base_velocity_m_s=[[0.,0.,0.] for _ in times],pitch_rad=[0. for _ in times],
        reward=[0.]+[1. for _ in times[1:]],done=[False]*(len(times)-1)+[True],
        metadata=dict(reference_pitch_rad=0.),
        termination=dict(terminated=terminated,truncated=not terminated,
                         reason='pitch' if terminated else 'time_limit',time_s=float(seconds)))


def fixture():
    policies=[dict(id='initial',source=dict(condition='initial'))]
    cases=[dict(policy_id='initial',initial_state_id=state,trace=trace()) for state in STATES]
    statuses=[]
    for condition,rate,base in (('high_lr',3e-5,.024),('low_lr',3e-6,.015)):
        for seed in range(3):
            identity=f'{condition}-seed-{seed}'
            policies.append(dict(id=identity,source=dict(condition=condition,seed=seed,learning_rate=rate)))
            cases.extend(dict(policy_id=identity,initial_state_id=state,trace=trace(displacement=base+seed*.001)) for state in STATES)
            statuses.append(dict(condition=condition,seed=seed,phase='completed',steps=2097152,
                                 reused=False,initial_array_digest='same-mature-policy'))
    return dict(kind='mature_balance_finetune_v1',policies=policies,cases=cases,
                run_statuses=statuses,experiment_manifest=dict(steps_per_run=2097152))


class MatureReportTests(unittest.TestCase):
    def test_post_step_position_reward_integrates_actual_dt_once(self):
        value=trace(2,.05)
        value['t_s']=[0.,.01,.03];value['termination']['time_s']=.03
        value['reward']=[0.,2.,3.]
        measured=measure_case(value)
        expected=-10*(1-math.exp(-1))*3
        self.assertAlmostEqual(measured['true_position_addition_return'],expected)
        self.assertEqual(measured['base_return'],5.)
        self.assertAlmostEqual(measured['objective_return'],5+expected)

    def test_120_seconds_are_not_capped_to_old_30_second_metric(self):
        measured=measure_case(trace(displacement=.008))
        self.assertEqual(measured['observed_episode_seconds'],120.)
        self.assertTrue(measured['reached_30s']);self.assertTrue(measured['reached_120s'])
        self.assertAlmostEqual(measured['rms_xy_displacement_m'],.008*math.sqrt(119.5/120))
        self.assertAlmostEqual(measured['last_10s_mean_xy_displacement_m'],.008)
        self.assertEqual(measured['first30s_in_band_seconds'],29.)
        self.assertEqual(measured['first30s_within_1cm_seconds'],29.)

    def test_horizon_failure_and_short_observations_are_distinct(self):
        short=measure_case(trace(2,0.,terminated=True))
        self.assertIsNone(short['last_10s_mean_xy_displacement_m'])
        self.assertFalse(short['reached_30s']);self.assertFalse(short['reached_120s'])
        at30=measure_case(trace(30,terminated=True))
        self.assertFalse(at30['reached_30s'])
        at120=measure_case(trace(120,terminated=True))
        self.assertTrue(at120['reached_30s']);self.assertFalse(at120['reached_120s'])

    def test_band_time_requires_both_endpoints_and_speed_constraint(self):
        value=trace(5,.005)
        value['t_s']=[0.,1.,2.,3.,30.,120.];value['termination']['time_s']=120.
        value['base_velocity_m_s'][2][0]=.2
        measured=measure_case(value)
        self.assertEqual(measured['first30s_in_band_seconds'],27.)
        self.assertEqual(measured['first30s_within_1cm_seconds'],27.)

    def test_pairs_three_seeds_after_primary_state_means(self):
        summary=summarize(fixture())
        self.assertEqual(len(summary['case_rows']),35)
        self.assertEqual(summary['failures'],[])
        self.assertTrue(summary['training_statuses']['verified'])
        self.assertAlmostEqual(summary['paired_stats']['primary']['first_120s']['last_10s_mean_xy_displacement_m']['median'],-.009)
        self.assertEqual(summary['paired_stats']['primary']['first_120s']['last_10s_mean_xy_displacement_m']['count'],3)
        self.assertAlmostEqual(summary['conditions']['low_lr']['seed_results'][1]['primary_delta_from_initial']['first_120s']['last_10s_mean_xy_displacement_m'],-.004)

    def test_early_failure_with_small_drift_cannot_win_horizon_comparison(self):
        bundle=fixture()
        item=next(c for c in bundle['cases'] if c['policy_id']=='low_lr-seed-0' and c['initial_state_id']=='nominal')
        item['trace']=trace(2,0.,terminated=True)
        summary=summarize(bundle)
        self.assertEqual(len(summary['failures']),1)
        self.assertIsNone(summary['paired_results'][0]['primary']['first_120s']['last_10s_mean_xy_displacement_m'])
        self.assertEqual(summary['paired_stats']['primary']['first_120s']['last_10s_mean_xy_displacement_m']['count'],2)

    def test_rejects_post_done_duplicate_cases_and_wrong_saved_measurement(self):
        value=trace();value['done'][1]=True
        with self.assertRaises(ValueError):measure_case(value)
        bundle=fixture();bundle['cases'].append(deepcopy(bundle['cases'][0]))
        with self.assertRaises(ValueError):summarize(bundle)
        bundle=fixture();bundle['cases'][0]['measurement']=dict(base_return=0.)
        with self.assertRaises(ValueError):summarize(bundle)
        bundle=fixture();bundle['run_statuses'][0]['initial_array_digest']='other'
        with self.assertRaises(ValueError):summarize(bundle)

    def test_progress_preserves_fixed_primary_cases_and_missing_horizons(self):
        bundle=fixture()
        values=[dict(initial_state_id=state,**measure_case(trace(30,.02))) for state in STATES]
        bundle['progress_records']={'low_lr-seed-0':[dict(steps=0,case_metrics=values),dict(steps=2097152,case_metrics=values)]}
        summary=summarize(bundle)
        rows=summary['progress_curves']['low_lr-seed-0']
        self.assertEqual([row['steps'] for row in rows],[0,2097152])
        self.assertAlmostEqual(rows[0]['primary']['windows']['first_30s']['last_10s_mean_xy_displacement_m'],.02)
        self.assertIsNone(rows[0]['primary']['windows']['first_120s']['last_10s_mean_xy_displacement_m'])

    def test_report_exports_all_cases_and_marks_both_horizons(self):
        with tempfile.TemporaryDirectory() as directory,patch('experiments.mature_balance_finetune.report._plot'):
            paths=write_report(fixture(),directory)
            content=Path(paths['report_html']).read_text(encoding='utf-8')
            self.assertIn('成熟模型',content);self.assertIn('前30秒',content);self.assertIn('120秒',content)
            self.assertIn('不能用早倒前的小位移算作改善',content)
            self.assertIn('不更新参考',content);self.assertIn('comparison-detail.png',content)
            with Path(paths['summary_csv']).open(encoding='utf-8',newline='') as stream:
                self.assertEqual(len(list(csv.DictReader(stream))),35)
            summary=json.loads(Path(paths['summary_json']).read_text(encoding='utf-8'))
            self.assertTrue(summary['training_statuses']['verified'])
