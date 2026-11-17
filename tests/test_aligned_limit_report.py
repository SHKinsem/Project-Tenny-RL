from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.aligned_limit_ablation.report import CONDITIONS, STATES, paired_summary, write_report


def fixture():
    initial_states=[dict(id=state,state=dict(qpos=[0.],qvel=[0.])) for state in STATES]
    digest=hashlib.sha256(json.dumps(initial_states,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()
    cases=[]
    values={'initial-limit1':2.7,'initial-limit2':2.7,'limit1':[1.,2.,2.5],'limit2':[1.2,1.8,2.6]}
    identities=[f'initial-{c}' for c in CONDITIONS]+[f'{c}-seed-{s}' for c in CONDITIONS for s in (0,1,2)]
    for identity in identities:
        condition,_,seed=identity.partition('-seed-')
        effective=values[identity] if not seed else values[condition][int(seed)]
        for state in STATES:
            trace=dict(t_s=[0.,1.,2.,4.],done=[False,False,False,True],
                base_position_m=[[0.,0.,.1],[0.,0.,.1],[.01,0.,.1],[.02,0.,.1]],
                base_velocity_m_s=[[0.,0.,0.]]*4,pitch_rad=[0.]*4,
                actuator_command_nm=[[0.,0.,0.,0.]]*4,
                metadata={'reference_pitch_rad':0.,'initial_states_sha256':digest},
                termination={'terminated':True,'truncated':False,'reason':'fixture'})
            cases.append(dict(policy_id=identity,label='<script>label</script>',initial_state_id=state,trace=trace,
                analysis={'windows':{'full_episode':{'observed_seconds':4.,'max_horizontal_displacement_m':.02,
                    'pitch_error_rms_deg':0.,'horizontal_speed_rms_m_s':0.}}},
                standing=dict(in_band_seconds_after_settling=effective,longest_in_band_seconds_after_settling=effective,
                              survival_seconds=4.,reached_horizon=False)))
    statuses=[dict(condition=c,seed=s,phase='completed',steps=2097152,initial_array_digest='same-start',reused=False)
              for c in CONDITIONS for s in (0,1,2)]
    return dict(schema_version=1,kind='balance_diagnostics',horizon_s=30.,cases=cases,
        initial_states=initial_states,initial_states_sha256=digest,run_statuses=statuses,
        experiment_manifest=dict(kind='aligned_torque_limit_ablation_v1',seeds=[0,1,2],steps_per_run=2097152,
                                 training_episode_seconds=30,torque_normalization_scales=[1.,1.,.0295,.0295]),
        alignment_validation=dict(deterministic_pair_max_abs_errors=dict(qpos=0.,qvel=0.,ctrl=0.),
                                  common_request_max_abs_error=0.))


class AlignedLimitReportTests(unittest.TestCase):
    def test_pairs_before_median_with_aligned_zero_step_and_six_fresh_runs(self):
        summary,rows,_=paired_summary(fixture())
        self.assertEqual(len(rows),40)
        self.assertAlmostEqual(summary['paired_delta_in_band_seconds']['median'],.1)
        self.assertAlmostEqual(summary['initial']['limit1']['primary']['mean_in_band_seconds'],2.7)
        self.assertAlmostEqual(summary['initial']['limit2']['primary']['mean_in_band_seconds'],2.7)
        self.assertAlmostEqual(summary['paired_delta_training_gain_in_band_seconds']['median'],.1)
        self.assertAlmostEqual(summary['conditions']['limit2']['delta_from_initial_in_band_seconds']['median'],-.9)
        self.assertEqual(summary['improved_seed_counts']['cumulative'],2)
        self.assertEqual(summary['matrix']['new_runs'],6)
        self.assertEqual(summary['matrix']['reused_runs'],0)
        self.assertFalse(any(row['reused'] for row in summary['conditions'].values()))
        self.assertTrue(all(value==0 for value in summary['zero_step_trace_max_abs_errors'].values()))

    def test_rejects_missing_duplicate_and_unknown_cases(self):
        for change in ('missing','duplicate','unknown'):
            value=fixture()
            if change=='missing':value['cases'].pop()
            elif change=='duplicate':value['cases'].append(deepcopy(value['cases'][0]))
            else:value['cases'][0]['policy_id']='initial'
            with self.subTest(change=change),self.assertRaises(ValueError):paired_summary(value)

    def test_rejects_budget_initializer_reuse_and_missing_statuses(self):
        for change in ({'steps':1},{'initial_array_digest':'other'},{'reused':True},{'phase':'failed'}):
            value=fixture();value['run_statuses'][0].update(change)
            with self.subTest(change=change),self.assertRaises(ValueError):paired_summary(value)
        value=fixture();del value['run_statuses']
        with self.assertRaises(ValueError):paired_summary(value)

    def test_rejects_physical_initial_state_mismatch(self):
        value=fixture();value['initial_states'][0]['state']['qpos'][0]=1.
        with self.assertRaises(ValueError):paired_summary(value)
        value=fixture();value['cases'][0]['trace']['metadata']['initial_states_sha256']='other'
        with self.assertRaises(ValueError):paired_summary(value)

    def test_rejects_changed_duration_and_reward_normalization(self):
        for field,new in (('kind','old_unaligned_experiment'),('training_episode_seconds',6),('torque_normalization_scales',[2.,2.,.0295,.0295])):
            value=fixture();value['experiment_manifest'][field]=new
            with self.subTest(field=field),self.assertRaises(ValueError):paired_summary(value)

    def test_rejects_trace_after_done_and_false_horizon_success(self):
        value=fixture();value['cases'][0]['trace']['done'][1]=True
        with self.assertRaises(ValueError):paired_summary(value)
        value=fixture();value['cases'][0]['standing']['reached_horizon']=True
        with self.assertRaises(ValueError):paired_summary(value)

    def test_requires_measured_alignment_evidence_inside_tolerance(self):
        for field,new in (('common_request_max_abs_error',1.1e-7),
                          ('common_request_max_abs_error',float('nan')),
                          ('common_request_max_abs_error',True),
                          ('deterministic_pair_max_abs_errors',{'qpos':1.1e-7}),
                          ('deterministic_pair_max_abs_errors',{})):
            value=fixture();value['alignment_validation'][field]=new
            with self.subTest(field=field,new=new),self.assertRaises(ValueError):paired_summary(value)
        value=fixture();del value['alignment_validation']
        with self.assertRaises(ValueError):paired_summary(value)
        value=fixture();value['alignment_validation']['common_request_max_abs_error']=5e-8
        self.assertEqual(paired_summary(value)[0]['alignment_validation']['common_request_max_abs_error'],5e-8)

    def test_report_independently_rejects_nonmatching_zero_step_physics(self):
        value=fixture()
        case=next(c for c in value['cases'] if c['policy_id']=='initial-limit2' and c['initial_state_id']=='nominal')
        case['trace']['base_position_m'][-1][0]+=2e-7
        with self.assertRaises(ValueError):paired_summary(value)
        value=fixture()
        case=next(c for c in value['cases'] if c['policy_id']=='initial-limit2' and c['initial_state_id']=='nominal')
        case['trace']['termination']['reason']='different_initial_failure'
        with self.assertRaises(ValueError):paired_summary(value)

    def test_report_escapes_labels_and_exports_two_initial_references(self):
        with tempfile.TemporaryDirectory() as directory,patch('experiments.aligned_limit_ablation.report._plot'):
            paths=write_report(fixture(),directory)
            content=Path(paths['report_html']).read_text(encoding='utf-8')
            self.assertNotIn('<script>',content);self.assertIn('&lt;script&gt;',content)
            self.assertIn('±1 N·m的物理对齐0步参考',content);self.assertIn('±2 N·m的物理对齐0步参考',content)
            self.assertIn('两组均为本轮新训练',content);self.assertIn('因果分解',content)
            self.assertNotIn('复用',content);self.assertIn('随机训练',content)
            self.assertIn('相同裁剪前请求',content)
            self.assertIn('comparison-detail.png',content)
            with Path(paths['summary_csv']).open(encoding='utf-8',newline='') as stream:
                self.assertEqual(len(list(csv.DictReader(stream))),40)
            summary=json.loads(Path(paths['paired_summary_json']).read_text(encoding='utf-8'))
            self.assertTrue(summary['matrix']['statuses_verified'])
