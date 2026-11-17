from copy import deepcopy
import json
import math
import unittest

from experiments.experiment_tools.paired import summarize


def fixture():
    conditions=[dict(id='control',label='Control'),dict(id='candidate',label='Candidate')]
    seeds=[0,1,2]
    partitions={'primary':['p1','p2'],'heldout':['h']}
    records=[]
    drifts={'control':[1.,2.,2.5],'candidate':[1.2,1.8,2.6]}
    balances={'control':[2.,3.,4.],'candidate':[3.,3.,2.]}
    for condition in ('control','candidate'):
        for seed in seeds:
            for case,offset in (('p1',-.2),('p2',.2),('h',10.)):
                records.append(dict(condition=condition,seed=seed,case_id=case,
                    values=dict(drift=drifts[condition][seed]+offset,balance=balances[condition][seed]),
                    eligible=dict(drift=True,balance=True)))
    return dict(schema_version=1,kind='paired_study',conditions=conditions,
        comparison=dict(baseline='control',treatment='candidate'),seeds=seeds,partitions=partitions,
        metrics=[dict(id='drift',unit='m',direction='min'),dict(id='balance',unit='s',direction='max')],
        records=records,reference_records=[dict(case_id=case,values=dict(drift=drift,balance=balance),
            eligible=dict(drift=True,balance=True)) for case,drift,balance in (('p1',1.3,1.),('p2',1.7,1.),('h',5.,2.))])


def record(study,condition='candidate',seed=0,case='p1'):
    return next(row for row in study['records'] if row['condition']==condition and
                type(row['seed']) is type(seed) and row['seed']==seed and row['case_id']==case)


class PairedStudyTests(unittest.TestCase):
    def test_seed_pairing_precedes_median_and_cases_are_not_repeats(self):
        summary=summarize(fixture())
        paired=summary['paired_summaries']['primary']['drift']
        self.assertAlmostEqual(paired['median'],.1)
        baseline=summary['condition_summaries']['control']['primary']['metrics']['drift']['median']
        treatment=summary['condition_summaries']['candidate']['primary']['metrics']['drift']['median']
        self.assertAlmostEqual(treatment-baseline,-.2)
        self.assertEqual(paired['count'],3)
        self.assertEqual(paired['improved_seed_count'],1)
        self.assertEqual(paired['degraded_seed_count'],2)
        self.assertEqual(summary['counts']['records'],18)

    def test_direction_and_ties_have_explicit_seed_counts(self):
        result=summarize(fixture())['paired_summaries']['primary']['balance']
        self.assertEqual(result['median'],0.)
        self.assertEqual((result['improved_seed_count'],result['degraded_seed_count'],result['tied_seed_count']),(1,1,1))

    def test_reference_is_averaged_with_the_same_declared_cases(self):
        summary=summarize(fixture())
        self.assertEqual(summary['references']['primary']['values']['drift'],1.5)
        rows=summary['delta_from_reference']['candidate']['primary']['seed_results']
        self.assertAlmostEqual(rows[0]['values']['drift'],-.3)
        self.assertAlmostEqual(rows[2]['values']['drift'],1.1)
        stats=summary['delta_from_reference']['candidate']['primary']['metrics']['drift']
        self.assertAlmostEqual(stats['median'],.3)
        self.assertEqual(stats['improved_seed_count'],1)

    def test_one_ineligible_case_invalidates_the_whole_seed_metric(self):
        study=fixture()
        failed=record(study)
        failed['values']['drift']=0.000001
        failed['eligible']['drift']=False
        summary=summarize(study)
        seed=summary['condition_summaries']['candidate']['primary']['seed_results'][0]
        self.assertIsNone(seed['values']['drift'])
        self.assertEqual(seed['missing_cases']['drift'],['p1'])
        self.assertEqual(seed['ineligible_cases']['drift'],['p1'])
        self.assertEqual(seed['missing_value_cases']['drift'],[])
        self.assertIsNotNone(seed['values']['balance'])
        self.assertIsNone(summary['paired_results']['primary'][0]['values']['drift'])
        self.assertIsNone(summary['paired_results']['primary'][0]['improved']['drift'])
        self.assertEqual(summary['paired_summaries']['primary']['drift']['count'],2)
        self.assertEqual(summary['paired_summaries']['primary']['drift']['missing_count'],1)
        self.assertEqual(summary['paired_summaries']['heldout']['drift']['count'],3)

    def test_null_values_are_unknown_even_if_an_eligibility_gate_passed(self):
        for eligible in (True,False):
            study=fixture();failed=record(study)
            failed['values']['drift']=None;failed['eligible']['drift']=eligible
            result=summarize(study)
            seed=result['condition_summaries']['candidate']['primary']['seed_results'][0]
            self.assertIsNone(seed['values']['drift'])
            self.assertEqual(seed['missing_value_cases']['drift'],['p1'])

    def test_different_missing_seeds_do_not_create_unpaired_success(self):
        study=fixture()
        for seed in (0,1,2):
            record(study,'candidate' if seed==0 else 'control',seed)['eligible']['drift']=False
        result=summarize(study)['paired_summaries']['primary']['drift']
        self.assertEqual(result['count'],0)
        self.assertEqual(result['missing_count'],3)
        self.assertIsNone(result['median'])
        self.assertEqual(result['improved_seed_count'],0)

    def test_missing_reference_metric_propagates_without_dropping_reference_case(self):
        study=fixture();study['reference_records'][0]['values']['drift']=None
        study['reference_records'][0]['eligible']['drift']=False
        summary=summarize(study)
        self.assertIsNone(summary['references']['primary']['values']['drift'])
        relative=summary['delta_from_reference']['candidate']['primary']['metrics']['drift']
        self.assertEqual(relative['count'],0);self.assertEqual(relative['missing_count'],3)
        self.assertEqual(summary['paired_summaries']['primary']['drift']['count'],3)

    def test_integer_and_string_seed_identity_is_preserved_for_json(self):
        study=fixture();study['seeds']=[0,'0']
        study['partitions']={'only':['case']};study['reference_records']=[];study['records']=[]
        for condition,values in (('control',[1.,10.]),('candidate',[2.,7.])):
            for seed,value in zip(study['seeds'],values):
                study['records'].append(dict(condition=condition,seed=seed,case_id='case',
                    values=dict(drift=value,balance=1.),eligible=dict(drift=True,balance=True)))
        summary=summarize(study);rows=summary['paired_results']['only']
        self.assertIs(type(rows[0]['seed']),int);self.assertIs(type(rows[1]['seed']),str)
        self.assertEqual([row['values']['drift'] for row in rows],[1.,-3.])
        self.assertEqual(summary['paired_summaries']['only']['drift']['median'],-1.)
        self.assertIsNone(summary['references']);self.assertIsNone(summary['delta_from_reference'])
        json.dumps(summary,allow_nan=False)

    def test_string_seeds_and_arbitrary_partition_sizes(self):
        study=dict(schema_version=1,kind='paired_study',conditions=[dict(id=c,label=c) for c in ('a','b','unused')],
            comparison=dict(baseline='a',treatment='b'),seeds=['first','last'],
            partitions={'warmup':['A','B','C'],'stress':['D'],'unseen':['E','F']},
            metrics=[dict(id='latency',unit='ms',direction='min')],records=[])
        for condition in ('a','b','unused'):
            values=[1.,3.,8.,10.,20.,30.] if condition!='b' else [2.,4.,6.,5.,10.,20.]
            for seed,offset in (('first',0.),('last',10.)):
                for case,value in zip('ABCDEF',values):
                    study['records'].append(dict(condition=condition,seed=seed,case_id=case,
                        values=dict(latency=value+offset),eligible=dict(latency=True)))
        result=summarize(study)
        self.assertEqual(result['paired_summaries']['warmup']['latency']['median'],0.)
        self.assertEqual(result['paired_summaries']['stress']['latency']['median'],-5.)
        self.assertEqual(result['paired_summaries']['unseen']['latency']['median'],-10.)
        self.assertEqual(result['condition_summaries']['a']['warmup']['seed_results'][0]['values']['latency'],4.)

    def test_duplicate_definitions_are_rejected(self):
        for target in ('conditions','seeds','metrics','within_partition','across_partitions'):
            study=fixture()
            if target in ('conditions','seeds','metrics'):study[target].append(deepcopy(study[target][0]))
            elif target=='within_partition':study['partitions']['primary'].append('p1')
            else:study['partitions']['heldout'].append('p2')
            with self.subTest(target=target),self.assertRaises(ValueError):summarize(study)

    def test_incomplete_duplicate_and_undeclared_record_rows_are_rejected(self):
        for target in ('missing','duplicate','condition','seed','case_id'):
            study=fixture()
            if target=='missing':study['records'].pop()
            elif target=='duplicate':study['records'].append(deepcopy(study['records'][0]))
            else:study['records'][0][target]='unknown'
            with self.subTest(target=target),self.assertRaises(ValueError):summarize(study)

    def test_nonfinite_and_boolean_numbers_are_rejected_even_if_ineligible(self):
        for value in (float('nan'),float('inf'),float('-inf'),True,False):
            study=fixture();row=record(study);row['values']['drift']=value;row['eligible']['drift']=False
            with self.subTest(value=value),self.assertRaises(ValueError):summarize(study)
        for seed in (True,False,1.0):
            study=fixture();study['seeds'][0]=seed
            with self.subTest(seed=seed),self.assertRaises(ValueError):summarize(study)

    def test_metric_values_and_eligibility_keys_are_explicit(self):
        for target in ('missing_value','missing_eligibility','extra_value','extra_eligibility','numeric_eligibility'):
            study=fixture();row=record(study)
            if target=='missing_value':del row['values']['drift']
            elif target=='missing_eligibility':del row['eligible']['drift']
            elif target=='extra_value':row['values']['typo']=1.
            elif target=='extra_eligibility':row['eligible']['typo']=True
            else:row['eligible']['drift']=1
            with self.subTest(target=target),self.assertRaises(ValueError):summarize(study)

    def test_nonempty_reference_requires_a_unique_complete_case_matrix(self):
        for target in ('missing','duplicate','unknown','missing_eligible'):
            study=fixture()
            if target=='missing':study['reference_records'].pop()
            elif target=='duplicate':study['reference_records'].append(deepcopy(study['reference_records'][0]))
            elif target=='unknown':study['reference_records'][0]['case_id']='unknown'
            else:del study['reference_records'][0]['eligible']['drift']
            with self.subTest(target=target),self.assertRaises(ValueError):summarize(study)

    def test_invalid_schema_comparison_and_direction_are_rejected(self):
        for target in ('version_bool','version','kind','same_condition','unknown_condition','direction'):
            study=fixture()
            if target=='version_bool':study['schema_version']=True
            elif target=='version':study['schema_version']=2
            elif target=='kind':study['kind']='unknown'
            elif target=='same_condition':study['comparison']['treatment']='control'
            elif target=='unknown_condition':study['comparison']['baseline']='unknown'
            else:study['metrics'][0]['direction']='smaller'
            with self.subTest(target=target),self.assertRaises(ValueError):summarize(study)

    def test_input_is_unchanged_and_output_does_not_alias_definitions(self):
        study=fixture();before=deepcopy(study);summary=summarize(study)
        self.assertEqual(study,before)
        summary['partitions']['primary'].append('another')
        summary['seeds'].append('another')
        summary['conditions'][0]['label']='changed'
        self.assertEqual(study,before)

    def test_large_finite_means_stay_finite_and_overflowing_differences_fail(self):
        study=fixture();study['reference_records']=[]
        for row in study['records']:row['values']['drift']=1e308
        summary=summarize(study)
        self.assertEqual(summary['condition_summaries']['control']['primary']['metrics']['drift']['median'],1e308)
        self.assertTrue(math.isfinite(summary['paired_summaries']['primary']['drift']['median']))
        json.dumps(summary,allow_nan=False)
        for row in study['records']:
            if row['condition']=='candidate':row['values']['drift']=-1e308
        with self.assertRaises(ValueError):summarize(study)
