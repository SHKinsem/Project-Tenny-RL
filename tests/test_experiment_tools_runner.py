import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from experiments.experiment_tools.freeze import freeze_inputs,verify_inputs
from experiments.experiment_tools.runner import run_matrix,validate_plan


def plan(jobs,**extra):
    return dict(schema_version=1,kind='experiment_jobs',max_total_seconds=5.,jobs=jobs,**extra)


def job(identity,code,*,timeout=2.,check=True,arguments=()):
    value=dict(id=identity,command=['{python}','-X','utf8','-c',code,'{run_dir}',*arguments],timeout_seconds=timeout)
    if check:value['expected_result']=dict(path='{run_dir}/status.json',checks={'phase':'completed','steps':123})
    return value


GOOD="import json,sys;from pathlib import Path;Path(sys.argv[1],'status.json').write_text(json.dumps({'phase':'completed','steps':123}),encoding='utf-8');print('finished',flush=True)"


class ExperimentToolsRunnerTest(unittest.TestCase):
    def test_sequential_unicode_paths_and_literal_argv_keep_trainer_status_separate(self):
        with tempfile.TemporaryDirectory(prefix='实验 输入 ') as folder:
            root=Path(folder);script=root/'作业 脚本.py';literal='{run_dir}/literal {value} $HOME $(anything) & data'
            script.write_text("import json,sys\nfrom pathlib import Path\np=Path(sys.argv[1]);p.joinpath('status.json').write_text(json.dumps({'phase':'completed','steps':123,'arg':sys.argv[2]}),encoding='utf-8')\nprint('stdout 标记',flush=True)\nprint('stderr 标记',file=sys.stderr,flush=True)\n",encoding='utf-8')
            first=dict(id='first',command=['{python}','-X','utf8',str(script),'{run_dir}',literal],timeout_seconds=2.,
                       expected_result=dict(path='{run_dir}/status.json',checks={'phase':'completed','steps':123,'arg':literal}))
            output=root/'输出 历史';status=run_matrix(plan([first,job('第二 任务',GOOD)]),output)
            self.assertEqual(status['phase'],'completed');self.assertEqual(status['completed_jobs'],2)
            self.assertEqual([row['phase'] for row in status['jobs']],['completed','completed'])
            saved=json.loads((output/'status.json').read_text(encoding='utf-8'))
            self.assertEqual(status,saved)
            self.assertLessEqual(status['jobs'][0]['finished_at'],status['jobs'][1]['started_at'])
            trainer=json.loads((output/'runs/first/status.json').read_text(encoding='utf-8'))
            self.assertEqual(trainer['arg'],literal)
            self.assertEqual(status['jobs'][0]['command'][-1],literal)
            self.assertNotIn('phase',json.loads((output/'plan.json').read_text(encoding='utf-8')))
            self.assertEqual(json.loads((output/'jobs/first.json').read_text(encoding='utf-8'))['phase'],'completed')
            log=(output/'logs/first.log').read_text(encoding='utf-8')
            self.assertIn('stdout 标记',log);self.assertIn('stderr 标记',log)

    def test_zero_exit_with_bad_result_is_failure_and_stops_following_job(self):
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'out'
            bad=GOOD.replace("'steps':123","'steps':12")
            status=run_matrix(plan([job('bad',bad),job('later',GOOD)]),output)
            self.assertEqual(status['phase'],'failed');self.assertEqual(status['error_kind'],'result_check_failed')
            self.assertEqual(status['jobs'][0]['returncode'],0)
            self.assertEqual(status['jobs'][1]['phase'],'not_started')
            self.assertFalse((output/'runs/later').exists())
            self.assertIn('result_check_failed',(output/'logs/bad.log').read_text(encoding='utf-8'))

    def test_nonzero_exit_is_recorded_with_log_and_no_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'out'
            status=run_matrix(plan([job('exit','import sys;print("error detail",flush=True);sys.exit(7)',check=False),job('later',GOOD)]),output)
            self.assertEqual(status['jobs'][0]['returncode'],7)
            self.assertEqual(status['error_kind'],'nonzero_exit')
            self.assertEqual(status['jobs'][1]['phase'],'not_started')
            self.assertIn('error detail',(output/'logs/exit.log').read_text(encoding='utf-8'))

    def test_missing_executable_and_missing_required_result_have_complete_failure_states(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);missing=dict(id='missing',command=[str(root/'absent executable')],timeout_seconds=1.)
            failed=run_matrix(plan([missing,job('later',GOOD)]),root/'launch')
            self.assertEqual(failed['error_kind'],'launch_error')
            self.assertFalse(failed['jobs'][0]['process_started'])
            self.assertIsNone(failed['jobs'][0]['returncode'])
            self.assertEqual(failed['jobs'][1]['phase'],'not_started')
            no_result=run_matrix(plan([job('no-result','print("done without result")')]),root/'result')
            self.assertEqual(no_result['error_kind'],'result_check_failed')
            self.assertEqual(no_result['jobs'][0]['returncode'],0)

    def test_explicit_literal_cwd_and_dotted_boolean_checks(self):
        with tempfile.TemporaryDirectory(prefix='工作 路径 ') as folder:
            root=Path(folder);working=root/'独立 cwd';working.mkdir()
            code="import json,sys;from pathlib import Path;Path(sys.argv[1],'status.json').write_text(json.dumps({'where':str(Path.cwd()),'metrics':{'good':True}}),encoding='utf-8')"
            value=job('cwd',code);value['cwd']=str(working)
            value['expected_result']=dict(path='status.json',checks={'where':str(working),'metrics.good':True})
            self.assertEqual(run_matrix(plan([value]),root/'ok')['phase'],'completed')
            value['expected_result']['checks']['metrics.good']=1
            self.assertEqual(run_matrix(plan([value]),root/'bool')['error_kind'],'result_check_failed')

    def test_job_timeout_and_total_budget_are_distinct_and_child_is_reaped(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);sleep='import time;print("sleeping",flush=True);time.sleep(2)'
            first=run_matrix(plan([job('slow',sleep,timeout=.15,check=False),job('later',GOOD)]),root/'job-timeout')
            self.assertEqual(first['error_kind'],'job_timeout');self.assertTrue(first['jobs'][0]['timed_out'])
            self.assertIsInstance(first['jobs'][0]['returncode'],int)
            self.assertEqual(first['jobs'][1]['phase'],'not_started')
            limited=plan([job('slow',sleep,timeout=2.,check=False)])
            limited['max_total_seconds']=.15
            second=run_matrix(limited,root/'total-timeout')
            self.assertEqual(second['error_kind'],'total_budget_exceeded')
            self.assertLessEqual(second['jobs'][0]['effective_timeout_seconds'],.15)
            self.assertIsInstance(second['jobs'][0]['returncode'],int)

    def test_freeze_tracks_file_bytes_tree_members_and_empty_directories(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);tree=root/'资料 树';tree.mkdir();(tree/'空目录').mkdir()
            protected=tree/'参数.json';protected.write_text('original',encoding='utf-8')
            snapshot=freeze_inputs([tree,protected]);summary=verify_inputs(snapshot)
            self.assertEqual(snapshot['hash_algorithm'],'sha256');self.assertEqual(summary['files_verified'],2)
            entry=next(item for item in snapshot['entries'] if item['kind']=='file')
            self.assertEqual(entry['sha256'],hashlib.sha256(b'original').hexdigest())
            protected.write_text('modified',encoding='utf-8')
            with self.assertRaises(ValueError):verify_inputs(snapshot)
            protected.write_text('original',encoding='utf-8');verify_inputs(snapshot)
            (tree/'新目录').mkdir()
            with self.assertRaises(ValueError):verify_inputs(snapshot)

    def test_changed_frozen_input_fails_after_job_and_preserves_logs(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);protected=root/'protected.txt';protected.write_text('before',encoding='utf-8')
            snapshot=freeze_inputs([protected])
            mutate=GOOD+";Path(sys.argv[2]).write_text('after',encoding='utf-8')"
            output=root/'out';status=run_matrix(plan([job('mutate',mutate,arguments=[str(protected)]),job('later',GOOD)],frozen_inputs=snapshot),output)
            self.assertEqual(status['phase'],'failed');self.assertEqual(status['error_kind'],'input_changed')
            self.assertEqual(status['jobs'][0]['returncode'],0)
            self.assertEqual(status['jobs'][0]['frozen_before']['phase'],'verified')
            self.assertEqual(status['jobs'][1]['phase'],'not_started')
            self.assertIn('input_changed',(output/'logs/mutate.log').read_text(encoding='utf-8'))

    def test_dry_run_is_pure_and_prechanged_input_never_launches_child(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);protected=root/'input';protected.write_text('before',encoding='utf-8')
            snapshot=freeze_inputs([protected]);protected.unlink()
            value=plan([job('first',GOOD)],frozen_inputs=snapshot)
            self.assertEqual(validate_plan(value),value)
            output=root/'out';self.assertFalse(output.exists())
            status=run_matrix(value,output)
            self.assertEqual(status['phase'],'failed');self.assertEqual(status['error_kind'],'input_changed')
            self.assertFalse(status['jobs'][0]['process_started'])
            self.assertIsNone(status['jobs'][0]['returncode'])

    def test_invalid_plan_and_nonempty_output_are_rejected_without_overwrite(self):
        for identity in ('../escape','CON','bad/name','trail.',''):
            with self.assertRaises(ValueError):validate_plan(plan([job(identity,GOOD)]))
        with self.assertRaises(ValueError):validate_plan(plan([job('same',GOOD),job('SAME',GOOD)]))
        invalid=plan([job('valid',GOOD)]);invalid['max_total_seconds']=float('nan')
        with self.assertRaises(ValueError):validate_plan(invalid)
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'old';output.mkdir();marker=output/'history.txt';marker.write_text('keep',encoding='utf-8')
            with self.assertRaises(ValueError):run_matrix(plan([job('valid',GOOD)]),output)
            self.assertEqual(marker.read_text(encoding='utf-8'),'keep')
            self.assertEqual(list(output.iterdir()),[marker])
