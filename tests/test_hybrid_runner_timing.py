"""Real shell + parser + log pipes, explicitly fake ML child processes.

These verify argument transport, fail-fast and concurrent-run exclusion only.
They are NOT JAX, MuJoCo or PPO tests.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
TORQUE = ROOT/'experiments'/'torque200'


@unittest.skipUnless(os.name == 'posix' and shutil.which('bash') and shutil.which('flock'), 'POSIX bash/flock required')
class RunnerTimingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.script = self.root/'project'/'experiments'/'torque200'/'hybrid-wsl.sh'
        self.script.parent.mkdir(parents=True)
        shutil.copyfile(TORQUE/'hybrid-wsl.sh', self.script)
        self.base = self.root/'runtime'; self.base.mkdir()
        (self.base/'src').symlink_to(TORQUE, target_is_directory=True)
        self.bin = self.root/'bin'; self.bin.mkdir()
        # rsync is deliberately stubbed. No installed ML dependencies or source
        # mutation are needed; all fake reports go to the temporary runtime.
        sync = self.bin/'rsync'
        sync.write_text('#!'+sys.executable+'\n'+r'''
import json, os, sys
with open(os.environ['TENNY_TEST_SYNC_CALLS'], 'a') as f:
    f.write(json.dumps(sys.argv[1:])+'\n')
if '/results/hybrid-' in sys.argv[-1] and os.environ.get('TENNY_TEST_FAIL_COLLECT'):
    sys.exit(23)
''')
        sync.chmod(0o755)
        py = self.bin/'fake-python'
        py.write_text('#!'+sys.executable+'\n'+r'''
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
if '-m' in args:
    index = args.index('-m')
    module = args[index+1]
    if module == 'hybrid.readiness':
        # The readiness module has its own evidence-integrity tests. This fake
        # models its subprocess contract to test real shell argument transport.
        import subprocess, uuid
        argv = args[index+2:]
        root = Path(argv[argv.index('--runtime-root')+1])
        mode = argv[argv.index('--mode')+1]
        hz = argv[argv.index('--control-hz')+1]
        result_path = Path(argv[argv.index('--result')+1])
        cached = root/('fake-ready-'+mode+'-'+hz+'.json')
        if '--force' not in argv and cached.exists():
            result = json.loads(cached.read_text()); result['cache_hit'] = True
            result_path.write_text(json.dumps(result)); sys.exit(0)
        out = root/'artifacts'/('readiness-'+uuid.uuid4().hex); out.mkdir(parents=True)
        result = dict(validation_path=str(out/'validation.json'), evidence_dir=str(out), cache_hit=False)
        result_path.write_text(json.dumps(result))
        common = ['--mode', mode, '--control-hz', hz, '--validation', str(out/'validation.json'),
                  '--steps', '16', '--num-envs', '2', '--eval-envs', '2', '--unroll-length', '4']
        commands = [
            ['hybrid.validate', '--output', str(out/'validation.json'), '--control-hz', hz],
            ['hybrid.train', *common, '--output', str(out/'first')],
            ['hybrid.train', *common, '--output', str(out/'restart'), '--restore', str(out/'first/checkpoints/16')]]
        for command in commands:
            code = subprocess.run([sys.argv[0], '-m', *command]).returncode
            if code: sys.exit(code)
        cached.write_text(json.dumps(result)); sys.exit(0)
    if module in ('hybrid.validate', 'hybrid.train'):
        argv = args[index+2:]
        with open(os.environ['TENNY_TEST_CALLS'], 'a') as f:
            f.write(json.dumps(dict(module=module, argv=argv))+'\n')
        out = Path(argv[argv.index('--output')+1])
        if module == 'hybrid.validate':
            if os.environ.get('TENNY_TEST_FAIL_VALIDATION'):
                print('TEST_VALIDATION_FAILURE', flush=True); sys.exit(7)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(dict(schema="hybrid-local-validation-v1", runtime_validated=True, elapsed_seconds=.01)))
            print('HYBRID_LOCAL_VALIDATION_OK', flush=True)
        else:
            from hybrid.train import parser
            a = parser().parse_args(argv)
            out.mkdir(parents=True)
            checkpoint=out/'checkpoints'/'16'; checkpoint.mkdir(parents=True)
            (out/'report.json').write_text(json.dumps(dict(checkpoint_validated=True,
                checkpoint=str(checkpoint), actual_progress_steps=a.steps, requested_steps=a.steps,
                configured_steps=a.steps, control_hz=a.control_hz,
                training_including_compile_seconds=.01, total_elapsed_seconds=.02)))
            print('HYBRID_TRAIN_RESTORE_EXPORT_OK', flush=True)
        sys.exit(0)
os.execv(sys.executable, [sys.executable]+args)
''')
        py.chmod(0o755)
        self.calls = self.root/'calls.jsonl'
        self.sync_calls = self.root/'sync-calls.jsonl'
        self.env = dict(os.environ, TENNY_PYTHON=str(py), TENNY_HYBRID_ROOT=str(self.base),
                        TENNY_TEST_CALLS=str(self.calls), TENNY_TEST_SYNC_CALLS=str(self.sync_calls),
                        PYTHONPATH=str(TORQUE),
                        PATH=str(self.bin)+os.pathsep+os.environ['PATH'])

    def run_script(self, *args):
        return subprocess.run(['bash', str(self.script), *args], env=self.env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=30)

    def read_calls(self):
        return [json.loads(line) for line in self.calls.read_text().splitlines()] if self.calls.exists() else []

    def test_all_phases_receive_selected_frequency_and_mode(self):
        for hz in ('100', '200'):
            with self.subTest(hz=hz):
                self.calls.unlink(missing_ok=True)
                self.sync_calls.unlink(missing_ok=True)
                result = self.run_script('--mode', 'matched_envelope', '--control-hz', hz, '--steps', '32')
                self.assertEqual(result.returncode, 0, result.stdout)
                calls = self.read_calls()
                self.assertEqual([c['module'] for c in calls], ['hybrid.validate']+['hybrid.train']*3)
                for call in calls:
                    argv = call['argv']
                    self.assertEqual(argv[argv.index('--control-hz')+1], hz)
                    if call['module'] == 'hybrid.train':
                        self.assertEqual(argv[argv.index('--mode')+1], 'matched_envelope')
                self.assertIn('--restore', calls[2]['argv'])
                self.assertNotIn('--restore', calls[3]['argv'])
                syncs = [json.loads(v) for v in self.sync_calls.read_text().splitlines()]
                self.assertEqual(len(syncs), 3)
                self.assertEqual(syncs[0][-1], str(self.base/'src')+'/')
                self.assertTrue(syncs[1][-1].endswith('/preparation/'))
                self.assertIn('/results/hybrid-', syncs[2][-1])

    def test_formal_training_reuses_ready_result_but_explicit_recheck_runs_gate(self):
        first = self.run_script('--mode', 'virtual_leg_force', '--control-hz', '100', '--steps', '32')
        self.assertEqual(first.returncode, 0, first.stdout)
        self.calls.unlink()
        second = self.run_script('--reuse-ready', '--mode', 'virtual_leg_force', '--control-hz', '100', '--steps', '32')
        self.assertEqual(second.returncode, 0, second.stdout)
        self.assertEqual([c['module'] for c in self.read_calls()], ['hybrid.train'])
        self.calls.unlink()
        third = self.run_script('--reuse-ready', '--force-prepare', '--mode', 'virtual_leg_force', '--control-hz', '100')
        self.assertEqual(third.returncode, 0, third.stdout)
        self.assertEqual([c['module'] for c in self.read_calls()], ['hybrid.validate']+['hybrid.train']*2)

    def test_validation_failure_preserves_exit_code_and_never_trains(self):
        self.env['TENNY_TEST_FAIL_VALIDATION']='1'
        result = self.run_script('--control-hz', '100')
        self.assertEqual(result.returncode, 7, result.stdout)
        self.assertEqual([c['module'] for c in self.read_calls()], ['hybrid.validate'])
        self.assertNotIn('HYBRID_RUN_COMPLETE', result.stdout)

    def test_invalid_frequency_rejected_before_ml_process(self):
        result = self.run_script('--control-hz', '150')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.read_calls(), [])

    def test_same_runtime_cannot_be_overwritten_by_second_run(self):
        import fcntl
        with open(self.base/'.runner.lock', 'w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            result = self.run_script('--control-hz', '100')
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn('already running', result.stdout)
        self.assertEqual(self.read_calls(), [])
        self.assertFalse(self.sync_calls.exists())

    def test_copy_failure_does_not_replace_validation_failure(self):
        self.env['TENNY_TEST_FAIL_VALIDATION']='1'
        self.env['TENNY_TEST_FAIL_COLLECT']='1'
        result = self.run_script('--control-hz', '100')
        self.assertEqual(result.returncode, 7, result.stdout)
        self.assertIn('Result collection failed (exit 23)', result.stdout)
        self.assertEqual([c['module'] for c in self.read_calls()], ['hybrid.validate'])

    def test_copy_failure_after_success_is_not_successful_exit(self):
        self.env['TENNY_TEST_FAIL_COLLECT']='1'
        result = self.run_script('--control-hz', '200')
        self.assertEqual(result.returncode, 23, result.stdout)
        self.assertIn('Result collection failed (exit 23)', result.stdout)
        self.assertEqual(len(self.read_calls()), 3)


if __name__ == '__main__':
    unittest.main()
