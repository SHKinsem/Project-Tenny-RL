"""Progress-only tests: no JAX, MuJoCo, training, or user processes."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

TORQUE_ROOT = Path(__file__).resolve().parents[1] / "experiments" / "torque200"
sys.path.insert(0, str(TORQUE_ROOT))
from hybrid.progress import ValidationProgress
from hybrid import validate


class FlushStream(io.StringIO):
    def __init__(self):
        super().__init__()
        self.flushes = 0

    def flush(self):
        self.flushes += 1
        super().flush()


class ProgressTests(unittest.TestCase):
    def test_stage_names_timings_counts_and_explicit_flush(self):
        out = FlushStream()
        with ValidationProgress(60, out) as p:
            p.stage("known samples", total=2)
            p.advance()
            p.advance()
            p.stage("compile", hint="No estimate")
        self.assertGreaterEqual(out.flushes, 6)
        self.assertIn("已完成 2/2 项", out.getvalue())
        self.assertIn("No estimate", out.getvalue())
        self.assertEqual([s['stage'] for s in p.history], ["known samples", "compile"])
        self.assertTrue(all(s['elapsed_seconds'] >= 0 for s in p.history))
        self.assertIsNone(p.failed_stage)
        self.assertFalse(p.worker.is_alive())

    def test_broken_output_preserves_validation_error_and_failure_history(self):
        class BrokenStream(io.StringIO):
            def write(self, text):
                raise BrokenPipeError("closed output")
        failure = AssertionError("real model failure")
        with self.assertRaises(AssertionError) as raised:
            with ValidationProgress(.01, BrokenStream()) as p:
                p.stage("actual check")
                raise failure
        self.assertIs(raised.exception, failure)
        self.assertEqual(p.log_error, "BrokenPipeError")
        self.assertEqual(p.history[-1]['status'], "失败")
        self.assertFalse(p.worker.is_alive())

    def test_incomplete_count_cannot_be_reported_complete(self):
        with self.assertRaisesRegex(ValueError, "all declared checks"):
            with ValidationProgress(60, io.StringIO()) as p:
                p.stage("counted", total=2)
                p.advance()
        self.assertEqual(p.history[-1]['status'], "失败")
        self.assertEqual(p.failed_stage, "counted")
        self.assertFalse(p.worker.is_alive())

    def test_exception_and_interrupt_not_swallowed(self):
        for exc in (RuntimeError("diagnostic failed"), KeyboardInterrupt()):
            with self.subTest(exc=type(exc).__name__):
                out = io.StringIO()
                with self.assertRaises(type(exc)) as raised:
                    with ValidationProgress(.01, out) as p:
                        p.stage("bad stage")
                        raise exc
                self.assertIs(raised.exception, exc)
                self.assertEqual(p.failed_stage, "bad stage")
                self.assertEqual(p.history[-1]['status'], "失败")
                self.assertFalse(p.worker.is_alive())
                self.assertNotIn("完成", out.getvalue())

    def test_no_more_heartbeats_after_exit(self):
        out = io.StringIO()
        with ValidationProgress(.01, out) as p:
            p.stage("waiting")
            time.sleep(.06)
        saved = out.getvalue()
        self.assertIn("心跳不代表检查通过或编译有进展", saved)
        time.sleep(.03)
        self.assertEqual(saved, out.getvalue())

    def test_bad_intervals_and_counts_rejected(self):
        for interval in (0, -1, float('inf'), float('nan')):
            with self.assertRaises(ValueError):
                ValidationProgress(interval)
        with ValidationProgress(60, io.StringIO()) as p:
            for total in (0, -1, 1.5):
                with self.assertRaises(ValueError):
                    p.stage("bad", total=total)
            p.stage("unknown")
            with self.assertRaises(ValueError):
                p.advance()
            p.stage("one", total=1)
            p.advance()
            with self.assertRaises(ValueError):
                p.advance()

    def test_piped_output_heartbeats_arrive_before_silent_operation_finishes(self):
        # Deliberately omit Python -u: each progress line must flush on its own.
        code = """from hybrid.progress import ValidationProgress
import time
with ValidationProgress(.05) as p:
    p.stage('silent fake compile')
    time.sleep(2)
print('OPERATION_RETURNED', flush=True)
"""
        proc = subprocess.Popen([sys.executable, '-c', code], cwd=TORQUE_ROOT,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf8')
        lines = queue.Queue()
        reader = threading.Thread(target=lambda: [lines.put(x) for x in proc.stdout], daemon=True)
        reader.start()
        try:
            first = lines.get(timeout=5)
            second = lines.get(timeout=5)
            self.assertIn("开始", first)
            self.assertIn("等待", second)
            self.assertNotIn("OPERATION_RETURNED", first+second)
            self.assertIsNone(proc.poll(), "Heartbeat must arrive before process exits")
            self.assertEqual(proc.wait(timeout=5), 0)
            reader.join(timeout=1)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)
            proc.stdout.close()

    def test_main_failure_reports_stage_and_never_reports_success(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'report.json'
            out = io.StringIO()
            def fail(report, progress):
                progress.stage('runtime compilation')
                raise RuntimeError('test failure')
            with mock.patch.object(sys, 'argv', ['validate', '--output', str(path)]), \
                 mock.patch.object(validate, 'static_audit', return_value={'runtime_validated': False}), \
                 mock.patch.object(validate, 'runtime_checks', side_effect=fail), \
                 redirect_stdout(out), self.assertRaisesRegex(RuntimeError, 'test failure'):
                validate.main()
            report = json.loads(path.read_text())
            self.assertFalse(report['runtime_validated'])
            self.assertEqual(report['failed_stage'], 'runtime compilation')
            self.assertEqual(report['stage_timings'][-1]['status'], '失败')
            self.assertNotIn('HYBRID_LOCAL_VALIDATION_OK', out.getvalue())

    def test_static_failure_also_writes_failure_report(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'report.json'
            with mock.patch.object(sys, 'argv', ['validate', '--output', str(path)]), \
                 mock.patch.object(validate, 'static_audit', side_effect=ValueError('bad model')), \
                 redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'bad model'):
                validate.main()
            report = json.loads(path.read_text())
            self.assertFalse(report['runtime_validated'])
            self.assertEqual(report['failed_stage'], '源码几何扫描和指纹')

    def test_static_only_remains_non_runtime_validation(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'report.json'
            out = io.StringIO()
            with mock.patch.object(sys, 'argv', ['validate', '--static-only', '--output', str(path)]), \
                 mock.patch.object(validate, 'static_audit', return_value={'runtime_validated': False}), \
                 mock.patch.object(validate, 'runtime_checks') as runtime, redirect_stdout(out):
                validate.main()
            runtime.assert_not_called()
            self.assertFalse(json.loads(path.read_text())['runtime_validated'])
            self.assertIn('HYBRID_STATIC_ONLY_NOT_TRAINING_READY', out.getvalue())


if __name__ == '__main__':
    unittest.main()
