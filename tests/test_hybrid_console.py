"""Pure logging/real subprocess tests. No JAX, MuJoCo or training claims."""
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

ROOT = Path(__file__).resolve().parents[1]
TORQUE = ROOT / 'experiments' / 'torque200'
sys.path.insert(0, str(TORQUE))
from hybrid.console import ConsoleView, capture, summary, MAX_JSON

DRIVER = 'E1114 23:55:39.720475   19240 cuda_executor.cc:1743] Could not get kernel mode driver version: ( INVALID_ARGUMENT: Version does not match the format X.Y.Z )\n'
PJRT = 'W1114 23:55:51.300626   19216 pjrt_executable.cc:642] Assume version compatibility. PjRt-IFRT does not track XLA executable versions.\n'


class ConsoleTests(unittest.TestCase):
    def view(self, **kw):
        out = io.StringIO()
        return ConsoleView('smoke', out, **kw), out

    def test_metrics_are_summarized_but_finite_error_never_hidden(self):
        v, out = self.view()
        event = dict(steps=16, elapsed_seconds=55.2, control_hz=100, aggregate_simulated_seconds=.16, metrics={'eval/episode_reward': -94.2, 'training/total_loss': .4, 'other': 123})
        v.line(json.dumps(event)+'\n')
        self.assertIn('实际步数 16', out.getvalue())
        self.assertIn('55.2s', out.getvalue())
        self.assertIn('控制 100 Hz', out.getvalue())
        self.assertIn('累计各环境仿真 0.160s', out.getvalue())
        self.assertNotIn('other', out.getvalue())
        event['metrics']['other'] = float('nan')
        v.line(json.dumps(event)+'\n')
        self.assertIn('NaN', out.getvalue())

    def test_unknown_json_and_unknown_errors_pass_through(self):
        v, out = self.view()
        for line in ('{"steps": 1, "elapsed_seconds": 2, "metrics": {}, "error": "bad"}\n',
                     'E1114 cuda_executor.cc:1743] CUDA_ERROR_OUT_OF_MEMORY\n',
                     'WARNING: a new warning\n', 'RuntimeError: actual failure\n',
                     DRIVER.rstrip()+' fatal extra detail\n'):
            v.line(line)
            self.assertIn(line.rstrip(), out.getvalue())

    def test_exact_duplicate_notices_only_are_counted(self):
        v, out = self.view()
        for _ in range(3):
            v.line(DRIVER)
            v.line(PJRT)
        v.finish()
        self.assertEqual(v.notices['驱动版本字符串格式提示'], 3)
        self.assertEqual(out.getvalue().count('重复项合并计数'), 1)
        self.assertIn('×3', out.getvalue())

    def test_stage_waits_count_and_errors(self):
        v, out = self.view()
        v.line('[23:55:35] 开始 | some compile | 阶段 0.0s | 总计 3.0s | long hint\n')
        v.line('[23:55:35] 完成 | some compile | 阶段 0.1s | 总计 3.1s\n')
        v.line('[23:55:35] 进度 | measured check | 阶段 1.0s | 总计 4.0s | 已完成 3/20 项\n')
        self.assertIn('开始：some compile', out.getvalue())
        self.assertIn('已完成 3/20 项', out.getvalue())
        self.assertNotIn('long hint', out.getvalue())
        v.last_visible -= 16
        v.wait_message()
        self.assertIn('心跳不代表取得进展', out.getvalue())
        v.line('[23:55:35] 失败 | bad check | 阶段 1.0s | 总计 4.0s\n')
        self.assertIn('失败：bad check', out.getvalue())

    def test_raw_bytes_complete_and_existing_log_never_truncated(self):
        raw = b'\xff\x00 raw\r\n'+DRIVER.encode()+b'{"steps":16,"elapsed_seconds":5,"metrics":{}}\n'
        with tempfile.TemporaryDirectory() as d:
            log = Path(d)/'full.log'
            v, _ = self.view()
            capture(io.BytesIO(raw), log, v)
            self.assertEqual(log.read_bytes(), raw)
            with self.assertRaises(FileExistsError):
                capture(io.BytesIO(b'replacement'), log, self.view()[0])
            self.assertEqual(log.read_bytes(), raw)

    def test_huge_json_integer_remains_visible_and_raw_tail_is_saved(self):
        raw = (json.dumps(dict(steps=16, elapsed_seconds=1, metrics={'huge': 10**400})) +
               '\nFATAL: later error\nTAIL\n').encode()
        with tempfile.TemporaryDirectory() as d:
            log = Path(d)/'full.log'
            v, out = self.view()
            capture(io.BytesIO(raw), log, v)
            self.assertEqual(log.read_bytes(), raw)
            self.assertIn('FATAL: later error', out.getvalue())
            self.assertIn('"huge":', out.getvalue())

    def test_broken_display_does_not_truncate_raw_log(self):
        class BrokenStream(io.StringIO):
            def write(self, text):
                raise BrokenPipeError('terminal closed')
        raw = b'first line\nFATAL: later error\nTAIL\n'
        with tempfile.TemporaryDirectory() as d:
            log = Path(d)/'full.log'
            with self.assertRaises(BrokenPipeError):
                capture(io.BytesIO(raw), log, ConsoleView('broken', BrokenStream()))
            self.assertEqual(log.read_bytes(), raw)

    def test_known_report_not_success_until_separate_summary(self):
        v, out = self.view()
        report = dict(checkpoint_validated=True, checkpoint='/tmp/test', actual_progress_steps=16, total_elapsed_seconds=69)
        for line in json.dumps(report, indent=2).splitlines(True):
            v.line(line)
        v.line('\n')
        v.line('HYBRID_TRAIN_RESTORE_EXPORT_OK\n')
        v.finish()
        self.assertNotIn('通过', out.getvalue())
        self.assertNotIn('checkpoint', out.getvalue())

    def test_unknown_report_field_not_hidden(self):
        v, out = self.view()
        report = dict(checkpoint_validated=True, checkpoint='/tmp/test', actual_progress_steps=16, total_elapsed_seconds=69, failure='bad')
        for line in json.dumps(report, indent=2).splitlines(True):
            v.line(line)
        v.finish()
        self.assertIn('"failure": "bad"', out.getvalue())

    def test_incomplete_and_interrupted_json_preserves_error(self):
        for error in ('Traceback (most recent call last):\n', 'F1114 native.cc:1] Check failed\n', 'E1114 CUDA is broken\n', 'LLVM ERROR: broken\n', 'fatal: broken\n', 'Segmentation fault\n'):
            v, out = self.view()
            v.line('{\n')
            v.line('  "checkpoint_validated": true,\n')
            v.line(error)
            self.assertIn(error.rstrip(), out.getvalue())
            self.assertIn('"checkpoint_validated"', out.getvalue())
        v, out = self.view()
        v.line('{\n')
        v.finish()
        self.assertIn('{', out.getvalue())

    def test_oversized_unparsed_json_does_not_buffer_forever(self):
        v, out = self.view()
        v.line('{\n')
        v.line('x'*MAX_JSON)
        self.assertFalse(v.pending)
        self.assertIn('x'*20, out.getvalue())

    def test_verbose_shows_original_unknown_content(self):
        v, out = self.view(verbose=True)
        raw = DRIVER + '{"steps":16,"elapsed_seconds":5,"metrics":{}}\n'
        for line in raw.splitlines(True):
            v.line(line)
        v.finish()
        self.assertEqual(out.getvalue(), raw)

    def test_success_summary_rejects_failed_missing_and_nonfinite_reports(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'report.json'
            for report in ({}, {'runtime_validated': False, 'schema':'hybrid-local-validation-v1'},
                           {'runtime_validated': True, 'schema':'hybrid-local-validation-v1', 'elapsed_seconds': float('nan')},
                           {'checkpoint_validated': True}):
                path.write_text(json.dumps(report))
                with self.assertRaises(ValueError):
                    summary('stage', path, 'full.log')
            with self.assertRaises(FileNotFoundError):
                summary('stage', Path(d)/'missing', 'full.log')

    def test_training_summary_steps_and_elapsed(self):
        report = dict(checkpoint_validated=True, actual_progress_steps=32, configured_steps=32, requested_steps=17,
                      total_elapsed_seconds=69.1, training_including_compile_seconds=53.6, warm_restart_changed_network_leaves=16)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'report.json'
            path.write_text(json.dumps(report))
            out = io.StringIO()
            with mock.patch('sys.stdout', out):
                summary('热启动', path, 'restart.log')
            self.assertIn('实际 32 步（请求 17 / 配置 32）', out.getvalue())
            self.assertIn('编译+训练 53.6s', out.getvalue())
            self.assertIn('优化器重新初始化', out.getvalue())

    def test_capture_pipe_keeps_early_heartbeat_before_child_output(self):
        with tempfile.TemporaryDirectory() as d:
            p = subprocess.Popen([sys.executable, '-u', '-m', 'hybrid.console', 'capture', '--label', 'sleep',
                                  '--log', str(Path(d)/'log'), '--heartbeat', '.05'], cwd=TORQUE,
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            q = queue.Queue()
            reader = threading.Thread(target=lambda: [q.put(x) for x in p.stdout], daemon=True)
            reader.start()
            try:
                line = q.get(timeout=5)
                self.assertIn('等待', line)
                self.assertIsNone(p.poll())
                p.stdin.write('RuntimeError: demonstrated\n')
                p.stdin.close()
                self.assertEqual(p.wait(timeout=5), 0)  # sink only; shell checks producer
                reader.join(1)
                self.assertIn('RuntimeError: demonstrated', ''.join(q.queue))
            finally:
                if p.poll() is None:
                    p.kill()
                    p.wait()
                p.stdout.close()
                if not p.stdin.closed:
                    p.stdin.close()

    def test_unterminated_bytes_are_saved_before_eof(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d)/'log'
            p = subprocess.Popen([sys.executable, '-u', '-m', 'hybrid.console', 'capture', '--label', 'bytes',
                                  '--log', str(log)], cwd=TORQUE,
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            try:
                data = b'FATAL: no newline but flushed'
                p.stdin.write(data)
                p.stdin.flush()
                deadline = time.monotonic()+3
                while time.monotonic() < deadline:
                    if log.exists() and log.read_bytes() == data:
                        break
                    time.sleep(.01)
                self.assertTrue(log.exists())
                self.assertEqual(log.read_bytes(), data)
                self.assertIsNone(p.poll())
                p.stdin.close()
                self.assertEqual(p.wait(timeout=5), 0)
                self.assertIn(data, p.stdout.read())
            finally:
                if p.poll() is None:
                    p.kill()
                    p.wait()
                p.stdout.close()
                if not p.stdin.closed:
                    p.stdin.close()

    def test_pending_unknown_json_is_revealed_before_heartbeat(self):
        v, out = self.view()
        v.line('{\n')
        v.line('  "unexpected": "value",\n')
        v.last_visible -= 16
        v.wait_message()
        self.assertIn('"unexpected"', out.getvalue())
        self.assertFalse(v.pending)

    def test_utf8_split_across_chunks_is_displayed_and_saved(self):
        class Chunked(io.BytesIO):
            def read1(self, size):
                return super().read1(1)
        raw = '正常汉字\n'.encode()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'log'
            v, out = self.view()
            capture(Chunked(raw), path, v)
            self.assertEqual(path.read_bytes(), raw)
            self.assertEqual(out.getvalue(), '正常汉字\n')

    def test_shell_pipeline_retains_producer_failure(self):
        # Execute the ACTUAL shell helper, no training or dependency imports.
        script = (TORQUE/'hybrid-wsl.sh').read_text()
        start = script.index('run_logged() {')
        helper = script[start:script.index('\n}', start) + 2]
        with tempfile.TemporaryDirectory() as d:
            logfile = Path(d)/'failure.log'
            code = 'print("HYBRID_TRAIN_RESTORE_EXPORT_OK"); print("RuntimeError: preserved failure"); raise SystemExit(7)'
            command = 'set -euo pipefail\nPY="$1"\n'+helper+'\nrun_logged failure "$2" "$3" "$PY" -c "$4"\necho SHOULD_NOT_RUN\n'
            p = subprocess.run(['bash', '-c', command, 'test', sys.executable, str(logfile), str(Path(d)/'missing'), code],
                               cwd=TORQUE, text=True, capture_output=True, timeout=10)
            self.assertEqual(p.returncode, 7)
            self.assertNotIn('SHOULD_NOT_RUN', p.stdout)
            self.assertNotIn('[通过]', p.stdout)
            self.assertIn('RuntimeError: preserved failure', p.stdout+p.stderr)
            self.assertIn(str(logfile), p.stderr)
            self.assertIn('HYBRID_TRAIN_RESTORE_EXPORT_OK', logfile.read_text())

    def test_shell_success_requires_report_and_sink_success(self):
        script = (TORQUE/'hybrid-wsl.sh').read_text()
        start = script.index('run_logged() {')
        helper = script[start:script.index('\n}', start) + 2]
        with tempfile.TemporaryDirectory() as d:
            for exists in (False, True):
                logfile = Path(d)/str(exists)
                if exists:
                    logfile.write_text('original')
                command = 'set -euo pipefail\nPY="$1"\n'+helper+'\nrun_logged test "$2" "$3" "$PY" -c "print(123)"\necho SHOULD_NOT_RUN\n'
                p = subprocess.run(['bash', '-c', command, 'test', sys.executable, str(logfile), str(Path(d)/'missing')],
                                   cwd=TORQUE, text=True, capture_output=True, timeout=10)
                self.assertNotEqual(p.returncode, 0)
                self.assertNotIn('SHOULD_NOT_RUN', p.stdout)
                self.assertNotIn('[通过]', p.stdout)
                if exists:
                    self.assertEqual(logfile.read_text(), 'original')


if __name__ == '__main__':
    unittest.main()
