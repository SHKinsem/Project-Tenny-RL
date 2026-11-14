"""Launcher recovery and selected-mode routing without invoking WSL or ML."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tenny import runner, registry


class ConsoleEntryTests(unittest.TestCase):
    def test_linux_selection_supports_hybrid_and_rejects_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            selected = Path(tmp) / 'selected-mode.json'
            with patch.object(runner, 'IS_WIN', False), patch.object(runner, 'SELECTED_MODE_FILE', str(selected)):
                self.assertEqual(runner.read_selected_mode(), 'original')
                selected.write_text(json.dumps({'mode': 'hybrid_virtual_leg_force_100'}))
                self.assertEqual(runner.read_selected_mode(), 'hybrid_virtual_leg_force_100')
                for bad in ('selected', 'hybrid_virtual_leg_force_150', 'unknown'):
                    selected.write_text(json.dumps({'mode': bad}))
                    with self.assertRaises(registry.TaskError):
                        runner.read_selected_mode()

    def test_windows_selection_failure_does_not_fallback(self):
        with patch.object(runner, 'IS_WIN', True), patch.object(runner, '_wsl', return_value=Mock(returncode=1, stdout='')):
            with self.assertRaises(registry.TaskError):
                runner.read_selected_mode()

    def test_training_reuses_readiness_but_explicit_test_does_not(self):
        tasks = registry.load()
        for name, expected in (('train-hybrid', True), ('test-hybrid', False)):
            argv = registry.build_args(tasks[name], registry.resolve_params(tasks[name], {}))
            self.assertEqual('--reuse-ready' in argv, expected)

    def test_snapshot_inventory_does_not_start_any_jobs(self):
        fake = Mock()
        fake.snapshot.return_value = {'id': 'one', 'task': 'web', 'status': 'running'}
        with patch.object(runner, '_jobs', {'one': fake}):
            self.assertEqual(runner.snapshots(), [fake.snapshot.return_value])
            fake.snapshot.assert_called_once_with()

    def test_selected_mode_is_resolved_before_launch(self):
        task = registry.load()['web']
        fake = Mock(id='one')
        with patch.object(runner, '_jobs', {}), patch.object(runner, 'check_available'), \
                patch.object(runner, 'read_selected_mode', return_value='hybrid_virtual_leg_force_100'), \
                patch.object(runner, 'wsl_ip', return_value='127.0.0.1'), \
                patch.object(runner, 'build_command', return_value=['fake']) as build, \
                patch.object(runner, 'Job', return_value=fake):
            runner.start('web', {'mode': 'selected'}, {'web': task})
            self.assertEqual(build.call_args.args[1]['mode'], 'hybrid_virtual_leg_force_100')


if __name__ == '__main__':
    unittest.main()
