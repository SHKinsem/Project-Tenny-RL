"""Real host-side timing/CLI/contract checks; no simulated dynamics claimed."""
from contextlib import redirect_stderr
import copy
import io
import json
import math
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "torque200"))
sys.path.insert(0, str(ROOT))
from hybrid.timing import Timing, configure_timing, validate_timing
from hybrid import core, train, deployment, validate
from tenny import registry


def bare_config():
    # Exercises the same attribute assignments used on ConfigDict without
    # substituting a pretend simulator or claiming JAX has been run.
    return SimpleNamespace(cmd_config=SimpleNamespace(), reward_config=SimpleNamespace())


class TimingTest(unittest.TestCase):
    def test_integer_decimation_and_wall_clock_durations(self):
        for hz, count in ((100, 4), (200, 2)):
            with self.subTest(hz=hz):
                t = Timing(hz)
                c = t.contract()
                self.assertEqual(t.n_substeps, count)
                self.assertEqual(t.n_substeps * t.sim_dt, t.ctrl_dt)
                self.assertEqual(c["episode_length"] * t.ctrl_dt, 6)
                self.assertEqual(c["command_period_steps"] * t.ctrl_dt, 3)
                self.assertEqual(c["command_start_steps"] * t.ctrl_dt, .2)
                self.assertEqual(t.steps(.1), hz // 10)
                self.assertEqual(t.steps(0), 0)

    def test_no_unsupported_or_noninteger_frequency(self):
        for hz in (0, -100, 50, 150, 400, 800, 200.0, "200", True, None, math.inf, math.nan):
            with self.subTest(hz=hz), self.assertRaises(ValueError):
                Timing(hz)

    def test_duration_and_delay_never_silently_rounded(self):
        for hz in (100, 200):
            t = Timing(hz)
            self.assertEqual(t.contract(10)["action_delay_steps"], hz // 100)
            for seconds in (-1, math.nan, math.inf, True, "1", .001):
                with self.subTest(hz=hz, seconds=seconds), self.assertRaises(ValueError):
                    t.steps(seconds)
            for delay in (-1, math.nan, math.inf, True, "10", .1):
                with self.subTest(hz=hz, delay=delay), self.assertRaises(ValueError):
                    t.contract(delay)
        with self.assertRaises(ValueError):
            Timing(100).contract(5)
        self.assertEqual(Timing(200).contract(5)["action_delay_steps"], 1)

    def test_discount_gae_and_integrated_cost_keep_physical_time(self):
        a, b = Timing(100).contract(), Timing(200).contract()
        self.assertAlmostEqual(a["discounting"], b["discounting"] ** 2)
        self.assertAlmostEqual(a["gae_lambda"], b["gae_lambda"] ** 2)
        for hz in (100, 200):
            c = Timing(hz).contract()
            self.assertAlmostEqual(c["discounting"] ** (hz * .5), .5)
            # Constant physical reward rate integrated for one second, using
            # inherited reward dt scaling and hybrid per-physics-step averaging.
            for raw_rate in (2., -3., 0.):
                total = hz * raw_rate * c["ctrl_dt"] / c["reward_reference_dt"]
                self.assertEqual(total, 100 * raw_rate)
        # For identical physical torque samples, grouping 2 vs4 physics steps
        # preserves integrated effort and first/second-difference costs.
        samples = [math.sin(i * .2) for i in range(40)]
        rates = []
        prev = prevprev = 0.
        for value in samples:
            rates.append(abs(value-prev)*4 + abs(value+prevprev-2*prev)*16)
            prevprev, prev = prev, value
        totals = []
        for hz in (100, 200):
            t = Timing(hz)
            totals.append(sum(sum(rates[i:i+t.n_substeps])/t.n_substeps * t.ctrl_dt/.01
                              for i in range(0, len(rates), t.n_substeps)))
        self.assertAlmostEqual(*totals)

    def test_config_recomputes_all_derived_fields_and_delay(self):
        for hz in (100, 200):
            cfg = configure_timing(bare_config(), hz, 10)
            self.assertEqual(validate_timing(cfg), Timing(hz))
            self.assertEqual(cfg.episode_length, 6*hz)
            self.assertEqual(cfg.action_delay_steps, hz//100)
            # Actual inherited command reset uses >, not >=.
            counter, sampling = 0, []
            for step in range(1, hz*7):
                counter += 1
                if counter == round(cfg.cmd_start_delay_s / cfg.ctrl_dt):
                    sampling.append(step)
                if counter > cfg.cmd_config.command_change_interval:
                    counter = 0
            self.assertEqual(sampling[:3], [round(.2*hz), round(3.2*hz), round(6.2*hz)])

    def test_partial_config_edits_fail_closed(self):
        changes = (("ctrl_dt", .02), ("sim_dt", .005), ("control_hz", 400),
                   ("episode_length", 1), ("episode_time", 3), ("action_repeat", 2),
                   ("cmd_start_delay_s", .1), ("action_delay_steps", 2),
                   ("action_delay_ms", 5), ("cmd_config.command_change_interval", 600),
                   ("reward_config.reference_dt", .005))
        for path, value in changes:
            cfg = configure_timing(bare_config(), 100)
            target = cfg
            *parents, field = path.split('.')
            for parent in parents:
                target = getattr(target, parent)
            setattr(target, field, value)
            with self.subTest(path=path), self.assertRaises(ValueError):
                validate_timing(cfg)


class FrequencyEntryTest(unittest.TestCase):
    def test_every_mode_can_select_both_rates_in_cli_and_launcher(self):
        tasks = registry.load()
        for task_id in ('test-hybrid', 'train-hybrid'):
            task = tasks[task_id]
            self.assertEqual(registry.resolve_params(task, {})['control_hz'], '200')
            for mode in core.MODES:
                for hz in (100, 200):
                    values = registry.resolve_params(task, dict(mode=mode, control_hz=hz))
                    argv = registry.build_args(task, values)
                    args = train.parser().parse_args(argv + ['--validation', 'check.json', '--output', 'run'])
                    self.assertEqual((args.mode, args.control_hz), (mode, hz))
            for hz in ('50', '150', '400', 'nan', '100; echo wrong'):
                with self.assertRaises(registry.TaskError):
                    registry.resolve_params(task, dict(control_hz=hz))

    def test_legacy_tasks_do_not_expose_frequency_override(self):
        tasks = registry.load()
        for key in ('web', 'test-torque200', 'test-original', 'train-timed', 'preview', 'sim'):
            with self.subTest(task=key), self.assertRaises(registry.TaskError):
                registry.resolve_params(tasks[key], dict(control_hz=100))

    def test_direct_train_parser_rejects_unsupported_frequency(self):
        base = ['--validation', 'check.json', '--output', 'run', '--control-hz']
        for hz in ('50', '150', '400', 'nan', '200.5'):
            with redirect_stderr(io.StringIO()), self.subTest(hz=hz), self.assertRaises(SystemExit):
                train.parser().parse_args(base+[hz])

    def test_validation_is_frequency_specific_and_static_never_grants_training(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(train, 'source_fingerprint', return_value='source'):
            p = Path(d)/'validation.json'
            for hz in (100, 200):
                report = dict(schema='hybrid-local-validation-v1', source_fingerprint='source',
                              runtime_validated=True, **Timing(hz).contract())
                p.write_text(json.dumps(report))
                self.assertEqual(train.check_validation(p, hz), report)
                with self.assertRaises(ValueError):
                    train.check_validation(p, 300-hz)
                for field in Timing(hz).contract():
                    bad = report.copy(); del bad[field]
                    p.write_text(json.dumps(bad))
                    with self.subTest(hz=hz, field=field), self.assertRaises(ValueError):
                        train.check_validation(p, hz)
                report['runtime_validated'] = False
                p.write_text(json.dumps(report))
                with self.assertRaises(ValueError):
                    train.check_validation(p, hz)

    def test_static_report_names_requested_rate_without_runtime_claim(self):
        for hz in (100, 200):
            report = validate.static_audit(hz)
            self.assertFalse(report['runtime_validated'])
            for key, value in Timing(hz).contract().items():
                self.assertEqual(report[key], value)


class FrequencyArtifactTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scan = core.scan_geometry(core.read_geometry(), samples=65)

    def test_restore_rejects_cross_frequency_even_with_valid_bytes(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(core, 'source_fingerprint', return_value='same'):
            root = Path(d); checkpoint = root/'checkpoints'/'16'; checkpoint.mkdir(parents=True)
            (checkpoint/'data').write_text('synthetic')
            (root/'report.json').write_text(json.dumps(dict(checkpoint_validated=True,
                checkpoint_name='16', checkpoint_sha256=core.digest_tree(checkpoint))))
            for mode in core.MODES:
                a = core.metadata(mode, self.scan, 100); b = core.metadata(mode, self.scan, 200)
                (root/'policy_metadata.json').write_text(json.dumps(a))
                train.restore_guard(checkpoint, a)
                with self.assertRaisesRegex(ValueError, 'control_hz|ctrl_dt'):
                    train.restore_guard(checkpoint, b)

    def test_deployment_uses_saved_rate_or_rejects_explicit_mismatch(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(core, 'source_fingerprint', return_value='same'), \
             mock.patch.object(deployment, 'scan_geometry', return_value=self.scan):
            root=Path(d); export=root/'policy'; export.write_text('synthetic')
            meta=root/'meta.json'
            kernel=ModuleType('legwheel.deployment.numpy_policy'); kernel.RL_Policy=mock.Mock()
            with mock.patch.dict(sys.modules, {kernel.__name__:kernel}):
                for hz in (100, 200):
                    contract=core.metadata('virtual_leg_force', self.scan, hz)
                    contract['export_sha256']=core.digest_file(export)
                    meta.write_text(json.dumps(contract))
                    policy=deployment.HybridPolicy(export, meta, 'virtual_leg_force')
                    self.assertEqual(policy.control_hz, hz)
                    self.assertEqual(policy.n_substeps, 400//hz)
                    self.assertEqual(policy.ctrl_dt, 1/hz)
                    with self.assertRaises(ValueError):
                        deployment.HybridPolicy(export, meta, 'virtual_leg_force', expected_control_hz=300-hz)
                    delayed = core.metadata('virtual_leg_force', self.scan, hz, action_delay_ms=10)
                    delayed['export_sha256'] = core.digest_file(export)
                    meta.write_text(json.dumps(delayed))
                    with self.assertRaisesRegex(ValueError, 'zero-delay'):
                        deployment.HybridPolicy(export, meta, 'virtual_leg_force')
                    contract.pop('control_hz'); meta.write_text(json.dumps(contract))
                    with self.assertRaises(ValueError):
                        deployment.HybridPolicy(export, meta, 'virtual_leg_force')


if __name__ == '__main__':
    unittest.main()
