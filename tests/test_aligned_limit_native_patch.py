"""Isolated source-transform scope; compiled numerical tests run in WSL separately."""
from pathlib import Path
import unittest

from experiments.aligned_limit_ablation.native_patch import (
    FROZEN_SOURCE_SHA256, LEG_ANCHOR_RATIO, patch_sources,
)


ROOT = Path(__file__).resolve().parents[1] / 'experiments/native_eval'


class AlignedNativePatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = {path: (ROOT/path).read_text(encoding='utf-8') for path in FROZEN_SOURCE_SHA256}

    def test_patch_keeps_legacy_wheel_and_dense_math(self):
        patched = patch_sources(self.original)
        old, new = self.original['cpp/policy.cpp'], patched['cpp/policy.cpp']
        dense_begin, dense_end = '    for (const auto& layer : spec_.layers) {\n      next.resize', '    action.resize(action_size());'
        old_dense = old.split(dense_begin, 1)[1].split(dense_end, 1)[0]
        new_dense = new.split(dense_begin, 1)[1].split('    std::vector<double> anchor_mean;', 1)[0]
        self.assertEqual(old_dense, new_dense)
        tail = '      const int index = spec_.skip_indices[output];'
        self.assertEqual(old.split(tail, 1)[1].split(' private:', 1)[0],
                         new.split(tail, 1)[1].split(' private:', 1)[0])
        self.assertIn('if (anchor_ && output < 2)', new)
        self.assertIn('(value - anchor_mu)', new)
        self.assertEqual(self.original, {path: (ROOT/path).read_text(encoding='utf-8') for path in FROZEN_SOURCE_SHA256})

    def test_policy_force_clip_precedes_jacobian_and_motor_clip(self):
        task = patch_sources(self.original)['cpp/legwheel_task.cpp']
        force = task.index('const double force = std::clamp(clipped[i] * policy_leg_force_scales_[i]')
        mapping = task.index('raw = latched ? 0.0 : j * force;')
        motor = task.index('std::clamp(raw, -torque_scales_[i], torque_scales_[i])')
        self.assertLess(force, mapping)
        self.assertLess(mapping, motor)
        self.assertIn('double raw = latched ? 0.0 : gain * clipped[i];', task)
        self.assertIn('if (has_policy_leg_force_scales_ && i < 2)', task)
        self.assertIn('policy_leg_force_scales requires virtual_leg_force mode', task)
        self.assertIn('data->qfrc_actuator[actual_v_[i]] / norm_scales_[i]', task)
        self.assertIn('current[i] = sent_sum[i] / substeps / norm_scales_[i]', task)
        # Motor permission guards still use execution scales, not policy/normalization scales.
        begin, end = '    const auto hips = named_ids<2>', '  std::string name() const override'
        self.assertEqual(self.original['cpp/legwheel_task.cpp'].split(begin, 1)[1].split(end, 1)[0],
                         task.split(begin, 1)[1].split(end, 1)[0])

    def test_anchor_contract_and_recursion_guards_are_generated(self):
        patched = patch_sources(self.original)
        policy, binding, header = (patched[path] for path in (
            'cpp/policy.cpp', 'cpp/bindings.cpp', 'include/native_eval/runtime.hpp'))
        self.assertEqual(LEG_ANCHOR_RATIO, 0.4999995231628418)
        self.assertIn('float leg_anchor_ratio = 0.4999995231628418f;', header)
        self.assertIn('std::shared_ptr<MlpSpec> leg_anchor;', header)
        self.assertIn('spec_.mean.size() != 24 || spec_.action_size != 4', policy)
        self.assertIn('anchor_spec.mean != spec_.mean || anchor_spec.stddev != spec_.stddev', policy)
        self.assertIn('spec_.leg_anchor_ratio <= 0.5f', policy)
        self.assertIn('anchor_->spec_.skip_indices[leg] != -1', policy)
        self.assertIn('anchor_->spec_.network_scales[leg] != 1.0f', policy)
        self.assertIn('if (!allow_leg_anchor && mapping.contains("leg_anchor"))', binding)
        self.assertIn('parse_policy(as_dict(mapping["leg_anchor"], "leg_anchor"), false)', binding)
        self.assertIn('leg_anchor_ratio requires leg_anchor', binding)

    def test_unknown_or_already_patched_sources_fail_closed(self):
        patched = patch_sources(self.original)
        with self.assertRaisesRegex(ValueError, 'already applied'):
            patch_sources(patched)
        for path in self.original:
            bad = dict(self.original, **{path: self.original[path]+'\n'})
            with self.subTest(path=path):
                with self.assertRaisesRegex(ValueError, 'differs from its frozen'):
                    patch_sources(bad)

    def test_crlf_and_additional_unmodified_files(self):
        original = {path: text.replace('\n', '\r\n') for path, text in self.original.items()}
        original['cpp/runtime.cpp'] = 'not owned by this patch\r\n'
        patched = patch_sources(original)
        expected = patch_sources(self.original)
        for path in FROZEN_SOURCE_SHA256:
            self.assertEqual(patched[path], expected[path])
        self.assertEqual(patched['cpp/runtime.cpp'], original['cpp/runtime.cpp'])

    def test_missing_and_non_text_source_rejected(self):
        bad = dict(self.original)
        bad.pop('cpp/policy.cpp')
        with self.assertRaisesRegex(ValueError, 'Missing frozen'):
            patch_sources(bad)
        bad['cpp/policy.cpp'] = b'not text'
        with self.assertRaises(TypeError):
            patch_sources(bad)
