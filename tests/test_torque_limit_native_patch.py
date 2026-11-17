"""Fail-closed patch scope checks; parent experiment validates compiled behavior."""
import hashlib
from pathlib import Path
import unittest

from experiments.torque_limit_ablation.native_patch import FROZEN_SOURCE_SHA256, patch_legwheel_source


SOURCE = Path(__file__).resolve().parents[1] / 'experiments/native_eval/cpp/legwheel_task.cpp'


class NativeNormalizationPatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = SOURCE.read_text(encoding='utf-8')

    def test_scope_preserves_physics_and_execution_limits(self):
        patched = patch_legwheel_source(self.original)
        # These complete sections must remain byte-for-byte the reviewed code.
        sections = (
            ('    const auto hips = named_ids<2>', '  std::string name() const override'),
            ('    for (int step = 0; step < substeps;', '      for (int i = 0; i < 4; ++i) {\n        const double actual'),
            ('  GeometryState geometry_state(', '  double hold_position('),
            ('  double get_reward(', '  std::array<LegGeometry, 2> legs_;'),
        )
        for begin, end in sections:
            with self.subTest(section=begin):
                original_section = self.original.split(begin, 1)[1].split(end, 1)[0]
                patched_section = patched.split(begin, 1)[1].split(end, 1)[0]
                self.assertEqual(original_section, patched_section)
        self.assertEqual(patched.count('/ norm_scales_[i]'), 2)
        self.assertNotIn('/ torque_scales_[i]', patched)
        self.assertEqual(SOURCE.read_text(encoding='utf-8'), self.original)

    def test_optional_vector_defaults_to_old_scales_and_checks_values(self):
        patched = patch_legwheel_source(self.original)
        self.assertIn('norm_scales_(p.vectors.count("torque_normalization_scales")', patched)
        self.assertIn('? vector_value<4>(p, "torque_normalization_scales")\n                       : torque_scales_)', patched)
        self.assertIn('std::isfinite(value) && value > 0', patched)
        self.assertIn('"torque_scales", "torque_normalization_scales",', patched)
        # The existing vector reader still rejects wrong dimensions/nonfinite values.
        self.assertIn('found->second.size() != N', patched)
        self.assertIn('if (!std::isfinite(out[i])) invalid("nonfinite config vector: " + key);', patched)

    def test_unknown_source_and_repeat_application_fail(self):
        for source in (self.original+'\n', self.original.replace('std::clamp(raw,', 'std::clamp(raw * 2,', 1)):
            with self.subTest(source_end=source[-20:]):
                with self.assertRaisesRegex(ValueError, 'differs from the frozen'):
                    patch_legwheel_source(source)
        with self.assertRaisesRegex(ValueError, 'already applied'):
            patch_legwheel_source(patch_legwheel_source(self.original))

    def test_crlf_is_normalized_for_the_frozen_identity(self):
        self.assertEqual(hashlib.sha256(self.original.encode('utf-8')).hexdigest(), FROZEN_SOURCE_SHA256)
        self.assertEqual(patch_legwheel_source(self.original.replace('\n', '\r\n')),
                         patch_legwheel_source(self.original))

    def test_source_must_be_text(self):
        with self.assertRaises(TypeError):
            patch_legwheel_source(self.original.encode('utf-8'))
