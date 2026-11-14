"""Dependency-free checks for the opt-in hybrid controller contract.

These tests exercise source kinematics and fail-closed host-side guards only.
They do not import JAX/MuJoCo, train a policy, or certify physical safety.
"""

import copy
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import sys
import tempfile
from types import ModuleType
import unittest
from unittest import mock
import xml.etree.ElementTree as ET

TORQUE_ROOT = Path(__file__).resolve().parents[1] / "experiments" / "torque200"
sys.path.insert(0, str(TORQUE_ROOT))

from hybrid import core, deployment, train, validate  # noqa: E402


class GeometryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.legs = core.read_geometry()

    def test_source_geometry_has_left_right_legs_and_unit_axes(self):
        self.assertEqual(tuple(leg.side for leg in self.legs), ("left", "right"))
        for leg in self.legs:
            self.assertEqual(len(leg.poly), 5)
            self.assertAlmostEqual(core.dot(leg.knee_axis, leg.knee_axis), 1.0)

    def test_constrained_jacobian_matches_length_finite_difference(self):
        lo, hi = core.HIP_RANGE
        for leg in self.legs:
            for n in range(11):
                q = lo + (hi - lo) * n / 10
                length, jacobian, partial = leg.evaluate(q)
                self.assertGreater(length, 0)
                self.assertTrue(all(math.isfinite(x) for x in (length, jacobian, partial)))
                for eps in (1e-6, 1e-5, 1e-4):
                    with self.subTest(side=leg.side, q=q, eps=eps):
                        finite_difference = (leg.evaluate(q + eps)[0] - leg.evaluate(q - eps)[0]) / (2 * eps)
                        self.assertAlmostEqual(jacobian, finite_difference, delta=2e-7)

    def test_negative_jacobian_means_increasing_hip_shortens_both_legs(self):
        lo, hi = core.HIP_RANGE
        for leg in self.legs:
            lengths = []
            for n in range(21):
                q = lo + (hi - lo) * n / 20
                length, jacobian, partial = leg.evaluate(q)
                self.assertLess(jacobian, 0)
                self.assertGreater(partial, 0)
                self.assertLess(leg.slope(q), 0)
                lengths.append(length)
            self.assertTrue(all(a > b for a, b in zip(lengths, lengths[1:])))

    def test_actual_knee_is_used_without_projecting_soft_equality(self):
        eps = 1e-6
        for leg in self.legs:
            qh = -0.23
            ideal = leg.coupled_knee(qh)
            for residual in (-0.004, 0.004):
                qk = ideal + residual
                length, jacobian, partial = leg.evaluate(qh, qk)
                with self.subTest(side=leg.side, residual=residual):
                    self.assertNotAlmostEqual(length, leg.evaluate(qh)[0], places=8)
                    fd = (leg.evaluate(qh, qk + eps)[0] - leg.evaluate(qh, qk - eps)[0]) / (2 * eps)
                    self.assertAlmostEqual(partial, fd, delta=1e-9)
                    self.assertAlmostEqual(jacobian, partial * leg.slope(qh), delta=1e-12)
                    # Holding the actual knee angle fixed makes length invariant
                    # to the hip rigid rotation, not to its coupled derivative.
                    self.assertEqual(length, leg.evaluate(qh + 0.01, qk)[0])

    def test_polynomial_uses_reference_offsets_and_full_derivative(self):
        leg = replace(self.legs[0], hip_ref=0.4, knee_ref=-0.7,
                      poly=(0.2, -1.3, 0.5, -0.2, 0.8))
        self.assertAlmostEqual(leg.coupled_knee(leg.hip_ref), leg.knee_ref + 0.2)
        for q in (-0.3, 0.4, 0.9):
            x = q - leg.hip_ref
            expected = leg.knee_ref + sum(a * x**i for i, a in enumerate(leg.poly))
            self.assertAlmostEqual(leg.coupled_knee(q), expected)
            fd = (leg.coupled_knee(q + 1e-6) - leg.coupled_knee(q - 1e-6)) / 2e-6
            self.assertAlmostEqual(leg.slope(q), fd, delta=1e-8)

    def test_scan_produces_consistent_finite_margin_and_force_scales(self):
        scan = core.scan_geometry(self.legs, samples=65)
        self.assertEqual([row["side"] for row in scan], ["left", "right"])
        for i, row in enumerate(scan):
            self.assertGreater(row["length_min_m"], 0)
            self.assertGreater(row["length_max_m"], row["length_min_m"])
            self.assertLess(row["jacobian_max"], 0)
            self.assertGreater(row["jmax_bound"], abs(row["jacobian_min"]))
            self.assertGreater(row["minimum_ratio"], 0.05)
            self.assertLess(row["finite_difference_max_error"], 2e-7)
            self.assertAlmostEqual(row["force_scale_n"] * row["jmax_bound"],
                                   0.9 * core.TORQUE_LIMITS[i])

    def test_scan_rejects_invalid_range_and_sample_count(self):
        for hip_range, samples in (((0, 0), 5), ((1, -1), 5),
                                   ((math.nan, 1), 5), ((0, math.inf), 5),
                                   (core.HIP_RANGE, 2)):
            with self.subTest(hip_range=hip_range, samples=samples):
                with self.assertRaises(ValueError):
                    core.scan_geometry(self.legs, hip_range, samples)

    def test_scan_rejects_zero_jacobian_sign_change_and_deadpoint_band(self):
        class SyntheticLeg:
            side = "left"

            def __init__(self, jacobian):
                self.jacobian = jacobian

            def evaluate(self, q):
                return 0.1, self.jacobian(q), 0.1

        for jacobian in (lambda q: 0.0, lambda q: q, lambda q: 0.001 + q**2):
            with self.subTest(jacobian=jacobian):
                with self.assertRaises(ValueError):
                    core.scan_geometry((SyntheticLeg(jacobian),), (-1, 1), 65)

    def test_model_edits_invalidating_reduced_fk_fail_closed(self):
        changes = (
            ("compiler", "angle", "degree"),
            ("compiler", "eulerseq", "xyz"),
            (".//body[@name='calf_left']", "quat", "1 0 0 0"),
            (".//joint[@name='thigh_left_joint']", "pos", "0.001 0 0"),
            (".//site[@name='wheel_left_site']", "pos", "0 0.001 0"),
            (".//joint[@name='calf_left_joint']", "type", "slide"),
            (".//joint[@name='calf_left_joint']", "axis", "0 0 0"),
            ("equality/joint", "active", "false"),
            ("equality/joint", "polycoef", "0 1 0 0"),
        )
        with tempfile.TemporaryDirectory() as directory:
            assets = Path(directory) / "assets"
            shutil.copytree(core.XML.parent, assets)
            path = assets / "modified.xml"
            for xpath, name, value in changes:
                with self.subTest(xpath=xpath, name=name, value=value):
                    tree = ET.parse(core.XML)
                    node = tree.find(xpath)
                    self.assertIsNotNone(node)
                    node.set(name, value)
                    tree.write(path)
                    with self.assertRaises(ValueError):
                        core.read_geometry(path)


class MappingTest(unittest.TestCase):
    def test_zero_action_has_no_force_bias_or_wheel_reaction(self):
        for mode in core.MODES:
            self.assertEqual(core.map_action((0, 0, 0, 0), (-0.1, -0.08), (9, 8), mode), (0, 0, 0, 0))
            self.assertEqual(core.map_action((0, 0, 1, -1), (-0.1, -0.08), (9, 8), mode),
                             (0, 0, core.TORQUE_LIMITS[2], -core.TORQUE_LIMITS[3]))

    def test_virtual_force_is_j_times_force_not_force_divided_by_j(self):
        action = (0.4, -0.7, 0.25, -0.5)
        jac = (-0.03, 0.06)
        scales = (2.0, 3.0)
        got = core.map_action(action, jac, scales, "virtual_leg_force")
        expected = (jac[0] * scales[0] * action[0], jac[1] * scales[1] * action[1],
                    core.TORQUE_LIMITS[2] * action[2], core.TORQUE_LIMITS[3] * action[3])
        for actual, wanted in zip(got, expected):
            self.assertAlmostEqual(actual, wanted)

    def test_deadpoint_and_tiny_jacobian_do_not_divide_or_amplify(self):
        for jac in ((0.0, 0.0), (-1e-15, 1e-15), (-1e-300, 1e-300)):
            for mode in ("virtual_leg_force", "matched_envelope"):
                with self.subTest(jac=jac, mode=mode):
                    got = core.map_action((1, 1, 0, 0), jac, (9, 8), mode)
                    self.assertTrue(all(math.isfinite(t) for t in got))
                    self.assertLessEqual(abs(got[0]), abs(jac[0]) * 9)
                    self.assertLessEqual(abs(got[1]), abs(jac[1]) * 8)
                    self.assertEqual(got[2:], (0, 0))

    def test_matched_envelope_is_equivalent_under_leg_sign_transform(self):
        rng = random.Random(1024)
        for _ in range(50):
            action = tuple(rng.uniform(-3, 3) for _ in range(4))
            jac = tuple(rng.uniform(-0.3, 0.3) for _ in range(2))
            scales = (9, 11)
            signed = tuple(action[i] * (1 if jac[i] >= 0 else -1) for i in range(2)) + action[2:]
            force = core.map_action(action, jac, scales, "virtual_leg_force")
            matched = core.map_action(signed, jac, scales, "matched_envelope")
            self.assertEqual(force, matched)

    def test_action_and_final_torques_are_bounded_for_every_mode(self):
        rng = random.Random(2048)
        for mode in core.MODES:
            for _ in range(50):
                action = tuple(rng.uniform(-5, 5) for _ in range(4))
                got = core.map_action(action, (-0.8, 0.4), (100, 100), mode)
                self.assertEqual(len(got), 4)
                self.assertTrue(all(abs(t) <= limit for t, limit in zip(got, core.TORQUE_LIMITS)))
                clipped = tuple(max(-1, min(1, a)) for a in action)
                self.assertEqual(got, core.map_action(clipped, (-0.8, 0.4), (100, 100), mode))

    def test_direct_torque_ignores_geometry_and_force_scale(self):
        action = (0.4, -0.7, 0.25, -0.5)
        expected = tuple(a * limit for a, limit in zip(action, core.TORQUE_LIMITS))
        for jac, scales in (((-0.1, -0.2), (9, 8)), ((0, 0), (2, 3))):
            self.assertEqual(core.map_action(action, jac, scales, "direct_torque"), expected)

    def test_invalid_mode_dimensions_and_nonpositive_calibration_are_rejected(self):
        valid = dict(action=(0, 0, 0, 0), jacobian=(-0.1, -0.1),
                     force_scales=(9, 9), mode="virtual_leg_force", limits=core.TORQUE_LIMITS)
        invalid = (
            ("mode", "legacy"), ("action", (0, 0, 0)), ("action", (0,) * 5),
            ("jacobian", (0,)), ("jacobian", (0,) * 3),
            ("force_scales", (9,)), ("force_scales", (9,) * 3),
            ("limits", (1, 1, 1)), ("limits", (1,) * 5),
            ("force_scales", (0, 9)), ("force_scales", (-1, 9)),
            ("limits", (0, 1, 1, 1)), ("limits", (-1, 1, 1, 1)),
        )
        for name, value in invalid:
            with self.subTest(name=name, value=value):
                with self.assertRaises(ValueError):
                    core.map_action(**{**valid, name: value})

    def test_nonfinite_control_inputs_are_rejected(self):
        valid = dict(action=(0, 0, 0, 0), jacobian=(-0.1, -0.1),
                     force_scales=(9, 9), mode="virtual_leg_force", limits=core.TORQUE_LIMITS)
        for name in ("action", "jacobian", "force_scales", "limits"):
            for value in (math.nan, math.inf, -math.inf):
                changed = (value,) + valid[name][1:]
                with self.subTest(name=name, value=value):
                    with self.assertRaises(ValueError):
                        core.map_action(**{**valid, name: changed})


class MetadataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scan = core.scan_geometry(core.read_geometry(), samples=65)

    def contract(self, mode="virtual_leg_force"):
        with mock.patch.object(core, "source_fingerprint", return_value="test-source-sha256"):
            return core.metadata(mode, self.scan)

    def test_metadata_describes_independent_four_action_hybrid(self):
        actual = self.contract()
        self.assertEqual(actual["action_order"], ["left_leg", "right_leg", "left_wheel", "right_wheel"])
        self.assertEqual(actual["actor_dim"], 24)
        self.assertEqual(actual["critic_dim"], 51)
        self.assertEqual(actual["ctrl_dt"], 0.005)
        self.assertEqual(actual["sim_dt"], 0.0025)
        self.assertEqual(actual["force_bias_n"], [0.0, 0.0])
        self.assertEqual(actual["force_scales_n"], [r["force_scale_n"] for r in self.scan])
        self.assertEqual(actual["torque_limits_nm"], list(core.TORQUE_LIMITS))
        self.assertEqual(actual["source_fingerprint"], "test-source-sha256")

    def test_every_expected_metadata_field_is_required_and_exact(self):
        expected = self.contract()
        core.require_metadata(copy.deepcopy(expected), expected)
        core.require_metadata({**expected, "extra_diagnostic": "allowed"}, expected)
        for key in expected:
            with self.subTest(key=key):
                absent = {k: v for k, v in expected.items() if k != key}
                with self.assertRaisesRegex(ValueError, key):
                    core.require_metadata(absent, expected)
                with self.assertRaisesRegex(ValueError, key):
                    core.require_metadata({**expected, key: None}, expected)

    def test_action_modes_and_legacy_metadata_cannot_cross_load(self):
        for actual_mode in core.MODES:
            for expected_mode in core.MODES:
                if actual_mode != expected_mode:
                    with self.subTest(actual=actual_mode, expected=expected_mode):
                        with self.assertRaisesRegex(ValueError, "action_mode"):
                            core.require_metadata(self.contract(actual_mode), self.contract(expected_mode))
        with self.assertRaises(ValueError):
            core.require_metadata({"actor_dim": 24}, self.contract())
        with self.assertRaises(ValueError):
            self.contract("legacy100Hz")

    def test_json_writer_replaces_complete_document_and_rejects_nan(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "metadata.json"
            core.write_json(path, {"complete": True})
            self.assertEqual(json.loads(path.read_text()), {"complete": True})
            core.write_json(path, {"replacement": [1, 2]})
            self.assertEqual(json.loads(path.read_text()), {"replacement": [1, 2]})
            self.assertFalse(path.with_name(path.name + ".tmp").exists())
            with self.assertRaises(ValueError):
                core.write_json(path, {"bad": math.nan})
            self.assertEqual(json.loads(path.read_text()), {"replacement": [1, 2]})


class TrainingGateTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.validation_path = self.root / "validation.json"

    def write_report(self, report):
        self.validation_path.write_text(json.dumps(report), encoding="utf8")

    def test_static_audit_is_explicitly_not_runtime_validation(self):
        report = validate.static_audit()
        self.assertIs(report["runtime_validated"], False)
        self.assertEqual(report["schema"], "hybrid-local-validation-v1")
        self.write_report(report)
        with self.assertRaises(ValueError):
            train.check_validation(self.validation_path)

    def test_only_successful_matching_runtime_report_passes_gate(self):
        valid = dict(schema="hybrid-local-validation-v1", runtime_validated=True,
                     source_fingerprint="current-source")
        with mock.patch.object(train, "source_fingerprint", return_value="current-source"):
            self.write_report(valid)
            self.assertEqual(train.check_validation(self.validation_path), valid)
            for changes in ({"runtime_validated": False}, {"runtime_validated": 1},
                            {"runtime_validated": "true"}, {"source_fingerprint": "stale"},
                            {"schema": "legacy"}):
                with self.subTest(changes=changes):
                    self.write_report({**valid, **changes})
                    with self.assertRaises(ValueError):
                        train.check_validation(self.validation_path)
            for key in valid:
                with self.subTest(missing=key):
                    self.write_report({k: v for k, v in valid.items() if k != key})
                    with self.assertRaises(ValueError):
                        train.check_validation(self.validation_path)

    def test_static_report_cannot_create_output_or_start_training(self):
        self.write_report(dict(schema="hybrid-local-validation-v1", runtime_validated=False,
                               source_fingerprint="current-source"))
        output = self.root / "must-not-exist"
        argv = ["hybrid.train", "--validation", str(self.validation_path), "--output", str(output)]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(train, "run") as run:
            with self.assertRaises(ValueError):
                train.main()
            run.assert_not_called()
        self.assertFalse(output.exists())

    def test_nonpositive_training_counts_are_rejected_before_validation(self):
        for flag in ("--steps", "--num-envs", "--eval-envs", "--unroll-length"):
            for count in ("0", "-1"):
                argv = ["hybrid.train", "--validation", str(self.validation_path),
                        "--output", str(self.root / "output"), flag, count]
                with self.subTest(flag=flag, count=count), mock.patch.object(sys, "argv", argv), \
                        mock.patch.object(train, "check_validation") as check, \
                        mock.patch.object(train, "run") as run:
                    with self.assertRaises(ValueError):
                        train.main()
                    check.assert_not_called()
                    run.assert_not_called()


class ArtifactBindingTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.checkpoint = self.root / "run" / "checkpoints" / "000016"
        self.checkpoint.mkdir(parents=True)
        (self.checkpoint / "params.bin").write_bytes(b"synthetic test parameters")
        (self.checkpoint / "nested").mkdir()
        (self.checkpoint / "nested" / "metadata.json").write_text('{"version": 1}')
        self.expected = {"schema": core.SCHEMA, "action_mode": "virtual_leg_force",
                         "observation_version": core.OBS_VERSION, "source_fingerprint": "source"}
        self.metadata_path = self.checkpoint.parent.parent / "policy_metadata.json"
        self.metadata_path.write_text(json.dumps(self.expected))
        self.report = dict(checkpoint_validated=True, checkpoint_name=self.checkpoint.name,
                           checkpoint_sha256=core.digest_tree(self.checkpoint))
        self.report_path = self.checkpoint.parent.parent / "report.json"
        self.report_path.write_text(json.dumps(self.report))

    def test_file_digest_hashes_exact_bytes(self):
        path = self.root / "export.txt"
        contents = b"test export\x00\xff\n"
        path.write_bytes(contents)
        self.assertEqual(core.digest_file(path), hashlib.sha256(contents).hexdigest())

    def test_checkpoint_digest_is_portable_but_binds_content_and_filenames(self):
        original = core.digest_tree(self.checkpoint)
        moved = self.root / "copy"
        shutil.copytree(self.checkpoint, moved)
        self.assertEqual(core.digest_tree(moved), original)
        (moved / "params.bin").rename(moved / "renamed.bin")
        self.assertNotEqual(core.digest_tree(moved), original)
        (moved / "renamed.bin").rename(moved / "params.bin")
        self.assertEqual(core.digest_tree(moved), original)
        (moved / "nested" / "metadata.json").write_text('{"version": 2}')
        self.assertNotEqual(core.digest_tree(moved), original)

    def test_checkpoint_digest_rejects_empty_missing_file_and_link_targets(self):
        empty = self.root / "empty"
        empty.mkdir()
        for path in (empty, self.root / "missing", self.checkpoint / "params.bin"):
            with self.subTest(path=path.name):
                with self.assertRaises(ValueError):
                    core.digest_tree(path)
        link = self.root / "link"
        try:
            link.symlink_to(self.checkpoint, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("Filesystem does not permit symlinks")
        with self.assertRaises(ValueError):
            core.digest_tree(link)
        (self.checkpoint / "nested-link").symlink_to(self.checkpoint / "params.bin")
        with self.assertRaises(ValueError):
            core.digest_tree(self.checkpoint)

    def test_verified_checkpoint_restores_after_portable_copy(self):
        train.restore_guard(self.checkpoint, self.expected)
        relocated = self.root / "relocated-run"
        shutil.copytree(self.checkpoint.parent.parent, relocated)
        train.restore_guard(relocated / "checkpoints" / self.checkpoint.name, self.expected)

    def test_restore_rejects_incomplete_unbound_and_different_checkpoint(self):
        invalid = ({"checkpoint_validated": False}, {"checkpoint_validated": 1},
                   {"checkpoint_name": "000017"}, {"checkpoint_sha256": "changed"})
        for change in invalid:
            with self.subTest(change=change):
                self.report_path.write_text(json.dumps({**self.report, **change}))
                with self.assertRaises(ValueError):
                    train.restore_guard(self.checkpoint, self.expected)
        for key in self.report:
            with self.subTest(missing=key):
                self.report_path.write_text(json.dumps({k: v for k, v in self.report.items() if k != key}))
                with self.assertRaises(ValueError):
                    train.restore_guard(self.checkpoint, self.expected)

    def test_restore_rejects_metadata_mismatch_before_checkpoint_use(self):
        for key in self.expected:
            with self.subTest(key=key):
                self.metadata_path.write_text(json.dumps({**self.expected, key: "incompatible"}))
                with self.assertRaisesRegex(ValueError, key):
                    train.restore_guard(self.checkpoint, self.expected)

    def test_restore_rejects_modified_checkpoint_bytes(self):
        (self.checkpoint / "params.bin").write_bytes(b"different synthetic parameters")
        with self.assertRaises(ValueError):
            train.restore_guard(self.checkpoint, self.expected)


class DeploymentContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.legs = core.read_geometry()
        cls.scan = core.scan_geometry(cls.legs, samples=65)

    def policy_without_network(self, mode="virtual_leg_force"):
        # This isolates the public torque adapter without constructing the
        # NumPy network or suggesting that a real export has been validated.
        policy = object.__new__(deployment.HybridPolicy)
        policy.legs = self.legs
        policy.scan = self.scan
        policy.mode = mode
        policy.meta = {"force_scales_n": [row["force_scale_n"] for row in self.scan]}
        return policy

    def test_adapter_maps_fresh_joint_positions_in_each_mode(self):
        qh = (-0.3, -0.1)
        qk = tuple(leg.coupled_knee(q) for leg, q in zip(self.legs, qh))
        action = (0.3, -0.2, 0.1, -0.1)
        jac = [leg.evaluate(qh[i], qk[i])[1] for i, leg in enumerate(self.legs)]
        for mode in core.MODES:
            policy = self.policy_without_network(mode)
            self.assertEqual(policy.torque(action, qh, qk, (0, 0), (0, 0)),
                             core.map_action(action, jac, policy.meta["force_scales_n"], mode))

    def test_adapter_rejects_bad_joint_dimensions_nonfinite_and_out_of_range(self):
        policy = self.policy_without_network()
        qh = (-0.2, -0.2)
        qk = tuple(leg.coupled_knee(q) for leg, q in zip(self.legs, qh))
        invalid = (((0,), qk), (qh, (0,)), ((math.nan, qh[1]), qk),
                   (qh, (math.inf, qk[1])), ((core.HIP_RANGE[0] - 1e-6, qh[1]), qk),
                   ((qh[0], core.HIP_RANGE[1] + 1e-6), qk))
        for hips, knees in invalid:
            with self.subTest(hips=hips, knees=knees):
                with self.assertRaises(ValueError):
                    policy.torque((0, 0, 0, 0), hips, knees, (0, 0), (0, 0))

    def test_adapter_rejects_excessive_position_residual_in_every_mode(self):
        hips = (-0.2, -0.2)
        knees = tuple(leg.coupled_knee(q) for leg, q in zip(self.legs, hips))
        for mode in core.MODES:
            for sign in (-1, 1):
                invalid = (knees[0] + sign * (core.MAX_EQ_POSITION + 1e-6), knees[1])
                with self.subTest(mode=mode, sign=sign):
                    with self.assertRaisesRegex(ValueError, "residual"):
                        self.policy_without_network(mode).torque((0, 0, 0, 0), hips, invalid, (0, 0), (0, 0))

    def test_adapter_validates_velocity_dimensions_finiteness_and_residual(self):
        hips = (-0.2, -0.2)
        knees = tuple(leg.coupled_knee(q) for leg, q in zip(self.legs, hips))
        invalid = (((0,), (0, 0)), ((0, 0), (0,)),
                   ((math.nan, 0), (0, 0)), ((0, 0), (math.inf, 0)),
                   ((0, 0), (100, 0)), ((100, 0), (0, 0)))
        for mode in core.MODES:
            policy = self.policy_without_network(mode)
            for hip_v, knee_v in invalid:
                with self.subTest(mode=mode, hip_v=hip_v, knee_v=knee_v):
                    with self.assertRaises(ValueError):
                        policy.torque((0, 0, 0, 0), hips, knees, hip_v, knee_v)
            hip_v = (0.2, -0.3)
            knee_v = tuple(leg.slope(q) * velocity for leg, q, velocity in zip(self.legs, hips, hip_v))
            self.assertEqual(policy.torque((0, 0, 0, 0), hips, knees, hip_v, knee_v), (0, 0, 0, 0))

    def test_adapter_rejects_metadata_and_export_mismatch_before_network_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            export = root / "policy.cpp.txt"
            export.write_text("synthetic export bytes, not a network")
            metadata_path = root / "policy_metadata.json"
            with mock.patch.object(core, "source_fingerprint", return_value="test-source"), \
                    mock.patch.object(deployment, "scan_geometry", return_value=self.scan):
                valid = core.metadata("virtual_leg_force", self.scan)
                valid["export_sha256"] = core.digest_file(export)
                invalid = ({"action_mode": "direct_torque"}, {"source_fingerprint": "stale"},
                           {"export_sha256": "not-this-export"}, {"export_sha256": None})
                # If guard ordering regresses, a prohibited kernel import raises
                # ModuleNotFoundError instead of the expected contract error.
                with mock.patch.dict(sys.modules, {"legwheel.deployment.numpy_policy": None}):
                    for change in invalid:
                        with self.subTest(change=change):
                            metadata_path.write_text(json.dumps({**valid, **change}))
                            with self.assertRaises(ValueError):
                                deployment.HybridPolicy(export, metadata_path, "virtual_leg_force")
                    metadata_path.write_text(json.dumps(valid))
                    export.write_text("modified export bytes")
                    with self.assertRaises(ValueError):
                        deployment.HybridPolicy(export, metadata_path, "virtual_leg_force")

    def test_matching_export_constructs_kernel_only_after_contract_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            export = root / "policy.cpp.txt"
            export.write_text("synthetic test export")
            metadata_path = root / "policy_metadata.json"
            kernel = ModuleType("legwheel.deployment.numpy_policy")
            kernel.RL_Policy = mock.Mock()
            with mock.patch.object(core, "source_fingerprint", return_value="test-source"), \
                    mock.patch.object(deployment, "scan_geometry", return_value=self.scan), \
                    mock.patch.dict(sys.modules, {kernel.__name__: kernel}):
                contract = core.metadata("virtual_leg_force", self.scan)
                contract["export_sha256"] = core.digest_file(export)
                metadata_path.write_text(json.dumps(contract))
                policy = deployment.HybridPolicy(export, metadata_path, "virtual_leg_force")
                kernel.RL_Policy.assert_called_once_with(path=export)
                self.assertEqual(policy.mode, "virtual_leg_force")

    def test_adapter_clears_history_before_each_single_frame_action(self):
        policy = self.policy_without_network()
        policy.policy = mock.Mock()
        policy.policy.run.return_value = (0.1, 0.2, 0.3, 0.4)
        observation = [0] * 24
        self.assertEqual(policy.action(observation), (0.1, 0.2, 0.3, 0.4))
        self.assertEqual(policy.policy.mock_calls,
                         [mock.call.clean_history(), mock.call.run(observation)])


if __name__ == "__main__":
    unittest.main()
