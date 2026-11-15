"""Pure synthetic continuation metadata; no artifacts, clocks, or process access."""

import copy
import json
import math
import unittest
from uuid import UUID

from experiments.torque200.workbench_api.checkpoint_metadata import (
    MAX_SAFE_STEPS,
    MetadataError,
    calculate_progress,
    capture_checkpoint,
    make_run_context,
    validate_checkpoint,
    validate_run_context,
)


def identity(number):
    return str(UUID(int=number))


class CheckpointMetadataTests(unittest.TestCase):
    def context(self, run_id="run_001", **options):
        parameters = dict(run_id=run_id, mode="original", clock_policy_id="training-budget-v1",
                          context_id=identity(1))
        parameters.update(options)
        return make_run_context(**parameters)

    def parent(self, *, seconds=600, steps=524288):
        return capture_checkpoint(self.context(), elapsed_seconds=seconds, steps=steps, checkpoint_id=identity(2))

    def continuation(self, parent=None):
        return self.context("run_002", origin="checkpoint", parent=self.parent() if parent is None else parent,
                            context_id=identity(3))

    def progress(self, context, *, seconds=0, steps=0, budget=600):
        return calculate_progress(context, elapsed_seconds=seconds, steps=steps, duration_seconds=budget)

    def test_new_context_and_serialized_checkpoint_round_trip(self):
        context = self.context()
        original = copy.deepcopy(context)
        checkpoint = capture_checkpoint(context, elapsed_seconds=600, steps=524288, checkpoint_id=identity(2))
        self.assertEqual(context, original)
        self.assertEqual(checkpoint["baseTrainedSeconds"], 0)
        self.assertEqual(checkpoint["baseTrainedSteps"], 0)
        self.assertEqual(checkpoint["trainedSeconds"], 600)
        self.assertEqual(checkpoint["trainedSteps"], 524288)
        self.assertIsNone(checkpoint["parentCheckpointId"])
        self.assertEqual(validate_checkpoint(json.loads(json.dumps(checkpoint, allow_nan=False))), checkpoint)
        self.assertEqual(validate_run_context(json.loads(json.dumps(context)), run_id="run_001", mode="original"), context)
        self.assertNotIn("alias", checkpoint)

    def test_600_plus_600_starts_at_50_percent_then_55_percent(self):
        context = self.continuation()
        initial = self.progress(context)
        self.assertEqual(initial["baseTrainedSeconds"], 600)
        self.assertEqual(initial["baseTrainedSteps"], 524288)
        self.assertEqual(initial["targetTrainedSeconds"], 1200)
        self.assertEqual(initial["segmentPercent"], 0)
        self.assertEqual(initial["cumulativePercent"], 50)
        self.assertEqual(initial["remainingSeconds"], 600)
        progressed = self.progress(context, seconds=60, steps=81920)
        self.assertEqual(progressed["trainedSeconds"], 660)
        self.assertEqual(progressed["trainedSteps"], 606208)
        self.assertEqual(progressed["segmentPercent"], 10)
        self.assertAlmostEqual(progressed["cumulativePercent"], 55)
        self.assertEqual(progressed["sourceCheckpointId"], identity(2))
        self.assertEqual(progressed["contextId"], identity(3))

    def test_multiple_midrun_snapshots_use_one_fixed_baseline(self):
        context = self.continuation()
        first = capture_checkpoint(context, elapsed_seconds=60, steps=81920, checkpoint_id=identity(4))
        second = capture_checkpoint(context, elapsed_seconds=120, steps=163840, checkpoint_id=identity(5))
        self.assertEqual(first["trainedSeconds"], 660)
        self.assertEqual(second["trainedSeconds"], 720)
        self.assertEqual(second["trainedSteps"], 688128)
        self.assertEqual(first["parentCheckpointId"], identity(2))
        self.assertEqual(second["parentCheckpointId"], identity(2))
        self.assertEqual(second["baseTrainedSeconds"], 600)
        self.assertEqual(context["parentCheckpoint"]["trainedSeconds"], 600)

    def test_successive_runs_accumulate_once_and_keep_parent_lineage(self):
        second_run = self.continuation()
        second_checkpoint = capture_checkpoint(second_run, elapsed_seconds=60, steps=81920, checkpoint_id=identity(4))
        third_run = self.context("run_003", origin="checkpoint", parent=second_checkpoint, context_id=identity(5))
        third_checkpoint = capture_checkpoint(third_run, elapsed_seconds=30, steps=40960, checkpoint_id=identity(6))
        self.assertEqual(third_checkpoint["baseTrainedSeconds"], 660)
        self.assertEqual(third_checkpoint["trainedSeconds"], 690)
        self.assertEqual(third_checkpoint["trainedSteps"], 647168)
        self.assertEqual(third_checkpoint["parentCheckpointId"], second_checkpoint["id"])
        self.assertEqual(third_checkpoint["createdByRunId"], "run_003")

    def test_identical_paused_reported_counters_never_advance(self):
        context = self.continuation()
        before = copy.deepcopy(context)
        first = self.progress(context, seconds=60, steps=81920)
        for _ in range(20):
            self.assertEqual(self.progress(context, seconds=60, steps=81920), first)
        self.assertEqual(context, before)

    def test_unlimited_budget_preserves_known_totals_without_target_or_percent(self):
        progress = self.progress(self.continuation(), seconds=60, steps=81920, budget=0)
        self.assertEqual(progress["trainedSeconds"], 660)
        self.assertEqual(progress["trainedSteps"], 606208)
        for field in ("targetTrainedSeconds", "remainingSeconds", "segmentPercent", "cumulativePercent"):
            self.assertIsNone(progress[field])

    def test_legacy_or_absent_context_preserves_unknown_history_but_segment_is_known(self):
        for context in (None, self.context(origin="legacy")):
            progress = self.progress(context, seconds=60, steps=81920, budget=120)
            self.assertEqual(progress["segmentPercent"], 50)
            self.assertEqual(progress["remainingSeconds"], 60)
            for field in ("baseTrainedSeconds", "baseTrainedSteps", "trainedSeconds", "trainedSteps",
                          "targetTrainedSeconds", "cumulativePercent", "sourceCheckpointId"):
                self.assertIsNone(progress[field])
        checkpoint = capture_checkpoint(self.context(origin="legacy"), elapsed_seconds=60, steps=81920)
        self.assertIsNone(checkpoint["trainedSeconds"])
        self.assertIsNone(checkpoint["trainedSteps"])

    def test_partial_unknown_time_does_not_hide_known_step_history(self):
        parent = self.parent(seconds=None, steps=524288)
        self.assertEqual(validate_checkpoint(parent)["trainedSteps"], 524288)
        context = self.continuation(parent)
        progress = self.progress(context, seconds=60, steps=81920)
        self.assertIsNone(progress["trainedSeconds"])
        self.assertEqual(progress["trainedSteps"], 606208)
        self.assertEqual(progress["segmentPercent"], 10)
        self.assertIsNone(progress["cumulativePercent"])
        checkpoint = capture_checkpoint(context, elapsed_seconds=60, steps=81920)
        self.assertIsNone(checkpoint["trainedSeconds"])
        self.assertEqual(checkpoint["trainedSteps"], 606208)

    def test_partial_unknown_steps_does_not_hide_known_time_history(self):
        progress = self.progress(self.continuation(self.parent(seconds=600, steps=None)), seconds=60, steps=81920)
        self.assertEqual(progress["trainedSeconds"], 660)
        self.assertAlmostEqual(progress["cumulativePercent"], 55)
        self.assertIsNone(progress["trainedSteps"])

    def test_current_unknown_counters_and_unknown_budget_are_independent(self):
        context = self.continuation()
        progress = self.progress(context, seconds=None, steps=81920, budget=600)
        self.assertIsNone(progress["trainedSeconds"])
        self.assertEqual(progress["trainedSteps"], 606208)
        self.assertEqual(progress["targetTrainedSeconds"], 1200)
        self.assertIsNone(progress["remainingSeconds"])
        unknown_budget = self.progress(context, seconds=60, steps=None, budget=None)
        self.assertEqual(unknown_budget["trainedSeconds"], 660)
        self.assertIsNone(unknown_budget["trainedSteps"])
        self.assertIsNone(unknown_budget["targetTrainedSeconds"])

    def test_over_budget_progress_is_not_clamped(self):
        progress = self.progress(self.continuation(), seconds=700, steps=81920, budget=600)
        self.assertEqual(progress["trainedSeconds"], 1300)
        self.assertGreater(progress["segmentPercent"], 100)
        self.assertGreater(progress["cumulativePercent"], 100)
        self.assertEqual(progress["remainingSeconds"], 0)

    def test_derived_addition_and_division_overflow_are_null_with_warnings(self):
        context = self.continuation(self.parent(seconds=1e308, steps=MAX_SAFE_STEPS))
        progress = self.progress(context, seconds=1e308, steps=1, budget=1e308)
        self.assertIsNone(progress["trainedSeconds"])
        self.assertIsNone(progress["trainedSteps"])
        self.assertIsNone(progress["targetTrainedSeconds"])
        self.assertGreaterEqual(len(progress["warnings"]), 3)
        json.dumps(progress, allow_nan=False)
        extreme = self.progress(self.context(), seconds=1e308, steps=1, budget=1e-308)
        self.assertEqual(extreme["trainedSeconds"], 1e308)
        self.assertIsNone(extreme["segmentPercent"])
        self.assertIsNone(extreme["cumulativePercent"])
        self.assertTrue(extreme["warnings"])
        json.dumps(extreme, allow_nan=False)
        with self.assertRaises(MetadataError):
            capture_checkpoint(context, elapsed_seconds=1e308, steps=1)

    def test_rejects_invalid_numeric_inputs_instead_of_zero_filling(self):
        context = self.context()
        for invalid in (True, False, -1, math.nan, math.inf, -math.inf, "60"):
            with self.subTest(seconds=invalid), self.assertRaises(MetadataError):
                self.progress(context, seconds=invalid)
            with self.subTest(budget=invalid), self.assertRaises(MetadataError):
                self.progress(context, budget=invalid)
        for invalid in (True, False, -1, 1.5, 2.0, math.nan, math.inf, MAX_SAFE_STEPS + 1, "10"):
            with self.subTest(steps=invalid), self.assertRaises(MetadataError):
                capture_checkpoint(context, elapsed_seconds=0, steps=invalid)

    def test_rejects_inconsistent_or_fabricated_unknown_totals(self):
        for field, value in (("trainedSeconds", 601), ("trainedSteps", 524289), ("trainedSeconds", None)):
            checkpoint = self.parent()
            checkpoint[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(MetadataError):
                validate_checkpoint(checkpoint)
        unknown = self.parent(seconds=None)
        unknown["trainedSeconds"] = 0
        with self.assertRaises(MetadataError):
            validate_checkpoint(unknown)

    def test_fractional_seconds_allow_only_representation_roundoff(self):
        checkpoint = self.parent(seconds=0.1, steps=1)
        context = self.continuation(checkpoint)
        result = capture_checkpoint(context, elapsed_seconds=0.2, steps=1)
        result["trainedSeconds"] = 0.3
        self.assertEqual(validate_checkpoint(result)["trainedSeconds"], 0.3)
        result["trainedSeconds"] = 0.31
        with self.assertRaises(MetadataError):
            validate_checkpoint(result)

    def test_uuid_canonical_form_self_parent_and_generated_identities(self):
        context = self.context(context_id=None)
        self.assertEqual(str(UUID(context["id"])), context["id"])
        checkpoint = capture_checkpoint(context, elapsed_seconds=0, steps=0)
        self.assertNotEqual(checkpoint["id"], context["id"])
        for invalid in ("not-a-uuid", identity(12).replace("-", ""), str(UUID(int=0xABCDEF1234)).upper(), True):
            with self.subTest(identity=invalid), self.assertRaises(MetadataError):
                self.context(context_id=invalid)
        checkpoint["parentCheckpointId"] = checkpoint["id"]
        with self.assertRaises(MetadataError):
            validate_checkpoint(checkpoint)
        parent = self.parent()
        with self.assertRaises(MetadataError):
            capture_checkpoint(self.continuation(parent), elapsed_seconds=1, steps=1, checkpoint_id=parent["id"])

    def test_schema_run_and_policy_identities_are_validated(self):
        context = self.context()
        for version in (True, 0, 2, None):
            broken = {**context, "schemaVersion": version}
            with self.subTest(version=version), self.assertRaises(MetadataError):
                validate_run_context(broken)
        with self.assertRaises(MetadataError):
            validate_run_context(context, run_id="another_run")
        for run_id in ("../escape", "a/b", "a\\b", "", ".", "x:drive"):
            with self.subTest(run_id=run_id), self.assertRaises(MetadataError):
                self.context(run_id)
        for policy in ("", "a/b", "a b", "p" * 65, True):
            with self.subTest(policy=policy), self.assertRaises(MetadataError):
                self.context(clock_policy_id=policy)

    def test_nil_uuid_cannot_stand_for_persistent_identity_or_unknown_parent(self):
        with self.assertRaises(MetadataError):
            self.context(context_id=identity(0))
        with self.assertRaises(MetadataError):
            capture_checkpoint(self.context(), elapsed_seconds=0, steps=0, checkpoint_id=identity(0))
        checkpoint = self.parent()
        checkpoint["parentCheckpointId"] = identity(0)
        with self.assertRaises(MetadataError):
            validate_checkpoint(checkpoint)

    def test_mode_frequency_and_clock_policy_must_match_parent(self):
        context = self.context()
        with self.assertRaises(MetadataError):
            validate_run_context(context, mode="torque200")
        context["controlHz"] = 200
        with self.assertRaises(MetadataError):
            validate_run_context(context)
        with self.assertRaises(MetadataError):
            self.context(mode="teacher")
        with self.assertRaises(MetadataError):
            self.context(origin="checkpoint", parent=self.parent(), clock_policy_id="another-policy")
        with self.assertRaises(MetadataError):
            self.context(mode="torque200", origin="checkpoint", parent=self.parent())

    def test_origin_parent_and_required_flat_metadata_contract(self):
        for origin in ("new", "legacy"):
            with self.subTest(origin=origin), self.assertRaises(MetadataError):
                self.context(origin=origin, parent=self.parent())
        with self.assertRaises(MetadataError):
            self.context(origin="checkpoint")
        with self.assertRaises(MetadataError):
            self.context(origin="other")
        incomplete = self.parent()
        incomplete.pop("trainedSeconds")
        with self.assertRaises(MetadataError):
            validate_checkpoint(incomplete)
        extra = self.parent()
        extra["internalObject"] = {"ignored": True}
        self.assertNotIn("internalObject", validate_checkpoint(extra))


if __name__ == "__main__":
    unittest.main()
