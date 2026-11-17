"""Saved diagnostic reports preserve physical meanings and untrusted labels."""
import copy
import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.balance_diagnostics.report import write_report


def sample_bundle():
    return {
        "schema_version": 1, "kind": "balance_diagnostics", "horizon_s": 30,
        "scope": "Nominal <fixed> replay",
        "cases": [{
            "label": '<script>alert("policy")</script>', "policy_id": "final",
            "initial_state_id": "nominal",
            "trace": {
                "schema_version": 1,
                "t_s": [0., 1., 3., 7.],
                "base_position_m": [[0., 0., 0.1], [0.03, 0.04, 0.1], [0., 0.1, 0.1], [0., .2, 0.1]],
                "base_velocity_m_s": [[0., 0., 0.], [0., 0.1, 0.], [0., 0.2, 0.], [0., .3, 0.]],
                "pitch_rad": [0., 0.1, .2, .3], "roll_rad": [0., 0., 0., 0.],
                "pitch_rate_rad_s": [0., .1, .05, .025],
                "wheel_angle_distance_m": [0., 1., 2., 3.],
                "actuator_command_nm": [[0.], [.1], [.2], [.3]],
                "reward": [0., 1., 2., 4.], "done": [False, False, False, True],
                "metadata": {"reference_pitch_rad": 0., "torque_limits_nm": [1.]},
                "termination": {"terminated": True, "truncated": False, "reason": "pitch", "time_s": 7.},
            },
        }],
    }


class ReportTests(unittest.TestCase):
    def render(self, bundle):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        with patch("experiments.balance_diagnostics.report._plot") as plot:
            paths = write_report(bundle, directory.name)
        return paths, plot

    def test_html_escapes_labels_and_keeps_relative_artifact_links(self):
        paths, plot = self.render(sample_bundle())
        content = Path(paths["report_html"]).read_text(encoding="utf-8")
        self.assertNotIn("<script>", content)
        self.assertIn("&lt;script&gt;", content)
        self.assertIn("Nominal &lt;fixed&gt; replay", content)
        for reference in ('href="results.json"', 'href="summary.csv"', 'src="comparison.png"'):
            self.assertIn(reference, content)
        self.assertNotIn("https://", content)
        self.assertIn("未运行新训练", content)
        self.assertIn('comparison-detail.png', content)

    def test_csv_separates_body_displacement_from_wheel_angle_distance(self):
        paths, _ = self.render(sample_bundle())
        with Path(paths["summary_csv"]).open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["policy_label"], sample_bundle()["cases"][0]["label"])
        self.assertAlmostEqual(float(rows[0]["first_6s_max_displacement_m"]), .1)
        self.assertAlmostEqual(float(rows[0]["max_displacement_m"]), .2)
        self.assertAlmostEqual(float(rows[0]["reward_per_second"]), 1.)
        self.assertEqual(rows[0]["termination"], "terminated:pitch")

    def test_weighted_rms_uses_duration_and_terminal_sample(self):
        paths, _ = self.render(sample_bundle())
        with Path(paths["summary_csv"]).open(encoding="utf-8", newline="") as stream:
            row = next(csv.DictReader(stream))
        expected = ((0.**2 + .1**2) / 2 * 1 + (.1**2 + .2**2) / 2 * 2 +
                    (.2**2 + .3**2) / 2 * 4) / 7
        expected = expected ** .5
        self.assertAlmostEqual(float(row["horizontal_speed_rms_m_s"]), expected)

    def test_rejects_trace_containing_automatic_reset_samples(self):
        bundle = sample_bundle()
        bundle["cases"][0]["trace"]["done"][1] = True
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, "first episode"):
            write_report(bundle, directory)

    def test_rejects_invalid_time_and_contradictory_termination(self):
        for mutate, error in (
            (lambda trace: trace["t_s"].__setitem__(2, 1.), "increasing"),
            (lambda trace: trace["pitch_rad"].__setitem__(1, float("nan")), "finite"),
            (lambda trace: trace["termination"].__setitem__("truncated", True), "both"),
        ):
            bundle = copy.deepcopy(sample_bundle())
            mutate(bundle["cases"][0]["trace"])
            with self.subTest(error=error), tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(ValueError, error):
                    write_report(bundle, directory)


if __name__ == "__main__":
    unittest.main()
