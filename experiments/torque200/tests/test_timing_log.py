"""Observable timing/accounting invariants; no training or accelerator imports."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from timing_log import RunTiming, main, rebuild_summary


class FakeClock:
    def __init__(self, initial=1000.0):
        self.value = initial

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


class CostlyStream:
    """Advance the independent clock for real writes/flushes on an actual file."""

    def __init__(self, stream, clock):
        self.stream = stream
        self.clock = clock

    def write(self, value):
        self.clock.advance(0.2)
        return self.stream.write(value)

    def flush(self):
        self.clock.advance(0.3)
        self.stream.flush()

    def close(self):
        self.clock.advance(0.1)
        self.stream.close()


class RunTimingTests(unittest.TestCase):
    def assertBalanced(self, summary):
        self.assertAlmostEqual(
            summary["process_wall_seconds"],
            summary["exclusive_seconds"] + summary["recorder_overhead_seconds"] + summary["unaccounted_seconds"],
        )
        self.assertAlmostEqual(summary["accounting_residual_seconds"], 0.0)
        self.assertAlmostEqual(
            summary["exclusive_seconds"],
            sum(row["total_seconds"] for row in summary["phases"].values()),
        )

    @staticmethod
    def events(directory):
        return [json.loads(line) for line in (Path(directory) / "timing.jsonl").read_text(encoding="utf-8").splitlines()]

    def test_nested_phases_are_exclusive_and_keep_one_clock_origin(self):
        with tempfile.TemporaryDirectory() as directory:
            clock = FakeClock(initial=87654.0)
            timing = RunTiming(directory, run_id="nested", mode="original", clock=clock, phase="prepare")
            clock.advance(2)
            timing.set_phase("train", steps=0, epoch=1)
            clock.advance(3)
            with timing.span("evaluate", steps=120, episode=1):
                clock.advance(5)
                with timing.span("copy", steps=120, target="actor"):
                    clock.advance(7)
                clock.advance(11)
            clock.advance(13)
            summary = timing.finish(steps=240)

            self.assertEqual(summary["process_wall_seconds"], 41)
            self.assertEqual(summary["phases"]["prepare"]["total_seconds"], 2)
            self.assertEqual(summary["phases"]["train"]["total_seconds"], 16)
            self.assertEqual(summary["phases"]["evaluate"]["total_seconds"], 16)
            self.assertEqual(summary["phases"]["copy"]["total_seconds"], 7)
            self.assertEqual(summary["phases"]["train"]["segments"], 2)
            self.assertEqual(summary["phases"]["evaluate"]["segments"], 2)
            segments = [event for event in self.events(directory) if event["event"] == "segment"]
            self.assertEqual([event["wall_start_seconds"] for event in segments], [0, 2, 5, 10, 17, 28])
            self.assertEqual([event["wall_end_seconds"] for event in segments], [2, 5, 10, 17, 28, 41])
            self.assertEqual([event["duration_seconds"] for event in segments], [2, 3, 5, 7, 11, 13])
            self.assertEqual([(event["steps_start"], event["steps_end"]) for event in segments], [(None, 0), (0, 120), (120, 120), (120, 120), (120, 120), (120, 240)])
            self.assertEqual(segments[-1]["metadata"], {"epoch": 1})
            self.assertEqual(segments[-2]["metadata"], {"episode": 1})
            self.assertEqual(summary["steps"], 240)
            self.assertBalanced(summary)

    def test_same_phase_creates_distinct_segments_and_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            clock = FakeClock()
            timing = RunTiming(directory, run_id="same", mode="torque200", clock=clock, phase="train")
            clock.advance(2)
            timing.set_phase("train", steps=20)
            clock.advance(4)
            summary = timing.finish(steps=40)
            row = summary["phases"]["train"]
            self.assertEqual((row["segments"], row["closed_segments"], row["open_segments"]), (2, 2, 0))
            self.assertEqual((row["mean_seconds"], row["min_seconds"], row["max_seconds"]), (3, 2, 4))
            self.assertEqual(row["process_wall_fraction"], 1)
            self.assertBalanced(summary)

    def test_snapshots_are_observations_and_do_not_reset_or_append(self):
        with tempfile.TemporaryDirectory() as directory:
            clock = FakeClock()
            timing = RunTiming(directory, run_id="observe", mode="original", clock=clock, phase="train")
            before = (Path(directory) / "timing.jsonl").read_bytes()
            clock.advance(4)
            first = timing.snapshot()
            clock.advance(6)
            second = timing.snapshot()
            self.assertEqual((Path(directory) / "timing.jsonl").read_bytes(), before)
            self.assertEqual(first["process_wall_seconds"], 4)
            self.assertEqual(second["process_wall_seconds"], 10)
            self.assertEqual(second["closed_exclusive_seconds"], 0)
            self.assertEqual(second["current_open_seconds"], 10)
            self.assertEqual(second["phases"]["train"]["segments"], 1)
            self.assertEqual(second["phases"]["train"]["open_segments"], 1)
            self.assertBalanced(second)
            self.assertEqual(timing.finish()["phases"]["train"]["total_seconds"], 10)

    def test_write_and_summary_io_are_excluded_and_carried_into_later_events(self):
        with tempfile.TemporaryDirectory() as directory:
            clock = FakeClock()
            timing = RunTiming(directory, run_id="io", mode="original", clock=clock, phase="prepare")
            timing._stream = CostlyStream(timing._stream, clock)
            original_atomic = timing._atomic_text

            def costly_atomic(path, text):
                clock.advance(0.25)
                original_atomic(path, text)

            with patch.object(timing, "_atomic_text", side_effect=costly_atomic):
                clock.advance(2)
                timing.set_phase("train", steps=0)
                clock.advance(3)
                timing.flush_summary()
                clock.advance(4)
                summary = timing.finish(steps=100)

            # 9 s business + .5 s transition + 1 s flush + .5 s last segment.
            self.assertAlmostEqual(summary["exclusive_seconds"], 9)
            self.assertAlmostEqual(summary["recorder_overhead_seconds"], 2)
            self.assertAlmostEqual(summary["process_wall_seconds"], 11)
            self.assertAlmostEqual(summary["phases"]["train"]["total_seconds"], 7)
            self.assertAlmostEqual(summary["phases"]["train"]["process_wall_fraction"], 7 / 11)
            self.assertAlmostEqual(summary["phases"]["prepare"]["process_wall_fraction"], 2 / 11)
            # Final marker, two reports, and stream close are a stated tail.
            self.assertAlmostEqual(clock.value - 1000 - summary["process_wall_seconds"], 1.1)
            self.assertIn("not included or assumed to be zero", summary["reporting_tail_note"])
            segments = [event for event in self.events(directory) if event["event"] == "segment"]
            self.assertAlmostEqual(segments[-1]["recorder_excluded_seconds"], 1)
            rebuilt = rebuild_summary(directory)
            self.assertAlmostEqual(rebuilt["recorder_overhead_seconds"], 2)
            self.assertAlmostEqual(rebuilt["process_wall_seconds"], 11)
            self.assertBalanced(summary)
            self.assertBalanced(rebuilt)

    def test_snapshot_bookkeeping_is_excluded_from_the_open_phase(self):
        with tempfile.TemporaryDirectory() as directory:
            clock = FakeClock()
            timing = RunTiming(directory, run_id="snapshot-cost", mode="original", clock=clock, phase="train")
            original_build = timing._build_summary

            def costly_build(*args, **kwargs):
                result = original_build(*args, **kwargs)
                clock.advance(0.5)
                return result

            clock.advance(2)
            with patch.object(timing, "_build_summary", side_effect=costly_build):
                summary = timing.snapshot()
            self.assertEqual(summary["current_open_seconds"], 2)
            self.assertEqual(summary["process_wall_seconds"], 2.5)
            self.assertEqual(summary["recorder_overhead_seconds"], 0.5)
            self.assertBalanced(summary)
            clock.advance(3)
            final = timing.finish()
            self.assertEqual(final["phases"]["train"]["total_seconds"], 5)
            self.assertEqual(final["recorder_overhead_seconds"], 0.5)
            self.assertBalanced(final)

    def test_initial_log_serialization_is_recorder_overhead(self):
        with tempfile.TemporaryDirectory() as directory:
            clock = FakeClock()
            original_dumps = json.dumps

            def costly_dumps(*args, **kwargs):
                clock.advance(0.25)
                return original_dumps(*args, **kwargs)

            with patch("timing_log.json.dumps", side_effect=costly_dumps):
                timing = RunTiming(directory, run_id="serialization", mode="original", clock=clock, phase="train")
                clock.advance(3)
                summary = timing.finish()
            self.assertEqual(summary["exclusive_seconds"], 3)
            self.assertEqual(summary["recorder_overhead_seconds"], 0.5)
            self.assertEqual(summary["process_wall_seconds"], 3.5)
            self.assertBalanced(summary)

    def test_exception_and_cancellation_restore_parent_and_record_their_type(self):
        for exception, expected in [(ValueError("broken"), "error"), (KeyboardInterrupt(), "cancelled"), (SystemExit(2), "cancelled")]:
            with self.subTest(exception=type(exception).__name__), tempfile.TemporaryDirectory() as directory:
                clock = FakeClock()
                timing = RunTiming(directory, run_id="aborted-span", mode="original", clock=clock, phase="train")
                clock.advance(1)
                with self.assertRaises(type(exception)):
                    with timing.span("evaluate", steps=100):
                        clock.advance(2)
                        raise exception
                self.assertEqual(timing.snapshot()["open_segment"]["phase"], "train")
                clock.advance(3)
                summary = timing.finish(outcome=expected, steps=100)
                row = summary["phases"]["evaluate"]
                self.assertEqual(row["outcomes"], {expected: 1})
                self.assertEqual(summary["phases"]["train"]["total_seconds"], 4)
                segment = next(event for event in self.events(directory) if event["event"] == "segment" and event["phase"] == "evaluate")
                self.assertEqual(segment["metadata"]["exception_type"], type(exception).__name__)
                self.assertEqual(rebuild_summary(directory)["outcome"], expected)
                self.assertBalanced(summary)

    def test_finish_is_idempotent_and_later_clock_time_does_not_extend_it(self):
        with tempfile.TemporaryDirectory() as directory:
            clock = FakeClock()
            timing = RunTiming(directory, run_id="final", mode="original", clock=clock)
            clock.advance(9)
            final = timing.finish("cancelled", steps=900)
            contents = {name: Path(path).read_bytes() for name, path in timing.paths.items()}
            clock.advance(100)
            self.assertIs(timing.finish("completed", steps=999), final)
            self.assertIs(timing.snapshot(), final)
            self.assertIs(timing.flush_summary(), final)
            self.assertEqual(final["process_wall_seconds"], 9)
            self.assertEqual(final["steps"], 900)
            self.assertEqual(final["outcome"], "cancelled")
            self.assertEqual({name: Path(path).read_bytes() for name, path in timing.paths.items()}, contents)
            with self.assertRaisesRegex(RuntimeError, "finished"):
                timing.set_phase("train")
            with self.assertRaisesRegex(RuntimeError, "finished"):
                with timing.span("train"):
                    pass

    def test_explicit_cooperative_cancel_types_are_not_classified_by_name(self):
        class RequestedStop(Exception):
            pass

        for configured, expected in [((), "error"), ((RequestedStop,), "cancelled")]:
            with self.subTest(configured=bool(configured)), tempfile.TemporaryDirectory() as directory:
                clock = FakeClock()
                timing = RunTiming(directory, run_id="cooperative", mode="original", clock=clock,
                                   cancelled_exceptions=configured)
                with self.assertRaises(RequestedStop):
                    with timing.span("train"):
                        clock.advance(3)
                        raise RequestedStop("requested")
                summary = timing.finish()
                self.assertEqual(summary["phases"]["train"]["outcomes"], {expected: 1})
                self.assertBalanced(summary)

    def test_running_checkpoints_do_not_double_count_or_depend_on_summary_files(self):
        with tempfile.TemporaryDirectory() as directory:
            clock = FakeClock()
            timing = RunTiming(directory, run_id="live", mode="original", clock=clock, phase="train")
            clock.advance(4)
            timing.flush_summary()
            clock.advance(6)
            timing.flush_summary()
            Path(timing.paths["timing_summary_file"]).write_text("stale and invalid", encoding="utf-8")
            summary = rebuild_summary(directory)
            self.assertFalse(summary["finished"])
            self.assertEqual(summary["process_wall_seconds"], 10)
            self.assertEqual(summary["exclusive_seconds"], 10)
            self.assertEqual(summary["phases"]["train"]["segments"], 1)
            self.assertEqual(summary["phases"]["train"]["closed_segments"], 0)
            self.assertBalanced(summary)
            timing.set_phase("evaluate", steps=100)
            transitioned = rebuild_summary(directory)
            self.assertEqual(transitioned["phases"]["train"]["total_seconds"], 10)
            self.assertEqual(transitioned["open_segment"]["phase"], "evaluate")
            self.assertFalse(transitioned["open_segment"]["observed"])
            self.assertIsNone(transitioned["open_segment"]["duration_seconds"])
            self.assertEqual(transitioned["current_open_seconds"], 0)
            self.assertBalanced(transitioned)
            clock.advance(2)
            timing.finish()
            self.assertEqual(rebuild_summary(directory)["exclusive_seconds"], 12)

    def test_truncated_final_line_retains_all_complete_events(self):
        for tail in (b'{"event":"seg', b'{"metadata":"\xe4\xb8'):
            with self.subTest(tail=tail), tempfile.TemporaryDirectory() as directory:
                clock = FakeClock()
                timing = RunTiming(directory, run_id="torn", mode="original", clock=clock, phase="save")
                clock.advance(7)
                timing.finish("error", steps=42)
                Path(timing.paths["timing_summary_file"]).unlink()
                Path(timing.paths["timing_text_file"]).unlink()
                with Path(timing.paths["timing_file"]).open("ab") as stream:
                    stream.write(tail)
                summary = rebuild_summary(directory)
                self.assertTrue(summary["finished"])
                self.assertEqual(summary["outcome"], "error")
                self.assertEqual(summary["steps"], 42)
                self.assertEqual(summary["exclusive_seconds"], 7)
                self.assertEqual(summary["ignored_trailing_lines"], 1)
                self.assertBalanced(summary)

    def test_middle_corruption_is_not_silently_discarded(self):
        with tempfile.TemporaryDirectory() as directory:
            clock = FakeClock()
            timing = RunTiming(directory, run_id="middle", mode="original", clock=clock)
            timing.finish()
            path = Path(timing.paths["timing_file"])
            lines = path.read_bytes().splitlines(keepends=True)
            path.write_bytes(lines[0] + b'{"corrupted":\n' + b"".join(lines[1:]))
            with self.assertRaisesRegex(ValueError, "line 2"):
                rebuild_summary(directory)

    def test_later_sessions_append_and_do_not_mix_clock_origins(self):
        with tempfile.TemporaryDirectory() as directory:
            first_clock = FakeClock(initial=50)
            first = RunTiming(directory, run_id="old", mode="original", clock=first_clock, phase="old-phase")
            first_clock.advance(10)
            first.finish()
            original = Path(first.paths["timing_file"]).read_bytes()
            second_clock = FakeClock(initial=9000)
            second = RunTiming(directory, run_id="new", mode="torque200", clock=second_clock, phase="new-phase")
            self.assertTrue(Path(second.paths["timing_file"]).read_bytes().startswith(original))
            second_clock.advance(3)
            second.finish()
            summary = rebuild_summary(directory)
            self.assertEqual(summary["run_id"], "new")
            self.assertEqual(summary["process_wall_seconds"], 3)
            self.assertEqual(set(summary["phases"]), {"new-phase"})
            self.assertEqual(summary["sessions_in_log"], 2)
            self.assertBalanced(summary)

    def test_atomic_summary_replace_does_not_truncate_an_existing_report(self):
        with tempfile.TemporaryDirectory() as directory:
            timing = RunTiming(directory, run_id="atomic", mode="original", clock=FakeClock())
            timing.flush_summary()
            path = Path(timing.paths["timing_summary_file"])
            previous = path.read_bytes()
            with patch("timing_log.os.replace", side_effect=OSError("replacement failed")):
                with self.assertRaisesRegex(OSError, "replacement failed"):
                    timing.flush_summary()
            self.assertEqual(path.read_bytes(), previous)
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])
            timing.finish()

    def test_cli_reads_jsonl_and_reports_old_runs_clearly(self):
        with tempfile.TemporaryDirectory() as directory:
            error = io.StringIO()
            with redirect_stderr(error):
                self.assertEqual(main([directory]), 2)
            self.assertIn("No timing.jsonl", error.getvalue())
            clock = FakeClock()
            timing = RunTiming(directory, run_id="cli", mode="original", clock=clock, phase="train")
            clock.advance(2)
            timing.finish()
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main([directory]), 0)
            self.assertIn("train", output.getvalue())
            self.assertIn("Recorder overhead", output.getvalue())
            self.assertIn("wall %", output.getvalue())

    def test_backwards_clock_is_rejected_instead_of_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            clock = FakeClock()
            timing = RunTiming(directory, run_id="backwards", mode="original", clock=clock)
            clock.advance(2)
            timing.snapshot()
            clock.advance(-1)
            with self.assertRaisesRegex(RuntimeError, "monotonic"):
                timing.snapshot()
            clock.advance(1)
            timing.finish()


if __name__ == "__main__":
    unittest.main()
