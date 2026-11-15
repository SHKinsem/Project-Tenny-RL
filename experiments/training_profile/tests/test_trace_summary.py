"""Measured-device identification, overlap accounting and semantic limits."""
import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trace_summary import summarize_trace


def metadata():
    return [
        {"ph": "M", "name": "process_name", "pid": 1, "args": {"name": "/device:GPU:0"}},
        {"ph": "M", "name": "thread_name", "pid": 1, "tid": 13, "args": {"name": "Stream #13(Compute)"}},
        {"ph": "M", "name": "thread_name", "pid": 1, "tid": 14, "args": {"name": "Stream #14(Compute)"}},
        {"ph": "M", "name": "thread_name", "pid": 1, "tid": 0, "args": {"name": "XLA Ops"}},
        {"ph": "M", "name": "process_name", "pid": 701, "args": {"name": "/host:CPU"}},
        {"ph": "M", "name": "thread_name", "pid": 701, "tid": 2, "args": {"name": "GPUExecutor"}},
    ]


def event(start, duration, stage="ppo.rollout/ppo.physics", *, pid=1, tid=13, name="kernel"):
    return {"ph": "X", "ts": start, "dur": duration, "pid": pid, "tid": tid, "name": name,
            "args": {"hlo_module": "pmap_training_epoch", "name": "jit(epoch)/" + stage}}


def window(start=0, duration=100, *, name="epoch", step=0):
    return {"ph": "X", "ts": start, "dur": duration, "pid": 701, "tid": 3,
            "name": name, "args": {"step_num": str(step)}}


def copy_event(start=10, duration=20, *, name="MemcpyD2H", details=None, pid=1, tid=13):
    return {"ph": "X", "pid": pid, "tid": tid, "ts": start, "dur": duration, "name": name,
            "args": {"memcpy_details": details if details is not None else
                     "kind_src:device kind_dst:host size:32 dest:0 async:1", "correlation_id": "17"}}


class TraceSummaryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def summarize(self, events, *, compressed=False, **options):
        file = Path(self.directory.name) / ("trace.json.gz" if compressed else "trace.json")
        text = json.dumps({"traceEvents": events, "displayTimeUnit": "ns"})
        if compressed:
            with gzip.open(file, "wt", encoding="utf8") as stream:
                stream.write(text)
        else:
            file.write_text(text, encoding="utf8")
        return summarize_trace(file, **options)

    def assertSeconds(self, actual, microseconds):
        self.assertAlmostEqual(actual, microseconds * 1e-6, places=12)

    def test_real_shape_metadata_and_args_name_classification(self):
        result = self.summarize(metadata() + [window(), event(10, 20)])
        self.assertTrue(result["supported"])
        self.assertSeconds(result["category_busy_seconds"]["rollout.physics"], 20)
        self.assertEqual(result["classification_coverage_fraction"], 1.)
        self.assertEqual(result["gpu_event_count"], 1)

    def test_host_cuda_and_gpu_xla_aggregate_are_not_device_execution(self):
        events = metadata() + [window(), event(0, 100, pid=701, tid=2, name="cuLaunchKernel"),
                               event(0, 100, tid=0, name="aggregate_xla"), event(10, 20)]
        result = self.summarize(events)
        self.assertSeconds(result["gpu_busy_seconds"], 20)
        self.assertEqual(result["gpu_event_count"], 1)

    def test_missing_device_metadata_is_unsupported_not_zero_time(self):
        result = self.summarize([window(), event(10, 20)])
        self.assertFalse(result["supported"])
        self.assertIsNone(result["gpu_busy_seconds"])
        self.assertIn("metadata", result["reason"])

    def test_host_only_trace_is_unsupported(self):
        result = self.summarize(metadata()[:0] + [window(), event(1, 10, pid=701, tid=2)])
        self.assertFalse(result["supported"])
        self.assertSeconds(result["host_window_seconds"], 100)

    def test_multistream_union_and_additive_overlap_partition(self):
        events = metadata() + [window(), event(0, 30), event(10, 30, "ppo.update/ppo.gradient", tid=14)]
        result = self.summarize(events)
        self.assertSeconds(result["gpu_busy_seconds"], 40)
        self.assertSeconds(result["gpu_event_duration_sum_seconds"], 60)
        self.assertSeconds(result["cross_category_overlap_seconds"], 20)
        self.assertSeconds(sum(result["category_exclusive_seconds"].values()), 20)
        self.assertAlmostEqual(sum(result["category_exclusive_seconds"].values()) +
                               result["cross_category_overlap_seconds"], result["gpu_busy_seconds"])
        self.assertSeconds(result["gpu_idle_or_untraced_within_host_window_seconds"], 60)

    def test_same_category_overlap_is_counted_once(self):
        result = self.summarize(metadata() + [window(), event(0, 30), event(10, 30, tid=14)])
        self.assertSeconds(result["category_exclusive_seconds"]["rollout.physics"], 40)
        self.assertSeconds(result["cross_category_overlap_seconds"], 0)

    def test_parent_child_stage_uses_leaf_once(self):
        item = event(0, 20, "ppo.rollout/ppo.physics")
        item["args"]["hlo_op"] = "fusion(ppo.rollout/ppo.physics)/ppo.rollout/ppo.physics"
        result = self.summarize(metadata() + [item])
        self.assertEqual(set(result["category_busy_seconds"]), {"rollout.physics"})
        self.assertSeconds(result["gpu_busy_seconds"], 20)
        self.assertEqual(result["top_kernels"][0]["stage_tags"], ["ppo.physics", "ppo.rollout"])

    def test_explicit_stage_prefix(self):
        result = self.summarize(metadata() + [event(0, 20, "measure.rollout/measure.physics")],
                                stage_prefix="measure.")
        self.assertEqual(set(result["category_busy_seconds"]), {"rollout.physics"})

    def test_gradient_forward_backward_remain_combined(self):
        result = self.summarize(metadata() + [event(0, 20, "ppo.update/ppo.gradient/ppo.actor_forward")])
        self.assertEqual(set(result["category_busy_seconds"]), {"update.gradient_combined"})

    def test_standalone_actor_loss_scope_is_not_sampling(self):
        result = self.summarize(metadata() + [event(0, 20, "ppo.actor_forward")])
        self.assertEqual(set(result["category_busy_seconds"]), {"update.actor_forward"})

    def test_cross_phase_fusion_is_mixed(self):
        result = self.summarize(metadata() + [event(0, 20, "ppo.rollout/ppo.physics + ppo.update/ppo.gradient")])
        self.assertEqual(set(result["category_busy_seconds"]), {"mixed"})
        self.assertEqual(result["classification_coverage_fraction"], 0.)

    def test_unknown_kernel_is_reported(self):
        result = self.summarize(metadata() + [event(0, 20, "dot_general")])
        self.assertEqual(result["classification_coverage_fraction"], 0.)
        self.assertSeconds(result["unknown_or_mixed_busy_seconds"], 20)
        self.assertEqual(result["top_kernels"][0]["categories"], {"unknown": 1})

    def test_chrome_timestamp_units_ignore_display_unit(self):
        result = self.summarize(metadata() + [event(0, 1_000_000)], compressed=True)
        self.assertEqual(result["display_time_unit"], "ns")
        self.assertSeconds(result["gpu_busy_seconds"], 1_000_000)
        self.assertEqual(result["gpu_busy_seconds"], 1.)

    def test_explicit_host_window_clips_events(self):
        result = self.summarize(metadata() + [window(20, 20), event(0, 30), event(30, 30)], window_name="epoch")
        self.assertSeconds(result["gpu_busy_seconds"], 20)
        self.assertSeconds(result["host_window_seconds"], 20)

    def test_disjoint_host_windows_exclude_interepoch_gap(self):
        result = self.summarize(metadata() + [window(0, 10), window(50, 10, step=1), event(0, 60)])
        self.assertSeconds(result["gpu_busy_seconds"], 20)
        self.assertSeconds(result["host_window_seconds"], 20)
        self.assertSeconds(result["host_window_envelope_seconds"], 60)

    def test_missing_explicit_window_does_not_silently_measure_whole_trace(self):
        result = self.summarize(metadata() + [event(0, 10)], window_name="missing")
        self.assertFalse(result["supported"])
        self.assertIsNone(result["gpu_busy_seconds"])
        self.assertIn("window", result["reason"])

    def test_no_host_window_does_not_invent_host_idle_time(self):
        result = self.summarize(metadata() + [event(0, 10)])
        self.assertTrue(result["supported"])
        self.assertIsNone(result["host_window_seconds"])
        self.assertIsNone(result["gpu_idle_or_untraced_within_host_window_seconds"])

    def test_invalid_and_empty_events_do_not_become_elapsed_time(self):
        events = [event(0, -1), event(0, 0), event(0, True), event(0, float("nan"))]
        result = self.summarize(metadata() + events)
        self.assertFalse(result["supported"])
        self.assertEqual(result["invalid_device_event_count"], 4)
        self.assertIsNone(result["gpu_busy_seconds"])

    def test_malformed_metadata_and_overflowing_timestamp_are_not_measured(self):
        broken = [{"ph": "M", "name": "thread_name", "args": None},
                  {"ph": "M", "name": "process_name", "args": {"name": "/device:GPU:0"}}]
        result = self.summarize(metadata() + broken + [event(1e308, 1e308)])
        self.assertFalse(result["supported"])
        self.assertEqual(result["invalid_device_event_count"], 1)

    def test_aggregate_stream_is_not_counted_twice(self):
        events = metadata() + [{"ph": "M", "name": "thread_name", "pid": 1, "tid": 99,
                                "args": {"name": "Stream #all"}}, event(0, 100, tid=99), event(10, 20)]
        self.assertSeconds(self.summarize(events)["gpu_busy_seconds"], 20)

    def test_other_top_level_stage_and_optimizer_leaf(self):
        result = self.summarize(metadata() + [event(0, 10, "ppo.batch_prepare"),
                                             event(10, 10, "ppo.update/ppo.optimizer")])
        self.assertEqual(set(result["category_busy_seconds"]), {"batch_prepare", "update.optimizer"})

    def test_bad_trace_or_prefix_rejected(self):
        file = Path(self.directory.name) / "bad.json"
        file.write_text('{"traceEvents": {}}')
        with self.assertRaises(ValueError):
            summarize_trace(file)
        with self.assertRaises(ValueError):
            summarize_trace(file, stage_prefix="")

    def hlo_file(self, text, name="compiled.hlo"):
        file = Path(self.directory.name) / name
        file.write_text(text, encoding="utf8")
        return file

    def hlo_event(self, *, op="fusion.1", module="pmap_training_epoch", name="fusion_1"):
        value = event(0, 20, "unlabelled", name=name)
        value["args"].update(hlo_op=op, hlo_module=module)
        return value

    def test_compiled_root_scope_wins_over_called_computation(self):
        file = self.hlo_file('''HloModule pmap_training_epoch
%compute (x: f32[]) -> f32[] {
 ROOT %add = f32[] add(%x, %x), metadata={op_name="ppo.update/ppo.gradient/add"}
}
ENTRY %main (x: f32[]) -> f32[] {
 ROOT %fusion.1 = f32[] fusion(%x), calls=%compute, metadata={op_name="ppo.rollout/ppo.physics/root"}
}
''')
        result = self.summarize(metadata() + [self.hlo_event()], compiled_hlo_path=file)
        self.assertEqual(set(result["category_busy_seconds"]), {"rollout.physics"})
        self.assertEqual(result["classification_sources"], {"hlo_op_exact": 1})
        self.assertIn("compiled_op_name", result["top_kernels"][0]["metadata_origins"])

    def test_compiled_calls_recover_shared_parent_but_not_false_leaf_precision(self):
        file = self.hlo_file('''HloModule pmap_training_epoch
%compute (x: f32[]) -> f32[] {
 %add = f32[] add(%x, %x), metadata={op_name="ppo.rollout/ppo.physics/add"}
 %mul = f32[] multiply(%x, %x), metadata={op_name="ppo.rollout/ppo.obs/mul"}
 ROOT %sum = f32[] add(%add, %mul)
}
ENTRY %main (x: f32[]) -> f32[] {
 ROOT %fusion.1 = f32[] fusion(%x), calls=%compute
}
''')
        result = self.summarize(metadata() + [self.hlo_event()], compiled_hlo_path=file)
        self.assertEqual(set(result["category_busy_seconds"]), {"rollout.mixed"})
        self.assertEqual(result["classification_coverage_fraction"], 0.)
        self.assertEqual(result["domain_classification_coverage_fraction"], 1.)
        self.assertTrue(any("ppo.rollout" in sample for sample in result["top_kernels"][0]["hlo_name_samples"]))

    def test_compiled_cross_parent_fusion_remains_mixed(self):
        file = self.hlo_file('''HloModule pmap_training_epoch
%compute (x: f32[]) -> f32[] {
 %add = f32[] add(%x, %x), metadata={op_name="ppo.rollout/ppo.physics/add"}
 ROOT %mul = f32[] multiply(%add, %x), metadata={op_name="ppo.update/ppo.gradient/mul"}
}
ENTRY %main (x: f32[]) -> f32[] {
 ROOT %fusion.1 = f32[] fusion(%x), calls=%compute
}
''')
        result = self.summarize(metadata() + [self.hlo_event()], compiled_hlo_path=file)
        self.assertEqual(set(result["category_busy_seconds"]), {"mixed"})
        self.assertEqual(result["domain_classification_coverage_fraction"], 0.)

    def test_unmatched_hlo_op_does_not_guess_using_matching_kernel_symbol(self):
        file = self.hlo_file('''HloModule pmap_training_epoch
ENTRY %main (x: f32[]) -> f32[] {
 ROOT %fusion.1 = f32[] negate(%x), metadata={op_name="ppo.rollout/ppo.physics/neg"}
}
''')
        result = self.summarize(metadata() + [self.hlo_event(op="other_graph_fusion")], compiled_hlo_path=file)
        self.assertEqual(set(result["category_busy_seconds"]), {"unknown"})
        self.assertEqual(result["classification_sources"], {"hlo_op_unmatched": 1})

    def test_kernel_symbol_fallback_retains_alias_ambiguity(self):
        file = self.hlo_file('''HloModule pmap_training_epoch
ENTRY %main (x: f32[]) -> f32[] {
 %fusion.1 = f32[] negate(%x), metadata={op_name="ppo.rollout/ppo.physics/neg", deduplicated_name="shared"}
 ROOT %fusion.2 = f32[] negate(%fusion.1), metadata={op_name="ppo.update/ppo.gradient/neg", deduplicated_name="shared"}
}
''')
        value = self.hlo_event(name="shared")
        value["args"].pop("hlo_op")
        result = self.summarize(metadata() + [value], compiled_hlo_path=file)
        self.assertEqual(set(result["category_busy_seconds"]), {"mixed"})
        self.assertEqual(result["classification_sources"], {"kernel_symbol_exact": 1})

    def test_command_buffer_group_uses_audited_exact_kernel_symbol(self):
        file = self.hlo_file('''HloModule pmap_training_epoch
ENTRY %main (x: f32[]) -> f32[] {
 ROOT %fusion.1 = f32[] negate(%x), metadata={op_name="ppo.rollout/ppo.physics/neg"}
}
''')
        result = self.summarize(metadata() + [self.hlo_event(op="command_buffer_83")], compiled_hlo_path=file)
        self.assertEqual(set(result["category_busy_seconds"]), {"rollout.physics"})
        self.assertEqual(result["classification_sources"], {"command_buffer_kernel_symbol": 1})

    def test_unlabelled_condition_inherits_explicit_callsite_scope(self):
        file = self.hlo_file('''HloModule pmap_training_epoch
%body (x: f32[]) -> f32[] {
 ROOT %fusion.1 = f32[] negate(%x)
}
ENTRY %main (x: f32[]) -> f32[] {
 ROOT %while = f32[] while(%x), body=%body, metadata={op_name="ppo.rollout/ppo.physics/while"}
}
''')
        result = self.summarize(metadata() + [self.hlo_event()], compiled_hlo_path=file)
        self.assertEqual(set(result["category_busy_seconds"]), {"rollout.physics"})
        self.assertIn("inherited_callsite", result["top_kernels"][0]["metadata_origins"])

    def test_shared_callsite_contexts_keep_cross_domain_ambiguity(self):
        file = self.hlo_file('''HloModule pmap_training_epoch
%shared (x: f32[]) -> f32[] {
 ROOT %fusion.1 = f32[] negate(%x)
}
ENTRY %main (x: f32[]) -> f32[] {
 %call1 = f32[] call(%x), to_apply=%shared, metadata={op_name="ppo.rollout/ppo.physics/call"}
 ROOT %call2 = f32[] call(%call1), to_apply=%shared, metadata={op_name="ppo.update/ppo.gradient/call"}
}
''')
        result = self.summarize(metadata() + [self.hlo_event()], compiled_hlo_path=file)
        self.assertEqual(set(result["category_busy_seconds"]), {"mixed"})

    def test_original_solver_metadata_is_evidence_separate_from_ppo_stage(self):
        file = self.hlo_file('''HloModule pmap_training_epoch
%body (x: f32[]) -> f32[] {
 ROOT %fusion.1 = f32[] negate(%x)
}
ENTRY %main (x: f32[]) -> f32[] {
 ROOT %while = f32[] while(%x), body=%body, metadata={op_name="jit(epoch)/step/forward/solve/while"}
}
''')
        result = self.summarize(metadata() + [self.hlo_event()], compiled_hlo_path=file)
        self.assertEqual(set(result["category_busy_seconds"]), {"unknown"})
        self.assertEqual(set(result["subsystem_busy_seconds"]), {"physics.constraint_solver"})
        self.assertSeconds(result["subsystem_busy_seconds"]["physics.constraint_solver"], 20)
        self.assertTrue(any("/step/forward/solve/" in sample
                            for sample in result["top_kernels"][0]["subsystem_source_samples"]))

    def test_module_mismatch_and_nonstring_operation_are_unmapped(self):
        file = self.hlo_file('''HloModule pmap_training_epoch
ENTRY %main (x: f32[]) -> f32[] {
 ROOT %fusion.1 = f32[] negate(%x), metadata={op_name="ppo.rollout/ppo.physics/neg"}
}
''')
        value = self.hlo_event(op=["fusion.1"])
        result = self.summarize(metadata() + [value], compiled_hlo_path=file)
        self.assertEqual(result["classification_sources"], {"invalid_hlo_op": 1})
        result = self.summarize(metadata() + [self.hlo_event(module="different")], compiled_hlo_path=file)
        self.assertEqual(result["classification_sources"], {"module_mismatch": 1})

    def test_trace_tags_have_priority_over_offline_hlo(self):
        file = self.hlo_file('''HloModule pmap_training_epoch
ENTRY %main (x: f32[]) -> f32[] {
 ROOT %fusion.1 = f32[] negate(%x), metadata={op_name="ppo.update/ppo.gradient/neg"}
}
''')
        value = self.hlo_event()
        value["args"]["name"] = "ppo.rollout/ppo.physics/neg"
        result = self.summarize(metadata() + [value], compiled_hlo_path=file)
        self.assertEqual(set(result["category_busy_seconds"]), {"rollout.physics"})
        self.assertEqual(result["classification_sources"], {"trace_stage_tags": 1})

    def test_annotated_callgraph_matches_exact_scope_stripped_original_path(self):
        original = self.hlo_file('''HloModule pmap_training_epoch
ENTRY %main (x: f32[]) -> f32[] {
 ROOT %fusion.1 = f32[] negate(%x), metadata={op_name="jit(epoch)/while/body/closed_call/while/body/add"}
}
''')
        annotated = self.hlo_file('''HloModule pmap_training_epoch
%rollout_body (x: f32[]) -> f32[] {
 ROOT %add = f32[] add(%x, %x), metadata={op_name="ppo.rollout/while/body/ppo.physics/add"}
}
%training (x: f32[]) -> f32[] {
 ROOT %while = f32[] while(%x), body=%rollout_body, metadata={op_name="ppo.rollout/while"}
}
%outer_body (x: f32[]) -> f32[] {
 ROOT %call = f32[] call(%x), to_apply=%training, metadata={op_name="while/body/closed_call"}
}
ENTRY %main (x: f32[]) -> f32[] {
 ROOT %while = f32[] while(%x), body=%outer_body, metadata={op_name="jit(epoch)/while"}
}
''', "annotated.hlo")
        result = self.summarize(metadata() + [self.hlo_event()], compiled_hlo_path=original,
                                annotated_mlir_path=annotated)
        self.assertEqual(set(result["category_busy_seconds"]), {"rollout.physics"})
        self.assertEqual(result["compiled_hlo_metadata"]["annotation_recovery"]["exact_op_names_with_stage_labels"], 1)
        self.assertIn("annotated_exact_op_name", result["top_kernels"][0]["metadata_origins"])

    def test_ambiguous_annotated_opname_is_not_arbitrarily_selected(self):
        original = self.hlo_file('''HloModule pmap_training_epoch
ENTRY %main (x: f32[]) -> f32[] {
 ROOT %fusion.1 = f32[] negate(%x), metadata={op_name="jit(epoch)/add"}
}
''')
        annotated = self.hlo_file('''HloModule pmap_training_epoch
ENTRY %main (x: f32[]) -> f32[] {
 %one = f32[] negate(%x), metadata={op_name="jit(epoch)/ppo.rollout/add"}
 ROOT %two = f32[] negate(%one), metadata={op_name="jit(epoch)/ppo.update/add"}
}
''', "annotated.hlo")
        result = self.summarize(metadata() + [self.hlo_event()], compiled_hlo_path=original,
                                annotated_mlir_path=annotated)
        self.assertEqual(set(result["category_busy_seconds"]), {"mixed"})
        self.assertEqual(result["compiled_hlo_metadata"]["annotation_recovery"]["ambiguous_exact_op_names"], 1)

    def test_explicit_source_range_is_audited_and_optional(self):
        file = self.hlo_file('''HloModule pmap_training_epoch
ENTRY %main (x: f32[]) -> f32[] {
 ROOT %fusion.1 = f32[] negate(%x), metadata={source_file="package/train.py" source_line=12}
}
''')
        result = self.summarize(metadata() + [self.hlo_event()], compiled_hlo_path=file,
                                source_stage_ranges=[{"path_suffix": "package/train.py", "start_line": 10,
                                                      "end_line": 15, "stage": "ppo.update"}])
        self.assertEqual(set(result["category_busy_seconds"]), {"update.other"})
        self.assertIn("explicit_source_range", result["top_kernels"][0]["metadata_origins"])

    def test_annotated_input_requires_compiled_hlo_and_matching_module(self):
        file = self.hlo_file("HloModule one\nENTRY %main () -> f32[] {\n ROOT %zero = f32[] constant(0)\n}\n")
        other = self.hlo_file("HloModule two\nENTRY %main () -> f32[] {\n ROOT %zero = f32[] constant(0)\n}\n", "other.hlo")
        with self.assertRaises(ValueError):
            self.summarize(metadata(), annotated_mlir_path=other)
        with self.assertRaises(ValueError):
            self.summarize(metadata(), compiled_hlo_path=file, annotated_mlir_path=other)

    def test_explicit_device_copy_on_mixed_compute_stream(self):
        result = self.summarize(metadata() + [window(), copy_event()])
        copy = result["transfers"]["device"]
        self.assertTrue(copy["supported"])
        self.assertEqual(copy["bytes"], 32)
        self.assertEqual(copy["by_direction"]["D2H"]["event_count"], 1)
        self.assertSeconds(copy["busy_seconds"], 20)

    def test_memcpy_post_kernel_and_name_only_events_are_not_transfers(self):
        post = copy_event(name="memcpy32_post")
        post["args"]["kernel_details"] = "kernel launch"
        name_only = event(40, 10, name="MemcpyD2H")
        result = self.summarize(metadata() + [window(), post, name_only])
        self.assertFalse(result["transfers"]["device"]["supported"])
        self.assertIsNone(result["transfers"]["device"]["busy_seconds"])

    def test_unknown_bytes_remain_null_with_known_lower_bound(self):
        unknown = copy_event(40, 10, details="kind_src:device kind_dst:host size:unknown")
        result = self.summarize(metadata() + [window(), copy_event(), unknown])
        value = result["transfers"]["device"]["by_direction"]["D2H"]
        self.assertEqual(value["known_bytes"], 32)
        self.assertIsNone(value["bytes"])
        self.assertEqual(value["unknown_byte_events"], 1)

    def test_unknown_direction_is_kept_separate(self):
        value = copy_event(name="Memcpy", details={"src_kind": "unknown", "dst_kind": "managed", "num_bytes": 64})
        result = self.summarize(metadata() + [window(), value])
        unknown = result["transfers"]["device"]["by_direction"]["unknown"]
        self.assertEqual(unknown["bytes"], 64)
        self.assertEqual(unknown["event_count"], 1)

    def test_conflicting_direction_evidence_stays_unknown(self):
        value = copy_event(name="MemcpyH2D")  # Details explicitly say device -> host.
        result = self.summarize(metadata() + [window(), value])
        self.assertEqual(result["transfers"]["device"]["by_direction"]["unknown"]["event_count"], 1)

    def test_copy_overlaps_use_union_without_double_counting_bytes(self):
        result = self.summarize(metadata() + [window(), copy_event(0, 30), copy_event(10, 30, tid=14)])
        value = result["transfers"]["device"]
        self.assertSeconds(value["busy_seconds"], 40)
        self.assertSeconds(value["event_duration_sum_seconds"], 60)
        self.assertEqual(value["bytes"], 64)

    def test_cross_window_copies_are_excluded_whole_not_prorated(self):
        events = [window(10, 10), window(30, 10, step=1), copy_event(15, 20), copy_event(10, 10)]
        result = self.summarize(metadata() + events)
        value = result["transfers"]["device"]
        self.assertEqual(value["event_count"], 1)
        self.assertEqual(value["bytes"], 32)
        self.assertEqual(value["partial_window_events_excluded"], 1)
        self.assertSeconds(value["busy_seconds"], 10)

    def test_host_copy_events_are_separate_from_device_execution(self):
        host = copy_event(0, 70, pid=701, tid=2)
        gpu = copy_event(30, 20)
        result = self.summarize(metadata() + [window(), host, gpu])
        self.assertEqual(result["transfers"]["host_api"]["event_count"], 1)
        self.assertSeconds(result["transfers"]["host_api"]["busy_seconds"], 70)
        self.assertSeconds(result["transfers"]["device"]["busy_seconds"], 20)

    def test_cuda_host_memcpy_without_device_details_stays_host_api(self):
        host = {"ph": "X", "pid": 701, "tid": 2, "ts": 10, "dur": 5,
                "name": "cuMemcpyHtoDAsync_v2", "args": {}}
        result = self.summarize(metadata() + [window(), host])
        self.assertTrue(result["transfers"]["host_api"]["supported"])
        self.assertEqual(result["transfers"]["host_api"]["by_direction"]["H2D"]["event_count"], 1)
        self.assertIsNone(result["transfers"]["host_api"]["bytes"])
        self.assertFalse(result["transfers"]["device"]["supported"])

    def test_explicit_host_sync_wait_api_union(self):
        sync = {"ph": "X", "pid": 701, "tid": 2, "ts": 10, "dur": 30,
                "name": "cuStreamSynchronize", "args": {}}
        wait = dict(sync, tid=3, ts=20, name="cudaStreamWaitEvent")
        gpu_named_sync = dict(sync, pid=1, tid=13)
        unrelated_wait = dict(sync, name="Wait for donation holds")
        result = self.summarize(metadata() + [window(), sync, wait, gpu_named_sync, unrelated_wait])
        value = result["host_sync_api"]
        self.assertTrue(value["supported"])
        self.assertEqual(value["event_count"], 2)
        self.assertSeconds(value["busy_seconds"], 40)
        self.assertSeconds(value["event_duration_sum_seconds"], 60)

    def test_absent_or_cross_window_sync_does_not_claim_zero_blocking(self):
        sync = {"ph": "X", "pid": 701, "tid": 2, "ts": 0, "dur": 50,
                "name": "cudaDeviceSynchronize", "args": {}}
        result = self.summarize(metadata() + [window(10, 10), sync])
        value = result["host_sync_api"]
        self.assertFalse(value["supported"])
        self.assertIsNone(value["busy_seconds"])
        self.assertEqual(value["partial_window_events_excluded"], 1)

    def test_invalid_size_units_or_fractional_bytes_remain_unknown(self):
        for details in ({"num_bytes": True}, {"size": -1}, {"num_bytes": "1.5"},
                        {"num_bytes": "4", "unit": "KB"}):
            with self.subTest(details=details):
                result = self.summarize(metadata() + [window(), copy_event(details=details)])
                self.assertIsNone(result["transfers"]["device"]["bytes"])

    def test_missing_window_cannot_count_entire_trace_transfers(self):
        result = self.summarize(metadata() + [copy_event()], window_name="not-present")
        self.assertFalse(result["transfers"]["device"]["supported"])
        self.assertIsNone(result["transfers"]["device"]["bytes"])


if __name__ == "__main__":
    unittest.main()
