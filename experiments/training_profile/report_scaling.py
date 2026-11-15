"""CPU-only replay of a completed scaling matrix and its device traces."""
import argparse
import json
from pathlib import Path

from run_profile import write_json
from trace_summary import summarize_trace


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path, help="Directory containing matrix.json")
    parser.add_argument("--output", type=Path, help="Summary JSON; defaults to the matrix directory")
    args = parser.parse_args(argv)
    matrix = args.matrix.expanduser().resolve()
    receipt = json.loads((matrix / "matrix.json").read_text())
    resources = [json.loads(line) for line in (matrix / "resources.jsonl").read_text().splitlines() if line]
    records, configs = [], []
    for case in receipt["cases"]:
        case_dir = Path(case["output"])
        record = {"num_envs": case["num_envs"], "exit_code": case.get("exit_code"), "passed": False}
        if not (case_dir / "result.json").exists():
            record["status"] = "partial"
            records.append(record)
            continue
        result = json.loads((case_dir / "result.json").read_text())
        contract = json.loads((case_dir / "contract.json").read_text())
        configs.append({key: value for key, value in contract["config"].items() if key != "num_envs"})
        summary = summarize_trace(result["trace_file"], window_name="scaling.host_epoch",
                                  compiled_hlo_path=case_dir / "compiled-hlo.txt.gz")
        write_json(case_dir / "trace-summary.json", summary)
        memory = [row["stats"] or {} for row in result["memory"]]
        memory_fields = sorted({key for row in memory for key in row})
        memory_peaks = {key: max(row[key] for row in memory if key in row)
                        for key in memory_fields if all(isinstance(row.get(key, 0), (int, float)) for row in memory)}
        gpu = [row for row in resources if row["num_envs"] == case["num_envs"] and "memory_used_mib" in row]
        record.update(status="complete", passed=result["passed"], performance=result["performance"],
            compile=result["compile"], diagnostic_env_steps=result["diagnostic_env_steps"],
            initial_parameter_max_abs=result["initial_parameter_max_abs"],
            initial_parameter_sha256=result["initial_parameter_sha256"],
            normalizer_max_abs_change=result["normalizer_max_abs_change"],
            production_and_checkpoint_unchanged=result["production_and_checkpoint_unchanged"],
            device_allocator_peak_fields=memory_peaks,
            gpu_total_memory_peak_mib=max((row["memory_used_mib"] for row in gpu), default=None),
            gpu_memory_total_mib=gpu[0]["memory_total_mib"] if gpu else None,
            resource_samples=len(gpu), boundary_transfers=result["boundary_transfers"],
            trace_epoch_seconds=result["trace_epoch_seconds"],
            trace_to_untraced_median_ratio=result["trace_to_untraced_median_ratio"],
            recovery_epoch_seconds=result["recovery_epoch_seconds"],
            trace_gpu_busy_seconds=summary.get("gpu_busy_seconds"),
            trace_gpu_event_count=summary.get("gpu_event_count"),
            transfers=summary.get("transfers"), host_sync_api=summary.get("host_sync_api"),
            subsystem_exclusive_seconds=summary.get("subsystem_exclusive_seconds"),
            subsystem_cross_overlap_seconds=summary.get("subsystem_cross_overlap_seconds"))
        records.append(record)
    complete = [row for row in records if row["passed"]]
    if complete:
        baseline_sps = complete[0]["performance"]["steps_per_second"]
        for row in complete:
            row["throughput_ratio_to_first_case"] = row["performance"]["steps_per_second"] / baseline_sps
    result = {"schema_version": 1, "matrix_passed": receipt.get("passed", False),
        "run_id": receipt["run_id"], "checkpoint_sha256": receipt["checkpoint_sha256"],
        "common_batch_size": receipt["common_batch_size"], "matrix_wall_seconds": receipt["wall_seconds"],
        "same_initial_parameters": receipt["same_initial_parameters"],
        "same_config_except_num_envs": bool(configs) and all(config == configs[0] for config in configs),
        "cases": records,
        "notes": ["Throughput uses only untraced fixed-work epochs; trace/API times are separate diagnostics.",
                  "The common batch differs from production; these ratios are not end-to-end training gains.",
                  "Device allocator peaks include setup/compilation/trace; nvidia-smi samples cover the whole GPU.",
                  "Different environment counts change sampled trajectories; no convergence claim or bitwise final equality.",
                  "Trace HLO is exported by the same case executable; absent synchronization APIs do not prove zero stalls."]}
    write_json(args.output or matrix / "scaling-summary.json", result)
    print(json.dumps({"matrix_passed": result["matrix_passed"], "same_config_except_num_envs": result["same_config_except_num_envs"],
        "cases": [{key: row.get(key) for key in ("num_envs", "passed", "performance", "throughput_ratio_to_first_case",
                      "device_allocator_peak_fields", "gpu_total_memory_peak_mib", "trace_to_untraced_median_ratio")}
                  for row in records]}, indent=2))


if __name__ == "__main__":
    main()
