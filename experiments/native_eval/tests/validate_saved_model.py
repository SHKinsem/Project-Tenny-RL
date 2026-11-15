"""Compare a descriptor with the frozen CPU/GPU benchmark; never train a policy."""
import argparse
import json
from pathlib import Path
import statistics
import time

import mujoco
import numpy as np

from native_eval import Evaluator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptor", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.reference_dir, args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    contract = json.loads((root / "contract.json").read_text())
    limits = contract["limits_declared_before_measurement"]
    gpu = json.loads((root / "gpu-timing.json").read_text())
    cpu = json.loads((root / "cpu-timing.json").read_text())
    start = time.perf_counter()
    engine = Evaluator(args.descriptor)
    setup_seconds = time.perf_counter() - start
    metadata = engine.descriptor.get("metadata", {})
    for key in ("checkpoint_sha256", "source_fingerprint"):
        if metadata.get(key) != contract[key]:
            raise ValueError(f"Descriptor differs from frozen reference: {key}")
    initial = dict(np.load(root / "initial.npz", allow_pickle=False))
    count, steps = contract["worlds"], contract["steps"]
    states = [dict(qpos=initial["qpos"][i], qvel=initial["qvel"][i],
                   extra={key: [float(initial[key][i])] for key in
                          ("position_start", "sinyaw_start", "sin_balance_zero")}) for i in range(count)]
    common = np.load(root / "common-policy.npz", allow_pickle=False)
    action_error = float(np.max(np.abs(engine.policy_actions(common["observation"]) - common["action"])))
    trials, trace = [], None
    for index in range(4):
        start = time.perf_counter()
        result = engine.evaluate(states, num_envs=count, steps=steps, episode_steps=steps,
                                 control_dt=1 / contract["control_hz"], trace_worlds=[0] if index == 0 else [])
        seconds = time.perf_counter() - start
        if index == 0:
            trace = result["traces"][0]
        result.pop("traces")
        trials.append({"index": index, "seconds": seconds, "includes_trace": index == 0, **result})
    gt = dict(np.load(root / "gpu-trace.npz", allow_pickle=False))
    ct = dict(np.load(root / "cpu-trace.npz", allow_pickle=False))
    model = mujoco.MjModel.from_xml_path(engine.descriptor["xml_path"])
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                            engine.descriptor["task"]["config"]["root_body"])
    pitch = -trace["xmat"][:, body, 2, 0]
    ref_returns = np.median([item["rewards"] for item in gpu["trials"][1:]], axis=0)
    returns = trials[1]["returns"]
    return_relative = float(np.max(np.abs(returns - ref_returns) / np.maximum(np.abs(ref_returns), 1)))
    first = {key: float(np.max(np.abs(trace[key][0] - gt[key][0, 0])))
             for key in ("observation", "action", "qpos", "qvel", "reward")}
    errors = {key: float(np.max(np.abs(trace[key] - gt[key][:, 0])))
              for key in ("observation", "action", "qpos", "qvel", "reward")}
    errors["pitch_sin"] = float(np.max(np.abs(pitch - gt["pitch_sin"][:, 0])))
    errors["base_position_m"] = float(np.max(np.abs(trace["qpos"][:, :3] - gt["qpos"][:, 0, :3])))
    cpu_errors = {key: float(np.max(np.abs(trace[key] - ct[key])))
                  for key in ("observation", "action", "qpos", "qvel", "reward")}
    length_error = int(np.max(np.abs(np.asarray(trials[1]["lengths"], dtype=np.int64) -
                                    np.asarray(gpu["trials"][2]["lengths"], dtype=np.int64))))
    gates = {
        "same_observation_policy": action_error <= limits["common_action_max_abs"],
        "one_step_qpos": first["qpos"] <= limits["one_step_qpos_max_abs"],
        "one_step_qvel": first["qvel"] <= limits["one_step_qvel_max_abs"],
        "episode_lengths": length_error <= limits["episode_length_steps_abs"],
        "no_early_termination_both": all(item == steps for item in gpu["trials"][2]["lengths"]) and
                                      not np.any(trials[1]["terminated"]),
        "return_relative": return_relative <= limits["return_relative"],
        "closed_loop_pitch": errors["pitch_sin"] <= limits["closed_loop_pitch_sin_max_abs"],
        "closed_loop_position": errors["base_position_m"] <= limits["closed_loop_base_position_m_max_abs"],
        "done_sequence": bool(np.array_equal(trace["done"], gt["done"][:, 0])),
    }
    native_median = statistics.median(item["seconds"] for item in trials[1:])
    gpu_median = statistics.median(item["seconds"] for item in gpu["trials"][1:])
    cpu_median = statistics.median(item["seconds"] for item in cpu["trials"][1:])
    summary = {"schema_version": 1, "model_run_id": metadata["model_run_id"],
               "checkpoint_sha256": metadata["checkpoint_sha256"], "source_fingerprint": metadata["source_fingerprint"],
               "workload": {"num_envs": count, "control_steps_per_world": steps,
                            "physics_steps": trials[1]["physics_steps"]},
               "setup_seconds": setup_seconds, "trials": trials, "gates": gates,
               "all_gates_passed": all(gates.values()), "limits_declared_before_measurement": limits,
               "common_action_max_abs": action_error, "first_step_gpu_max_abs": first,
               "closed_loop_gpu_max_abs": errors, "closed_loop_python_cpu_max_abs": cpu_errors,
               "return_relative_max": return_relative, "native_warm_median_seconds": native_median,
               "previous_gpu_warm_median_seconds": gpu_median, "previous_python_cpu_warm_median_seconds": cpu_median,
               "speedup_over_previous_gpu": gpu_median / native_median,
               "speedup_over_previous_python_cpu": cpu_median / native_median,
               "timing_scope": "Same frozen work and host; historical baselines, no trace in warm trials; not end-to-end training speed"}
    np.savez(out / "native-trace.npz", **{key: value for key, value in trace.items() if key != "world"})
    (out / "comparison.json").write_text(json.dumps(summary, default=lambda value: value.tolist(), indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "trials"}, indent=2), flush=True)
    assert all(gates.values()), gates


if __name__ == "__main__":
    main()
