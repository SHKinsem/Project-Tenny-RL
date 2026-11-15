"""Task-specific descriptor export. The generic evaluator does not import this."""
from dataclasses import asdict
import hashlib
import importlib
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET


def _arrays(path):
    import numpy as np
    text = path.read_text(encoding="utf8")
    result = {}
    for name in ("state_mean", "state_std", "weights", "biases"):
        match = re.search(r"float " + name + r"\[\]\s*=\s*\{(.*?)\};", text, re.S)
        if match is None:
            raise ValueError("Missing exported array: " + name)
        tokens = match.group(1).strip().split(",")
        if tokens[-1].strip() == "":
            tokens.pop()
        values = [float(token.strip().removesuffix("f").removesuffix("F")) for token in tokens]
        result[name] = np.asarray(values, dtype=np.float32)
        if not np.isfinite(result[name]).all():
            raise ValueError("Nonfinite exported policy values")
    return result


def policy_descriptor(export_path, metadata, ppo_config):
    """Decode a supported export using artifact dimensions, not fixed layer widths."""
    import numpy as np
    export_path = Path(export_path)
    if hashlib.sha256(export_path.read_bytes()).hexdigest() != metadata.get("export_sha256"):
        raise ValueError("Policy export hash differs from its metadata")
    arrays = _arrays(export_path)
    action_size = len(metadata["action_order"])
    hidden = ppo_config["network_factory"]["policy_hidden_layer_sizes"]
    dimensions = [int(metadata["actor_dim"]), *map(int, hidden), 2 * action_size]
    if any(size <= 0 for size in dimensions):
        raise ValueError("Invalid policy layer sizes")
    weights, biases = arrays["weights"], arrays["biases"]
    if (weights.size != sum(a*b for a, b in zip(dimensions, dimensions[1:])) or
            biases.size != sum(dimensions[1:]) or arrays["state_mean"].size != dimensions[0] or
            arrays["state_std"].size != dimensions[0] or np.any(arrays["state_std"] <= 0)):
        raise ValueError("Policy arrays do not match the declared architecture")
    layers, wi, bi = [], 0, 0
    for index, (inputs, outputs) in enumerate(zip(dimensions, dimensions[1:])):
        # Saved stochastic heads contain mean and scale logits. Deterministic
        # evaluation uses only the first action_size rows of the final layer.
        used_outputs = action_size if index == len(dimensions)-2 else outputs
        layers.append({"weights": weights[wi:wi+inputs*outputs].reshape(outputs, inputs)[:used_outputs].tolist(),
                       "bias": biases[bi:bi+used_outputs].tolist(),
                       "activation": "silu" if index < len(dimensions)-2 else "identity"})
        wi += inputs*outputs
        bi += outputs
    result = {"type": "mlp", "mean": arrays["state_mean"].tolist(), "std": arrays["state_std"].tolist(),
              "layers": layers, "action_size": action_size, "output_activation": "tanh"}
    return _apply_head(result, metadata)


def _apply_head(result, metadata):
    head = metadata.get("policy_head")
    if head == "wheel_logit_residual_v1":
        if metadata.get("observation_version") != "torque24-mean-sent-v1" or len(result["mean"]) != 24 or result["action_size"] != 4:
            raise ValueError("Unsupported observation layout for saved residual head")
        result.update(skip_indices=[-1, -1, 22, 23], network_scales=[1., 1., metadata["wheel_residual_scale"],
                      metadata["wheel_residual_scale"]], skip_scales=[0., 0., 1., 1.],
                      skip_transform="atanh", skip_clip_epsilon=metadata["previous_torque_clip"])
    elif head not in (None, "standard_mlp"):
        raise ValueError("Unsupported saved policy head: " + str(head))
    return result


def policy_template(metadata, ppo_config):
    """Architecture and head only; live Brax parameters must replace every array."""
    action_size = len(metadata["action_order"])
    dimensions = [int(metadata["actor_dim"]),
                  *map(int, ppo_config["network_factory"]["policy_hidden_layer_sizes"]), action_size]
    if any(size <= 0 for size in dimensions):
        raise ValueError("Invalid policy layer sizes")
    layers = [{"weights": [[0.] * inputs for _ in range(outputs)], "bias": [0.] * outputs,
               "activation": "silu" if index < len(dimensions)-2 else "identity"}
              for index, (inputs, outputs) in enumerate(zip(dimensions, dimensions[1:]))]
    result = {"type": "mlp", "mean": [0.] * dimensions[0], "std": [1.] * dimensions[0],
              "layers": layers, "action_size": action_size, "output_activation": "tanh"}
    return _apply_head(result, metadata)


def _production_core(runtime_src):
    runtime_src = Path(runtime_src).expanduser().resolve()
    sys.path.insert(0, str(runtime_src))
    try:
        core = importlib.import_module("hybrid.core")
    finally:
        sys.path.pop(0)
    if not Path(core.__file__).resolve().is_relative_to(runtime_src):
        raise RuntimeError("A different hybrid.core is already imported; export in a fresh process")
    return core


def task_descriptor_from_config(cfg, meta, runtime_src):
    """Validate the live task contract, independently of a completed policy export."""
    import numpy as np
    runtime_src = Path(runtime_src).expanduser().resolve()
    if (meta.get("schema") != "wheel-center-axial-v1" or meta.get("observation_version") != "torque24-mean-sent-v1"
            or meta.get("action_mode") not in ("virtual_leg_force", "direct_torque", "matched_envelope")
            or meta.get("actor_dim") != 24 or meta.get("action_order") !=
            ["left_leg", "right_leg", "left_wheel", "right_wheel"]):
        raise ValueError("Unsupported task/observation contract")
    core = _production_core(runtime_src)
    if core.source_fingerprint() != meta.get("source_fingerprint"):
        raise ValueError("Runtime source differs from the saved model")
    if (cfg["noise_config"]["level"] != 0 or cfg["push_config"]["enable"] or
            cfg["cmd_config"]["cmd_prob"] != 0 or cfg["action_delay_steps"] != 0 or
            cfg["action_repeat"] != 1 or cfg["obs_history_len"] != 1 or cfg["last_act_obs"] or
            cfg["hold_error_clip"]["enable"] or cfg["obs_normalize"]["state_mean"] is not None or
            cfg["obs_normalize"]["privileged_state_mean"] is not None or
            cfg["reward_config"]["termination_penalty_enable"] or cfg["reward_config"]["do_deconflict"]):
        raise ValueError("Saved task uses unsupported noise/push/history/delay/normalization/reward settings")
    if (any(float(value) != 0 for value in meta.get("force_bias_n", [0., 0.])) or
            cfg["wheel_pos_nosin"] or float(cfg["wheel_sin_scale"]) != .1):
        raise ValueError("Unsupported force bias or wheel-position observation settings")
    if meta.get("training_profile") not in (None, "standing_v2"):
        raise ValueError("Unsupported saved training profile")
    xml = Path(cfg["_XML_PATH"].removeprefix("unserializable object: "))
    if not xml.is_absolute():
        xml = runtime_src / xml
    xml = xml.resolve()
    legs = core.read_geometry(xml)
    scans = core.scan_geometry(legs, hip_range=tuple(meta["hip_range"]))

    def flatten(leg):
        item = asdict(leg)
        return [item["hip_ref"], item["knee_ref"], *item["poly"], *item["knee_offset"],
                *[x for row in item["knee_rotation"] for x in row], *item["knee_anchor"],
                *item["knee_axis"], *item["wheel_center"]]

    joints = {part: [f"{part}_{side}_joint" for side in ("left", "right")]
              for part in ("thigh", "calf", "wheel")}
    actuator_nodes = ET.parse(xml).getroot().find("actuator")
    actuator_map = {node.get("joint"): node.get("name") for node in actuator_nodes}
    joint_order = [*joints["thigh"], *joints["wheel"]]
    if any(not actuator_map.get(name) for name in joint_order):
        raise ValueError("Actuator names must resolve to the configured joints")
    ranges = cfg["cmd_config"]
    command_scales = []
    for name in ("lin_vel_x", "ang_vel_yaw", "height_value", "roll_value"):
        lo, hi = np.asarray(ranges[name], dtype=np.float32)
        command_scales.append(float((hi-lo)/np.float32(2)))
    task = {
        "hip_joints": joints["thigh"], "knee_joints": joints["calf"], "wheel_joints": joints["wheel"],
        "actuators": [actuator_map[name] for name in joint_order], "wheel_bodies": ["wheel_left", "wheel_right"],
        "root_body": "root", "gyro_sensor": "root_gyro", "local_linvel_sensor": "root_local_linvel",
        "action_mode": meta["action_mode"], "leg_geometry_left": flatten(legs[0]), "leg_geometry_right": flatten(legs[1]),
        "jmax": [row["jmax_bound"] for row in scans], "force_scales": list(meta["force_scales_n"]),
        "torque_scales": list(meta["torque_limits_nm"]), "hip_range": list(meta["hip_range"]),
        "thigh_range": list(cfg["action_range"]["thigh"]), "command_scales": command_scales,
        "wheel_radius": cfg["wheel_radius"], "sinyaw_scale": cfg["sinyaw_scale"],
        "cmd_lin_vel_x_zero_margin": cfg["cmd_lin_vel_x_zero_margin"],
        "cmd_ang_vel_zero_margin": cfg["cmd_ang_vel_zero_margin"],
        "sinpitch_zero": cfg["sinpitch_zero"], "sinyaw": cfg["sinyaw"],
        "wheel_sinpos_start": cfg["wheel_sinpos_start"], "geometry_schema": meta["schema"],
        "observation_version": meta["observation_version"],
        "equality_position_limit": meta["validity"]["equality_position_rad"],
        "length_velocity_limit": meta["validity"]["length_velocity_m_s"],
        "minimum_jacobian_ratio": meta["validity"]["minimum_jacobian_ratio"],
        "noise_level": 0., "command_probability": 0., "action_delay_steps": 0.,
        "action_repeat": 1., "obs_history_len": 1., "push_enabled": False,
        "domain_randomization": False, "last_act_obs": False, "hold_error_clip": False,
        "obs_normalization": False, "termination_penalty": False, "do_deconflict": False,
    }
    for name in ("reference_dt", "sigma", "sigma_wheel", "height_scale", "vel_scale", "roll_scale", "pitch_scale",
                 "zero_keep_scale", "energy_scale", "action_rate_scale", "pitch_velocity_penalty_scale",
                 "roll_thigh_coupling", "height_feasible_projection"):
        task[name] = cfg["reward_config"][name]
    for name in ("lin_vel_x", "ang_vel_yaw", "height_value", "roll_value"):
        task["enable_"+name] = ranges["enable"][name]
    return {"schema_version": 1, "xml_path": str(xml), "physics_dt": cfg["sim_dt"],
            "task": {"type": "legwheel", "config": task},
            "eval_defaults": {"num_envs": 1, "rollout_steps": meta["episode_length"],
                              "episode_steps": meta["episode_length"], "control_dt": cfg["ctrl_dt"], "seed": 0},
            "metadata": {"source_fingerprint": meta["source_fingerprint"],
                         "action_order": meta["action_order"], "task_schema": meta["schema"]}}


def descriptor_from_run(run_dir, runtime_src=None):
    """Export a proven deterministic hybrid task without importing JAX/Brax."""
    run = Path(run_dir).expanduser().resolve()
    runtime_src = Path(runtime_src or run.parents[3] / "src").expanduser().resolve()
    meta = json.loads((run / "policy_metadata.json").read_text(encoding="utf8"))
    cfg = json.loads((run / "env_config.json").read_text(encoding="utf8"))
    ppo = json.loads((run / "ppo_config.json").read_text(encoding="utf8"))
    report = json.loads((run / "report.json").read_text(encoding="utf8"))
    if not report.get("completed") or not report.get("checkpoint_validated"):
        raise ValueError("Require a completed and verified saved run")
    result = task_descriptor_from_config(cfg, meta, runtime_src)
    result["policy"] = policy_descriptor(run / "policy.cpp.txt", meta, ppo)
    result["metadata"].update(model_run_id=run.name, checkpoint_sha256=report["checkpoint_sha256"],
                              export_sha256=meta["export_sha256"])
    return result
