"""Validated host-side adapter, not an embedded firmware/real-motor deployment.

Call torque at 400 Hz with fresh joint positions, holding action for the saved ctrl_dt (5 or 10 ms).
Deployment must reproduce observation timing: mean of n_substeps FINAL SENT torques /
TORQUE_LIMITS, after delay/filter/limit. No contact-force sensor is required.
"""
import json
from pathlib import Path

from hybrid.core import (HIP_RANGE, read_geometry, scan_geometry, metadata,
                         require_metadata, map_action, digest_file, validate_positions, MAX_LENGTH_VELOCITY)


class HybridPolicy:
    def __init__(self, export_path, metadata_path, expected_mode, expected_control_hz=None):
        # Check semantics before importing or constructing the old NumPy kernel.
        self.legs = read_geometry()
        self.scan = scan_geometry(self.legs)
        self.meta = json.loads(Path(metadata_path).read_text(encoding="utf8"))
        if self.meta.get("action_delay_ms") != 0.0:
            raise ValueError("HybridPolicy supports only the zero-delay training contract; no hidden delay queue")
        # Default derives the rate from the saved contract, never a UI preset.
        hz = self.meta.get("control_hz") if expected_control_hz is None else expected_control_hz
        require_metadata(self.meta, metadata(expected_mode, self.scan, hz))
        self.control_hz = self.meta["control_hz"]
        self.ctrl_dt = self.meta["ctrl_dt"]
        self.sim_dt = self.meta["sim_dt"]
        self.n_substeps = self.meta["n_substeps"]
        if self.meta.get("export_sha256") != digest_file(export_path):
            raise ValueError("Export bytes are not bound to this policy metadata")
        from legwheel.deployment.numpy_policy import RL_Policy
        self.policy = RL_Policy(path=Path(export_path))
        self.mode = expected_mode

    def action(self, observation):
        self.policy.clean_history()  # This contract has exactly one 24D frame.
        return self.policy.run(observation)

    def torque(self, action, hip_positions, knee_positions, hip_velocities, knee_velocities):
        import math
        jac = validate_positions(self.legs, hip_positions, knee_positions,
                                 [r["jmax_bound"] for r in self.scan])
        if len(hip_velocities) != 2 or len(knee_velocities) != 2:
            raise ValueError("Expected left/right hip and knee velocities")
        if any(not math.isfinite(float(v)) for v in (*hip_velocities, *knee_velocities)):
            raise ValueError("Nonfinite joint velocity")
        for i, leg in enumerate(self.legs):
            partial = leg.evaluate(hip_positions[i], knee_positions[i])[2]
            residual_v = knee_velocities[i]-leg.slope(hip_positions[i])*hip_velocities[i]
            if abs(partial*residual_v) > MAX_LENGTH_VELOCITY:
                raise ValueError("Constraint length-velocity residual exceeds diagnostic limit")
        return map_action(action, jac, self.meta["force_scales_n"], self.mode)
