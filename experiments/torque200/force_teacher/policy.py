"""Precision-preserving export and strict reader for the force-only residual head."""
import json
from pathlib import Path
import numpy as np

from force_teacher.contract import metadata, require_metadata, file_digest
from force_teacher.networks import transform_logits


def _validated_parameters(params):
    stats, actor = params[:2]
    mean = np.asarray(stats.mean["state"], dtype=np.float32)
    std = np.asarray(stats.std["state"], dtype=np.float32)
    if mean.shape != (24,) or std.shape != (24,) or np.any(std <= 0):
        raise ValueError("Invalid force export normalization")
    layers = sorted(actor["params"].items())
    if len(layers) != 3:
        raise ValueError("Expected three force actor layers")
    kernels, biases = [], []
    for (_, layer), shape in zip(layers, ((24,64), (64,64), (64,8)), strict=True):
        kernel, bias = np.asarray(layer["kernel"], dtype=np.float32), np.asarray(layer["bias"], dtype=np.float32)
        if kernel.shape != shape or bias.shape != (shape[1],):
            raise ValueError("Invalid force actor parameter shape")
        kernels.append(kernel.T.reshape(-1))
        biases.append(bias)
    arrays = dict(state_mean=mean, state_std=std, weights=np.concatenate(kernels), biases=np.concatenate(biases))
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Nonfinite force export parameters")
    return arrays


def _write_export(params, out_path, checkpoint_label):
    arrays = _validated_parameters(params)
    lines = [f"// Extracted from checkpoint: {checkpoint_label}",
             "// Force actor requires policy_metadata.json and the wheel_logit_residual_v1 runtime."]
    for name, values in arrays.items():
        lines.append(f"float {name}[] = {{")
        # >=9 significant digits round-trip float32, including tiny normalizer std.
        for offset in range(0, len(values), 12):
            lines.append("    " + ", ".join(format(float(x), ".10g") for x in values[offset:offset+12]) + ",")
        lines.append("};\n")
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf8")
    return path


def export_params(checkpoint, out_path):
    """Same four C arrays/layout as legacy export, with float32 round-trip precision."""
    from legwheel.deployment.extract import load
    return _write_export(load(str(checkpoint)), out_path, str(checkpoint))


class ForcePolicy:
    def __init__(self, export_path, metadata_path, original_root=None):
        self.meta = json.loads(Path(metadata_path).read_text(encoding="utf8"))
        require_metadata(self.meta, metadata(original_root))
        if self.meta.get("export_sha256") != file_digest(export_path):
            raise ValueError("Force export bytes are not bound to metadata")
        from legwheel.deployment.numpy_policy import load_exported
        self.arrays = load_exported(Path(export_path))
        if any(not np.isfinite(value).all() for value in self.arrays.values() if isinstance(value, np.ndarray)):
            raise ValueError("Nonfinite force policy parameters")
        if np.any(self.arrays["std"] <= 0):
            raise ValueError("Invalid force normalization std")
        self.control_hz = self.meta["control_hz"]

    def clean_history(self):
        pass  # The previous executed torque is explicitly present in observation.

    def action(self, observation):
        obs = np.asarray(observation, dtype=np.float32)
        if obs.shape != (24,) or not np.isfinite(obs).all():
            raise ValueError("Invalid force policy observation")
        a = self.arrays
        x = (obs - a["mean"]) / a["std"]
        for i in (1,2,3):
            x = a[f"w{i}"] @ x + a[f"b{i}"]
            if i < 3:
                x = x / (1 + np.exp(-np.clip(x, -80, 80)))
        # Use raw logits; never invert the already squashed legacy actor output.
        action = np.tanh(transform_logits(x, obs)[:4]).astype(np.float32)
        if not np.isfinite(action).all():
            raise FloatingPointError("Nonfinite force policy action")
        return action

    def run(self, observation):
        return self.action(observation)
