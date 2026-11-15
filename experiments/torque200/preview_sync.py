"""On-demand actor-only snapshots; no periodic export or training checkpoint."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import uuid
import numpy as np
from control_modes import mode_spec


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class SnapshotPolicy:
    def __init__(self, arrays, mode, experiment_config=None):
        self.mode = mode
        self.experiment_config = None
        if mode == "leg_force_bc":
            from force_teacher.options import normalize_options, legacy_options
            self.experiment_config = legacy_options() if experiment_config is None else normalize_options(experiment_config)
        size = mode_spec(mode)["obs_dim"]
        shapes = {"mean": (size,), "std": (size,)}
        for i, (a, b) in enumerate(zip((size, 64, 64), (64, 64, 8))):
            shapes[f"w{i}"] = (a, b)
            shapes[f"b{i}"] = (b,)
        self.arrays = {}
        for key, shape in shapes.items():
            value = np.asarray(arrays[key], dtype=np.float32)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid preview parameter: {key}")
            self.arrays[key] = value.copy()
        if np.any(self.arrays["std"] <= 0):
            raise ValueError("Invalid preview normalization")

    @classmethod
    def from_params(cls, params, mode, experiment_config=None):
        stats, actor = params[:2]
        arrays = {"mean": stats.mean["state"], "std": stats.std["state"]}
        layers = sorted(actor["params"].items())
        if len(layers) != 3:
            raise ValueError("Expected three actor layers")
        for i, (_, layer) in enumerate(layers):
            arrays[f"w{i}"] = layer["kernel"]
            arrays[f"b{i}"] = layer["bias"]
        return cls(arrays, mode, experiment_config)

    def clean_history(self):
        pass  # Both supported actors use one observation frame.

    def run(self, observation):
        a = self.arrays
        x = np.asarray(observation, dtype=np.float32)
        if x.shape != a["mean"].shape:
            raise ValueError("Wrong preview observation shape")
        raw_observation = x
        x = (x - a["mean"]) / a["std"]
        for i in range(3):
            x = x @ a[f"w{i}"] + a[f"b{i}"]
            if i < 2:
                x = x / (1 + np.exp(-np.clip(x, -80, 80)))
        if self.mode == "leg_force_bc":
            from force_teacher.networks import transform_logits
            x = transform_logits(x, raw_observation, xp=np,
                wheel_head=self.experiment_config["wheel_head"],
                wheel_residual_scale=self.experiment_config["wheel_residual_scale"])
        result = np.tanh(x[:4])
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite preview action")
        return result


def snapshot_fingerprint(mode, original_root=None):
    """Exactly one contract owns each mode, including teacher-backed hybrids."""
    spec = mode_spec(mode)
    if mode == "leg_force_bc":
        from force_teacher.contract import source_fingerprint
        return source_fingerprint(original_root)
    if spec["teacher"]:
        from wheel_teacher.contract import source_fingerprint
        return source_fingerprint(original_root)
    if spec["hybrid"]:
        from hybrid.core import source_fingerprint
        return source_fingerprint()
    return None


def read_snapshot(path, mode, request_id, *, original_root=None):
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        if metadata.get("request_id") != request_id:
            return None  # An older request must never satisfy a new open.
        if metadata.get("mode") != mode:
            raise ValueError("Preview snapshot mode mismatch")
        fingerprint = snapshot_fingerprint(mode, original_root)
        if fingerprint is not None and metadata.get("source_fingerprint") != fingerprint:
            raise ValueError("Preview snapshot source fingerprint mismatch")
        config = None
        if mode == "leg_force_bc":
            from force_teacher.options import options_from_metadata
            config = options_from_metadata(metadata)
            sidecar = Path(path).parent / "policy_metadata.json"
            if sidecar.is_file():
                saved = options_from_metadata(json.loads(sidecar.read_text(encoding="utf8")))
                if config != saved:
                    raise ValueError("Preview snapshot experiment configuration mismatch")
        return SnapshotPolicy(data, mode, config), metadata


class SnapshotPublisher:
    def __init__(self, run_dir, mode, original_root=None, experiment_config=None):
        self.run_dir, self.mode = Path(run_dir), mode
        self.original_root = original_root
        self.experiment_config = experiment_config
        self.handled = None
        self.count = 0

    def service(self, steps, params):
        request_path = self.run_dir / "preview.request.json"
        try:
            request = json.loads(request_path.read_text())
        except (FileNotFoundError, ValueError):
            return None
        request_id = request.get("request_id")
        if not request_id or request_id == self.handled or request.get("mode") != self.mode:
            return None
        started = time.perf_counter()
        response = {"request_id": request_id, "mode": self.mode,
                    "run_id": self.run_dir.name, "steps": int(steps), "synced_at": utc_now()}
        fingerprint = snapshot_fingerprint(self.mode, self.original_root)
        if fingerprint is not None:
            response["source_fingerprint"] = fingerprint
            response.update(action_mode=mode_spec(self.mode)["action_mode"],
                            control_hz=mode_spec(self.mode)["control_hz"])
        temporary = self.run_dir / ("preview-" + uuid.uuid4().hex + ".tmp")
        try:
            config = None
            if self.mode == "leg_force_bc":
                from force_teacher.options import normalize_options, options_from_metadata
                sidecar = self.run_dir / "policy_metadata.json"
                saved = options_from_metadata(json.loads(sidecar.read_text(encoding="utf8"))) if sidecar.is_file() else None
                config = normalize_options(self.experiment_config) if self.experiment_config is not None else saved
                if config is None:
                    raise ValueError("Force snapshot needs the model's saved experiment configuration")
                if saved is not None and config != saved:
                    raise ValueError("Publisher experiment configuration differs from saved model")
                response.update(teacher_schema="leg-force-configurable-bc-v4", experiment_config=config)
            actor = SnapshotPolicy.from_params(params, self.mode, config)
            # One small uncompressed NPZ, atomically replaced; no critic/optimizer.
            with temporary.open("wb") as stream:
                np.savez(stream, metadata=json.dumps(response), **actor.arrays)
            temporary.replace(self.run_dir / "preview-policy.npz")
            response.update(bytes=(self.run_dir / "preview-policy.npz").stat().st_size,
                            publish_seconds=time.perf_counter() - started)
        except Exception as exc:
            response["error"] = str(exc)
        finally:
            temporary.unlink(missing_ok=True)
        temporary = self.run_dir / ("preview-response-" + uuid.uuid4().hex + ".tmp")
        temporary.write_text(json.dumps(response), encoding="utf8")
        temporary.replace(self.run_dir / "preview.response.json")
        self.handled = request_id
        self.count += 1
        return response
