"""Hybrid-compatible dynamics metadata plus independently bound teacher semantics."""
import hashlib
import json
from pathlib import Path
from hybrid.core import (metadata as hybrid_metadata, read_geometry, scan_geometry,
                         source_fingerprint as hybrid_fingerprint)
from wheel_teacher.contract import (original_path, teacher_policy_path, AUTHOR_CHECKPOINT,
                                    file_digest, tree_digest)
from force_teacher.networks import POLICY_HEAD_VERSION, WHEEL_RESIDUAL_SCALE, PREVIOUS_TORQUE_CLIP

MODE = "leg_force_bc"
SCHEMA = "leg-force-reachable-teacher-bc-v3"
ROOT = Path(__file__).resolve().parents[1]
digest_file, digest_tree = file_digest, tree_digest


def source_fingerprint(original_root=None):
    digest = hashlib.sha256(bytes.fromhex(hybrid_fingerprint()))
    roots = [ROOT / "force_teacher", ROOT / "wheel_teacher",
             original_path(original_root) / "mujoco_legwheel_rl_public/legwheel",
             original_path(original_root) / "mujoco_legwheel_rl_public/assets",
             original_path(original_root) / "mujoco_esp32_robot_deploy_public/host"]
    for index, root in enumerate(roots):
        if not root.is_dir():
            raise FileNotFoundError(f"Missing force teacher source group {index}")
        for path in sorted(root.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                digest.update(f"{index}/{path.relative_to(root).as_posix()}".encode() + b"\0")
                digest.update(bytes.fromhex(file_digest(path)))
    policy = teacher_policy_path(original_root)
    for path in (policy, policy.with_suffix(".h")):
        digest.update(path.name.encode() + bytes.fromhex(file_digest(path)))
    return digest.hexdigest()


def metadata(original_root=None):
    result = hybrid_metadata("virtual_leg_force", scan_geometry(read_geometry()), 100)
    result.update(mode=MODE, teacher_schema=SCHEMA, force_source_fingerprint=source_fingerprint(original_root),
                  teacher_checkpoint=AUTHOR_CHECKPOINT, teacher_sha256=file_digest(teacher_policy_path(original_root)),
                  teacher_label_semantics="projected reachable oracle on true motor student states; 100 Hz held axial force and wheel torque; 400 Hz Jacobian refresh",
                  unreachable_label_policy="explicit force projection with saturation/error reporting; retain reachable executed student trajectories",
                  teacher_information="author 26D policy and private wheel-target memory; student observes only deployable 24D state",
                  teacher_collection_version=2, training_profile="standing_v2",
                  evaluation_policy="deterministic", normalization_policy="fixed_bc",
                  policy_head=POLICY_HEAD_VERSION, wheel_residual_scale=WHEEL_RESIDUAL_SCALE,
                  previous_torque_clip=PREVIOUS_TORQUE_CLIP, export_float_precision="float32-roundtrip-10sig")
    return result


def require_metadata(actual, expected):
    for key, value in expected.items():
        if actual.get(key) != value:
            raise ValueError(f"Incompatible force teacher model: {key}")


def verified_report(run, report, original_root=None):
    run = Path(run).resolve()
    expected = metadata(original_root)
    actual = json.loads((run / "policy_metadata.json").read_text(encoding="utf8"))
    require_metadata(actual, expected)
    if (report.get("mode") != MODE or report.get("checkpoint_validated") is not True or
            report.get("source_fingerprint") != expected["source_fingerprint"]):
        raise ValueError("Not a verified leg-force teacher checkpoint")
    checkpoint, export = Path(report["checkpoint"]).resolve(), Path(report["export_path"]).resolve()
    if not checkpoint.is_relative_to(run) or not export.is_relative_to(run):
        raise ValueError("Artifacts escaped run directory")
    if (report.get("checkpoint_sha256") != tree_digest(checkpoint) or
            report.get("export_sha256") != file_digest(export) or
            actual.get("export_sha256") != report["export_sha256"]):
        raise ValueError("Force teacher artifact bytes changed")


def restore_guard(checkpoint, original_root=None):
    checkpoint = Path(checkpoint).resolve()
    run = checkpoint.parent.parent
    report = json.loads((run / "report.json").read_text(encoding="utf8"))
    if Path(report.get("checkpoint", "")).resolve() != checkpoint:
        raise ValueError("Not the verified checkpoint path")
    verified_report(run, report, original_root)
