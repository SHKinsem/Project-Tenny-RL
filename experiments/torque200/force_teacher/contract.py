"""Hybrid-compatible dynamics metadata plus independently bound teacher semantics."""
import hashlib
import json
from pathlib import Path
from hybrid.core import (metadata as hybrid_metadata, read_geometry, scan_geometry,
                         source_fingerprint as hybrid_fingerprint)
from wheel_teacher.contract import (original_path, teacher_policy_path, AUTHOR_CHECKPOINT,
                                    file_digest, tree_digest)
from force_teacher.networks import POLICY_HEAD_VERSION, WHEEL_RESIDUAL_SCALE, PREVIOUS_TORQUE_CLIP
from force_teacher.options import normalize_options, legacy_options, options_from_metadata

MODE = "leg_force_bc"
SCHEMA = "leg-force-configurable-bc-v4"
LEGACY_SCHEMA = "leg-force-reachable-teacher-bc-v3"
# Explicitly verified production v3 source pair. This is a migration allowlist,
# not permission to accept arbitrary historical source fingerprints.
LEGACY_SOURCE_PAIRS = {("4ca2bc7387038679a16eaedde390b4f4db7b6e678007d83fba268ccc9137960a",
                        "234757fe38df11388b3303f3a6686f9e4877727769b0ca762968a979f1a94487")}
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


def metadata(original_root=None, experiment_config=None):
    options = normalize_options(experiment_config)
    result = hybrid_metadata("virtual_leg_force", scan_geometry(read_geometry()), 100)
    result.update(mode=MODE, teacher_schema=SCHEMA, force_source_fingerprint=source_fingerprint(original_root),
                  teacher_checkpoint=AUTHOR_CHECKPOINT, teacher_sha256=file_digest(teacher_policy_path(original_root)),
                  teacher_label_semantics="projected reachable oracle on true motor student states; 100 Hz held axial force and wheel torque; 400 Hz Jacobian refresh",
                  unreachable_label_policy="explicit force projection with saturation/error reporting; retain reachable executed student trajectories",
                  teacher_information="author 26D policy and private wheel-target memory; student observes only deployable 24D state",
                  teacher_collection_version=2, training_profile="standing_v2" if options["standing_only"] else "commands_v1",
                  evaluation_policy="deterministic" if options["deterministic_eval"] else "stochastic",
                  normalization_policy="fixed_bc" if options["fixed_normalization"] else "updating_welford",
                  policy_head=POLICY_HEAD_VERSION if options["wheel_head"] == "residual" else "absolute_tanh_v1",
                  wheel_residual_scale=options["wheel_residual_scale"],
                  previous_torque_clip=PREVIOUS_TORQUE_CLIP, export_float_precision="float32-roundtrip-10sig")
    if options["teacher_source"] == "reference":
        result.update(teacher_label_semantics="reference servo trajectories; measured cycle force fit and mean wheel torque",
                      unreachable_label_policy="discard unreachable reference cycles; no silent clipping",
                      teacher_collection_version=1)
    result["experiment_config"] = options
    result["experiment_config_sha256"] = hashlib.sha256(json.dumps(options,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    return result


def require_metadata(actual, expected):
    for key, value in expected.items():
        if actual.get(key) != value:
            raise ValueError(f"Incompatible force teacher model: {key}")


def validate_metadata(actual, original_root=None, experiment_config=None):
    """Validate new sources or one explicitly migrated v3 source pair and semantics."""
    options = options_from_metadata(actual)
    if experiment_config is not None and options != normalize_options(experiment_config):
        raise ValueError("Resume configuration conflicts with saved experiment")
    expected = metadata(original_root, options)
    if actual.get("teacher_schema") == LEGACY_SCHEMA:
        pair = (actual.get("source_fingerprint"), actual.get("force_source_fingerprint"))
        if pair not in LEGACY_SOURCE_PAIRS:
            raise ValueError("Unrecognized legacy force source version")
        expected.pop("experiment_config")
        expected.pop("experiment_config_sha256")
        expected.update(teacher_schema=LEGACY_SCHEMA,source_fingerprint=pair[0],force_source_fingerprint=pair[1])
    unknown = set(actual) - set(expected) - {"export_sha256"}
    if unknown:
        raise ValueError("Unknown force metadata fields: " + ", ".join(sorted(unknown)))
    require_metadata(actual, expected)
    return options


def verified_report(run, report, original_root=None, experiment_config=None):
    run = Path(run).resolve()
    actual = json.loads((run / "policy_metadata.json").read_text(encoding="utf8"))
    options = validate_metadata(actual, original_root, experiment_config)
    if (report.get("mode") != MODE or report.get("checkpoint_validated") is not True or
            report.get("source_fingerprint") != actual["source_fingerprint"]):
        raise ValueError("Not a verified leg-force teacher checkpoint")
    checkpoint, export = Path(report["checkpoint"]).resolve(), Path(report["export_path"]).resolve()
    if not checkpoint.is_relative_to(run) or not export.is_relative_to(run):
        raise ValueError("Artifacts escaped run directory")
    if (report.get("checkpoint_sha256") != tree_digest(checkpoint) or
            report.get("export_sha256") != file_digest(export) or
            actual.get("export_sha256") != report["export_sha256"]):
        raise ValueError("Force teacher artifact bytes changed")
    return options


def restore_guard(checkpoint, original_root=None, experiment_config=None):
    checkpoint = Path(checkpoint).resolve()
    run = checkpoint.parent.parent
    report = json.loads((run / "report.json").read_text(encoding="utf8"))
    if Path(report.get("checkpoint", "")).resolve() != checkpoint:
        raise ValueError("Not the verified checkpoint path")
    return verified_report(run, report, original_root, experiment_config)
