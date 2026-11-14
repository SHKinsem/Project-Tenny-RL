"""Auditable per-stage evidence; reports and old policy metadata stay immutable."""
import json
import math
from pathlib import Path

from hybrid.core import ROOT, digest_file, digest_tree, require_metadata, source_fingerprint
from hybrid.timing import Timing

SCHEMA = "hybrid-stage-success-v1"
ATTESTATION = "hybrid-source-attestation-v1"
# Unknown new hybrid modules are included by default. Only known orchestration
# and display files are excluded from the numerical stage they cannot affect.
ORCHESTRATION = {"hybrid/readiness.py", "hybrid/stage_evidence.py", "hybrid/console.py"}
VALIDATION_EXCLUDED = ORCHESTRATION | {
    "hybrid/train.py", "hybrid/checkpoints.py", "hybrid/deployment.py", "hybrid/managed_training.py"}
TRAIN_EXCLUDED = ORCHESTRATION | {"hybrid/validate.py", "hybrid/progress.py"}
TRAIN_ROOTS = {"verify_artifacts.py", "console_train.py", "control_modes.py", "preview_sync.py"}


def read(path):
    path = Path(path)
    if path.is_symlink() or path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("Invalid evidence file")
    return json.loads(path.read_text(encoding="utf8"),
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def inventory():
    paths = [*ROOT.joinpath("assets").rglob("*"), *ROOT.joinpath("hybrid").glob("*.py"),
             *ROOT.joinpath("legwheel").rglob("*.py"), ROOT / "hybrid-wsl.sh",
             *(ROOT / name for name in TRAIN_ROOTS)]
    return {p.relative_to(ROOT).as_posix(): digest_file(p) for p in sorted(paths) if p.is_file()}


def dependencies(stage, files=None):
    if stage not in ("validation", "first", "restart"):
        raise ValueError("Unknown acceptance stage")
    files = inventory() if files is None else files
    excluded = VALIDATION_EXCLUDED if stage == "validation" else TRAIN_EXCLUDED
    return {name: value for name, value in files.items()
            if (name.startswith("assets/") or
                (name.startswith("legwheel/") and name.endswith(".py")) or
                (name.startswith("hybrid/") and name.endswith(".py") and name not in excluded) or
                (stage != "validation" and name in TRAIN_ROOTS))}


def stage_identity(stage, expected, first=None):
    value = dict(stage=stage, sources=dependencies(stage), runtime=expected["runtime"],
                 timing=expected["timing"])
    if stage != "validation":
        value["mode"] = expected["mode"]
    if stage == "restart":
        if not first:
            raise ValueError("Restart requires verified first-training evidence")
        value["first"] = dict(checkpoint_sha256=first["checkpoint_sha256"],
                              report_sha256=first["report_sha256"])
    return json.loads(json.dumps(value))


def _finite(value, maximum):
    return (not isinstance(value, bool) and isinstance(value, (float, int)) and
            math.isfinite(value) and 0 <= value <= maximum)


def validation_report(path, expected, fingerprint):
    report = read(path)
    if (report.get("schema") != "hybrid-local-validation-v1" or
            report.get("runtime_validated") is not True or report.get("error") or
            report.get("source_fingerprint") != fingerprint):
        raise ValueError("Dynamic validation or source provenance is incomplete")
    require_metadata(report, expected["timing"])
    packages = dict(expected["runtime"]["packages"])
    if not report.get("runtime_packages"):
        raise ValueError("Validation package provenance is missing")
    for name, version in report["runtime_packages"].items():
        if packages.get(name.lower().replace("_", "-")) != version:
            raise ValueError("Validation runtime packages changed")
    return report


def training_report(path, expected, fingerprint, stage):
    path = Path(path)
    report, meta = read(path), read(path.parent / "policy_metadata.json")
    require_metadata(meta, dict(action_mode=expected["mode"], source_fingerprint=fingerprint,
                                **expected["timing"]))
    require_metadata(report, dict(action_mode=expected["mode"], control_hz=expected["timing"]["control_hz"]))
    if (report.get("checkpoint_validated") is not True or report.get("all_parameters_finite") is not True or
            report.get("error") or not _finite(report.get("parameter_tree_max_abs_error"), 0) or
            not _finite(report.get("restore_max_abs_error"), 1e-6) or
            not _finite(report.get("export_max_abs_error"), 1e-3) or
            not _finite(report.get("adapter_max_abs_error"), 1e-5) or
            not report.get("finite_training_losses_checked") or
            type(report.get("actual_progress_steps")) is not int or report["actual_progress_steps"] < 16):
        raise ValueError("Short training acceptance is incomplete")
    if stage == "restart" and (report.get("warm_restart_changed_network_leaves", 0) <= 0 or
                                report.get("warm_restart_network_max_abs_change", 0) <= 0):
        raise ValueError("Warm restart did not demonstrate a network update")
    checkpoint = Path(report["checkpoint"])
    if (not checkpoint.is_absolute() or not checkpoint.resolve().is_relative_to(path.parent.resolve()) or
            checkpoint.name != report.get("checkpoint_name")):
        raise ValueError("Checkpoint is outside its accepted run")
    if digest_tree(checkpoint) != report.get("checkpoint_sha256"):
        raise ValueError("Checkpoint content changed")
    export_hash = digest_file(path.parent / "policy.cpp.txt")
    if export_hash != report.get("export_sha256") or export_hash != meta.get("export_sha256"):
        raise ValueError("Export content changed")
    return report


def record(stage, report_path, attestation_path, expected, first=None):
    """Called only after both child and logging processes exit successfully."""
    report_path, attestation_path = Path(report_path).resolve(), Path(attestation_path).resolve()
    attestation = read(attestation_path)
    value = dict(schema=SCHEMA, stage=stage, identity=stage_identity(stage, expected, first),
                 report_path=str(report_path), report_sha256=digest_file(report_path),
                 attestation_path=str(attestation_path), attestation_sha256=digest_file(attestation_path),
                 producer_exit_code=0, logger_exit_code=0)
    if stage != "validation":
        checked = training_report(report_path, expected, attestation["source_fingerprint"], stage)
        value.update(checkpoint=checked["checkpoint"], checkpoint_sha256=checked["checkpoint_sha256"],
                     metadata_sha256=digest_file(report_path.parent / "policy_metadata.json"),
                     export_sha256=checked["export_sha256"])
    else:
        validation_report(report_path, expected, attestation["source_fingerprint"])
    verify(value, expected, stage, first)
    return value


def verify(value, expected, stage, first=None):
    if (value.get("schema") != SCHEMA or value.get("stage") != stage or
            type(value.get("producer_exit_code")) is not int or value["producer_exit_code"] != 0 or
            type(value.get("logger_exit_code")) is not int or value["logger_exit_code"] != 0 or
            value.get("identity") != stage_identity(stage, expected, first)):
        raise ValueError("Acceptance stage identity/exit proof mismatch")
    attestation_path = Path(value["attestation_path"])
    report_path = Path(value["report_path"])
    if (digest_file(attestation_path) != value["attestation_sha256"] or
            digest_file(report_path) != value["report_sha256"]):
        raise ValueError("Acceptance report/attestation changed")
    attestation = read(attestation_path)
    if (attestation.get("schema") != ATTESTATION or
            dependencies(stage, attestation["files"]) != value["identity"]["sources"]):
        raise ValueError("Accepted source inventory does not match current stage dependencies")
    if stage == "validation":
        return validation_report(report_path, expected, attestation["source_fingerprint"])
    report = training_report(report_path, expected, attestation["source_fingerprint"], stage)
    if (report["checkpoint"] != value["checkpoint"] or
            report["checkpoint_sha256"] != value["checkpoint_sha256"] or
            digest_file(report_path.parent / "policy_metadata.json") != value["metadata_sha256"] or
            report["export_sha256"] != value["export_sha256"]):
        raise ValueError("Training stage artifact binding changed")
    return report


def load_validation_reference(path, control_hz, runtime=None):
    value = read(path)
    if runtime is None:
        from hybrid.readiness import runtime_identity
        runtime = runtime_identity()
    expected = dict(runtime=runtime, timing=Timing(control_hz).contract())
    # Return the original report; never replace its source fingerprint or claim
    # that a new simulator run occurred. Compatibility comes from this record.
    return verify(value, expected, "validation")


def check_restore_reference(path, checkpoint, expected):
    from hybrid.readiness import runtime_identity
    value = read(path)
    stage_expected = dict(mode=expected["action_mode"], timing=Timing(expected["control_hz"]).contract(),
                          runtime=runtime_identity())
    report = verify(value, stage_expected, "first")
    if Path(report["checkpoint"]).resolve() != Path(checkpoint).resolve():
        raise ValueError("Restore reference names a different checkpoint")
    actual = read(Path(checkpoint).parent.parent / "policy_metadata.json")
    # Only source identity is delegated to the stage proof. All semantic fields
    # still need exact equality, including observation, limits and timing.
    require_metadata(actual, {k: v for k, v in expected.items() if k != "source_fingerprint"})
    return report
