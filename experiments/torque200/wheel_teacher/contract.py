"""Pure standard-library identity for the wheel-torque imitation experiment."""
import hashlib
import json
from pathlib import Path

MODE = "wheel_torque_bc"
SCHEMA = "wheel-torque-bc-v1"
OBSERVATION_VERSION = "original20-wheel-torque-history4-velocity2-v1"
WHEEL_TORQUE_LIMIT = 0.0295
CONTROL_HZ = 100
PHYSICS_HZ = 400
AUTHOR_CHECKPOINT = "ckpts/newobs-norecovery/000403660800"
ROOT = Path(__file__).resolve().parents[1]


def original_path(original_root=None):
    return Path(original_root or Path.home() / "projects/tenny-reference").resolve()


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_digest(path):
    root = Path(path)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("Checkpoint must be a real directory")
    digest = hashlib.sha256()
    files = sorted(root.rglob("*"))
    if any(p.is_symlink() for p in files):
        raise ValueError("Checkpoint symlinks are not allowed")
    files = [p for p in files if p.is_file()]
    if not files:
        raise ValueError("Empty checkpoint")
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(bytes.fromhex(file_digest(path)))
    return digest.hexdigest()


digest_file = file_digest
digest_tree = tree_digest


def teacher_policy_path(original_root=None):
    path = original_path(original_root) / "mujoco_esp32_robot_deploy_public/src/policy/rl_policy.cpp"
    if not path.is_file():
        raise FileNotFoundError("Author pretrained policy is missing")
    text = path.read_text(encoding="utf8")
    header = path.with_suffix(".h").read_text(encoding="utf8")
    if AUTHOR_CHECKPOINT not in text or AUTHOR_CHECKPOINT not in header:
        raise ValueError("Teacher must be the explicitly selected author pretrained checkpoint")
    return path


def source_fingerprint(original_root=None):
    original = original_path(original_root)
    roots = [original / "mujoco_legwheel_rl_public/legwheel",
             original / "mujoco_legwheel_rl_public/assets", ROOT / "wheel_teacher"]
    digest = hashlib.sha256()
    for index, root in enumerate(roots):
        if not root.is_dir():
            raise FileNotFoundError(f"Missing source group {index}")
        for path in sorted(root.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                digest.update(f"{index}/{path.relative_to(root).as_posix()}".encode() + b"\0")
                digest.update(bytes.fromhex(file_digest(path)))
    for name in ("console_train.py", "control_modes.py", "preview_sync.py", "verify_artifacts.py"):
        path = ROOT / name
        digest.update(name.encode() + b"\0" + bytes.fromhex(file_digest(path)))
    policy = teacher_policy_path(original)
    for path in (policy, policy.with_suffix(".h")):
        digest.update(path.name.encode() + b"\0" + bytes.fromhex(file_digest(path)))
    return digest.hexdigest()


def metadata(original_root=None):
    return dict(schema=SCHEMA, mode=MODE, action_mode=MODE, control_hz=CONTROL_HZ,
                physics_hz=PHYSICS_HZ, ctrl_dt=.01, sim_dt=.0025, n_substeps=4,
                actor_dim=26, critic_dim=55, observation_version=OBSERVATION_VERSION,
                action_order=["left_leg_delta_position", "right_leg_delta_position",
                              "left_wheel_torque", "right_wheel_torque"],
                wheel_torque_limits_nm=[WHEEL_TORQUE_LIMIT] * 2,
                leg_action_scale_rad=.25, leg_position_range_rad=[-.5, 0.],
                action_delay_steps=0, teacher_checkpoint=AUTHOR_CHECKPOINT,
                teacher_sha256=file_digest(teacher_policy_path(original_root)),
                source_fingerprint=source_fingerprint(original_root))


def require_metadata(actual, expected):
    for key, value in expected.items():
        if actual.get(key) != value:
            raise ValueError(f"Incompatible wheel torque model: {key}")


def verified_report(run, report, original_root=None):
    run = Path(run).resolve()
    meta = json.loads((run / "policy_metadata.json").read_text(encoding="utf8"))
    expected = metadata(original_root)
    require_metadata(meta, expected)
    if (report.get("mode") != MODE or report.get("checkpoint_validated") is not True or
            report.get("source_fingerprint") != expected["source_fingerprint"]):
        raise ValueError("Not a validated wheel torque checkpoint")
    checkpoint = Path(report["checkpoint"]).resolve()
    export = Path(report["export_path"]).resolve()
    if not checkpoint.is_relative_to(run) or not export.is_relative_to(run):
        raise ValueError("Artifacts escaped run directory")
    if (report.get("checkpoint_sha256") != tree_digest(checkpoint) or
            report.get("export_sha256") != file_digest(export) or
            meta.get("export_sha256") != report["export_sha256"]):
        raise ValueError("Checkpoint or export content changed")


def restore_guard(checkpoint, original_root=None):
    checkpoint = Path(checkpoint).resolve()
    run = checkpoint.parent.parent
    report = json.loads((run / "report.json").read_text(encoding="utf8"))
    if Path(report.get("checkpoint", "")).resolve() != checkpoint:
        raise ValueError("Requested checkpoint is not the verified checkpoint")
    verified_report(run, report, original_root)
