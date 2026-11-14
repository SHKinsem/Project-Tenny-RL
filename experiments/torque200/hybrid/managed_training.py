"""Console lifecycle around the shared hybrid readiness gate (no ML at import)."""
from contextlib import contextmanager, ExitStack
import os
from pathlib import Path

from control_modes import mode_spec


def runtime_root():
    return Path(os.environ.get("TENNY_HYBRID_ROOT", str(Path.home() / "projects/tenny-hybrid200"))).resolve()


def environment_pair(mode, cfg, legacy_factory, hybrid_factory=None, hybrid_config_factory=None):
    """Brax must evaluate hybrid actions in the identical action/timing contract.

    Legacy passes None so its established train_ppo evaluation default is retained.
    Factories make this routing testable without importing or compiling the simulator.
    """
    spec = mode_spec(mode)
    if not spec["hybrid"]:
        return {"env": legacy_factory(config=cfg), "eval_env": None}
    if hybrid_factory is None or hybrid_config_factory is None:
        raise ValueError("Hybrid training requires explicit hybrid evaluation factories")
    return {"env": hybrid_factory(spec["action_mode"], cfg),
            "eval_env": hybrid_factory(spec["action_mode"], hybrid_config_factory(spec["control_hz"]))}


@contextmanager
def runtime_locks(*roots):
    """Do not synchronize source over, or start two jobs in, a live runtime."""
    import fcntl
    with ExitStack() as stack:
        for root in sorted({Path(p).resolve() for p in roots}, key=str):
            root.mkdir(parents=True, exist_ok=True)
            stream = stack.enter_context((root / ".runner.lock").open("a+b"))
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("Another task owns this hybrid runtime; wait for it to finish") from exc
        yield


def prepare(args, monitor, ensure=None):
    """A cancelled/failed gate never starts formal training or consumes its budget."""
    if ensure is None:
        from hybrid.readiness import ensure_ready
        ensure = ensure_ready
    spec = mode_spec(args.mode)
    monitor.publish(phase="preparing", stage="readiness", control_hz=spec["control_hz"],
                    action_mode=spec["action_mode"], training_clock_started=False)

    def cancelled():
        return (args.run_dir / "stop.request").exists()

    def status(event):
        monitor.publish(phase="preparing", stage=event.get("stage", "readiness"),
                        preparation_message=event.get("message"),
                        readiness_cache_hit=event.get("cache_hit"),
                        readiness_evidence_dir=event.get("evidence_dir"))

    if cancelled():
        monitor.check_stop()
    try:
        result = ensure(runtime_root(), spec["action_mode"], spec["control_hz"],
                        force=getattr(args, "force_revalidate", False),
                        on_status=status, cancelled=cancelled)
    except Exception as exc:
        # Keep the optional readiness module lazy for host-only lifecycle tests.
        if cancelled():
            monitor.check_stop()
        raise
    if cancelled():
        monitor.check_stop()
    monitor.publish(readiness_cache_hit=result["cache_hit"], readiness_cache_key=result["cache_key"],
                    readiness_evidence_dir=result["evidence_dir"],
                    readiness_validation_path=result["validation_path"], preparation_message="准备检查已通过")
    monitor.begin_training_clock()
    return result


def run_managed(args, monitor, train):
    from hybrid.readiness import prepare_environment
    environment = prepare_environment(runtime_root())
    os.environ.pop("LD_LIBRARY_PATH", None)
    os.environ.update(environment)
    with runtime_locks(runtime_root(), args.torque_root):
        prepare(args, monitor)
        train(args, monitor)


def verified_report(run, mode, report):
    """Validate semantic and file identity before exposing a model for resume/preview."""
    import json
    from hybrid.core import (metadata, read_geometry, scan_geometry, require_metadata,
                             digest_file, digest_tree)
    spec = mode_spec(mode)
    if not spec["hybrid"] or report.get("checkpoint_validated") is not True:
        raise ValueError("Not a validated hybrid checkpoint")
    run = Path(run).resolve()
    meta = json.loads((run / "policy_metadata.json").read_text(encoding="utf8"))
    expected = metadata(spec["action_mode"], scan_geometry(read_geometry()), spec["control_hz"])
    require_metadata(meta, expected)
    if (report.get("action_mode") != spec["action_mode"] or
            report.get("control_hz") != spec["control_hz"] or
            report.get("source_fingerprint") != expected["source_fingerprint"]):
        raise ValueError("Hybrid report identity mismatch")
    checkpoint, export = Path(report["checkpoint"]).resolve(), Path(report["export_path"]).resolve()
    if not checkpoint.is_relative_to(run) or not export.is_relative_to(run):
        raise ValueError("Hybrid artifacts escaped their run")
    if (report.get("checkpoint_name") != checkpoint.name or
            report.get("checkpoint_sha256") != digest_tree(checkpoint) or
            report.get("export_sha256") != digest_file(export) or
            meta.get("export_sha256") != report["export_sha256"]):
        raise ValueError("Hybrid checkpoint/export bytes changed")


def verification_signature(run):
    """Cheap change detector for repeated UI polls; hashes are checked on every change."""
    from hybrid.core import ROOT
    paths = [*ROOT.joinpath("assets").rglob("*"), *ROOT.joinpath("hybrid").glob("*.py"),
             *ROOT.joinpath("legwheel").rglob("*.py"), *ROOT.glob("*.py"), ROOT / "hybrid-wsl.sh",
             *Path(run).rglob("*")]
    rows = []
    for path in paths:
        if path.is_file():
            stat = path.stat()
            rows.append((str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, path.is_symlink()))
    return tuple(sorted(rows))
