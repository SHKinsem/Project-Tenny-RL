"""Reusable, evidence-bound local acceptance; never a substitute for training.

A cache hit checks source, Python/packages, accelerator and runtime settings plus
the completed reports and artifacts. It imports no JAX and compiles no model.
Callers own the runtime's .runner.lock; this module owns only .readiness.lock.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import uuid

from hybrid.core import (MODES, ROOT, source_fingerprint, digest_file, digest_tree,
                         write_json, require_metadata)
from hybrid.timing import Timing

SCHEMA = "hybrid-readiness-v1"


class ReadinessCancelled(RuntimeError):
    pass


def prepare_environment(runtime_root, source=None):
    """Shared CLI/console defaults; preserve explicit device/backend choices.

    Console callers select CUDA before this helper because their parent preview
    process deliberately runs on CPU. The CLI also supports an explicit CPU gate.
    """
    env = dict(os.environ if source is None else source)
    env.pop("LD_LIBRARY_PATH", None)
    for key, value in (("JAX_PLATFORMS", "cuda"), ("CUDA_VISIBLE_DEVICES", "0"),
                       ("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.35"),
                       ("OMP_NUM_THREADS", "4"), ("OPENBLAS_NUM_THREADS", "1")):
        env.setdefault(key, value)
    env.update(XLA_PYTHON_CLIENT_PREALLOCATE="false", MPLBACKEND="Agg",
               JAX_COMPILATION_CACHE_DIR=str(Path(runtime_root) / "jax-cache"),
               PYTHONUNBUFFERED="1")
    return env


def runtime_identity():
    """Inspect installed metadata and the driver, without initializing JAX."""
    packages = sorted((d.metadata["Name"].lower().replace("_", "-"), d.version)
                      for d in importlib.metadata.distributions() if d.metadata["Name"])
    settings = {k: v for k, v in os.environ.items()
                if k.startswith(("JAX_", "XLA_", "CUDA_", "NVIDIA_"))
                and k not in {"JAX_COMPILATION_CACHE_DIR"}}
    settings["LD_LIBRARY_PATH"] = os.environ.get("LD_LIBRARY_PATH", "")
    backend = os.environ.get("JAX_PLATFORMS", "cuda")
    devices = []
    if any(name in backend for name in ("cuda", "gpu")):
        try:
            result = subprocess.run([
                "nvidia-smi", "--query-gpu=uuid,name,driver_version,compute_cap",
                "--format=csv,noheader,nounits"], capture_output=True, text=True,
                check=True, timeout=10)
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError("Cannot identify the GPU/driver for reusable readiness") from exc
        devices = sorted(line.strip() for line in result.stdout.splitlines() if line.strip())
        if not devices:
            raise RuntimeError("No GPU identity returned; readiness is not reusable")
    return dict(python=str(Path(sys.executable).resolve()), version=sys.version,
                prefix=str(Path(sys.prefix).resolve()), platform=platform.platform(),
                machine=platform.machine(), backend=backend, devices=devices,
                packages=packages, settings=settings)


def identity(mode, control_hz):
    if mode not in MODES:
        raise ValueError("Unknown hybrid action mode")
    return dict(schema=SCHEMA, source_fingerprint=source_fingerprint(),
                mode=mode, timing=Timing(control_hz).contract(), runtime=runtime_identity())


def cache_key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _read(path):
    path = Path(path)
    if path.is_symlink() or path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("Invalid readiness report")
    return json.loads(path.read_text(encoding="utf8"),
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def _finite(value, maximum):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and 0 <= value <= maximum)


def evidence_manifest(directory, expected):
    """Validate completed semantic checks and bind all deployable artifacts."""
    directory = Path(directory).resolve()
    report = _read(directory / "validation.json")
    if (report.get("schema") != "hybrid-local-validation-v1" or
            report.get("runtime_validated") is not True or report.get("error") or
            report.get("source_fingerprint") != expected["source_fingerprint"]):
        raise ValueError("Full validation is not complete for this source")
    require_metadata(report, expected["timing"])
    packages = dict(expected["runtime"]["packages"])
    for name, version in report.get("runtime_packages", {}).items():
        if packages.get(name.lower().replace("_", "-")) != version:
            raise ValueError("Validation packages do not match current Python")
    if not report.get("runtime_packages"):
        raise ValueError("Validation package provenance is missing")
    files = {"validation.json": digest_file(directory / "validation.json")}
    trees = {}
    for phase in ("first", "restart"):
        run = directory / phase
        trained = _read(run / "report.json")
        metadata = _read(run / "policy_metadata.json")
        require_metadata(metadata, dict(source_fingerprint=expected["source_fingerprint"],
                                       action_mode=expected["mode"], **expected["timing"]))
        required = dict(action_mode=expected["mode"], control_hz=expected["timing"]["control_hz"])
        require_metadata(trained, required)
        if (trained.get("checkpoint_validated") is not True or trained.get("all_parameters_finite") is not True or
                trained.get("error") or not _finite(trained.get("parameter_tree_max_abs_error"), 0) or
                not _finite(trained.get("restore_max_abs_error"), 1e-6) or
                not _finite(trained.get("export_max_abs_error"), 1e-3) or
                not _finite(trained.get("adapter_max_abs_error"), 1e-5) or
                not trained.get("finite_training_losses_checked") or
                not isinstance(trained.get("actual_progress_steps"), int) or
                trained["actual_progress_steps"] < 16):
            raise ValueError("Short training acceptance is incomplete")
        if phase == "restart" and (trained.get("warm_restart_changed_network_leaves", 0) <= 0 or
                                    trained.get("warm_restart_network_max_abs_change", 0) <= 0):
            raise ValueError("Warm restart did not demonstrate a network update")
        checkpoint = Path(trained["checkpoint"])
        if (not checkpoint.is_absolute() or not checkpoint.resolve().is_relative_to(run.resolve()) or
                checkpoint.name != trained.get("checkpoint_name")):
            raise ValueError("Checkpoint is outside its acceptance run")
        tree_hash = digest_tree(checkpoint)
        export_hash = digest_file(run / "policy.cpp.txt")
        if (tree_hash != trained.get("checkpoint_sha256") or
                export_hash != trained.get("export_sha256") or
                export_hash != metadata.get("export_sha256")):
            raise ValueError("Checkpoint/export content changed after acceptance")
        trees[checkpoint.relative_to(directory).as_posix()] = tree_hash
        for name in ("report.json", "policy_metadata.json", "policy.cpp.txt"):
            relative = f"{phase}/{name}"
            files[relative] = digest_file(directory / relative)
    return dict(files=files, trees=trees)


def _check_cancel(cancelled):
    if cancelled and cancelled():
        raise ReadinessCancelled("Preparation cancelled; no new readiness result was published")


@contextmanager
def _cache_lock(path, cancelled):
    # Real preparation is Linux-only; nonblocking acquisition allows cancellation.
    import fcntl
    with Path(path).open("a+b") as stream:
        while True:
            _check_cancel(cancelled)
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(0.2)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _terminate(process):
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait()
    except ProcessLookupError:
        pass


def _run_stage(label, module, argv, log, report, cancelled):
    """Preserve the existing raw logs, concise console and child exit status."""
    from hybrid.console import summary, failure
    print(f"[准备] {label} | 完整日志：{log}", flush=True)
    producer = subprocess.Popen([sys.executable, "-u", "-m", module, *argv],
                                cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                start_new_session=os.name == "posix")
    sink = None
    try:
        command = [sys.executable, "-u", "-m", "hybrid.console", "capture",
                   "--label", label, "--log", str(log)]
        if os.environ.get("TENNY_VERBOSE") == "1":
            command.append("--verbose")
        sink = subprocess.Popen(command, stdin=producer.stdout, cwd=ROOT,
                                start_new_session=os.name == "posix")
        producer.stdout.close()
        while producer.poll() is None:
            _check_cancel(cancelled)
            if sink.poll() not in (None, 0):
                raise RuntimeError("Readiness log collector failed before its producer")
            time.sleep(0.2)
        code = producer.returncode
        while sink.poll() is None:
            _check_cancel(cancelled)
            time.sleep(0.1)
        if code or sink.returncode:
            code = code or sink.returncode
            failure(label, log, code)
            raise subprocess.CalledProcessError(code, [module, *argv])
        summary(label, report, log)
    finally:
        _terminate(producer)
        if sink is not None:
            # Give the logger a chance to retain the final producer bytes.
            try:
                sink.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _terminate(sink)
        if producer.stdout and not producer.stdout.closed:
            producer.stdout.close()


def _stage_target(root, stage, expected, first=None):
    from hybrid.stage_evidence import stage_identity
    return root / "readiness-cache" / "stages-v1" / stage / (cache_key(stage_identity(stage, expected, first)) + ".json")


def _cached_stage(root, stage, expected, first=None):
    from hybrid.stage_evidence import verify
    target = _stage_target(root, stage, expected, first)
    try:
        index = _read(target)
        path = Path(index["reference_path"])
        if (not path.resolve().is_relative_to(root / "artifacts") or
                digest_file(path) != index["reference_sha256"]):
            raise ValueError("Stage reference escaped artifacts or changed")
        value = _read(path)
        verify(value, expected, stage, first)
        return dict(value=value, path=path)
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _audited_validation(root, expected):
    """Import only an explicitly audited old partial run, never a bare report."""
    from hybrid.stage_evidence import record, ATTESTATION
    for path in sorted((root / "artifacts").glob("readiness-*/source-manifest.json"), reverse=True)[:128]:
        try:
            att = _read(path)
            report, identity_path = path.parent / "validation.json", path.parent / "identity.json"
            old = _read(identity_path)
            proof = att["exit_evidence"]
            log = Path(proof["log"])
            if (att.get("schema") != ATTESTATION or type(att.get("validation_exit_code")) is not int or
                    att["validation_exit_code"] != 0 or
                    att["validation_sha256"] != digest_file(report) or
                    att["identity_sha256"] != digest_file(identity_path) or
                    att["source_fingerprint"] != old["source_fingerprint"] or
                    old["runtime"] != json.loads(json.dumps(expected["runtime"])) or
                    old["timing"] != expected["timing"] or digest_file(log) != proof["sha256"]):
                continue
            text = log.read_text(encoding="utf8", errors="replace")
            # Old orchestration checks BOTH process exits before this success
            # summary and before it starts the first training subprocess.
            markers = ("[通过] 预检", "[准备] 新训练")
            if text.find(markers[0]) < 0 or text.find(markers[1]) <= text.find(markers[0]):
                continue
            value = record("validation", report, path, expected)
            return dict(value=value, path=None)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return None


def discover_ready(runtime_root, mode, control_hz):
    """Read-only planning: no subprocesses, cache writes or JAX initialization."""
    root, expected = Path(runtime_root).resolve(), identity(mode, control_hz)
    found = {}
    for stage in ("validation", "first", "restart"):
        first = found.get("first", {}).get("value")
        if stage == "restart" and first is None:
            continue
        item = _cached_stage(root, stage, expected, first)
        if stage == "validation" and item is None and not _legacy_revoked(root, expected).exists():
            item = _audited_validation(root, expected)
        if item:
            found[stage] = item
    return dict(reused_stages=list(found), missing_stages=[s for s in ("validation", "first", "restart") if s not in found],
                validation_report_path=found.get("validation", {}).get("value", {}).get("report_path"))


def _publish_stage(root, stage, expected, item, first=None):
    target = _stage_target(root, stage, expected, first)
    write_json(target, dict(reference_path=str(item["path"]), reference_sha256=digest_file(item["path"])))


def _legacy_revoked(root, expected):
    key = cache_key(dict(runtime=expected["runtime"], timing=expected["timing"]))
    return root / "readiness-cache" / "stages-v1" / ("force-" + key + ".json")


def _revoke_stages(root, expected):
    """A forced failed recheck cannot silently fall back to older successes."""
    cache = root / "readiness-cache" / "stages-v1"
    for path in cache.glob("*/*.json"):
        try:
            value = _read(_read(path)["reference_path"])
            old = value["identity"]
            if (old["runtime"] == json.loads(json.dumps(expected["runtime"])) and old["timing"] == expected["timing"]
                    and old.get("mode", expected["mode"]) == expected["mode"]):
                path.unlink()
        except (OSError, ValueError, KeyError, TypeError):
            pass
    # Explicit force also revokes re-import of old audited validation records.
    write_json(_legacy_revoked(root, expected),
               dict(forced_at=datetime.now(timezone.utc).isoformat()))


def ensure_ready(runtime_root, mode, control_hz, *, force=False, on_status=None, cancelled=None):
    """Publish each proven stage immediately; resume after interruption safely."""
    from hybrid.stage_evidence import inventory, stage_identity, record, ATTESTATION
    started, root = time.monotonic(), Path(runtime_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    _check_cancel(cancelled)
    if on_status:
        on_status(dict(phase="preparing", stage="readiness_identity", message="逐阶段检查已完成的准备记录", cache_hit=False))
    expected = identity(mode, control_hz)
    with _cache_lock(root / ".readiness.lock", cancelled):
        if force:
            _revoke_stages(root, expected)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        directory = root / "artifacts" / ("readiness-" + stamp + "-" + uuid.uuid4().hex[:8])
        directory.mkdir(parents=True)
        write_json(directory / "identity.json", expected)
        selected, reused = {}, []
        try:
            for stage, label in (("validation", "预检"), ("first", "新训练"), ("restart", "热启动")):
                _check_cancel(cancelled)
                first = selected.get("first", {}).get("value")
                item = None if force else _cached_stage(root, stage, expected, first)
                revoked = _legacy_revoked(root, expected)
                if not force and stage == "validation" and item is None and not revoked.exists():
                    item = _audited_validation(root, expected)
                if item:
                    if item["path"] is None:
                        item["path"] = directory / "validation-reference.json"
                        write_json(item["path"], item["value"])
                        _publish_stage(root, stage, expected, item, first)
                    selected[stage] = item
                    reused.append(stage)
                    if on_status:
                        on_status(dict(phase="preparing", stage="readiness_" + stage, message="复用已通过的" + label,
                                       cache_hit=True, reused_stages=list(reused), evidence_dir=str(directory)))
                    continue
                before = identity(mode, control_hz)
                sources = inventory()
                if before["runtime"] != expected["runtime"]:
                    raise RuntimeError("Runtime changed during preparation")
                report = directory / "validation.json" if stage == "validation" else directory / stage / "report.json"
                if stage == "validation":
                    module, argv = "hybrid.validate", ["--output", str(report), "--control-hz", str(control_hz)]
                else:
                    module = "hybrid.train"
                    argv = ["--mode", mode, "--control-hz", str(control_hz),
                            "--validation", str(selected["validation"]["path"]), "--output", str(report.parent),
                            "--steps", "16", "--num-envs", "2", "--eval-envs", "2", "--unroll-length", "4", "--seed", "0"]
                    if stage == "restart":
                        argv += ["--restore", first["checkpoint"], "--restore-stage-reference", str(selected["first"]["path"])]
                if on_status:
                    on_status(dict(phase="preparing", stage="readiness_" + stage, message=label,
                                   cache_hit=False, reused_stages=list(reused), evidence_dir=str(directory)))
                _run_stage(label, module, argv, directory / (stage + ".log"), report, cancelled)
                # A completed subprocess is certified before honoring a stop that
                # arrived just after its exit. Incomplete children never get here.
                if identity(mode, control_hz) != before or inventory() != sources:
                    raise RuntimeError("Source/runtime changed during preparation")
                attestation_path = directory / (stage + "-source-manifest.json")
                write_json(attestation_path, dict(schema=ATTESTATION, source_fingerprint=before["source_fingerprint"],
                           files=sources, identity_sha256=digest_file(directory / "identity.json")))
                value = record(stage, report, attestation_path, expected, first)
                item = dict(value=value, path=directory / (stage + "-reference.json"))
                write_json(item["path"], value)
                _publish_stage(root, stage, expected, item, first)
                selected[stage] = item
            _check_cancel(cancelled)
        except BaseException as exc:
            write_json(directory / "readiness-failure.json", dict(error=str(exc), cancelled=isinstance(exc, ReadinessCancelled),
                       completed_stages=list(selected), reused_stages=reused))
            raise
        result = dict(validation_path=str(selected["validation"]["path"]), evidence_dir=str(directory),
                      cache_hit=len(reused) == 3, reused_stages=reused,
                      cache_key=cache_key({stage: item["value"]["identity"] for stage, item in selected.items()}),
                      preparation_seconds=time.monotonic()-started,
                      stage_references={stage: str(item["path"]) for stage, item in selected.items()})
        write_json(directory / "readiness-complete.json", result)
        if on_status:
            on_status(dict(phase="preparing", stage="readiness_ready", message="三阶段准备检查已通过", **result))
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--control-hz", type=int, choices=(100, 200), required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args(argv)
    os.environ.update(prepare_environment(args.runtime_root))
    os.environ.pop("LD_LIBRARY_PATH", None)
    def status(state):
        print("[准备] " + state["message"], flush=True)
        write_json(args.result, dict(ready=False, **state))
    try:
        result = ensure_ready(args.runtime_root, args.mode, args.control_hz, force=args.force,
                              on_status=status)
        write_json(args.result, result)
        print("[准备完成] " + ("复用已通过检查" if result["cache_hit"] else "完整检查通过"), flush=True)
    except subprocess.CalledProcessError as exc:
        return exc.returncode if exc.returncode > 0 else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
