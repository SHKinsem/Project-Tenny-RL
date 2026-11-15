"""Bounded checkpoint storage, registration and transactional model import.

This module never loads model parameters. The caller supplies the project's
real metadata/source/hash validator before an imported run is published.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import uuid


MAX_CONFIG_BYTES = 4 * 1024 * 1024
MAX_IMPORT_BYTES = 2 * 1024 * 1024 * 1024
MAX_IMPORT_FILES = 10_000
MAX_IMPORT_DEPTH = 24
MAX_REGISTERED_ROOTS = 16
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}\Z")
_MODE = re.compile(r"[a-z][a-z0-9_]{0,79}\Z")
_CONFIGS = ("policy_metadata.json", "env_config.json", "ppo_config.json", "runtime_profile.json",
            "workbench-run.json", "request.json", "status.json")


class StorageError(ValueError):
    """The directory or model cannot be safely used as checkpoint storage."""


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise StorageError("模型配置不能含有重复的 JSON 字段。")
        value[key] = item
    return value


def _finite_float(text):
    value = float(text)
    if not math.isfinite(value):
        raise StorageError("模型配置中的数值必须为有限值。")
    return value


def _reject_constant(_text):
    raise StorageError("模型配置中的数值必须为有限值。")


def _plain_path(value, *, exists=False, directory=True):
    if not isinstance(value, (str, os.PathLike)) or not str(value) or len(str(value)) > 4096:
        raise StorageError("请选择有效的绝对目录。")
    path = Path(value).expanduser()
    if not path.is_absolute() or ".." in path.parts:
        raise StorageError("目录必须使用不含上级跳转的绝对路径。")
    if path == Path(path.anchor):
        raise StorageError("不能使用文件系统根目录。")
    # Reject links before resolving them; accepting the resolved target would
    # make a later source/target swap indistinguishable from a normal folder.
    for part in (path, *path.parents):
        if part.is_symlink():
            raise StorageError("模型和保存目录不能包含符号链接。")
    if exists and not (path.is_dir() if directory else path.is_file()):
        raise StorageError("所选目录或文件不存在。")
    return path.resolve()


def _read_json(path):
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_CONFIG_BYTES:
                raise StorageError("模型配置过大或不是常规文件。")
            content = stream.read(MAX_CONFIG_BYTES + 1)
        if len(content) > MAX_CONFIG_BYTES:
            raise StorageError("模型配置超出读取限制。")
        result = json.loads(content.decode("utf-8-sig"), object_pairs_hook=_unique_object,
                            parse_float=_finite_float, parse_constant=_reject_constant)
        if not isinstance(result, dict):
            raise StorageError("模型配置必须为 JSON 对象。")
        return result
    except StorageError:
        raise
    except (OSError, UnicodeError, ValueError) as error:
        raise StorageError("无法读取完整模型配置。") from error


def _write_json(path, value):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class _CopyBudget:
    def __init__(self, root):
        self.files = self.bytes = self.entries = 0
        self.root, self.digest = root, hashlib.sha256()

    def file(self, source, destination):
        _plain_path(source, exists=True, directory=False)
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        descriptor = os.open(source, flags)
        with os.fdopen(descriptor, "rb") as incoming:
            before = os.fstat(incoming.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise StorageError("模型中只能包含常规文件和目录。")
            self.files += 1
            self.bytes += before.st_size
            if self.files > MAX_IMPORT_FILES or self.bytes > MAX_IMPORT_BYTES:
                raise StorageError("模型超过导入文件数或容量限制。")
            destination.parent.mkdir(parents=True, exist_ok=True)
            copied = 0
            digest = hashlib.sha256()
            with destination.open("xb") as outgoing:
                while block := incoming.read(min(1024 * 1024, before.st_size - copied + 1)):
                    copied += len(block)
                    if copied > before.st_size:
                        raise StorageError("模型文件正在变化，请保存完成后再导入。")
                    outgoing.write(block)
                    digest.update(block)
                outgoing.flush()
                os.fsync(outgoing.fileno())
            after = os.fstat(incoming.fileno())
            if copied != before.st_size or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                    after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
                raise StorageError("模型文件正在变化，请保存完成后再导入。")
            self.digest.update(destination.relative_to(self.root).as_posix().encode("utf-8") + b"\0")
            self.digest.update(digest.digest())

    def tree(self, source, destination, depth=0):
        _plain_path(source, exists=True)
        if depth > MAX_IMPORT_DEPTH:
            raise StorageError("模型目录层级超过限制。")
        destination.mkdir()
        with os.scandir(source) as iterator:
            entries = []
            for item in iterator:
                entries.append(item)
                self.entries += 1
                if self.entries > MAX_IMPORT_FILES:
                    raise StorageError("模型目录项目数超过限制。")
        for item in sorted(entries, key=lambda entry: entry.name):
            target = destination / item.name
            if item.is_symlink():
                raise StorageError("模型目录不能包含符号链接。")
            if item.is_dir(follow_symlinks=False):
                self.tree(Path(item.path), target, depth + 1)
            elif item.is_file(follow_symlinks=False):
                self.file(Path(item.path), target)
            else:
                raise StorageError("模型中包含不支持的特殊文件。")


class WorkbenchStorage:
    def __init__(self, artifacts):
        self.artifacts = _plain_path(artifacts)
        self.default_root = self.artifacts / "console"
        self.import_root = self.artifacts / "workbench-imports"
        self.config_path = self.default_root / "workbench-storage.json"

    def _configuration(self):
        if not self.config_path.exists():
            return {"schemaVersion": 1, "saveRoot": str(self.default_root), "roots": []}
        result = _read_json(self.config_path)
        roots = result.get("roots")
        if (result.get("schemaVersion") != 1 or not isinstance(roots, list)
                or len(roots) > MAX_REGISTERED_ROOTS):
            raise StorageError("保存目录记录无效，请重新选择目录。")
        _plain_path(result.get("saveRoot"))
        for root in roots:
            _plain_path(root)
        return result

    def save_root(self):
        root = _plain_path(self._configuration()["saveRoot"])
        if root != self.default_root and not root.is_dir():
            raise StorageError("已选择的保存目录当前不可访问，请重新选择目录。")
        return root

    def registered_roots(self):
        config = self._configuration()
        values = [self.default_root, self.import_root, config["saveRoot"], *config["roots"]]
        return tuple(dict.fromkeys(_plain_path(value) for value in values))

    def describe(self):
        root = self.save_root()
        return {"schemaVersion": 1, "path": str(root), "saveRoot": str(root),
                "defaultPath": str(self.default_root), "importRoot": str(self.import_root),
                "wslDistro": os.environ.get("WSL_DISTRO_NAME"),
                "configured": root != self.default_root,
                "roots": [str(value) for value in self.registered_roots()]}

    def configure(self, path):
        selected = _plain_path(path, exists=True)
        current = self._configuration()
        roots = list(dict.fromkeys([current["saveRoot"], *current["roots"], str(selected)]))
        roots = [root for root in roots if Path(root) not in {self.default_root, self.import_root}]
        if len(roots) > MAX_REGISTERED_ROOTS:
            raise StorageError("已登记的保存目录过多，请先整理已有目录。")
        self.default_root.mkdir(parents=True, exist_ok=True)
        _write_json(self.config_path, {"schemaVersion": 1, "saveRoot": str(selected), "roots": roots})
        return self.describe()

    def resolve_run(self, path, mode):
        if not isinstance(mode, str) or not _MODE.fullmatch(mode):
            raise StorageError("训练模式无效。")
        run = _plain_path(path, exists=True)
        if not _NAME.fullmatch(run.name) or not any(run.parent == root / mode for root in self.registered_roots()):
            raise StorageError("模型不在已登记的模式目录中。")
        return run

    def resolve_folder(self, path):
        candidate = _plain_path(path, exists=True)
        for root in self.registered_roots():
            if candidate == root:
                return candidate
            if not candidate.is_relative_to(root):
                continue
            parts = candidate.relative_to(root).parts
            if (parts and _MODE.fullmatch(parts[0]) and len(parts) <= 4
                    and (len(parts) < 2 or _NAME.fullmatch(parts[1]))
                    and (len(parts) < 3 or parts[2] == "checkpoints")
                    and (len(parts) < 4 or parts[3].isdigit())):
                return candidate
        raise StorageError("只能打开已登记的保存目录或模型目录。")

    @staticmethod
    def _model_source(path):
        source = _plain_path(path, exists=True)
        if source.name.isdigit() and source.parent.name == "checkpoints":
            run, checkpoint = source.parent.parent, source
        elif (source / "report.json").is_file():
            run = source
            report = _read_json(run / "report.json")
            declared = report.get("checkpoint")
            checkpoint = Path(declared) if isinstance(declared, str) else Path()
            if not checkpoint.is_absolute():
                checkpoint = run / checkpoint
            # Exported directories may have been moved from another machine.
            # Numeric checkpoint identity can be rebased, but never inferred.
            if not checkpoint.is_relative_to(run) and checkpoint.parent.name == "checkpoints":
                checkpoint = run / "checkpoints" / checkpoint.name
        else:
            raise StorageError("请选择完整训练目录或其中的数字 checkpoint 目录。")
        if not _NAME.fullmatch(run.name):
            raise StorageError("训练目录名称无效。")
        checkpoint = _plain_path(checkpoint, exists=True)
        if checkpoint.parent != run / "checkpoints" or not checkpoint.name.isdigit():
            raise StorageError("checkpoint 必须是训练目录中的数字目录。")
        return run, checkpoint

    @staticmethod
    def _rebase_records(run, source, checkpoint):
        report = _read_json(run / "report.json")
        old_export = report.get("export_path")
        export_name = Path(old_export).name if isinstance(old_export, str) else ""
        if not export_name or export_name in {".", ".."}:
            raise StorageError("模型缺少明确的导出文件。")
        report.update(checkpoint=str(run / "checkpoints" / checkpoint.name), export_path=str(run / export_name),
                      run_dir=str(run), run_id=source.name, report_path=str(run / "report.json"))
        _write_json(run / "report.json", report)
        for name in ("status.json", "request.json"):
            path = run / name
            if path.exists():
                value = _read_json(path)
                for field in ("checkpoint", "latest_checkpoint"):
                    if field in value:
                        value[field] = str(run / "checkpoints" / checkpoint.name)
                if "export_path" in value:
                    value["export_path"] = str(run / export_name)
                if "report_path" in value:
                    value["report_path"] = str(run / "report.json")
                value.update(run_dir=str(run), run_id=source.name)
                value.pop("pid", None)
                _write_json(path, value)
        return export_name

    def import_checkpoint(self, path, mode, validator):
        if not isinstance(mode, str) or not _MODE.fullmatch(mode) or not callable(validator):
            raise StorageError("导入模式或验证器无效。")
        source, checkpoint = self._model_source(path)
        report = _read_json(source / "report.json")
        if report.get("mode") != mode:
            raise StorageError("导入模型与当前训练模式不一致。")
        declared_checkpoint = report.get("checkpoint")
        if not isinstance(declared_checkpoint, str) or Path(declared_checkpoint).name != checkpoint.name:
            raise StorageError("请选择有完整验证报告的 checkpoint。")
        export_value = report.get("export_path")
        if not isinstance(export_value, str) or not export_value:
            raise StorageError("模型缺少有效导出文件。")
        export_name = Path(export_value).name
        if not export_name or export_name in _CONFIGS or export_name == "report.json":
            raise StorageError("模型缺少有效导出文件。")
        for name in ("policy_metadata.json", "env_config.json", "ppo_config.json", "status.json"):
            _read_json(source / name)
        _plain_path(source / export_name, exists=True, directory=False)
        parent = self.import_root / mode
        _plain_path(parent)
        parent.mkdir(parents=True, exist_ok=True)
        destination = parent / source.name
        temporary = parent / (".import_" + uuid.uuid4().hex)
        temporary.mkdir()
        budget = _CopyBudget(temporary)
        published = False
        try:
            (temporary / "checkpoints").mkdir()
            budget.tree(checkpoint, temporary / "checkpoints" / checkpoint.name)
            for name in (*_CONFIGS, "report.json"):
                candidate = source / name
                if candidate.exists():
                    budget.file(candidate, temporary / name)
                    _read_json(temporary / name)
            sidecar = checkpoint.with_name(checkpoint.name + ".workbench.json")
            if sidecar.exists():
                _read_json(sidecar)
                budget.file(sidecar, temporary / "checkpoints" / sidecar.name)
            budget.file(source / export_name, temporary / export_name)
            header = (source / export_name).with_suffix(".h")
            if header.exists() and header.name != export_name:
                budget.file(header, temporary / header.name)
            source_digest = budget.digest.hexdigest()
            if destination.exists():
                provenance = _read_json(destination / "workbench-import.json")
                verified = validator(destination, mode)
                if provenance.get("sourceDigest") != source_digest or not verified:
                    raise StorageError("已存在不同内容的同名训练目录，不能覆盖。")
                shutil.rmtree(temporary)
                return {"ok": True, "alreadyImported": True, "runId": source.name,
                        "runDir": str(destination),
                        "checkpoint": str(destination / "checkpoints" / checkpoint.name),
                        "copiedBytes": 0, "copiedFiles": 0,
                        "model": verified if isinstance(verified, dict) else None}
            self._rebase_records(temporary, source, checkpoint)
            if not validator(temporary, mode):
                raise StorageError("模型未通过实际兼容性或文件完整性验证。")
            # Serialize the final paths before publication. Validation above
            # used staging paths; the second validation below checks final IO.
            self._rebase_records_for_publish(temporary, source, checkpoint, destination)
            _write_json(temporary / "workbench-import.json", {
                "schemaVersion": 1, "sourceRunId": source.name, "checkpointName": checkpoint.name,
                "mode": mode, "copiedBytes": budget.bytes, "copiedFiles": budget.files,
                "sourceDigest": source_digest})
            if destination.exists():
                raise StorageError("导入目标已存在，请重新选择。")
            temporary.rename(destination)
            published = True
            verified = validator(destination, mode)
            if not verified:
                raise StorageError("导入后的模型未通过完整性验证。")
            return {"ok": True, "runId": source.name, "runDir": str(destination),
                    "checkpoint": str(destination / "checkpoints" / checkpoint.name),
                    "copiedBytes": budget.bytes, "copiedFiles": budget.files,
                    "model": verified if isinstance(verified, dict) else None}
        except Exception:
            # Only our unique, verified-under-parent staging/published target
            # is removed; source and any pre-existing destination are untouched.
            cleanup = destination if published else temporary
            if cleanup.parent == parent and cleanup.is_dir() and not cleanup.is_symlink():
                shutil.rmtree(cleanup)
            raise

    @staticmethod
    def _rebase_records_for_publish(temporary, source, checkpoint, destination):
        for name in ("report.json", "status.json", "request.json"):
            path = temporary / name
            if not path.exists():
                continue
            value = _read_json(path)
            for key in ("checkpoint", "latest_checkpoint", "export_path", "report_path"):
                item = value.get(key)
                if isinstance(item, str):
                    old = Path(item)
                    if old.is_relative_to(temporary):
                        value[key] = str(destination / old.relative_to(temporary))
            value.update(run_dir=str(destination), run_id=source.name)
            _write_json(path, value)
