"""Stat-only invalidation for previously verified dashboard read results.

These signatures never authorize a training/import action and never replace the
project's byte/shape/source validators. Uninspectable or linked dependencies are
uncacheable, so the caller falls back to the full validator.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import stat

from control_modes import mode_spec


MAX_SIGNATURE_ENTRIES = 100_000
MAX_SIGNATURE_DEPTH = 64


@dataclass(frozen=True)
class Dependency:
    path: Path
    recursive: bool = False
    suffix: str | None = None
    ignore_generated: bool = False


class _Uncacheable(Exception):
    pass


def _is_link(info):
    return (stat.S_ISLNK(info.st_mode)
            or bool(getattr(info, "st_file_attributes", 0) & 0x400))


def filesystem_signature(dependencies, *, max_entries=MAX_SIGNATURE_ENTRIES):
    """Return stable path/type/stat rows without reading file contents.

    Directory rows and entry names detect additions/removals. ctime and inode
    also detect same-size edits with restored mtime and atomic replacements on
    the Linux runner. Links/reparse points, special files and scan races never
    produce a reusable signature.
    """
    rows = {}
    ancestors = {}
    visited = 0

    def add(path, info):
        if _is_link(info) or not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
            raise _Uncacheable
        rows[str(path)] = (str(path), info.st_mode, info.st_size, info.st_mtime_ns,
                           info.st_ctime_ns, info.st_dev, info.st_ino, info.st_nlink)
        if len(rows) + len(ancestors) > max_entries:
            raise _Uncacheable

    try:
        for dependency in dependencies:
            root = Path(dependency.path).absolute()
            for parent in root.parents:
                if str(parent) in ancestors:
                    continue
                try:
                    info = parent.lstat()
                except FileNotFoundError:
                    ancestors[str(parent)] = (str(parent), "missing-parent")
                    continue
                if _is_link(info) or not stat.S_ISDIR(info.st_mode):
                    raise _Uncacheable
                # Ancestor identity matters, unrelated sibling edits do not.
                ancestors[str(parent)] = (str(parent), info.st_mode, info.st_dev,
                                          info.st_ino, info.st_nlink)
                if len(rows) + len(ancestors) > max_entries:
                    raise _Uncacheable
            try:
                info = root.lstat()
            except FileNotFoundError:
                rows[str(root)] = (str(root), "missing")
                continue
            add(root, info)
            if not stat.S_ISDIR(info.st_mode):
                continue
            stack = [(root, 0)]
            while stack:
                directory, depth = stack.pop()
                if depth > MAX_SIGNATURE_DEPTH:
                    raise _Uncacheable
                with os.scandir(directory) as entries:
                    for entry in entries:
                        visited += 1
                        if visited > max_entries:
                            raise _Uncacheable
                        if dependency.ignore_generated and (entry.name == "__pycache__" or entry.name.endswith(".pyc")):
                            continue
                        path = Path(entry.path)
                        info = entry.stat(follow_symlinks=False)
                        if _is_link(info):
                            raise _Uncacheable
                        if stat.S_ISDIR(info.st_mode):
                            if dependency.recursive:
                                add(path, info)
                                stack.append((path, depth + 1))
                        elif dependency.suffix is None or path.suffix == dependency.suffix:
                            add(path, info)
        return (tuple(sorted(ancestors.values())), tuple(sorted(rows.values())))
    except (OSError, ValueError, _Uncacheable):
        return None


def source_dependencies(mode, original_root):
    """Mirror the actual loaded verifier roots, including release/model splits."""
    spec = mode_spec(mode)
    dependencies = []
    if spec["hybrid"]:
        from hybrid.core import ROOT
        dependencies.extend((
            Dependency(ROOT / "assets", recursive=True),
            Dependency(ROOT / "hybrid", suffix=".py", ignore_generated=True),
            Dependency(ROOT / "legwheel", recursive=True, suffix=".py"),
            *(Dependency(ROOT / name) for name in ("verify_artifacts.py", "hybrid-wsl.sh",
                                                   "console_train.py", "control_modes.py", "preview_sync.py")),
        ))
    if spec["teacher"]:
        from wheel_teacher.contract import ROOT as WHEEL_ROOT
        reference = Path(original_root)
        dependencies.extend((
            Dependency(WHEEL_ROOT / "wheel_teacher", recursive=True, ignore_generated=True),
            Dependency(reference / "mujoco_legwheel_rl_public" / "legwheel", recursive=True, ignore_generated=True),
            Dependency(reference / "mujoco_legwheel_rl_public" / "assets", recursive=True, ignore_generated=True),
            Dependency(reference / "mujoco_esp32_robot_deploy_public" / "host", recursive=True, ignore_generated=True),
            Dependency(reference / "mujoco_esp32_robot_deploy_public" / "src" / "policy" / "rl_policy.cpp"),
            Dependency(reference / "mujoco_esp32_robot_deploy_public" / "src" / "policy" / "rl_policy.h"),
        ))
        if mode == "leg_force_bc":
            from force_teacher.contract import ROOT as FORCE_ROOT
            dependencies.extend((Dependency(FORCE_ROOT / "force_teacher", recursive=True, ignore_generated=True),
                                 Dependency(FORCE_ROOT / "wheel_teacher", recursive=True, ignore_generated=True)))
        else:
            dependencies.extend((
                Dependency(WHEEL_ROOT / "wheel_teacher", recursive=True, ignore_generated=True),
                *(Dependency(WHEEL_ROOT / name) for name in ("console_train.py", "control_modes.py",
                                                             "preview_sync.py", "verify_artifacts.py")),
            ))
    return tuple(dict.fromkeys(dependencies))


def source_signature(mode, original_root):
    return filesystem_signature(source_dependencies(mode, original_root))
