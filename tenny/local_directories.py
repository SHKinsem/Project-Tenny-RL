"""Native directory operations for the loopback console bridge.

Only directory selection and opening are exposed. All process arguments are
separate argv values; no command interpreter evaluates a selected path.
"""
from __future__ import annotations

import importlib.util
import json
import ntpath
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import subprocess
import sys
import threading


class DirectoryError(ValueError):
    pass


def _text_path(value):
    if not isinstance(value, str) or not value or len(value) > 4096 or any(ord(c) < 32 for c in value):
        raise DirectoryError("目录路径无效。")
    return value


def windows_to_backend(value, distro):
    """Map only drive paths and this WSL distribution's UNC directories."""
    value = _text_path(value)
    windows = PureWindowsPath(value)
    if ".." in windows.parts or not windows.is_absolute():
        raise DirectoryError("请选择绝对目录，不能使用上级跳转。")
    if value.startswith(("\\\\?\\", "\\\\.\\")):
        raise DirectoryError("不支持设备目录。")
    drive = windows.drive
    if re.fullmatch(r"[A-Za-z]:", drive):
        suffix = "/".join(windows.parts[1:])
        return f"/mnt/{drive[0].lower()}/" + suffix
    if drive.startswith("\\\\"):
        components = drive[2:].split("\\")
        if (len(components) != 2 or components[0].lower() not in {"wsl$", "wsl.localhost"}
                or components[1].casefold() != distro.casefold()):
            raise DirectoryError("请选择当前 WSL 的目录或本机磁盘目录。")
        return "/" + "/".join(windows.parts[1:])
    raise DirectoryError("目录不属于可访问的本机磁盘或 WSL。")


def backend_to_windows(value, distro):
    value = _text_path(value)
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts or "\\" in value:
        raise DirectoryError("后端目录路径无效。")
    if len(path.parts) >= 3 and path.parts[1] == "mnt" and re.fullmatch(r"[A-Za-z]", path.parts[2]):
        return str(PureWindowsPath(path.parts[2].upper() + ":\\", *path.parts[3:]))
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_. -]{0,79}", distro):
        raise DirectoryError("WSL 发行版名称无效。")
    return str(PureWindowsPath("\\\\wsl.localhost\\" + distro + "\\", *path.parts[1:]))


def _default_distro():
    configured = os.environ.get("TENNY_WSL_DISTRO")
    if configured:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_. -]{0,79}", configured):
            raise DirectoryError("WSL 发行版配置无效。")
        return configured
    try:
        result = subprocess.run(["wsl.exe", "--exec", "printenv", "WSL_DISTRO_NAME"],
                                capture_output=True, timeout=10, check=True,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        value = result.stdout.decode("utf-8-sig").strip()
    except (OSError, subprocess.SubprocessError, UnicodeError) as error:
        raise DirectoryError("无法识别当前 WSL 发行版。") from error
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_. -]{0,79}", value):
        raise DirectoryError("无法识别当前 WSL 发行版。")
    return value


class LocalDirectories:
    def __init__(self, *, distro=None, picker=None, opener=None, platform=None):
        self.platform = platform or sys.platform
        self._distro = distro
        self._target_verified = False
        self._picker = picker or self._pick
        self._opener = opener or self._open
        self._selection_lock = threading.Lock()

    @property
    def distro(self):
        if self._distro is None:
            self._distro = _default_distro()
        return self._distro

    def capabilities(self):
        configured = self._distro or os.environ.get("TENNY_WSL_DISTRO")
        known_distro = isinstance(configured, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_. -]{0,79}", configured)
        available = self.platform == "win32" and bool(known_distro)
        select = available and (self._picker != self._pick or importlib.util.find_spec("tkinter") is not None)
        return {"available": available, "select": select, "open": available,
                "wslDistro": configured if known_distro else None, "distroVerified": self._target_verified,
                "reason": None if available else ("原生目录操作需要 Windows 本机桥接。" if self.platform != "win32"
                                                   else "尚未确认目标 WSL 发行版，请等待后端连接。")}

    def bind_target_distro(self, value):
        """Use the fixed upstream's identity; never guess from WSL's default."""
        if value is None:
            return
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_. -]{0,79}", value):
            raise DirectoryError("后端 WSL 发行版名称无效。")
        configured = self._distro or os.environ.get("TENNY_WSL_DISTRO")
        if configured and configured.casefold() != value.casefold():
            raise DirectoryError("本机桥接与训练服务的 WSL 发行版不同，请检查桥接配置。")
        self._distro = value
        self._target_verified = True

    def select(self, *, purpose, initial=None):
        if purpose not in {"save", "import"}:
            raise DirectoryError("目录选择用途无效。")
        if not self.capabilities()["select"]:
            raise DirectoryError("当前桥接不支持原生目录选择。")
        initial = backend_to_windows(initial, self.distro) if initial else None
        if not self._selection_lock.acquire(blocking=False):
            raise DirectoryError("已有目录选择窗口，请先完成或取消。")
        try:
            selected = self._picker(purpose, initial)
            if not selected:
                return {"cancelled": True, "path": None, "displayPath": None}
            backend = windows_to_backend(selected, self.distro)
            return {"cancelled": False, "path": backend, "displayPath": ntpath.normpath(selected)}
        finally:
            self._selection_lock.release()

    def open(self, validated_path):
        if not self.capabilities()["open"]:
            raise DirectoryError("当前桥接不支持打开本机目录。")
        native = backend_to_windows(validated_path, self.distro)
        self._opener(native)
        return {"ok": True, "path": validated_path, "displayPath": native}

    @staticmethod
    def _pick(purpose, initial):
        argv = [sys.executable, str(Path(__file__).resolve()), "--pick-dir", "--purpose", purpose]
        if initial:
            argv.extend(["--initial", initial])
        try:
            result = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8",
                                    timeout=300, check=True,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            response = json.loads(result.stdout)
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            raise DirectoryError("目录选择窗口未完成，请重试。") from error
        if response.get("error"):
            raise DirectoryError("无法打开原生目录选择窗口。")
        return response.get("path")

    @staticmethod
    def _open(path):
        if not os.path.isdir(path):
            raise DirectoryError("目录不可访问，可能已移动或 WSL 未连接。")
        try:
            os.startfile(path, "explore")
        except OSError as error:
            raise DirectoryError("无法在资源管理器中打开该目录。") from error


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pick-dir", action="store_true", required=True)
    parser.add_argument("--purpose", choices=("save", "import"), required=True)
    parser.add_argument("--initial")
    options = parser.parse_args(argv)
    try:
        import tkinter
        from tkinter import filedialog
        root = tkinter.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        try:
            path = filedialog.askdirectory(parent=root, mustexist=True,
                title="选择 checkpoint 保存目录" if options.purpose == "save" else "选择 checkpoint 或训练目录",
                **({"initialdir": options.initial} if options.initial else {}))
        finally:
            root.destroy()
        response = {"path": path or None}
    except Exception:
        response = {"path": None, "error": "directory_dialog_unavailable"}
    print(json.dumps(response, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
