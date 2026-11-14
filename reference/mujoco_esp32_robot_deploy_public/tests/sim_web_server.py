#!/usr/bin/env python3
"""Web + sim：真机同一套 Control 页，旁边弹出 MuJoCo GLFW 窗口。

推荐：bash sim_web.sh
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FW_VENV = ROOT / ".venv" / "bin" / "python"
RL_VENV = (
    Path(os.environ.get("LEGWHEEL_ROOT", str(Path.home() / "Work/robotics/workspace/legwheel_rl")))
    / ".venv"
    / "bin"
    / "python"
)


def _as_bool(value: str) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _sim_python() -> Path:
    if FW_VENV.exists():
        return FW_VENV
    if RL_VENV.exists():
        return RL_VENV
    sys.exit(
        "找不到仿真 venv。\n"
        "在本仓库执行:\n"
        "  uv venv --python 3.12 .venv\n"
        "  uv pip install --python .venv/bin/python -r requirements-sim.txt"
    )


def _reexec_venv() -> None:
    target = _sim_python()
    if Path(sys.executable).resolve() == target.resolve():
        return
    os.execv(str(target), [str(target), str(Path(__file__).resolve()), *sys.argv[1:]])


def _apply_runtime(jax_platforms: str) -> None:
    os.environ["JAX_PLATFORMS"] = jax_platforms
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.1")
    os.environ.pop("MUJOCO_GL", None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--jax-platforms", type=str, default="cpu")
    ap.add_argument("--noise", type=str, default="false")
    ap.add_argument("--random-init", type=str, default="false")
    ap.add_argument("--hold-clip-wheel", type=float, default=5.0)
    ap.add_argument("--hold-clip-sinyaw", type=float, default=1.0)
    args = ap.parse_args()

    _reexec_venv()
    _apply_runtime(args.jax_platforms)

    sys.path.insert(0, str(ROOT / "host"))

    from sim.viewer import run

    run(
        host=args.host,
        port=args.port,
        noise=_as_bool(args.noise),
        random_init=_as_bool(args.random_init),
        hold_clip_wheel=args.hold_clip_wheel,
        hold_clip_sinyaw=args.hold_clip_sinyaw,
    )


if __name__ == "__main__":
    main()
