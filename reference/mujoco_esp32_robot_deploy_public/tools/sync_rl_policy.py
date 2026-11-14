#!/usr/bin/env python3
"""把训练仓 checkpoint 导进 src/policy/rl_policy.cpp。

框架契约：sim / real 只认这一份数组。默认导 DEPLOY_CKPT。
换策略先改 host/policy/ckpt.py 和 src/policy/rl_policy.h，再跑本脚本。

  python tools/sync_rl_policy.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

FW = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FW / "host"))
from policy.ckpt import DEPLOY_CKPT, POLICY_CPP, POLICY_H  # noqa: E402

MARKER = "unsigned int timestamp;"
START = "// Extracted from checkpoint:"


def _rewrite_header(ckpt_rel: str) -> None:
    text = POLICY_H.read_text()
    new, n = __import__("re").subn(
        r'#define\s+DEPLOY_CKPT\s+"[^"]+"',
        f'#define DEPLOY_CKPT "{ckpt_rel}"',
        text,
        count=1,
    )
    if n != 1:
        sys.exit(f"{POLICY_H}: 无法更新 DEPLOY_CKPT")
    POLICY_H.write_text(new)


def _rewrite_py_contract(ckpt_rel: str) -> None:
    path = FW / "host" / "policy" / "ckpt.py"
    text = path.read_text()
    new, n = __import__("re").subn(
        r'DEPLOY_CKPT = "[^"]+"',
        f'DEPLOY_CKPT = "{ckpt_rel}"',
        text,
        count=1,
    )
    if n != 1:
        sys.exit(f"{path}: 无法更新 DEPLOY_CKPT")
    path.write_text(new)


def main() -> None:
    rl = Path(os.environ.get("LEGWHEEL_ROOT", str(Path.home() / "Work/robotics/workspace/legwheel_rl")))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=DEPLOY_CKPT, help="相对 LEGWHEEL_ROOT 或绝对路径")
    ap.add_argument("--python", default=str(rl / ".venv" / "bin" / "python"))
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    ckpt_rel = args.ckpt if not ckpt.is_absolute() else DEPLOY_CKPT
    if ckpt.is_absolute():
        marker = "/ckpts/"
        idx = str(ckpt).find(marker)
        ckpt_rel = str(ckpt)[idx + 1 :] if idx >= 0 else DEPLOY_CKPT
        ckpt_path = ckpt
    else:
        ckpt_rel = str(ckpt)
        ckpt_path = rl / ckpt
    if not ckpt_path.exists():
        sys.exit(f"ckpt 不存在: {ckpt_path}")
    py = Path(args.python)
    if not py.exists():
        sys.exit(f"找不到训练仓 python: {py}")

    out = Path("/tmp/rl_policy_params.cpp.txt")
    export = rl / "scripts" / "export_params.py"
    subprocess.check_call([str(py), str(export), "--ckpt", str(ckpt_path), "--out", str(out)])
    exported = out.read_text().rstrip() + "\n"
    # 契约用相对路径，不要把本机绝对路径写进固件。
    lines = exported.splitlines(True)
    if lines and "Extracted from checkpoint:" in lines[0]:
        lines[0] = f"{START} {ckpt_rel}\n"
        exported = "".join(lines)

    text = POLICY_CPP.read_text()
    start = text.find(START)
    end = text.find(MARKER)
    if start < 0 or end < 0:
        sys.exit(f"{POLICY_CPP}: 找不到 splice 标记")
    POLICY_CPP.write_text(text[:start] + exported + "\n\n" + text[end:])
    if ckpt_rel != DEPLOY_CKPT:
        _rewrite_header(ckpt_rel)
        _rewrite_py_contract(ckpt_rel)
        print(f"已把框架契约 DEPLOY_CKPT 改成 {ckpt_rel}")
    print(f"已写入 {POLICY_CPP}")
    print(f"{START} {ckpt_rel}")


if __name__ == "__main__":
    main()
