"""框架契约：sim 与 real 共用这一份部署 ckpt。

训练仓里的 orbax 目录只是导出源。运行时权重在 src/policy/rl_policy.cpp；
host/policy 从该文件读数组。仿真禁止另 load JAX ckpt。

更新：tools/sync_rl_policy.py（默认就把 DEPLOY_CKPT 导进去）。
本文件的 DEPLOY_CKPT 必须与 src/policy/rl_policy.h 的同名宏一致。
"""

from __future__ import annotations

import re
from pathlib import Path

# 相对训练仓根目录。换策略只改这一处和 rl_policy.h，再跑 sync。
DEPLOY_CKPT = "ckpts/newobs-norecovery/000403660800"

POLICY_CPP = Path(__file__).resolve().parents[2] / "src" / "policy" / "rl_policy.cpp"
POLICY_H = Path(__file__).resolve().parents[2] / "src" / "policy" / "rl_policy.h"


def ckpt_from_cpp_comment(text: str | None = None) -> str:
    if text is None:
        text = POLICY_CPP.read_text()
    for line in text.splitlines():
        if "Extracted from checkpoint:" in line:
            path = line.split("Extracted from checkpoint:", 1)[1].strip()
            # 允许绝对路径，契约只认相对训练仓的尾缀。
            for sep in ("/ckpts/", "ckpts/"):
                idx = path.find(sep)
                if idx >= 0:
                    rel = path[idx:]
                    return rel[1:] if rel.startswith("/") else rel
            return path
    raise ValueError(f"{POLICY_CPP}: 没有 Extracted from checkpoint 注释")


def ckpt_from_header(text: str | None = None) -> str:
    if text is None:
        text = POLICY_H.read_text()
    match = re.search(r'#define\s+DEPLOY_CKPT\s+"([^"]+)"', text)
    if not match:
        raise ValueError(f"{POLICY_H}: 没有 DEPLOY_CKPT")
    return match.group(1)


def assert_shared_ckpt() -> str:
    """三处必须是同一份：Python 契约、固件宏、cpp 导出注释。"""
    from_h = ckpt_from_header()
    from_cpp = ckpt_from_cpp_comment()
    if from_h != DEPLOY_CKPT:
        raise AssertionError(f"rl_policy.h DEPLOY_CKPT={from_h!r} != {DEPLOY_CKPT!r}")
    if from_cpp != DEPLOY_CKPT:
        raise AssertionError(f"rl_policy.cpp 导出注释={from_cpp!r} != {DEPLOY_CKPT!r}")
    return DEPLOY_CKPT
