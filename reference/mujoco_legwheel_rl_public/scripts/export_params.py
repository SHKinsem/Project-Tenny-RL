#!/usr/bin/env python3
"""导出部署参数 CLI:checkpoint → C 数组文件(默认 deploy/params.cpp.txt)。

示例:
    python scripts/export_params.py --ckpt ckpts/20260111-2109/000801792000
    python scripts/export_params.py --ckpt ckpts/... --out deploy/params_new.cpp.txt
"""

import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=str, required=True, help="checkpoint 路径")
    ap.add_argument("--out", type=str, default=None, help="输出文件路径 (默认 deploy/params.cpp.txt)")
    args = ap.parse_args()

    from legwheel.deployment.extract import export_params

    export_params(args.ckpt, args.out)


if __name__ == "__main__":
    main()
