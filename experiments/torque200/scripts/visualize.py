#!/usr/bin/env python3
"""命令用例评估 CLI:跟踪曲线 + 演示视频,输出到 test/<时间戳>/。

示例:
    python scripts/visualize.py --ckpt ckpts/20260222-1708/000403660800
    python scripts/visualize.py --ckpt ckpts/... --no-push --duration 10
"""

import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=str, required=True, help="checkpoint 路径")
    ap.add_argument("--no-push", action="store_true", help="关闭随机推力")
    ap.add_argument("--duration", type=float, default=6.0, help="每个用例时长 (秒)")
    ap.add_argument("--no-writefile", action="store_true", help="不写 state_obs_output.txt")
    args = ap.parse_args()

    from legwheel.evaluation.visualize import evaluate

    evaluate(
        args.ckpt,
        push=not args.no_push,
        duration=args.duration,
        writefile=not args.no_writefile,
    )


if __name__ == "__main__":
    main()
