#!/usr/bin/env python3
"""训练入口。

示例:
    python scripts/train.py
    python scripts/train.py --iters 2 --num-evals 10 --maxstep 4e8
    python scripts/train.py --restore ckpts/20260222-1708/000403660800
"""

import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from legwheel.utils import setup_logging


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iters", type=int, default=1, help="课程学习迭代次数")
    ap.add_argument("--num-evals", type=int, default=10, help="每次迭代的评估次数")
    ap.add_argument(
        "--maxstep", type=float, default=4e8, help="总环境步数下限 (maxstep_candidate)"
    )
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument(
        "--restore", type=str, default=None, help="恢复训练的 checkpoint 路径"
    )
    ap.add_argument(
        "--save-dir", type=str, default=None, help="checkpoint 保存目录 (默认 ckpts/<时间戳>)"
    )
    ap.add_argument(
        "--no-inference", action="store_true", help="训练后不跑推理渲染 (远程无显示时用)"
    )
    ap.add_argument(
        "--seed", type=int, default=None, help="训练 seed (默认沿用 curriculum_config 的 0)"
    )
    ap.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="env 配置覆盖 (dotted key, 可多次), 如 reward_config.vel_scale=3.0",
    )
    ap.add_argument(
        "--ppo-override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="PPO 超参覆盖 (可多次), 如 half_life_s=0.8 / entropy_cost=3e-3",
    )
    args = ap.parse_args()

    setup_logging()

    from legwheel.config import apply_config_overrides, default_config, parse_override_pairs
    from legwheel.training.ppo import curriculum_learning, inference

    env_cfg = None
    env_overrides = parse_override_pairs(args.override)
    if env_overrides:
        env_cfg = apply_config_overrides(default_config(), env_overrides)
    ppo_overrides = parse_override_pairs(args.ppo_override)

    make_inference_fn, params, metrics, ppo_params = curriculum_learning(
        iter_num=args.iters,
        num_evals=args.num_evals,
        maxstep_candidate=args.maxstep,
        learning_rate=args.learning_rate,
        restore_checkpoint_path=args.restore,
        save_checkpoint_path=args.save_dir,
        env_cfg=env_cfg,
        ppo_overrides=ppo_overrides,
        seed=args.seed,
    )
    if not args.no_inference:
        inference(make_inference_fn, params, ppo_params)


if __name__ == "__main__":
    main()
