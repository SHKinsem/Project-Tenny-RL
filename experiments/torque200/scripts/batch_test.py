#!/usr/bin/env python3
"""批量测试 & 失败分析工具 CLI。

用法:
  ─── 批量测试模式 ───

    python scripts/batch_test.py --ckpt_path ./ckpts/20250724-2106/000801792000 --num_envs 4096
    python scripts/batch_test.py --ckpt_path ./ckpts/... --num_envs 8192 --base_seed 123 --episode_multiplier 2
    python scripts/batch_test.py --ckpt_path ./ckpts/... --no_push            # 禁用随机推力
    python scripts/batch_test.py --ckpt_path ./ckpts/... --no_random_init    # 禁用随机初始化
    python scripts/batch_test.py --ckpt_path ./ckpts/... --save_dir my_results

  ─── 重现模式 ───

    python scripts/batch_test.py --ckpt_path ./ckpts/... --replay \
        --failure_file failures/failures_xxx.pkl --failure_idx 0
    python scripts/batch_test.py --ckpt_path ./ckpts/... --replay \
        --failure_file failures/failures_xxx.pkl --replay_top_n 10
    python scripts/batch_test.py --ckpt_path ./ckpts/... --replay \
        --failure_file failures/failures_xxx.pkl --sort_by reward
    python scripts/batch_test.py --ckpt_path ./ckpts/... --replay \
        --failure_file failures/failures_xxx.pkl --failure_idx 0 --no_push
    python scripts/batch_test.py --ckpt_path ./ckpts/... --replay \
        --failure_file failures/failures_xxx.pkl --failure_idx 0 --no_saved_actions

视频渲染特性:
    - 绿色 3D 箭头:(vx, vy) 速度向量,始终显示,长度正比于速度大小
    - 红色 3D 箭头:push 推力向量,仅在 push 生效时显示
    - OSD 信息叠加:步数、姿态角、命令、动作、控制值、陀螺仪、速度、关节等

精确重现机制:
    批量测试时保存完整的 batch_context(所有环境的 RNG key 和每步动作),
    重现时使用与原始 batch_test 完全相同的 vmap 计算图重跑,确保浮点运算
    bit-identical。若 batch_context 不可用,回退到 saved-action 单环境回放模式。
"""

import argparse
import logging
import os
import sys
from datetime import datetime

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("batch_test")


def main():
    from legwheel.evaluation.failures import (
        load_failures,
        plot_failure_distribution,
        save_failures,
    )
    from legwheel.evaluation.replay import replay_failure, replay_top_failures
    from legwheel.evaluation.runner import batch_test, print_stats

    parser = argparse.ArgumentParser(description="批量测试 & 失败分析工具")
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="训练好的 checkpoint 路径")
    parser.add_argument("--num_envs", type=int, default=4096,
                        help="并发环境数 (默认 4096)")
    parser.add_argument("--base_seed", type=int, default=42,
                        help="基础随机种子")
    parser.add_argument("--episode_multiplier", type=int, default=1,
                        help="episode 长度倍数 (默认 1 = 原始长度)")
    parser.add_argument("--no_push", action="store_true",
                        help="禁用随机推力")
    parser.add_argument("--no_random_init", action="store_true",
                        help="禁用随机初始化")

    # 重现模式
    parser.add_argument("--replay", action="store_true",
                        help="进入重现模式")
    parser.add_argument("--failure_file", type=str, default=None,
                        help="失败记录文件路径 (.pkl)")
    parser.add_argument("--failure_idx", type=int, default=None,
                        help="重现第几个失败 (按 fail_step 排序后的索引)")
    parser.add_argument("--replay_top_n", type=int, default=5,
                        help="重现前 N 个失败 (默认 5)")
    parser.add_argument("--sort_by", type=str, default="earliest",
                        choices=["earliest", "reward"],
                        help="失败排序方式")
    parser.add_argument("--save_dir", type=str, default="failures",
                        help="输出目录")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="env 配置覆盖 (dotted key, 可多次), 用于按 trial 训练配置评估"
        " (如 vel_pid.target_scale=44.4)",
    )

    parser.add_argument(
        "--no_saved_actions",
        action="store_true",
        help="重现时不使用保存的动作序列（改用 policy 重新推理；将跳过精确 vmap 重现）",
    )

    args = parser.parse_args()
    push = not args.no_push
    random_init = not args.no_random_init
    use_saved_actions = not args.no_saved_actions

    from legwheel.config import parse_override_pairs

    env_overrides = parse_override_pairs(args.override)

    if args.replay:
        # ── 重现模式 ──
        if not args.failure_file:
            logger.error("重现模式需要指定 --failure_file")
            sys.exit(1)
        failures, stats = load_failures(args.failure_file)
        logger.info("已加载 %d 条失败记录", len(failures))
        print_stats(stats)

        # ── 查找批量上下文文件（用于精确 vmap 重现） ──
        batch_context_file = stats.get("batch_context_file")
        if batch_context_file and not os.path.exists(batch_context_file):
            # 尝试相对于 pkl 文件所在目录查找
            alt_path = os.path.join(
                os.path.dirname(args.failure_file),
                os.path.basename(batch_context_file),
            )
            if os.path.exists(alt_path):
                batch_context_file = alt_path
            else:
                logger.warning("批量上下文文件不存在: %s (将使用 saved-action 模式)",
                              batch_context_file)
                batch_context_file = None
        if batch_context_file:
            logger.info("找到批量上下文文件: %s (将使用精确 vmap 重现)", batch_context_file)

        if args.failure_idx is not None:
            sorted_f = sorted(failures, key=lambda x: x.fail_step)
            if args.failure_idx >= len(sorted_f):
                logger.error("failure_idx=%d 超出范围 (共 %d 条)", args.failure_idx, len(sorted_f))
                sys.exit(1)
            f = sorted_f[args.failure_idx]
            video_dir = os.path.join(args.save_dir, "videos")
            os.makedirs(video_dir, exist_ok=True)
            video_path = os.path.join(video_dir,
                f"replay_env{f.env_idx}_step{f.fail_step}.mp4")
            replay_failure(
                ckpt_path=args.ckpt_path,
                failure=f,
                push=push,
                render=True,
                verbose=True,
                save_video_path=video_path,
                use_saved_actions=use_saved_actions,
                batch_context_file=batch_context_file,
            )
        else:
            replay_top_failures(
                ckpt_path=args.ckpt_path,
                failures=failures,
                n=args.replay_top_n,
                sort_by=args.sort_by,
                save_dir=os.path.join(args.save_dir, "videos"),
                push=push,
                use_saved_actions=use_saved_actions,
                batch_context_file=batch_context_file,
            )
    else:
        # ── 批量测试模式 ──
        failures, stats, batch_context = batch_test(
            ckpt_path=args.ckpt_path,
            num_envs=args.num_envs,
            random_init=random_init,
            push=push,
            base_seed=args.base_seed,
            episode_multiplier=args.episode_multiplier,
            overrides=env_overrides,
        )
        print_stats(stats)

        if failures:
            # 保存（含批量上下文，用于精确重现）
            filepath = save_failures(failures, stats, save_dir=args.save_dir,
                                     batch_context=batch_context)
            batch_context_file = stats.get("batch_context_file")
            # 绘图
            plot_path = os.path.join(args.save_dir,
                f"failure_analysis_{datetime.now().strftime('%Y%m%d-%H%M%S')}.png")
            plot_failure_distribution(failures, save_path=plot_path)
            # 自动重现前 N 个
            logger.info("\n自动重现最早失败的 %d 个例子 ...", min(args.replay_top_n, len(failures)))
            replay_top_failures(
                ckpt_path=args.ckpt_path,
                failures=failures,
                n=args.replay_top_n,
                sort_by=args.sort_by,
                save_dir=os.path.join(args.save_dir, "videos"),
                push=push,
                use_saved_actions=use_saved_actions,
                batch_context_file=batch_context_file,
            )
        else:
            logger.info("🎉 所有 %d 个环境均成功完成 episode！", args.num_envs)


if __name__ == "__main__":
    main()
