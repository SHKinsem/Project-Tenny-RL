"""失败记录:数据结构、保存/加载与分布可视化。"""

import os
import pickle
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


@dataclass
class FailureRecord:
    """一个失败 episode 的完整信息"""
    env_idx: int           # 环境编号 (0 ~ num_envs-1)
    rng_seed: int          # 该环境的随机种子（仅用于显示）
    command: np.ndarray    # 初始 command (4,)
    fail_step: int         # 失败发生的步数
    episode_reward: float  # 截至失败的累计奖励
    sinpitch_at_fail: float
    sinroll_at_fail: float
    terminal_reason: str   # 'pitch' / 'roll' / 'nan' / 'unknown'
    qpos_at_fail: np.ndarray
    qvel_at_fail: np.ndarray
    # ── 以下字段用于精确重现 ──
    reset_rng: np.ndarray  # 实际传给 env.reset 的 JAX PRNGKey, shape (2,)
    actions: np.ndarray    # 失败前的完整动作序列, shape (fail_step, action_size)


def plot_failure_distribution(failures: List[FailureRecord], save_path: Optional[str] = None):
    """绘制失败分析图"""
    if not failures:
        logger.info("无失败记录，跳过绘图。")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Failure Analysis  (n={len(failures)})", fontsize=14)

    # 1. 失败步数分布
    fail_steps = [f.fail_step for f in failures]
    axes[0, 0].hist(fail_steps, bins=50, edgecolor="black", alpha=0.7)
    axes[0, 0].set_xlabel("Fail Step")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title("Fail Step Distribution")

    # 2. 失败原因饼图
    reasons = [f.terminal_reason for f in failures]
    reason_counts = {}
    for r in reasons:
        reason_counts[r] = reason_counts.get(r, 0) + 1
    axes[0, 1].pie(reason_counts.values(), labels=reason_counts.keys(),
                   autopct="%1.1f%%", startangle=90)
    axes[0, 1].set_title("Failure Reason")

    # 3. sinpitch 分布
    sinpitch_vals = [f.sinpitch_at_fail for f in failures]
    axes[0, 2].hist(sinpitch_vals, bins=50, edgecolor="black", alpha=0.7, color="orange")
    axes[0, 2].axvline(-0.6, color="red", linestyle="--", label="pitch threshold -0.6")
    axes[0, 2].axvline(0.85, color="red", linestyle="--", label="pitch threshold 0.85")
    axes[0, 2].legend()
    axes[0, 2].set_xlabel("sinPitch at Fail")
    axes[0, 2].set_title("sinPitch Distribution")

    # 4. sinroll 分布
    sinroll_vals = [f.sinroll_at_fail for f in failures]
    axes[1, 0].hist(sinroll_vals, bins=50, edgecolor="black", alpha=0.7, color="green")
    axes[1, 0].axvline(-0.6, color="red", linestyle="--", label="roll threshold ±0.6")
    axes[1, 0].axvline(0.6, color="red", linestyle="--")
    axes[1, 0].legend()
    axes[1, 0].set_xlabel("sinRoll at Fail")
    axes[1, 0].set_title("sinRoll Distribution")

    # 5. command 分布 (lin_vel_x vs ang_vel_yaw)
    cmds = np.array([f.command for f in failures])
    scatter = axes[1, 1].scatter(cmds[:, 0], cmds[:, 1], c=[f.fail_step for f in failures],
                                  cmap="viridis", alpha=0.6, s=10)
    plt.colorbar(scatter, ax=axes[1, 1], label="fail step")
    axes[1, 1].set_xlabel("cmd lin_vel_x")
    axes[1, 1].set_ylabel("cmd ang_vel_yaw")
    axes[1, 1].set_title("Command at Failure")

    # 6. 累计奖励分布
    rewards = [f.episode_reward for f in failures]
    axes[1, 2].hist(rewards, bins=50, edgecolor="black", alpha=0.7, color="purple")
    axes[1, 2].set_xlabel("Cumulative Reward")
    axes[1, 2].set_title("Failed Episode Reward")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("失败分析图已保存: %s", save_path)
    plt.show()


# ──────────────────────────────────────────────────────────────
# 保存 / 加载失败记录
# ──────────────────────────────────────────────────────────────
def save_failures(failures: List[FailureRecord], stats: Dict[str, Any],
                  save_dir: str = "failures",
                  batch_context: Optional[Dict[str, np.ndarray]] = None) -> str:
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # 保存批量上下文（用于精确 vmap 重现）
    if batch_context is not None:
        context_filename = f"batch_context_{timestamp}.npz"
        context_filepath = os.path.join(save_dir, context_filename)
        np.savez_compressed(
            context_filepath,
            env_rngs=batch_context["env_rngs"],
            all_actions=batch_context["all_actions"],
        )
        stats["batch_context_file"] = context_filepath
        size_mb = (batch_context["all_actions"].nbytes + batch_context["env_rngs"].nbytes) / 1e6
        logger.info("批量上下文已保存: %s (%.1f MB)", context_filepath, size_mb)

    filepath = os.path.join(save_dir, f"failures_{timestamp}.pkl")
    data = {"failures": failures, "stats": stats}
    with open(filepath, "wb") as f:
        pickle.dump(data, f)
    logger.info("失败记录已保存: %s  (共 %d 条)", filepath, len(failures))

    # 同时保存人类可读摘要
    summary_path = os.path.join(save_dir, f"summary_{timestamp}.txt")
    with open(summary_path, "w") as f:
        f.write(f"Batch Test Summary  {timestamp}\n")
        f.write("=" * 60 + "\n")
        for k, v in stats.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nAll {len(failures)} failures (sorted by fail_step):\n")
        sorted_failures = sorted(failures, key=lambda x: x.fail_step)
        for i, rec in enumerate(sorted_failures):
            f.write(f"  [{i}] env={rec.env_idx}, seed={rec.rng_seed}, "
                    f"step={rec.fail_step}, reason={rec.terminal_reason}, "
                    f"cmd={np.array2string(rec.command, precision=4)}, "
                    f"pitch={rec.sinpitch_at_fail:.4f}, roll={rec.sinroll_at_fail:.4f}, "
                    f"reward={rec.episode_reward:.3f}\n")
    logger.info("摘要已保存: %s", summary_path)
    return filepath


def load_failures(filepath: str) -> Tuple[List[FailureRecord], Dict[str, Any]]:
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data["failures"], data["stats"]
