"""训练进度回调:实时打印/绘制评估曲线并保存 progress png。"""

import os

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output, display


def make_progress_fn(env_cfg, ppo_params, iter_now: int):
    """构造 brax progress_fn 回调。维护 reward 曲线(fig)与分项奖励曲线(ax2),
    每次评估把最新图保存到 ppo_params.save_checkpoint_path。"""
    x_data, y_data, y_dataerr = [], [], []
    times = []
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    def progress(num_steps, metrics):
        clear_output(wait=True)
        ax1.cla()
        ax2.cla()
        for leg in fig.legends:
            leg.remove()

        for k, v in metrics.items():
            print(f"{k}: {v}")

        from datetime import datetime

        times.append(datetime.now())
        x_data.append(num_steps)
        y_data.append(metrics["eval/episode_reward"])
        y_dataerr.append(metrics["eval/episode_reward_std"])

        xmax = max(1000, num_steps * 1.1)
        ax1.set_xlim([0, xmax])
        ax1.set_xlabel("# environment steps")
        ax1.set_ylabel("reward per episode")
        # brax 0.12.1 只输出 avg_episode_length;std_episode_length 仅新版 brax 提供,
        # 只显示实际存在的指标,缺失时不再显示 nan。
        title_line2 = ", ".join(
            f"{key.split('/')[-1]}:{metrics[key]:.3f}"
            for key in ("eval/avg_episode_length", "eval/std_episode_length")
            if key in metrics
        )
        ax1.set_title(
            f"iter={iter_now}, y={y_data[-1]:.3f}, eval:{len(y_data)}-{num_steps}\n" + title_line2
        )

        num_annotations = min(10, len(y_data))
        if len(y_data) > num_annotations:
            num_annotations = num_annotations - 1
        step = max(1, int(np.ceil(len(y_data) / num_annotations)))
        indices = list(range(0, len(y_data), step))
        if len(y_data) - 1 not in indices:
            indices.append(len(y_data) - 1)

        ax1.errorbar(
            x_data,
            y_data,
            yerr=y_dataerr,
            color="blue",
            errorevery=step,
            label="total reward",
        )
        ax1.scatter(
            [x_data[i] for i in indices], [y_data[i] for i in indices], color="blue"
        )

        for i in indices:
            offset = 10 if i % 2 == 0 else -15
            ax1.annotate(
                f"{y_data[i]:.2f}",
                (x_data[i], y_data[i]),
                textcoords="offset points",
                xytext=(0, offset),
                ha="center",
            )

        reward_metrics = {
            "eval/episode_reward/pitch_cost": {"color": "black", "name": "pitch"},
            "eval/episode_reward/roll_cost": {"color": "red", "name": "roll"},
            "eval/episode_reward/tracking_lin_vel": {
                "color": "green",
                "name": "tracking_lin_vel",
            },
            "eval/episode_reward/tracking_wheels_sinposition": {
                "color": "blue",
                "name": "tracking_wheels_sinposition",
            },
            "eval/episode_reward/tracking_sinyaw": {
                "color": "yellow",
                "name": "tracking_sinyaw",
            },
            "eval/episode_reward/tracking_ang_vel": {
                "color": "orange",
                "name": "tracking_ang_vel",
            },
            "eval/episode_reward/tracking_height_jnt": {
                "color": "cyan",
                "name": "tracking_height_jnt",
            },
            "eval/episode_reward/tracking_height_action": {
                "color": "gray",
                "name": "tracking_height_action",
            },
            "eval/episode_reward/action_rate": {
                "color": "magenta",
                "name": "action_rate",
            },
            "eval/episode_reward/energy_cost": {"color": "purple", "name": "energy"},
            "eval/episode_reward/pitch_velocity_penalty": {
                "color": "brown",
                "name": "pitch_velocity_penalty",
            },
        }

        for metric_key, props in reward_metrics.items():
            if metric_key in metrics:
                if not hasattr(progress, metric_key):
                    setattr(progress, metric_key, [])
                metric_data = getattr(progress, metric_key)
                metric_data.append(metrics[metric_key])
                ax2.plot(x_data, metric_data, color=props["color"], label=props["name"])

        ax2.set_ylabel("other rewards")
        episode_length = env_cfg.episode_length
        ax2.set_ylim(0, episode_length)
        fig.legend(
            loc="upper left",
            bbox_to_anchor=(0, 1),
            bbox_transform=ax1.transAxes,
            framealpha=0.5,
        )
        ax1.grid(True, linestyle="--", alpha=0.7)
        display(fig)

        # Save the latest progress plot into the current checkpoint folder
        try:
            save_dir = None
            if ppo_params is not None and hasattr(ppo_params, "save_checkpoint_path"):
                save_dir = ppo_params.save_checkpoint_path
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                fig.savefig(
                    os.path.join(save_dir, f"progress_{iter_now}.png"),
                    dpi=150,
                    bbox_inches="tight",
                )
        except Exception as e:
            print(f"Warning: failed to save progress figure: {e}")

    return progress
