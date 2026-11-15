"""Read one verified training run's saved learning metrics; no training or merging."""
import json
import math
from pathlib import Path


def load_learning_data(artifacts: Path, mode="torque200") -> dict:
    if mode == "original":
        return load_original_learning_data(artifacts)
    if mode != "torque200":
        raise ValueError("Unknown mode")
    root = Path(artifacts).resolve()

    def trusted(path):
        path = Path(path).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Artifact path escapes root")
        return path

    def read(path):
        path = trusted(path)
        if path.stat().st_size > 2 * 1024 * 1024:
            raise ValueError("Artifact JSON too large")
        def reject(value):
            raise ValueError(f"Non-finite JSON constant: {value}")
        return json.loads(path.read_text(encoding="utf8"), parse_constant=reject)

    def number(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return float(value) if math.isfinite(value) else None

    live_file = root / "training-live.json"
    if live_file.is_file():
        try:
            live = read(live_file)
            directory = trusted(live["run_dir"])
            progress_file = trusted(live["progress_file"])
            if progress_file != directory / "progress.json":
                raise ValueError("Unexpected training progress path")
            config = read(directory / "env_config.json")
            ppo = read(directory / "ppo_config.json")
            dt = number(config.get("ctrl_dt"))
            if dt != 0.005:
                raise ValueError("Unexpected control timestep")
            events = read(progress_file)
            if not isinstance(events, list) or len(events) > 2000:
                raise ValueError("Invalid live progress")
            points = []
            for event in events:
                steps = number(event.get("steps"))
                if steps is None or steps < 0 or int(steps) != steps:
                    raise ValueError("Invalid live step")
                metrics = event["metrics"]
                length = number(metrics.get("eval/avg_episode_length"))
                points.append({"steps": int(steps),
                               "episode_seconds": length * dt if length is not None else None,
                               "episode_reward": number(metrics.get("eval/episode_reward")),
                               "value_loss": number(metrics.get("training/v_loss")),
                               "policy_loss": number(metrics.get("training/policy_loss"))})
            if any(b["steps"] <= a["steps"] for a, b in zip(points, points[1:])):
                raise ValueError("Non-increasing live progress")
            if not points:
                return {"available": False, "metadata": {"message": "本次训练准备中，等待首次评估记录。"}, "points": []}
            return {"available": True, "metadata": {
                "run": str(directory.relative_to(root)), "checkpoint": live.get("checkpoint"),
                "progress_file": str(progress_file), "ctrl_dt": dt,
                "episode_target_seconds": float(config["episode_length"]) * dt,
                "eval_envs": int(ppo["num_eval_envs"]), "point_count": len(points),
                "eval_point_count": sum(p["episode_seconds"] is not None for p in points),
                "phase": live.get("phase"), "source": "本次定时训练的实时记录",
                "limitations": ["仅显示本次运行；初始参数来自已有模型，步数从本轮零点起算。",
                                 "奖励和回合时长仅在真实评估时记录，不用训练loss补造评估点。",
                                 "评估环境少，均值有噪声；回合时长不是成功率。",
                                 "Loss下降不等于学会平衡；最终模型保存验证状态见训练进度。",
                                 "曲线每10秒刷新，画面关闭不影响更新。"]}, "points": points}
        except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError):
            return {"available": False, "metadata": {"message": "本次训练记录暂不可读，等待下一次保存。"}, "points": []}

    candidates = []
    try:
        latest = read(root / "performance-latest.json")
        if latest.get("completed") is True:
            checkpoint = trusted(latest["checkpoint"])
            if checkpoint.parent.parent.parent.name.startswith("performance-"):
                candidates.append((checkpoint, "已验证的性能对照训练模型"))
    except (OSError, ValueError, KeyError, TypeError):
        pass
    for run in sorted(root.glob("smoke-*"), reverse=True)[:64]:
        try:
            verified = read(run / "parameter-verification.json")
            if verified.get("both_parameter_trees_verified") is True:
                report = read(run / "resumed/report.json")
                candidates.append((trusted(report["checkpoint"]), "已验证的短训练模型"))
        except (OSError, ValueError, KeyError, TypeError):
            continue

    for checkpoint, source in candidates:
        try:
            directory = trusted(checkpoint.parent.parent)
            if not checkpoint.is_dir() or not checkpoint.name.isdigit():
                continue
            report = read(directory / "report.json")
            if (report.get("all_parameters_finite") is not True or
                    report.get("parameter_tree_max_abs_error") != 0 or
                    trusted(report["checkpoint"]) != checkpoint):
                continue
            export_error = number(report.get("export_max_abs_error"))
            export_atol = number(report.get("export_atol"))
            if export_error is None or export_atol is None or export_error > export_atol:
                continue
            config = read(directory / "env_config.json")
            ppo = read(directory / "ppo_config.json")
            dt = number(config.get("ctrl_dt"))
            if dt != 0.005:
                continue
            events = read(directory / "progress.json")
            if not isinstance(events, list) or not 1 <= len(events) <= 2000:
                continue
            points = []
            for event in events:
                steps = number(event.get("steps"))
                if steps is None or steps < 0 or int(steps) != steps:
                    raise ValueError("Invalid step count")
                metrics = event["metrics"]
                length = number(metrics.get("eval/avg_episode_length"))
                points.append({"steps": int(steps),
                               "episode_seconds": length * dt if length is not None else None,
                               "episode_reward": number(metrics.get("eval/episode_reward")),
                               "value_loss": number(metrics.get("training/v_loss")),
                               "policy_loss": number(metrics.get("training/policy_loss"))})
            if (any(b["steps"] <= a["steps"] for a, b in zip(points, points[1:])) or
                    points[-1]["steps"] != int(checkpoint.name) or
                    not any(p["episode_seconds"] is not None for p in points)):
                continue
            return {"available": True, "metadata": {
                "run": str(directory.relative_to(root)), "checkpoint": str(checkpoint),
                "progress_file": str(directory / "progress.json"), "ctrl_dt": dt,
                "episode_target_seconds": float(config["episode_length"]) * dt,
                "eval_envs": int(ppo["num_eval_envs"]), "point_count": len(points),
                "source": source,
                "limitations": ["只展示这一个运行的已保存数据，不拼接其他轮次。",
                                "评估环境少，均值有噪声；回合时长不是成功率。",
                                "Loss 下降不等于机器人学会平衡。",
                                "吞吐探针跳过了评估，不用于判断控制效果改善。",
                                "每10秒重新读取保存记录，不表示正在训练。"]}, "points": points}
        except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError):
            continue
    return {"available": False, "metadata": {"message": "暂无包含评估指标的已验证训练记录。"}, "points": []}


def load_original_learning_data(artifacts: Path) -> dict:
    """Original GPU smoke evidence, explicitly separate from the displayed author policy."""
    root = Path(artifacts).resolve()
    source = "原版短训练记录，与作者预训练展示模型不同"

    def read(path):
        path = path.resolve()
        if not path.is_relative_to(root) or path.stat().st_size > 2 * 1024 * 1024:
            raise ValueError("Invalid original artifact path")
        def reject(value):
            raise ValueError("Non-finite JSON")
        return json.loads(path.read_text(encoding="utf8"), parse_constant=reject)

    def number(value):
        return float(value) if (not isinstance(value, bool) and isinstance(value, (float, int))
                                and math.isfinite(value)) else None

    for directory in sorted(root.glob("train-gpu-*"), reverse=True)[:64]:
        try:
            report = read(directory / "report.json")
            restore = number(report.get("restore_max_abs_error"))
            error = number(report.get("export_max_abs_error"))
            atol = number(report.get("export_atol"))
            if restore is None or restore > 1e-6 or error is None or atol is None or error > atol:
                continue
            checkpoint = Path(report["checkpoint"]).resolve()
            if (not checkpoint.is_relative_to(directory.resolve()) or
                    not checkpoint.is_dir() or not checkpoint.name.isdigit()):
                continue
            config = read(directory / "env_config.json")
            ppo = read(directory / "ppo_config.json")
            dt = number(config.get("ctrl_dt"))
            if dt != 0.01:
                continue
            events = read(directory / "progress.json")
            if not isinstance(events, list) or not 1 <= len(events) <= 2000:
                continue
            points = []
            for event in events:
                steps = number(event.get("steps"))
                if steps is None or steps < 0 or int(steps) != steps:
                    raise ValueError("Invalid step count")
                metrics = event["metrics"]
                length = number(metrics.get("eval/avg_episode_length"))
                points.append({"steps": int(steps),
                               "episode_seconds": length * dt if length is not None else None,
                               "episode_reward": number(metrics.get("eval/episode_reward")),
                               "value_loss": number(metrics.get("training/v_loss")),
                               "policy_loss": number(metrics.get("training/policy_loss"))})
            if (any(b["steps"] <= a["steps"] for a, b in zip(points, points[1:])) or
                    points[-1]["steps"] != int(checkpoint.name) or
                    not any(p["episode_seconds"] is not None for p in points)):
                continue
            return {"available": True, "metadata": {
                "run": directory.name, "checkpoint": str(checkpoint),
                "progress_file": str(directory / "progress.json"), "ctrl_dt": dt,
                "episode_target_seconds": float(config["episode_length"]) * dt,
                "eval_envs": int(ppo["num_eval_envs"]), "point_count": len(points),
                "source": source, "limitations": [source,
                    "短训练仅验证训练、保存、恢复和导出流程，不能证明策略收敛。",
                    "只读取通过恢复及导出核对的原版GPU运行；不读取全扭矩训练状态。",
                    "每10秒重新读取保存记录，不表示正在训练。"]}, "points": points}
        except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError):
            continue
    return {"available": False, "metadata": {
        "message": source + "。暂无包含评估指标的已验证原版训练记录。"}, "points": []}


def load_console_learning_data(run_dir: Path, mode: str) -> dict:
    """Read the bounded live window; full history stays in metrics.jsonl."""
    waiting = {"available": False, "metadata": {"message": "本轮训练正在准备，等待首次记录。"}, "points": []}
    try:
        from control_modes import mode_spec
        spec = mode_spec(mode)
        directory = Path(run_dir).resolve()
        def read(name):
            path = directory / name
            if path.stat().st_size > 4 * 1024 * 1024:
                raise ValueError("Live snapshot too large")
            return json.loads(path.read_text(encoding="utf8"), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
        profile = {}
        if mode == "leg_force_bc":
            for name in ("status.json", "report.json"):
                if (directory / name).is_file():
                    value = read(name)
                    if isinstance(value, dict):
                        profile = value
                        break
            if (profile.get("bc_gate_passed") is False and profile.get("training_clock_started") is False
                    and (profile.get("phase") == "failed" or profile.get("stage") == "teacher_gate_failed")):
                return {"available": False, "metadata": {
                    "message": "BC闭环站立验收未通过，本轮未进入PPO；没有PPO学习曲线。请查看准备阶段错误。"}, "points": []}
        config, ppo, events = read("env_config.json"), read("ppo_config.json"), read("progress.json")
        dt = config["ctrl_dt"]
        if dt != 1 / spec["control_hz"]:
            raise ValueError("Mode mismatch")
        if spec["hybrid"] and not spec["teacher"]:
            policy = read("policy_metadata.json")
            if policy.get("action_mode") != spec["action_mode"] or policy.get("control_hz") != spec["control_hz"]:
                raise ValueError("Hybrid action mode or frequency mismatch")
        if spec["teacher"]:
            policy = read("policy_metadata.json")
            if policy.get("mode") != mode or policy.get("control_hz") != spec["control_hz"] or policy.get("actor_dim") != spec["obs_dim"]:
                raise ValueError("Teacher student metadata mismatch")
        if not isinstance(events, list) or len(events) > 2000:
            raise ValueError("Invalid progress window")
        def number(value):
            return float(value) if isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) else None
        points = []
        for event in events:
            steps = event["steps"]
            if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
                raise ValueError("Invalid step")
            metrics = event["metrics"]
            length = number(metrics.get("eval/avg_episode_length"))
            measured = [key for key, source in (
                ("episode_seconds", "eval/avg_episode_length"), ("episode_reward", "eval/episode_reward"),
                ("value_loss", "training/v_loss"), ("policy_loss", "training/policy_loss")) if source in metrics]
            points.append({"steps": steps, "episode_seconds": None if length is None else length * dt,
                "episode_reward": number(metrics.get("eval/episode_reward")),
                "value_loss": number(metrics.get("training/v_loss")), "policy_loss": number(metrics.get("training/policy_loss")),
                "measured": measured})
        if not points:
            return waiting
        if any(b["steps"] <= a["steps"] for a, b in zip(points, points[1:])):
            raise ValueError("Non-increasing steps")
        return {"available": True, "metadata": {"run": directory.name,
            "progress_file": str(directory / "progress.json"), "ctrl_dt": dt,
            "episode_target_seconds": config["episode_length"] * dt,
            "eval_envs": ppo.get("num_eval_envs", 2), "point_count": len(points),
            "eval_point_count": sum(p["episode_seconds"] is not None for p in points),
            "source": "站立阶段PPO记录（确定性站立评估）" if profile.get("training_profile") == "standing_v2" and profile.get("evaluation_policy") == "deterministic" else "控制台本轮训练记录",
            "training_profile": profile.get("training_profile"), "evaluation_policy": profile.get("evaluation_policy"),
            "limitations": (["本轮为站立阶段，评估使用确定性策略；不代表速度或转向任务已通过。"] if profile.get("training_profile") == "standing_v2" and profile.get("evaluation_policy") == "deterministic" else []) + [
                "曲线最多显示最近2000个记录点；完整记录保存在本轮metrics.jsonl。",
                "续训步数从本轮零点开始；不同运行不会拼接。",
                "奖励与回合长度只使用实际评估，缺失值不会补造。",
                f"每次{ppo.get('num_eval_envs', 2)}个评估环境；回合时长不等于成功率，loss不代表平衡能力。",
                "展示模型独立加载，训练结束后点击载入最新模型查看成果。"]}, "points": points}
    except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError):
        return waiting
