"""Render saved balance diagnostics without loading policies or contacting services.

Matplotlib is imported only when plotting. The JSON trace remains the source of
truth; HTML and CSV are reviewable views and can be regenerated independently.
"""
from __future__ import annotations

from collections.abc import Mapping
import csv
from html import escape
import math
from pathlib import Path


_CSV_FIELDS = (
    "policy_label", "policy_id", "initial_state_id", "observed_seconds",
    "termination", "first_6s_max_displacement_m", "max_displacement_m",
    "pitch_error_rms_deg", "horizontal_speed_rms_m_s", "reward_per_second",
)


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _series(case):
    trace = case["trace"]
    times = [_number(value, "t_s") for value in trace["t_s"]]
    if (len(times) < 2 or times[0] != 0 or
            any(right <= left for left, right in zip(times, times[1:]))):
        raise ValueError("Report trace needs time zero and strictly increasing post-step samples")
    count = len(times)
    for key in ("base_position_m", "base_velocity_m_s", "pitch_rad", "roll_rad",
                "pitch_rate_rad_s", "wheel_angle_distance_m", "reward", "done"):
        if len(trace[key]) != count:
            raise ValueError(f"Report trace {key} length differs from t_s")
    if any(type(value) is not bool for value in trace["done"]):
        raise ValueError("Report trace done values must be boolean")
    if any(trace["done"][:-1]):
        raise ValueError("Report trace contains samples after the first episode ended")
    positions, velocities = [], []
    for key, destination in (("base_position_m", positions), ("base_velocity_m_s", velocities)):
        for row in trace[key]:
            if len(row) != 3:
                raise ValueError(f"Report trace {key} needs three coordinates per sample")
            destination.append([_number(value, key) for value in row])
    reference = _number(trace["metadata"]["reference_pitch_rad"], "reference_pitch_rad")
    origin = positions[0]
    x = [row[0] - origin[0] for row in positions]
    y = [row[1] - origin[1] for row in positions]
    distance = [math.hypot(dx, dy) for dx, dy in zip(x, y)]
    speed = [math.hypot(row[0], row[1]) for row in velocities]
    pitch = [math.degrees(_number(value, "pitch_rad") - reference) for value in trace["pitch_rad"]]
    pitch_rate = [_number(value, "pitch_rate_rad_s") for value in trace["pitch_rate_rad_s"]]
    wheel = [_number(value, "wheel_angle_distance_m") for value in trace["wheel_angle_distance_m"]]
    wheel = [value - wheel[0] for value in wheel]
    reward = [_number(value, "reward") for value in trace["reward"]]
    return dict(t=times, x=x, y=y, distance=distance, speed=speed,
                pitch=pitch, pitch_rate=pitch_rate, wheel=wheel, reward=reward)


def _summary(case, series):
    from .metrics import analyze_trace

    trace = case["trace"]
    termination = trace["termination"]
    terminated = termination.get("terminated") is True
    truncated = termination.get("truncated") is True
    if terminated and truncated:
        raise ValueError("An episode cannot be both terminated and truncated")
    reason = str(termination.get("reason") or ("time_limit" if truncated else "incomplete"))
    if terminated:
        ending = "terminated:" + reason
    elif truncated:
        ending = "truncated:" + reason
    else:
        ending = "incomplete:" + reason
    analysis = case.get("analysis")
    if analysis is None:
        analysis = analyze_trace(trace)
    windows = analysis["windows"]
    first, full = windows["first_6s"], windows["full_episode"]
    duration = _number(full["observed_seconds"], "observed_seconds")
    if not math.isclose(duration, series["t"][-1], rel_tol=0, abs_tol=1e-9):
        raise ValueError("Saved analysis duration differs from the trace")
    return {
        "policy_label": str(case["label"]), "policy_id": str(case["policy_id"]),
        "initial_state_id": str(case["initial_state_id"]), "observed_seconds": duration,
        "termination": ending,
        "first_6s_max_displacement_m": first["max_horizontal_displacement_m"],
        "max_displacement_m": full["max_horizontal_displacement_m"],
        "pitch_error_rms_deg": full["pitch_error_rms_deg"],
        "horizontal_speed_rms_m_s": full["horizontal_speed_rms_m_s"],
        "reward_per_second": full["reward_per_second"],
    }


def _plot(bundle, prepared, destination, *, view_seconds=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    policies = list(dict.fromkeys(str(case["policy_id"]) for case, _, _ in prepared))
    states = list(dict.fromkeys(str(case["initial_state_id"]) for case, _, _ in prepared))
    palette = plt.get_cmap("tab10")
    colors = {policy: palette(index % 10) for index, policy in enumerate(policies)}
    figure, axes = plt.subplots(len(states), 5, figsize=(20, 3.6 * len(states)), squeeze=False)
    horizon = float(bundle["horizon_s"]) if view_seconds is None else view_seconds
    for row, state in enumerate(states):
        panels = axes[row]
        for case, series, summary in prepared:
            if str(case["initial_state_id"]) != state:
                continue
            color = colors[str(case["policy_id"])]
            # IDs are stable ASCII-friendly identities; labels in the HTML may
            # contain Chinese or arbitrary user text without needing plot fonts.
            label = str(case.get("plot_label", str(case["policy_id"])[:20]))
            time = series["t"]
            panels[0].plot(series["x"], series["y"], color=color, label=label, linewidth=1.4)
            panels[0].plot(series["x"][0], series["y"][0], "o", color=color, markersize=3)
            panels[1].plot(time, series["distance"], color=color, label=label + " body", linewidth=1.3)
            panels[1].plot(time, [abs(value) for value in series["wheel"]], "--", color=color,
                           label=label + " |wheel-angle|", linewidth=1.0, alpha=.75)
            panels[2].plot(time, series["pitch"], color=color, label=label, linewidth=1.3)
            panels[3].plot(time, series["speed"], color=color, label=label, linewidth=1.3)
            panels[4].plot(time, series["pitch_rate"], color=color, label=label, linewidth=1.3)
            if summary["termination"].startswith("terminated:"):
                for panel, value in zip(panels, (series["y"][-1], series["distance"][-1],
                                                series["pitch"][-1], series["speed"][-1],
                                                series["pitch_rate"][-1])):
                    endpoint_x = series["x"][-1] if panel is panels[0] else time[-1]
                    panel.plot(endpoint_x, value, "x", color=color, markersize=6)
        titles = ("Body path", "Body |xy| / |wheel-angle distance|", "Pitch error to reset reference",
                  "Horizontal body speed", "Pitch rate")
        ylabels = ("World y offset (m)", "Distance (m)", "Pitch error (deg)", "Speed (m/s)", "Pitch rate (rad/s)")
        for column, panel in enumerate(panels):
            panel.set_title(titles[column], fontsize=10)
            panel.set_ylabel(ylabels[column], fontsize=9)
            panel.set_xlabel("World x offset (m)" if column == 0 else "Time (s)", fontsize=9)
            panel.tick_params(labelsize=8)
            panel.grid(alpha=.18)
            panel.spines[["top", "right"]].set_visible(False)
            if column:
                panel.axvline(6.0, color="0.5", linestyle=":", linewidth=1)
                panel.set_xlim(0, horizon)
            else:
                panel.set_aspect("equal", adjustable="datalim")
            if row == 0:
                panel.legend(fontsize=7, loc="best")
        panels[0].text(0, 1.19, "Initial state: " + state, transform=panels[0].transAxes,
                       fontsize=11, fontweight="bold")
    figure.suptitle("Frozen-policy balance diagnostics | x = task termination | dotted line = 6 s",
                   fontsize=13, y=1.005)
    figure.tight_layout()
    try:
        figure.savefig(destination, dpi=160, bbox_inches="tight", facecolor="white")
    finally:
        plt.close(figure)


def write_report(bundle, output_dir):
    """Write comparison.png, report.html and summary.csv from a saved bundle.

    The caller writes ``results.json`` alongside these views. No model is loaded,
    no browser is opened, and no service is contacted here.
    """
    if (not isinstance(bundle, Mapping) or bundle.get("schema_version") != 1 or
            bundle.get("kind") != "balance_diagnostics"):
        raise ValueError("Expected balance_diagnostics bundle schema_version=1")
    if _number(bundle["horizon_s"], "horizon_s") <= 0:
        raise ValueError("horizon_s must be positive")
    cases = bundle.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Report needs at least one case")
    prepared = []
    for case in cases:
        series = _series(case)
        prepared.append((case, series, _summary(case, series)))
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    figure_path = destination / "comparison.png"
    _plot(bundle, prepared, figure_path)
    detail_path = destination / "comparison-detail.png"
    detail_seconds = min(float(bundle["horizon_s"]), max(series["t"][-1] for _, series, _ in prepared) * 1.05)
    _plot(bundle, prepared, detail_path, view_seconds=detail_seconds)
    csv_path = destination / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(summary for _, _, summary in prepared)
    headers = ("策略", "初态", "存活（秒）", "结束原因", "前6秒最大位移（米）", "全程最大位移（米）",
               "俯仰误差RMS（度）", "平面速度RMS（米/秒）", "奖励/秒")
    rows = []
    for _, _, summary in prepared:
        cells = [summary["policy_label"], summary["initial_state_id"], f'{summary["observed_seconds"]:.2f}',
                 summary["termination"], f'{summary["first_6s_max_displacement_m"]:.4f}',
                 f'{summary["max_displacement_m"]:.4f}', f'{summary["pitch_error_rms_deg"]:.2f}',
                 f'{summary["horizontal_speed_rms_m_s"]:.4f}', f'{summary["reward_per_second"]:.3f}']
        rows.append("<tr>" + "".join("<td>" + escape(cell) + "</td>" for cell in cells) + "</tr>")
    table_head = "".join("<th>" + escape(value) + "</th>" for value in headers)
    horizon = float(bundle["horizon_s"])
    content = f'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>固定策略平衡诊断</title>
<style>body{{font:16px/1.65 system-ui,sans-serif;color:#222;background:white;margin:36px auto;max-width:1500px;padding:0 24px}}
h1{{font-size:26px;margin-bottom:12px}}p{{max-width:1000px}}img{{display:block;width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}
th,td{{padding:10px 12px;text-align:right;border-bottom:1px solid #ddd}}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2),th:nth-child(4),td:nth-child(4){{text-align:left}}
.table{{overflow-x:auto}}a{{color:#24577d}}small{{color:#666}}</style>
<h1>固定策略平衡诊断</h1>
<p>{escape(str(bundle.get("scope", "")))}</p>
<p>最长回放 {horizon:g} 秒，仅统计首回合；达到上限表示截断，提前终止保留原任务判据。30 秒存活本身不等于稳定或收敛。此报告使用固定策略，未运行新训练。</p>
<p>实线位移来自机身实际平面位移的模长；虚线为轮关节角乘轮半径推算距离的绝对值，不能替代机身真实位移。带符号的轮角推算量保留在 JSON。所有位置以各自初态为零点；俯仰误差相对该回合保存的平衡参考。图中叉号为任务终止，6 秒虚线用于查看原指标的观察边界，终止后的空白表示无首回合数据。</p>
<div class="table"><table><thead><tr>{table_head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p><small>RMS 按实际采样间隔加权；前6秒栏只使用已观测样本，提前失败不补零。事件阈值仅作描述性诊断提示，并非经验证的成功标准；先后关系不能单独证明因果。本结果限于列出的标称及固定扰动工况，不代表随机初态、抗推力或实机鲁棒性。</small></p>
<p><a href="results.json">完整轨迹与指标 JSON</a> · <a href="summary.csv">摘要 CSV</a> · <a href="comparison.png">完整曲线图</a></p>
<h2>已观测时间段放大</h2>
<img src="comparison-detail.png" alt="已观测时间段的机身轨迹、位移、俯仰、速度与角速度对照">
<h2>完整评估上限</h2>
<img src="comparison.png" alt="30秒评估上限内的轨迹，终止后不延续曲线">
</html>'''
    html_path = destination / "report.html"
    html_path.write_text(content + "\n", encoding="utf-8")
    return {"report_html": str(html_path), "comparison_png": str(figure_path),
            "comparison_detail_png": str(detail_path), "summary_csv": str(csv_path)}
