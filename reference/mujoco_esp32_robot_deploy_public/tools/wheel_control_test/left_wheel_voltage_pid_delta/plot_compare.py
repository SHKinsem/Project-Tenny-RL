#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


REQUIRED_COLS = {"t_s", "battery_v", "control_v", "target_vel", "current_vel"}


def read_pid_csv(path: Path):
    t_s = []
    batt_v = []
    u_v = []
    target = []
    current = []
    raw = []

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or not REQUIRED_COLS.issubset(set(reader.fieldnames)):
            raise ValueError(f"CSV header must contain {sorted(REQUIRED_COLS)}; got {reader.fieldnames}")

        has_raw = reader.fieldnames is not None and ("raw_vel" in set(reader.fieldnames))

        for row in reader:
            t_s.append(float(row["t_s"]))
            batt_v.append(float(row["battery_v"]))
            u_v.append(float(row["control_v"]))
            target.append(float(row["target_vel"]))
            current.append(float(row["current_vel"]))
            if has_raw:
                raw.append(float(row["raw_vel"]))

    return t_s, batt_v, u_v, target, current, (raw if raw else None)


def apply_time_window(t_s, *series, tmin=None, tmax=None):
    if tmin is None and tmax is None:
        return (t_s, *series)

    tmin = float(tmin) if tmin is not None else None
    tmax = float(tmax) if tmax is not None else None

    idx = []
    for i, t in enumerate(t_s):
        if tmin is not None and t < tmin:
            continue
        if tmax is not None and t > tmax:
            continue
        idx.append(i)

    if not idx:
        raise ValueError("No samples in the requested time window")

    def pick(arr):
        return [arr[i] for i in idx]

    out = [pick(t_s)]
    out.extend(pick(s) for s in series)
    return tuple(out)


def default_out_path(csv_paths):
    stems = [p.stem for p in csv_paths]
    base = "compare_" + "_vs_".join(stems)
    base = base[:120]
    return Path("image") / (base + ".png")


def main():
    ap = argparse.ArgumentParser(description="Plot multiple pid_log CSVs on one figure for comparison")
    ap.add_argument("csv", type=Path, nargs="+", help="One or more pid_log.csv files")
    ap.add_argument("--labels", type=str, default=None, help="Comma-separated labels (same count as CSVs)")
    ap.add_argument("--tmin", type=float, default=None, help="Plot from this time (s)")
    ap.add_argument("--tmax", type=float, default=None, help="Plot until this time (s)")
    ap.add_argument("--save", type=Path, default=None, help="Save figure path. If omitted, saves to image/compare_*.png")
    args = ap.parse_args()

    csv_paths = [p for p in args.csv]
    labels = None
    if args.labels:
        provided = [s.strip() for s in args.labels.split(",") if s.strip()]
        stems = [p.stem for p in csv_paths]
        if not provided:
            labels = stems
        elif len(provided) < len(csv_paths):
            labels = provided + stems[len(provided) :]
        else:
            labels = provided[: len(csv_paths)]
    else:
        labels = [p.stem for p in csv_paths]

    fig, (ax_raw, ax_curr, ax_uq, ax_batt) = plt.subplots(4, 1, sharex=True, figsize=(10, 8))

    target_plotted_raw = False
    target_plotted_curr = False

    for path, label in zip(csv_paths, labels):
        t_s, batt_v, u_v, target, current, raw = read_pid_csv(path)
        if raw is not None:
            t_s, batt_v, u_v, target, current, raw = apply_time_window(
                t_s, batt_v, u_v, target, current, raw, tmin=args.tmin, tmax=args.tmax
            )
        else:
            t_s, batt_v, u_v, target, current = apply_time_window(
                t_s, batt_v, u_v, target, current, tmin=args.tmin, tmax=args.tmax
            )

        curr_line = ax_curr.plot(t_s, current, label=label)
        color = curr_line[0].get_color()
        linestyle = curr_line[0].get_linestyle()
        if raw is not None:
            ax_raw.plot(
                t_s,
                raw,
                label=f"{label} raw",
                color=color,
                linestyle=linestyle,
            )
        ax_uq.plot(t_s, u_v, label=label)
        ax_batt.plot(t_s, batt_v, label=label)

        if raw is not None and not target_plotted_raw:
            ax_raw.plot(t_s, target, color="k", linestyle="--", linewidth=1.2, label="target")
            target_plotted_raw = True
        if not target_plotted_curr:
            ax_curr.plot(t_s, target, color="k", linestyle="--", linewidth=1.2, label="target")
            target_plotted_curr = True

    ax_raw.set_ylabel("raw vel (rad/s)")
    ax_raw.grid(True, alpha=0.3)
    ax_raw.legend(loc="best")

    ax_curr.set_ylabel("current vel (rad/s)")
    ax_curr.grid(True, alpha=0.3)
    ax_curr.legend(loc="best")

    ax_uq.set_ylabel("Uq (V)")
    ax_uq.grid(True, alpha=0.3)
    ax_uq.legend(loc="best")

    ax_batt.set_ylabel("battery (V)")
    ax_batt.set_xlabel("time (s)")
    ax_batt.grid(True, alpha=0.3)
    ax_batt.legend(loc="best")

    fig.tight_layout()

    out_path = args.save if args.save else default_out_path(csv_paths)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)


if __name__ == "__main__":
    main()
