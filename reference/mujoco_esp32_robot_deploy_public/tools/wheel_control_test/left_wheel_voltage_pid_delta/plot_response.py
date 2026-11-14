#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_csv(path: Path):
    t_s = []
    batt_v = []
    u_v = []
    target = []
    current = []
    raw = []

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"t_s", "battery_v", "control_v", "target_vel", "current_vel"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"CSV header must contain {sorted(required)}; got {reader.fieldnames}")

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


def main():
    ap = argparse.ArgumentParser(description="Plot control response curves from pid_log CSV")
    ap.add_argument("csv", type=Path, help="Path to pid_log.csv")
    ap.add_argument("--tmin", type=float, default=None, help="Plot from this time (s)")
    ap.add_argument("--tmax", type=float, default=None, help="Plot until this time (s)")
    ap.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Save figure to file (png/pdf/svg). If omitted, saves to image/<csvname>.png",
    )
    args = ap.parse_args()

    t_s, batt_v, u_v, target, current, raw = read_csv(args.csv)
    if raw is not None:
        t_s, batt_v, u_v, target, current, raw = apply_time_window(
            t_s, batt_v, u_v, target, current, raw, tmin=args.tmin, tmax=args.tmax
        )
    else:
        t_s, batt_v, u_v, target, current = apply_time_window(
            t_s, batt_v, u_v, target, current, tmin=args.tmin, tmax=args.tmax
        )

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(10, 7))

    ax1.plot(t_s, target, label="target_vel (rad/s)")
    ax1.plot(t_s, current, label="current_vel (rad/s)")
    if raw is not None:
        ax1.plot(t_s, raw, linewidth=1.0, alpha=0.6, label="raw_vel (rad/s)")
    ax1.set_ylabel("velocity (rad/s)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")

    ax2.plot(t_s, u_v, color="tab:orange", label="control_v (V)")
    ax2.set_ylabel("Uq (V)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")

    ax3.plot(t_s, batt_v, color="tab:green", label="battery_v (V)")
    ax3.set_ylabel("battery (V)")
    ax3.set_xlabel("time (s)")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="best")

    fig.tight_layout()

    out_path = args.save if args.save else (Path("image") / (args.csv.stem + ".png"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)


if __name__ == "__main__":
    main()
