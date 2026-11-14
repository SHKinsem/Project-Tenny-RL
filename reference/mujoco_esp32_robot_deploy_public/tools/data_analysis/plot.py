import os
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np


def load_history(filename: str) -> np.ndarray:
	"""Load history data file as a float array.

	The expected format of each row is:
	time, info["command"](4), gyro(3), sinyaw(1), sinpitch(1), sinroll(1),
	jnts(4), velocity(4), wheel_linvel(2), previous_wheels_policyout(4),
	wheel_ctrl_vel_error(2), thigh_ctrl(2), wheel_ctrl(2)
	-> total 31 columns.
	"""
	base_dir = os.path.dirname(__file__)
	path = os.path.join(base_dir, "data", filename)
	data = np.loadtxt(path, delimiter=",")
	return data


def plot_group(
	time: np.ndarray,
	values: np.ndarray,
	labels,
	title: str,
	out_dir: str | None = None,
	filename: str | None = None,
):
	"""Plot a group of signals sharing the same time axis and optionally save.

	If ``out_dir`` is provided, the figure is saved into that directory with
	``filename`` (or a name derived from ``title`` if not given).
	"""
	if values.ndim == 1:
		values = values[:, None]

	plt.figure(figsize=(16, 4))
	for i in range(values.shape[1]):
		plt.plot(
			time,
			values[:, i],
			label=str(labels[i]),
			marker="o",
			markersize=2,
			linestyle="-",
		)
	plt.xlabel("time [s]")
	plt.ylabel(title)
	plt.title(title)

	ax = plt.gca()
	# If the visible time span is less than 1 second, use a
	# finer x-axis grid (≤ 10 ms spacing).
	if time.size > 1 and (time[-1] - time[0]) < 1.0:
		ax.xaxis.set_major_locator(MultipleLocator(0.02))  # 20 ms
		ax.xaxis.set_minor_locator(MultipleLocator(0.01))  # 10 ms
		ax.grid(True, which="both", alpha=0.3)
	else:
		ax.grid(True)
	plt.legend()
	plt.tight_layout()

	if out_dir is not None:
		os.makedirs(out_dir, exist_ok=True)
		if filename is None:
			# basic sanitising of title for a file name
			filename = (
				title.replace("[", "_")
				.replace("]", "")
				.replace(" ", "_")
				+ ".png"
			)
		path = os.path.join(out_dir, filename)
		plt.savefig(path)
		plt.close()


def main():
	# choose history file (can be overridden by CLI arg 1)
	# optional CLI usage:
	#   python3 plot.py [history_filename] [t_start] [t_end]
	# where t_start/t_end are in seconds (relative time axis)
	if len(sys.argv) > 1:
		history_filename = sys.argv[1]
	else:
		history_filename = "history (24).txt"

	# optional time range arguments
	t_start = None
	t_end = None
	if len(sys.argv) > 2:
		try:
			t_start = float(sys.argv[2])
		except ValueError:
			print(f"Warning: invalid t_start '{sys.argv[2]}', ignored.")
	if len(sys.argv) > 3:
		try:
			t_end = float(sys.argv[3])
		except ValueError:
			print(f"Warning: invalid t_end '{sys.argv[3]}', ignored.")

	# load data
	data = load_history(history_filename)
	# remove the second column 
	data = np.delete(data, 1, axis=1)

	# image root directory: image/<history_basename>
	base_dir = os.path.dirname(__file__)
	history_basename = os.path.splitext(os.path.basename(history_filename))[0]
	image_root = os.path.join(base_dir, "image", history_basename)

	# use relative time in seconds as x-axis
	t = (data[:, 0] - data[0, 0]) / 1000.0

	# apply time range mask if specified
	mask = np.ones_like(t, dtype=bool)
	if t_start is not None:
		mask &= t >= t_start
	if t_end is not None:
		mask &= t <= t_end

	if not np.any(mask):
		print("No data points in the specified time range; nothing to plot.")
		return

	t = t[mask]
	data = data[mask, :]

	# decide image output directory based on actual plotted time range
	# subfolder name: <t_start>_<t_end> (seconds), dots replaced for safety
	start_label = f"{t[0]:.3f}"
	end_label = f"{t[-1]:.3f}"
	range_folder = f"{start_label}_{end_label}".replace(".", "p")
	image_dir = os.path.join(image_root, range_folder)

	print(
		f"Plotting time range from {t[0]:.3f}s to {t[-1]:.3f}s "
		f"(requested: start={t_start}, end={t_end})",
	)

	# column layout inferred from control.cpp / rl_policy.cpp
	# 0: time
	# 1-4: info["command"] (4)
	# 5-7: gyro (3)
	# 8: sinyaw
	# 9: sinpitch
	# 10: sinroll
	# 11-14: jnts (4)
	# 15-18: velocity (4)
	# 19-20: wheel_linvel (2)
	# 21-24: previous_wheels_policyout (4)
	# 25-26: wheel_ctrl_vel_error (2)
	# 27-28: thigh_ctrl (2)
	# 29-30: wheel_ctrl (2)

	dt = data[:, 0:1]
	dt = np.diff(dt, axis=0, prepend=dt[0:1, :])
	dt = dt[1:]
	plot_group(
		t[1:],
		dt,
		["dt"],
		"dt",
		image_dir,
		"dt.png",
	)

	# info["command"]
	cmd = data[:, 1:5]
	plot_group(
		t,
		cmd,
		["cmd0", "cmd1", "cmd2", "cmd3"],
		"info[command]",
		image_dir,
		"info_command.png",
	)

	# gyro
	noisy_gyro = data[:, 5:8]
	plot_group(
		t,
		noisy_gyro,
		["gyro_x", "gyro_y", "gyro_z"],
		"gyro",
		image_dir,
		"gyro.png",
	)

	# sin of yaw/pitch/roll
	noisy_sin = data[:, 8:11]
	plot_group(
		t,
		noisy_sin,
		["sinyaw", "sinpitch", "sinroll"],
		"sin(yaw,pitch,roll)",
		image_dir,
		"sin_ypr.png",
	)

	# jnts: left thigh, left wheel, right thigh, right wheel
	noisy_jnts = data[:, 11:15]
	plot_group(
		t,
		noisy_jnts,
		["thigh_left", "wheel_left", "thigh_right", "wheel_right"],
		"jnts",
		image_dir,
		"jnts.png",
	)

	# velocity: left thigh vel, left wheel vel, right thigh vel, right wheel vel
	noisy_vel = data[:, 15:19]
	plot_group(
		t,
		noisy_vel,
		[
			"thigh_left_vel",
			"wheel_left_vel",
			"thigh_right_vel",
			"wheel_right_vel",
		],
		"velocity",
		image_dir,
		"velocity.png",
	)

	# wheel_linvel
	noisy_wheel_linvel = data[:, 19:21]
	plot_group(
		t,
		noisy_wheel_linvel,
		["wheel_left_linvel", "wheel_right_linvel"],
		"wheel_linvel",
		image_dir,
		"wheel_linvel.png",
	)

	# previous_wheels_policyout (history of previous wheel actions)
	prev_policy = data[:, 21:25]
	plot_group(
		t,
		prev_policy,
		[
			"prev_wheel_cmd0",
			"prev_wheel_cmd1",
			"prev_wheel_cmd2",
			"prev_wheel_cmd3",
		],
		"previous_wheels_policyout",
		image_dir,
		"previous_wheels_policyout.png",
	)

	# wheel_ctrl_vel_error
	wheel_ctrl_err = data[:, 25:27]
	plot_group(
		t,
		wheel_ctrl_err,
		["wheel_left_ctrl_vel_err", "wheel_right_ctrl_vel_err"],
		"wheel_ctrl_vel_error",
		image_dir,
		"wheel_ctrl_vel_error.png",
	)

	# thigh_ctrl (two thighs)
	thigh_ctrl = data[:, 27:29]
	plot_group(
		t,
		thigh_ctrl,
		["thigh_left_ctrl", "thigh_right_ctrl"],
		"thigh_ctrl",
		image_dir,
		"thigh_ctrl.png",
	)

	# wheel_ctrl (two wheels)
	wheel_ctrl = data[:, 29:31]
	plot_group(
		t,
		wheel_ctrl,
		["wheel_left_ctrl", "wheel_right_ctrl"],
		"wheel_ctrl",
		image_dir,
		"wheel_ctrl.png",
	)

	# plt.show()


if __name__ == "__main__":
	main()
