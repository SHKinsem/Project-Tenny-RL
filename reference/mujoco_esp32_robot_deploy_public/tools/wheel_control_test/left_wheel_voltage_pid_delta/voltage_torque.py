
import numpy as np
import matplotlib.pyplot as plt


def plot_piecewise_linear(voltage, torque):
	"""原有的分段线性插值绘图：全量数据，双子图。"""
	# 原始数据：电压(V) 与 转矩(N·m)
	# 为了进行分段线性拟合，先按自变量升序排序
	idx_v = np.argsort(voltage)
	voltage_sorted = voltage[idx_v]
	torque_sorted = torque[idx_v]

	idx_t = np.argsort(torque)
	torque_sorted_2 = torque[idx_t]
	voltage_sorted_2 = voltage[idx_t]

	# 使用分段线性插值生成平滑曲线
	v_fit = np.linspace(voltage_sorted.min(), voltage_sorted.max(), 200)
	t_fit = np.interp(v_fit, voltage_sorted, torque_sorted)

	t_fit2 = np.linspace(torque_sorted_2.min(), torque_sorted_2.max(), 200)
	v_fit2 = np.interp(t_fit2, torque_sorted_2, voltage_sorted_2)

	# 画 1×2 子图
	fig, axes = plt.subplots(1, 2, figsize=(10, 4))

	# 子图1：Voltage -> Torque
	ax1 = axes[0]
	ax1.scatter(voltage, torque, color="blue", label="measure data")
	ax1.plot(v_fit, t_fit, color="red", label="piecewise linear fit")
	ax1.set_xlabel("Voltage / V")
	ax1.set_ylabel("Torque / N·m")
	ax1.set_title("Torque vs Voltage")
	ax1.grid(True, linestyle="--", alpha=0.4)
	ax1.legend()
	ax1.text(
		0.05,
		0.95,
		"T(V): piecewise linear fit (through all measured points)",
		transform=ax1.transAxes,
		fontsize=8,
		verticalalignment="top",
		bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
	)

	# 子图2：Torque -> Voltage
	ax2 = axes[1]
	ax2.scatter(torque, voltage, color="green", label="measure data")
	ax2.plot(t_fit2, v_fit2, color="orange", label="piecewise linear fit")
	ax2.set_xlabel("Torque / N·m")
	ax2.set_ylabel("Voltage / V")
	ax2.set_title("Voltage vs Torque")
	ax2.grid(True, linestyle="--", alpha=0.4)
	ax2.legend()
	ax2.text(
		0.05,
		0.95,
		"V(T): segment linear fit (through all measured points)",
		transform=ax2.transAxes,
		fontsize=8,
		verticalalignment="top",
		bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
	)

	# 保存图片
	output_path = "voltage_torque_fit.png"
	fig.tight_layout()
	fig.savefig(output_path, dpi=200)


def plot_linear_fit_below_4v(voltage, torque):
	"""只对电压<=4V的数据进行线性拟合，并画左右子图（含公式和误差）。"""
	# 只选取电压<=4V的数据
	mask = voltage <= 4.0
	voltage_sub = voltage[mask]
	torque_sub = torque[mask]
	if voltage_sub.size < 2:
		return

	# 拟合 Torque = a * Voltage + b
	a, b = np.polyfit(voltage_sub, torque_sub, 1)
	v_line = np.linspace(voltage_sub.min(), voltage_sub.max(), 200)
	t_line = a * v_line + b
	# 计算 Torque 拟合误差（RMSE 与 R^2）
	t_pred = a * voltage_sub + b
	res_t = torque_sub - t_pred
	rmse_t = np.sqrt(np.mean(res_t ** 2))
	ss_res_t = np.sum(res_t ** 2)
	ss_tot_t = np.sum((torque_sub - torque_sub.mean()) ** 2)
	r2_t = 1.0 - ss_res_t / ss_tot_t if ss_tot_t > 0 else np.nan

	# 拟合 Voltage = c * Torque + d（反向拟合）
	c, d = np.polyfit(torque_sub, voltage_sub, 1)
	t_line2 = np.linspace(torque_sub.min(), torque_sub.max(), 200)
	v_line2 = c * t_line2 + d
	# 计算 Voltage 拟合误差（RMSE 与 R^2）
	v_pred = c * torque_sub + d
	res_v = voltage_sub - v_pred
	rmse_v = np.sqrt(np.mean(res_v ** 2))
	ss_res_v = np.sum(res_v ** 2)
	ss_tot_v = np.sum((voltage_sub - voltage_sub.mean()) ** 2)
	r2_v = 1.0 - ss_res_v / ss_tot_v if ss_tot_v > 0 else np.nan

	fig, axes = plt.subplots(1, 2, figsize=(10, 4))

	# 左子图：Torque vs Voltage（包含<=4V直线拟合）
	ax1 = axes[0]
	ax1.scatter(voltage, torque, color="blue", label="measure data")
	ax1.plot(v_line, t_line, color="red", label="linear fit (V<=4V)")
	ax1.set_xlabel("Voltage / V")
	ax1.set_ylabel("Torque / N·m")
	ax1.set_title("Torque vs Voltage (V<=4V)")
	ax1.grid(True, linestyle="--", alpha=0.4)
	ax1.legend()
	# 在图中显示拟合公式及误差: T = a V + b, RMSE, R^2
	eq1 = (
		f"T = {a:.5f} * V + {b:.5f}\n"
		f"RMSE = {rmse_t:.5e}, R^2 = {r2_t:.4f}"
	)
	ax1.text(
		0.05,
		0.95,
		eq1,
		transform=ax1.transAxes,
		fontsize=8,
		verticalalignment="top",
		bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
	)

	# 右子图：Voltage vs Torque（包含<=4V直线拟合）
	ax2 = axes[1]
	ax2.scatter(torque, voltage, color="green", label="measure data")
	ax2.plot(t_line2, v_line2, color="orange", label="linear fit (V<=4V)")
	ax2.set_xlabel("Torque / N·m")
	ax2.set_ylabel("Voltage / V")
	ax2.set_title("Voltage vs Torque (V<=4V)")
	ax2.grid(True, linestyle="--", alpha=0.4)
	ax2.legend()
	# 在图中显示拟合公式及误差: V = c T + d, RMSE, R^2
	eq2 = (
		f"V = {c:.5f} * T + {d:.5f}\n"
		f"RMSE = {rmse_v:.5e}, R^2 = {r2_v:.4f}"
	)
	ax2.text(
		0.05,
		0.95,
		eq2,
		transform=ax2.transAxes,
		fontsize=8,
		verticalalignment="top",
		bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
	)

	output_path = "voltage_torque_linear_lt4.png"
	fig.tight_layout()
	fig.savefig(output_path, dpi=200)
	# 需要时可以显示
	# plt.show()


def main():
	# 原始数据：电压(V) 与 转矩(N·m)
	voltage = np.array([
		7.0,
		6.0,
		5.0,
		4.0,
		3.5,
		3.0,
		2.5,
		2.0,
		1.5,
		1.0,
		0.5,
		0.3,
		0.2,
		0.1,
		0.0,
	])

	torque = np.array([
		0.0295,
		0.0281,
		0.0268,
		0.0246,
		0.0221,
		0.0192,
		0.0155,
		0.0124,
		0.00970,
		0.00647,
		0.00274,
		0.00157,
		0.00108,
		0.000490,
		0.0,
	])

	# 原有：全量数据分段线性插值
	plot_piecewise_linear(voltage, torque)
	# 新增：仅对 V<4V 的线性拟合，并在图中显示公式
	plot_linear_fit_below_4v(voltage, torque)


if __name__ == "__main__":
	main()

