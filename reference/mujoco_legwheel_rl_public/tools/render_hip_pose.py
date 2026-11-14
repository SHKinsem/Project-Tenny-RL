import os
os.environ['MUJOCO_GL'] = 'osmesa'

import mujoco
import matplotlib.pyplot as plt


# 1. 加载模型
xmlfilename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "legwheel_robot2_stl_nolinkage6_terrain.xml")
model = mujoco.MjModel.from_xml_path(xmlfilename)
data = mujoco.MjData(model)

# 2. 物理状态同步 (必须在 update_scene 之前)
mujoco.mj_forward(model, data)

# 3. 创建渲染器
renderer = mujoco.Renderer(model, width=1920, height=1080)

# 4. 配置相机
camera = mujoco.MjvCamera()
camera.distance = 0.5        # 距离物体 0.5 米
camera.lookat = [0, 0, 0.1]  # 盯着机器人
camera.azimuth = 90          # 水平角度
camera.elevation = -20       # 仰角

# 5. 配置渲染选项（可选：即使没灯也能看到轮廓）
scene_option = mujoco.MjvOption()
# 如果你想强制看到几何体边缘，可以取消下面这行的注释
# scene_option.flags[mujoco.mjtVisFlag.mjVIS_WIREFRAME] = True


def get_hip_actuator_indices(model):
	"""返回左右 hip 位置型执行器在 ctrl 里的索引。"""
	hip_actuator_names = ["thigh_left_pos", "thigh_right_pos"]
	indices = []
	for name in hip_actuator_names:
		a_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
		indices.append(a_id)
	return indices


hip_actuator_indices = get_hip_actuator_indices(model)

# 记录初始姿态，用于每次仿真前复位
qpos_initial = data.qpos.copy()


def render_with_hip_angle(angle_rad, n_steps=1000):
	"""给定 hip 目标角度（弧度），通过 actuator 施加控制并渲染图像。"""
	# 恢复到初始状态
	data.qpos[:] = qpos_initial
	data.qvel[:] = 0
	data.act[:] = 0
	data.ctrl[:] = 0

	# 通过位置执行器施加 hip 目标值
	for idx in hip_actuator_indices:
		data.ctrl[idx] = angle_rad

	# 让 PD 执行器驱动关节收敛一段时间
	for _ in range(n_steps):
		mujoco.mj_step(model, data)

	renderer.update_scene(data, camera=camera, scene_option=scene_option)
	pixels = renderer.render()
	print(f"hip(ctrl)={angle_rad:.2f} 时图像像素最大值: {pixels.max()}")
	return pixels


if __name__ == "__main__":
	# 要渲染的两个 hip 姿态
	hip_values = [0.0, -0.5]

	fig, axes = plt.subplots(1, len(hip_values), figsize=(8, 4))
	if len(hip_values) == 1:
		axes = [axes]

	for ax, angle in zip(axes, hip_values):
		pixels = render_with_hip_angle(angle)
		ax.imshow(pixels)
		ax.axis('off')
		ax.set_title(f"hip={angle}")

	plt.tight_layout()
	plt.savefig('render_hip_poses.png', bbox_inches='tight', pad_inches=0)