# 当前参考机器人的接触力动作设计

2026-11-14。设计分析完成；未实现控制器、未启动训练。依据 reference/mujoco_legwheel_rl_public 与 experiments/torque200 的 XML、env.py、config.py、rewards.py。

## 模型事实

- 左右各一个主动 thigh、一个受 equality 约束的 calf、一个驱动 wheel，共四个执行器。
- calf=g(thigh)=-0.0004-2.1405q+0.2637q²-1.2386q³-0.3494q⁴；此模型关节参考角为零。
- g'=-2.1405+0.5274q-3.7158q²-1.3976q³。
- 总质量 0.31354 kg，轮半径 0.0195 m，平均双支撑为 1.5379137 N/轮。
- 全扭矩配置的最终限幅为髋 ±1 Nm、轮 ±0.0295 Nm；XML 的轮 ±0.04 Nm 被 env 覆盖。髋 ±1 Nm 是此前实验暂定值，不是已标定硬件能力。
- 原版髋为位置执行器（最大力矩 ±24 Nm），全扭矩版为 motor。两者都保留 hip armature=20、damping=1、stiffness=1，不能视为轻质理想连杆。

## 动作定义

四维 a=[a_tL,a_nL,a_tR,a_nR]，各分量 [-1,1]。定义为地面对机器人的期望反力，不是机器人对地面的力。
采用地面法向 n 及滚动方向 t，不随机身 pitch 旋转。第一阶段限定平地双轮接触，转弯时 t 随轮朝向变化。
初始尺度候选：Fn=1.5379137*(1+a_n) N，Ft=1.0*a_t N。法向范围约 [0,3.076] N/轮，仅为首轮实验尺度，并非能力认证。
先投影到摩擦锥、执行器可实现范围，再映射。Fx/Fz 两轴表述仅适用于轮竖直的平地近似。侧向力不是独立动作，但动力学中不能忽略侧向约束反力。
Fn 共模调支撑，差模调横滚；Ft 共模调推进/平衡，差模调偏航。不能再独立指定任意 pitch 力矩；接触力的作用线决定整机外力矩，内部运动另影响机身姿态。

## 几何映射

必须对轮地接触处的瞬时轮体材料点计算 Jacobian，不能使用轮心 Jacobian 代替，也不能对“每时刻最低点位置”直接求导来代表材料点速度。
J_eff,h=J_contact,h+g'(q_h)*J_contact,k；J_eff,w=J_contact,w。
若 f 是地面对机器人的力，接触项的执行器前馈为 -J_eff^T f；完整动力学还有惯性、重力、被动力项。
平地机身水平、q_h=-0.25 时左腿的接触力项为：

```
[tau_hip  ]   [-0.0475827  -0.0983237] [Ft]
[tau_wheel] = [ 0.0195      0        ] [Fn]
```

单位 Nm/N。右腿基本相同。数值不是完整扭矩指令，未含动力学补偿。
例如 Ft=0、Fn=1.5379137 时，髋接触项约 -0.1512 Nm；不能据此宣称该扭矩可稳定站立。
轮准静态牵引上界约 0.0295/0.0195=1.51282 N，实际还受摩擦、惯性和损耗影响。

## 下层实现层级

轻量第一版：在物理子步重算 Jacobian，施加接触力前馈及经过验证的偏置补偿，限幅；策略周期只更新期望力。此版本应叫虚拟接触力动作，不能声称精确跟踪真实接触力。
真实力跟踪版：优化 ddq、tau、contact force f、闭链约束反力 lambda，最小化接触力跟踪误差及扭矩/加速度正则。

```
M ddq + h = S^T tau + Jc^T f + Je^T lambda + tau_passive
Je ddq + dot(Je) dq = closure stabilization
rolling/contact acceleration constraints
Fn >= 0, norm(Ftangential) <= mu Fn, actuator limits
```

接触力是软目标，动力学与限幅必须可行；不施加任意全姿态硬跟踪。转弯时将侧向反力作为求解量而非动作。滚动接触的几何点随轮转动变化，不能照搬固定足端约束的导数。
MuJoCo equality/contact 是软约束，刚性约束下层存在模型失配，应测真实反力并检查闭链残差。离地后期望接触力为零；空中姿态与落地准备需要独立的非接触控制逻辑，四维接触力接口不覆盖任意空中动作。

## 接入与验证

- 在独立实验中替换 torque200/env.py 的 action*torque_scales；不改 reference，不混用已有权重。
- actor 四维维数可保持，但动作语义和 previous-action 观测需明确改为力；扭矩正则仍用实际输出扭矩，动作变化正则用归一化力。保留姿态、关节、轮速等状态观测。
- 首轮与 torque200 保持相同时序，隔离动作参数化的效果；与原版100Hz对比需注明控制器和时序同时不同。
- 已通过：check-contact-force-map.py 在已有 WSL Python/MuJoCo 上运行，左右腿 q=-0.5,-0.25,0 共六点，有限差分最大误差 <9e-12 m/rad，二维矩阵均非奇异。检查只移除渲染资产，保留惯性和碰撞几何；不代表全行程/倾斜接触验证。
- 未验证：支撑力阶跃、推进力阶跃、摩擦投影、扭矩饱和、真实接触力跟踪、受扰平衡与训练收敛。特别先审查 armature=20 是否为刻意等效参数；不能未经验证直接删除或把补偿当作免费力矩。

参考定义：https://mujoco.readthedocs.io/en/latest/XMLreference.html （equality/joint 与 armature）；https://mujoco.readthedocs.io/en/latest/computation.html （动力学与约束）。
