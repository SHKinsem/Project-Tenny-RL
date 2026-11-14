"""域随机化:质量/质心、关节参数、执行器增益、地面摩擦、初始位姿。

原 domain_randomize.py 原样搬移,由 brax randomization_fn 按 env 批量调用。
"""

"""域随机化:质量/质心、关节参数、执行器增益、地面摩擦、初始位姿。

原 domain_randomize.py 原样搬移,由 brax randomization_fn 按 env 批量调用。
"""

import jax
import jax.numpy as jp
import numpy as np
import mujoco
from mujoco import mjx
from scipy.spatial.transform import Rotation as Rotater


def euler_to_quaternion(yaw, pitch, roll):
    """
    将欧拉角转换为四元数
    
    参数:
        order: 欧拉角的旋转顺序，默认为'ZYX'（yaw, pitch, roll）
    
    返回:
        quaternion: 形状为(4,)的数组，包含[w, x, y, z]四元数
    """
    
    cr = jp.cos(roll * 0.5)
    sr = jp.sin(roll * 0.5)
    cp = jp.cos(pitch * 0.5)
    sp = jp.sin(pitch * 0.5)
    cy = jp.cos(yaw * 0.5)
    sy = jp.sin(yaw * 0.5)
    
    # ZYX顺序 (yaw -> pitch -> roll)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return jp.array([w, x, y, z])
    

def default_randomization_config() -> dict:
    """域随机化的默认开关与范围(与原 domain_randomize 函数体内硬编码值一致)。

    C 层参数(见 docs/hyperparam_tuning_plan.md 第 1 节)不参与 reward 搜索,
    参数化仅便于做鲁棒性消融;tune.py 可通过 cfg 注入修改。
    """
    return dict(
        # 开关
        use_mass_ipos=True,
        enable_thigh_jnt_random=True,
        enable_wheel_jnt_random=True,
        enable_angXYZ_random=True,
        enable_geom_friction_random=True,
        enable_actuator_random=True,
        # 机器人整机约 0.3kg、轮半径 19.5mm,质心/质量扰动范围需保守
        torso_ipos_range=[-0.003, 0.003],
        body_mass_ratio_range=[0.9, 1.1],
        torso_mass_range=[-0.01, 0.01],
        thigh_frictionloss_ratio_range=[0.95, 1.05],
        wheel_frictionloss_ratio_range=[0.5, 2.0],
        thigh_armature_ratio_range=[0.95, 1.05],
        wheel_armature_ratio_range=[0.5, 2.0],
        thigh_stiffness_ratio_range=[0.95, 1.05],
        wheel_stiffness_ratio_range=[0.5, 2.0],
        thigh_damping_ratio_range=[0.9, 1.1],
        wheel_damping_ratio_range=[0.5, 2.0],
        geom_friction_range=[0.8, 1.2],
        # 执行器随机化(大腿 kp/kv、轮电机强度)
        thigh_kp_ratio_range=[0.9, 1.1],
        thigh_kv_ratio_range=[0.9, 1.1],
        wheel_motor_strength_ratio_range=[0.8, 1.2],
        # 初始位姿
        position_range=[-8.0, 8.0],
        angZ_range=[-np.pi, np.pi],
        angY_range=[-np.pi / 8, np.pi / 8],
        angX_range=[-np.pi / 18, np.pi / 18],
    )


def domain_randomize(model: mjx.Model, rng: jax.Array, cfg: dict | None = None):
  if cfg is None:
    cfg = default_randomization_config()
  # 域随机化开关：质量/质心、关节参数、执行器增益默认全部开启（sim2real）
  use_mass_ipos = cfg["use_mass_ipos"]
  enable_thigh_jnt_random = cfg["enable_thigh_jnt_random"]
  enable_wheel_jnt_random = cfg["enable_wheel_jnt_random"]
  enable_angXYZ_random = cfg["enable_angXYZ_random"]
  enable_geom_friction_random = cfg["enable_geom_friction_random"]
  enable_actuator_random = cfg["enable_actuator_random"]

  torso_ipos_range = cfg["torso_ipos_range"]
  body_mass_ratio_range = cfg["body_mass_ratio_range"]
  torso_mass_range = cfg["torso_mass_range"]

  FLOOR_GEOM_ID = mjx.name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
  TORSO_BODY_ID = mjx.name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'root')
  thigh_left_jnt_id = mjx.name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'thigh_left_joint')
  thigh_right_jnt_id = mjx.name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'thigh_right_joint')
  wheel_left_jnt_id = mjx.name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'wheel_left_joint')
  wheel_right_jnt_id = mjx.name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'wheel_right_joint')
  thigh_jnt_ids = jp.array([thigh_left_jnt_id, thigh_right_jnt_id], dtype=jp.int32)
  wheel_jnt_ids = jp.array([wheel_left_jnt_id, wheel_right_jnt_id], dtype=jp.int32)
  thigh_left_act_id = mjx.name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'thigh_left_pos')
  thigh_right_act_id = mjx.name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'thigh_right_pos')
  wheel_left_act_id = mjx.name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'wheel_left_motor')
  wheel_right_act_id = mjx.name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'wheel_right_motor')
  thigh_act_ids = jp.array([thigh_left_act_id, thigh_right_act_id], dtype=jp.int32)
  wheel_act_ids = jp.array([wheel_left_act_id, wheel_right_act_id], dtype=jp.int32)
  thigh_frictionloss_ratio_range = cfg["thigh_frictionloss_ratio_range"]
  wheel_frictionloss_ratio_range = cfg["wheel_frictionloss_ratio_range"]
  thigh_armature_ratio_range = cfg["thigh_armature_ratio_range"]
  wheel_armature_ratio_range = cfg["wheel_armature_ratio_range"]
  thigh_stiffness_ratio_range = cfg["thigh_stiffness_ratio_range"]
  wheel_stiffness_ratio_range = cfg["wheel_stiffness_ratio_range"]
  thigh_damping_ratio_range = cfg["thigh_damping_ratio_range"]
  wheel_damping_ratio_range = cfg["wheel_damping_ratio_range"]
  geom_friction_range = cfg["geom_friction_range"]
  # 执行器随机化：
  # - 大腿 position 执行器：kp/kv 各自 *U(0.9, 1.1)
  #   (MuJoCo 约定：gainprm[0]=kp, biasprm=[0, -kp, -kv])
  # - 轮电机（motor）：输出力矩比例 *U(0.8, 1.2)（gainprm[0]=gear）
  thigh_kp_ratio_range = cfg["thigh_kp_ratio_range"]
  thigh_kv_ratio_range = cfg["thigh_kv_ratio_range"]
  wheel_motor_strength_ratio_range = cfg["wheel_motor_strength_ratio_range"]

  position_range = cfg["position_range"]
  angZ_range = cfg["angZ_range"]
  angY_range = cfg["angY_range"]
  angX_range = cfg["angX_range"]

  @jax.vmap
  def rand_dynamics(rng):
    geom_friction = model.geom_friction
    dof_frictionloss = model.dof_frictionloss
    jnt_stiffness = model.jnt_stiffness
    dof_armature = model.dof_armature
    dof_damping = model.dof_damping
    body_ipos = model.body_ipos
    body_mass = model.body_mass
    body_inertia = model.body_inertia
    actuator_gainprm = model.actuator_gainprm
    actuator_biasprm = model.actuator_biasprm
    qpos0 = model.qpos0

    if enable_geom_friction_random:
        rng, key = jax.random.split(rng)
        geom_friction = model.geom_friction.at[FLOOR_GEOM_ID, 0].set(
            jax.random.uniform(key, minval=geom_friction_range[0], maxval=geom_friction_range[1])
        )

    # Scale static friction: *U(0.9, 1.1).
    
    if enable_thigh_jnt_random:
      rng, key = jax.random.split(rng)
      frictionloss = dof_frictionloss[thigh_jnt_ids+5] * jax.random.uniform(
          key, shape=(2,), minval=thigh_frictionloss_ratio_range[0], maxval=thigh_frictionloss_ratio_range[1]
      )
      dof_frictionloss = dof_frictionloss.at[thigh_jnt_ids+5].set(frictionloss)

    if enable_wheel_jnt_random:
      rng, key = jax.random.split(rng)
      frictionloss = dof_frictionloss[wheel_jnt_ids+5] * jax.random.uniform(
          key, shape=(2,), minval=wheel_frictionloss_ratio_range[0], maxval=wheel_frictionloss_ratio_range[1]
      )
      dof_frictionloss = dof_frictionloss.at[wheel_jnt_ids+5].set(frictionloss)

    # Scale armature: *U(0.0, 1.1).
    if enable_thigh_jnt_random:
      rng, key = jax.random.split(rng)
      armature = dof_armature[thigh_jnt_ids+5] * jax.random.uniform(
          key, shape=(2,), minval=thigh_armature_ratio_range[0], maxval=thigh_armature_ratio_range[1]
      )
      dof_armature = dof_armature.at[thigh_jnt_ids+5].set(armature)

    if enable_wheel_jnt_random:
      rng, key = jax.random.split(rng)
      armature = dof_armature[wheel_jnt_ids+5] * jax.random.uniform(
          key, shape=(2,), minval=wheel_armature_ratio_range[0], maxval=wheel_armature_ratio_range[1]
      )
      dof_armature = dof_armature.at[wheel_jnt_ids+5].set(armature)

    # stiffness
    if enable_thigh_jnt_random:
      rng, key = jax.random.split(rng)
      jnt_stiffness = jnt_stiffness.at[thigh_jnt_ids].set(
          jnt_stiffness[thigh_jnt_ids] * jax.random.uniform(
              key, shape=(2,), minval=thigh_stiffness_ratio_range[0], maxval=thigh_stiffness_ratio_range[1]
          )
      )

    if enable_wheel_jnt_random:
      rng, key = jax.random.split(rng)
      jnt_stiffness = jnt_stiffness.at[wheel_jnt_ids].set(
          jnt_stiffness[wheel_jnt_ids] * jax.random.uniform(
              key, shape=(2,), minval=wheel_stiffness_ratio_range[0], maxval=wheel_stiffness_ratio_range[1]
          )
      )

    # damping
    if enable_thigh_jnt_random:
      rng, key = jax.random.split(rng)
      dof_damping = dof_damping.at[thigh_jnt_ids+5].set(
         dof_damping[thigh_jnt_ids+5] * jax.random.uniform(
            key, shape=(2,), minval=thigh_damping_ratio_range[0], maxval=thigh_damping_ratio_range[1]))

    if enable_wheel_jnt_random:
      rng, key = jax.random.split(rng)
      dof_damping = dof_damping.at[wheel_jnt_ids+5].set(
         dof_damping[wheel_jnt_ids+5] * jax.random.uniform(
            key, shape=(2,), minval=wheel_damping_ratio_range[0], maxval=wheel_damping_ratio_range[1]))

    # actuator gain/strength randomization
    if enable_actuator_random:
      rng, key = jax.random.split(rng)
      kp_ratio = jax.random.uniform(
          key, shape=(2,), minval=thigh_kp_ratio_range[0], maxval=thigh_kp_ratio_range[1])
      rng, key = jax.random.split(rng)
      kv_ratio = jax.random.uniform(
          key, shape=(2,), minval=thigh_kv_ratio_range[0], maxval=thigh_kv_ratio_range[1])
      actuator_gainprm = actuator_gainprm.at[thigh_act_ids, 0].set(
          actuator_gainprm[thigh_act_ids, 0] * kp_ratio)
      actuator_biasprm = actuator_biasprm.at[thigh_act_ids, 1].set(
          actuator_biasprm[thigh_act_ids, 1] * kp_ratio)
      actuator_biasprm = actuator_biasprm.at[thigh_act_ids, 2].set(
          actuator_biasprm[thigh_act_ids, 2] * kv_ratio)

      rng, key = jax.random.split(rng)
      motor_ratio = jax.random.uniform(
          key, shape=(2,), minval=wheel_motor_strength_ratio_range[0], maxval=wheel_motor_strength_ratio_range[1])
      actuator_gainprm = actuator_gainprm.at[wheel_act_ids, 0].set(
          actuator_gainprm[wheel_act_ids, 0] * motor_ratio)

    if use_mass_ipos:
      # Jitter center of mass positiion: +U(-0.05, 0.05).
      rng, key = jax.random.split(rng)
      dpos = jax.random.uniform(key, (3,), minval=torso_ipos_range[0], maxval=torso_ipos_range[1])
      body_ipos = model.body_ipos.at[TORSO_BODY_ID].set(
          model.body_ipos[TORSO_BODY_ID] + dpos
      )

      # Scale all link masses: *U(0.9, 1.1).
      rng, key = jax.random.split(rng)
      dmass = jax.random.uniform(
          key, shape=(model.nbody,), minval=body_mass_ratio_range[0], maxval=body_mass_ratio_range[1]
      )
      body_mass = model.body_mass.at[:].set(model.body_mass * dmass)

      # Add mass to torso: +U(-1.0, 1.0).
      rng, key = jax.random.split(rng)
      dmass = jax.random.uniform(key, minval=torso_mass_range[0], maxval=torso_mass_range[1])
      body_mass = body_mass.at[TORSO_BODY_ID].set(
          body_mass[TORSO_BODY_ID] + dmass
      )

      # Recompute diagonal body inertia based on updated masses.
      # Scale each body's inertia proportionally to its mass change,
      # while avoiding division-by-zero for massless bodies.
      mass_scale = jp.where(model.body_mass > 0.0,
                            body_mass / model.body_mass,
                            1.0)
      body_inertia = model.body_inertia * mass_scale[:, None]


    # spread on the xy plane

    rng, key = jax.random.split(rng)
    qpos0 = qpos0.at[:2].set(
        qpos0[:2]
        + jax.random.uniform(key, shape=(2,), minval=position_range[0], maxval=position_range[1])
    )

    # random the yaw(orientation)
    if enable_angXYZ_random:
        rng, key = jax.random.split(rng)
        ang_z = jax.random.uniform(key, shape=(1,), minval=angZ_range[0], maxval=angZ_range[1])
        rng, key = jax.random.split(rng)
        ang_y = jax.random.uniform(key, shape=(1,), minval=angY_range[0], maxval=angY_range[1]) # -30~+30
        rng, key = jax.random.split(rng)
        ang_x = jax.random.uniform(key, shape=(1,), minval=angX_range[0], maxval=angX_range[1]) # -10~+10
        # quat = yaw_to_quaternion(ang_z[0])
        quat = euler_to_quaternion(ang_z[0], ang_y[0], ang_x[0])
        qpos0 = qpos0.at[3:7].set(quat)

    # return (geom_friction, body_ipos, body_mass, qpos0, dof_frictionloss, dof_armature, jnt_stiffness, dof_damping)
    return (geom_friction, body_ipos, body_mass, body_inertia,
            qpos0, dof_frictionloss, dof_armature, jnt_stiffness, dof_damping,
            actuator_gainprm, actuator_biasprm)

  (friction,
     body_ipos,
     body_mass,
     body_inertia,
     qpos0,
     dof_frictionloss,
     dof_armature,
     jnt_stiffness,
     dof_damping,
     actuator_gainprm,
     actuator_biasprm) = rand_dynamics(rng)

  in_axes = jax.tree_util.tree_map(lambda x: None, model)
  in_axes = in_axes.tree_replace({
      "geom_friction": 0,
      "body_ipos": 0,
      "body_mass": 0,
      "body_inertia": 0,
      "qpos0": 0,
      "dof_frictionloss": 0,
      "dof_armature": 0,
      "jnt_stiffness": 0,
      "dof_damping":0,
      "actuator_gainprm": 0,
      "actuator_biasprm": 0,
  })

  model = model.tree_replace({
      "geom_friction": friction,
      "body_ipos": body_ipos,
      "body_mass": body_mass,
      "body_inertia": body_inertia,
      "qpos0": qpos0,
      "dof_frictionloss": dof_frictionloss,
      "dof_armature": dof_armature,
      "jnt_stiffness": jnt_stiffness,
      "dof_damping": dof_damping,
      "actuator_gainprm": actuator_gainprm,
      "actuator_biasprm": actuator_biasprm,
  })

  return model, in_axes