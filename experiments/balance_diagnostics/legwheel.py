"""Convert native legwheel traces to explicit post-integration SI measurements."""
import math


def quaternion_euler(quaternion):
    """MuJoCo wxyz quaternion to ZYX roll/pitch/yaw, in radians."""
    if len(quaternion) != 4 or not all(math.isfinite(float(v)) for v in quaternion):
        raise ValueError('Expected a finite wxyz quaternion')
    norm = math.sqrt(sum(float(v) ** 2 for v in quaternion))
    if norm < 1e-12:
        raise ValueError('Quaternion norm must be positive')
    w, x, y, z = (float(v) / norm for v in quaternion)
    return (math.atan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y)),
            math.asin(max(-1., min(1., 2 * (w*y - z*x)))),
            math.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z)))


def model_layout(model, task):
    """Resolve task names; other robot topologies need their own trace adapter."""
    import mujoco
    if task.get('type') != 'legwheel':
        raise ValueError('No balance trace adapter for task ' + str(task.get('type')))
    cfg = task['config']
    body = model.body(cfg['root_body']).id
    joint = int(model.body_jntadr[body])
    if model.body_jntnum[body] != 1 or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError('Legwheel balance measurements require one free root joint')
    return dict(root_qpos=int(model.jnt_qposadr[joint]), root_qvel=int(model.jnt_dofadr[joint]),
                root_body=body, wheel_qpos=[int(model.jnt_qposadr[model.joint(name).id])
                                          for name in cfg['wheel_joints']],
                wheel_radius_m=float(cfg['wheel_radius']), torque_limits_nm=cfg['torque_scales'])


def normalized_trace(raw, initial, result, world, dt, layout):
    """Keep only the first episode; native traces may continue through auto-resets."""
    import numpy as np
    length = int(result['lengths'][world])
    if length < 1 or length > len(raw['qpos']):
        raise ValueError('Invalid first-episode trace length')
    terminated = bool(result['terminated'][world])
    truncated = bool(result['truncated'][world])
    if result['cancelled'] or terminated == truncated:
        raise ValueError('Require a completed terminated or truncated first episode')
    done = np.asarray(raw['done'][:length], dtype=bool)
    if not done[-1] or done[:-1].any():
        raise ValueError('Trace includes a reset or omits its first terminal step')
    qpos = np.vstack([initial['qpos'], raw['qpos'][:length]])
    qvel = np.vstack([initial['qvel'], raw['qvel'][:length]])
    q, v = layout['root_qpos'], layout['root_qvel']
    if qpos.shape[1] < q + 7 or qvel.shape[1] < v + 6:
        raise ValueError('Root layout exceeds recorded state dimensions')
    angles = np.asarray([quaternion_euler(row[q+3:q+7]) for row in qpos])
    positions, velocities = qpos[:, q:q+3], qvel[:, v:v+3]
    # MuJoCo free-joint angular qvel is in the local body frame. ZYX theta-dot.
    pitch_rate = (np.cos(angles[:, 0]) * qvel[:, v+4] -
                  np.sin(angles[:, 0]) * qvel[:, v+5])
    displacement = positions - positions[0]
    forward = displacement[:, 0] * math.cos(angles[0, 2]) + displacement[:, 1] * math.sin(angles[0, 2])
    wheel = np.mean(qpos[:, layout['wheel_qpos']], axis=1) * layout['wheel_radius_m']
    reference_sin = float(initial['extra']['sin_balance_zero'][0])
    if not -1 <= reference_sin <= 1:
        raise ValueError('Invalid saved pitch reference')
    return dict(schema_version=1, t_s=(np.arange(length+1) * dt).tolist(),
        base_position_m=positions.tolist(), base_velocity_m_s=velocities.tolist(),
        pitch_rad=angles[:, 1].tolist(), roll_rad=angles[:, 0].tolist(),
        pitch_rate_rad_s=pitch_rate.tolist(), forward_displacement_m=forward.tolist(),
        wheel_angle_distance_m=(wheel-wheel[0]).tolist(),
        actuator_command_nm=np.vstack([np.zeros(raw['ctrl'].shape[1]), raw['ctrl'][:length]]).tolist(),
        reward=[0., *map(float, raw['reward'][:length])], done=[False, *done.tolist()],
        termination=dict(terminated=terminated, truncated=truncated,
                         reason=result['reasons'][world], time_s=length*dt),
        metadata=dict(reference_pitch_rad=math.asin(reference_sin),
            torque_limits_nm=list(layout['torque_limits_nm']),
            position_frame='World frame; free-root position, measured from the supplied reset.',
            attitude_phase='ZYX angles from post-integration qpos, not native raw xmat.',
            pitch_rate_source='ZYX pitch derivative from post-step local free-joint angular qvel.',
            wheel_distance_source='Mean relative wheel-joint angle times radius; a proxy, not world displacement.',
            command_source='Last physics-substep motor ctrl; not mean effort or evidence of clipping.',
            first_episode_only=True))
