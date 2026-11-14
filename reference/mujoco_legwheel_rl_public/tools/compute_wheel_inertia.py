import os

import numpy as np
import mujoco


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(THIS_DIR, "..", "assets", "legwheel_robot2_stl_nolinkage6_terrain.xml")


def compute_wheel_inertia():
    """Compute total equivalent inertia around each wheel joint at current pose.

    Current implementation uses the default qpos (all zeros).
    The result is taken from the joint's diagonal entry of the mass matrix M(q).
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model XML not found at {MODEL_PATH}")

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    # Use default qpos (all zeros) and propagate to get consistent kinematics.
    mujoco.mj_forward(model, data)

    # Get joint ids for the wheel joints.
    j_left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "wheel_left_joint")
    j_right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "wheel_right_joint")

    # Hip joints (for varying thigh_left_pos/thigh_right_pos).
    j_hip_left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "thigh_left_joint")
    j_hip_right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "thigh_right_joint")

    # Map joint ids to DOF indices.
    dof_left = model.jnt_dofadr[j_left]
    dof_right = model.jnt_dofadr[j_right]

    # Hip position actuators (use ctrl to set hip angle, do not write qpos directly).
    a_hip_left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "thigh_left_pos")
    a_hip_right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "thigh_right_pos")

    # Get site ids for the wheel contact points (used for inverted pendulum axis).
    s_left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "wheel_left_site")
    s_right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "wheel_right_site")

    # Build full mass matrix M(q) from the compact representation qM
    # and show local joint inertia (wheel body + armature only).
    M = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, M, data.qM)

    Jloc_left = M[dof_left, dof_left]
    Jloc_right = M[dof_right, dof_right]

    print(f"Local joint inertia (wheel only, current pose):")
    print(f"  Wheel left  J_local = {Jloc_left:.6e} kg·m^2")
    print(f"  Wheel right J_local = {Jloc_right:.6e} kg·m^2")

    # For sanity check, also show wheel body's own inertia and joint armature.
    b_left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wheel_left")
    b_right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wheel_right")

    # body_inertia is (Ixx, Iyy, Izz) around the body COM principal axes.
    I_body_left = model.body_inertia[b_left]
    I_body_right = model.body_inertia[b_right]

    # dof_armature is the additional inertia on each DOF (e.g. motor rotor).
    arm_left = model.dof_armature[dof_left]
    arm_right = model.dof_armature[dof_right]

    print("\nBreakdown (wheel body about its own axis + armature):")
    print(
        f"  Left wheel:  Izz(body) = {I_body_left[2]:.6e}, "
        f"armature = {arm_left:.6e}, sum = {I_body_left[2] + arm_left:.6e}"
    )
    print(
        f"  Right wheel: Izz(body) = {I_body_right[2]:.6e}, "
        f"armature = {arm_right:.6e}, sum = {I_body_right[2] + arm_right:.6e}"
    )

    # Generic: compute whole-robot inertia about an arbitrary spatial axis.

    def total_inertia_about_axis(axis_world: np.ndarray,
                                 anchor_world: np.ndarray,
                                 verbose: bool = False,
                                 label: str = "") -> float:
        """Sum inertia of all bodies about a given spatial axis.

        Axis is defined by unit direction `axis_world` and a point
        `anchor_world` that lies on the axis. For each body we take:

            J_i = u^T I_world u + m * d^2

        where u is axis_world, I_world is inertia tensor about COM in world
        frame, and d is perpendicular distance from axis to COM.
        """

        axis_world = axis_world.astype(float)
        axis_world /= np.linalg.norm(axis_world)

        per_body = []  # (name, mass, d, J_i)
        J_total = 0.0
        for b in range(model.nbody):
            m_b = model.body_mass[b]
            if m_b == 0.0:
                continue

            # COM position in world.
            com = data.xipos[b]

            # Rotation matrix of body frame to world.
            R = data.xmat[b].reshape(3, 3)

            # Inertia about COM in body frame -> world frame.
            I_body = np.diag(model.body_inertia[b])
            I_world = R @ I_body @ R.T

            # Vector from axis point to COM.
            r = com - anchor_world
            # Component perpendicular to axis.
            r_par = axis_world * np.dot(axis_world, r)
            r_perp = r - r_par
            d2 = np.dot(r_perp, r_perp)
            d = float(np.sqrt(d2))

            # Inertia of this body about the axis.
            J_i = float(axis_world @ (I_world @ axis_world) + m_b * d2)
            J_total += J_i

            if verbose:
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
                per_body.append((name, float(m_b), d, J_i))

        if verbose:
            print(f"\nPer-body contributions {label}:")
            print("  body_name           mass(kg)    d_to_axis(m)    J_i(kg·m^2)")
            for name, m_b, d, J_i in sorted(per_body, key=lambda x: -abs(x[3])):
                print(f"  {name:16s}  {m_b:8.4f}   {d:10.4e}   {J_i:10.4e}")

        return J_total

    # Helper: set symmetric hip target via position actuators and let dynamics converge.

    qpos_initial = data.qpos.copy()

    def set_hip_via_ctrl(angle_rad: float, n_steps: int = 1000) -> None:
        """Set hip angle (left/right symmetric) using position actuators via ctrl.

        This mimics the control method used in render_hip_pose.py: we do not
        write hip qpos directly, but instead set actuator targets and run a
        short simulation so that PD controllers bring the joints to the
        desired angle.
        """

        # Reset to initial pose and clear velocities/controls.
        data.qpos[:] = qpos_initial
        data.qvel[:] = 0.0
        data.act[:] = 0.0
        data.ctrl[:] = 0.0

        # Apply symmetric hip targets through actuators.
        data.ctrl[a_hip_left] = angle_rad
        data.ctrl[a_hip_right] = angle_rad

        # Let the system evolve so joints reach the target.
        for _ in range(n_steps):
            mujoco.mj_step(model, data)

    # -------- Whole-robot inertia about each wheel's rotation axis --------

    print("\nWhole-robot inertia about each wheel rotation axis:")
    print("(axis = wheel hinge axis in world frame; anchor = wheel joint anchor)")

    # Scan over hip angles (leg length) with thigh_left = thigh_right,
    # same convention as in view_hip_pose.py (e.g. -0.5: leg extended, 0.0: retracted).
    print("\nScan over symmetric hip angle thigh_left = thigh_right (radians):")
    print("  angle(rad)    J_left_total(kg·m^2)   J_right_total(kg·m^2)")

    angles = np.linspace(-0.5, 0.0, 6)

    for ang in angles:
        # Set hip angle symmetrically using actuators (do not touch qpos directly).
        set_hip_via_ctrl(float(ang))

        # Wheel joint rotation axes and anchors in world coordinates.
        axis_left = data.xaxis[j_left].copy()
        anchor_left = data.xanchor[j_left].copy()
        axis_right = data.xaxis[j_right].copy()
        anchor_right = data.xanchor[j_right].copy()

        J_left_total = total_inertia_about_axis(axis_left, anchor_left)
        J_right_total = total_inertia_about_axis(axis_right, anchor_right)
        print(f"  {ang:8.4f}   {J_left_total:16.6e}   {J_right_total:16.6e}")

    # Additionally, provide detailed per-body breakdown at two representative
    # leg lengths: hip = -0.5 rad (extended) and hip = 0.0 rad (retracted).

    for ang, desc in [(-0.5, "hip = -0.5"), (0.0, "hip = 0.0")]:
        # Use actuator control to realize the desired hip angle.
        set_hip_via_ctrl(float(ang))

        axis_left = data.xaxis[j_left].copy()
        anchor_left = data.xanchor[j_left].copy()

        _ = total_inertia_about_axis(axis_left, anchor_left, verbose=True, label=f"(left wheel axis, {desc})")

        # Also print simple geometric distances root<->wheel for sanity check.
        b_root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "root")
        root_pos = data.xipos[b_root].copy()

        # Use left wheel site as representative wheel center.
        pos_left = data.site_xpos[s_left].copy()
        wheel_pos = pos_left

        diff = root_pos - wheel_pos
        dist_euclid = float(np.linalg.norm(diff))
        # Vertical height difference and horizontal distance in world XZ-plane.
        dz = float(root_pos[2] - wheel_pos[2])
        dxz = float(np.linalg.norm(diff[[0, 2]]))

        print("\nGeometric distances (" + desc + "):")
        print(f"  root_pos   = {root_pos}")
        print(f"  wheel_pos  = {wheel_pos}")
        print(f"  |root-wheel| (3D)   = {dist_euclid*1000:.2f} mm")
        print(f"  vertical dz         = {dz*1000:.2f} mm")
        print(f"  distance in XZ-plane= {dxz*1000:.2f} mm")


if __name__ == "__main__":
    compute_wheel_inertia()
