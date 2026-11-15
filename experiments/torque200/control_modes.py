"""Shared console mode identities; no ML imports or implicit compatibility."""
HYBRID_ACTIONS = ("virtual_leg_force", "matched_envelope", "direct_torque")
LEGACY_MODES = ("original", "torque200")
HYBRID_MODES = tuple(f"hybrid_{action}_{hz}" for action in HYBRID_ACTIONS for hz in (100, 200))
TEACHER_MODE = "wheel_torque_bc"
FORCE_TEACHER_MODE = "leg_force_bc"
MODES = LEGACY_MODES + HYBRID_MODES + (TEACHER_MODE, FORCE_TEACHER_MODE)


def mode_spec(mode):
    if mode not in MODES:
        raise ValueError("Unknown control mode")
    if mode == FORCE_TEACHER_MODE:
        return dict(mode=mode, hybrid=True, teacher=True, action_mode="virtual_leg_force",
                    control_hz=100, obs_dim=24)
    if mode == TEACHER_MODE:
        return dict(mode=mode, hybrid=False, teacher=True, action_mode=None,
                    control_hz=100, obs_dim=26)
    if mode in LEGACY_MODES:
        return dict(mode=mode, hybrid=False, teacher=False, action_mode=None,
                    control_hz=100 if mode == "original" else 200,
                    obs_dim=26 if mode == "original" else 24)
    action, hz = mode.removeprefix("hybrid_").rsplit("_", 1)
    return dict(mode=mode, hybrid=True, teacher=False, action_mode=action, control_hz=int(hz), obs_dim=24)
