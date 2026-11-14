"""Shared console mode identities; no ML imports or implicit compatibility."""
HYBRID_ACTIONS = ("virtual_leg_force", "matched_envelope", "direct_torque")
LEGACY_MODES = ("original", "torque200")
HYBRID_MODES = tuple(f"hybrid_{action}_{hz}" for action in HYBRID_ACTIONS for hz in (100, 200))
MODES = LEGACY_MODES + HYBRID_MODES


def mode_spec(mode):
    if mode not in MODES:
        raise ValueError("Unknown control mode")
    if mode in LEGACY_MODES:
        return dict(mode=mode, hybrid=False, action_mode=None,
                    control_hz=100 if mode == "original" else 200,
                    obs_dim=26 if mode == "original" else 24)
    action, hz = mode.removeprefix("hybrid_").rsplit("_", 1)
    return dict(mode=mode, hybrid=True, action_mode=action, control_hz=int(hz), obs_dim=24)
