"""Use the same deployed mode definitions in both supported package layouts."""

if "." in __package__:
    from ..control_modes import MODES, mode_spec
else:
    from control_modes import MODES, mode_spec

__all__ = ["MODES", "mode_spec"]
