"""仿真本体：MuJoCo MJX 环境。训练仍在 legwheel_rl；这里只跑推理。"""

from legwheel.config import default_config
from legwheel.env import LegWheelRobot

__all__ = ["LegWheelRobot", "default_config"]
