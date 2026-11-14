"""LegWheelRobot 主包:MuJoCo MJX 环境 + Brax PPO 训练。"""

from legwheel.config import default_config
from legwheel.env import LegWheelRobot

__all__ = ["LegWheelRobot", "default_config"]
