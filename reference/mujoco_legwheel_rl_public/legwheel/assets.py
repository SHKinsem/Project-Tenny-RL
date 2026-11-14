"""包内资产(XML / mesh / hfield)路径解析与 MuJoCo 模型加载。

所有路径基于包自身位置解析,与运行时 cwd 无关。
"""

from pathlib import Path
from typing import Optional

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
XML_PATH = ASSETS_DIR / "legwheel_robot2_stl_nolinkage6_terrain.xml"


def load_mj_model(xml_path: Optional[Path] = None, timestep: Optional[float] = None):
    """加载 MuJoCo 模型。include/mesh/hfield 相对 XML 所在目录解析,
    纹理等附加资产由 mujoco_playground 的 dm_control_suite common 提供。"""
    import mujoco
    from mujoco_playground._src.dm_control_suite import common

    path = Path(xml_path) if xml_path is not None else XML_PATH
    model = mujoco.MjModel.from_xml_path(str(path), assets=common.get_assets())
    if timestep is not None:
        model.opt.timestep = timestep
    return model
