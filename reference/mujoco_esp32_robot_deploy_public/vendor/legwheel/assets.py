"""包内资产(XML / mesh / hfield)路径解析与 MuJoCo 模型加载。

路径相对 vendor/，不依赖训练仓 cwd。天空盒和网格都在 vendor/assets/。
"""

from pathlib import Path
from typing import Optional

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
XML_PATH = ASSETS_DIR / "legwheel_robot2_stl_nolinkage6_terrain.xml"


def load_mj_model(xml_path: Optional[Path] = None, timestep: Optional[float] = None):
    import mujoco

    path = Path(xml_path) if xml_path is not None else XML_PATH
    model = mujoco.MjModel.from_xml_path(str(path))
    if timestep is not None:
        model.opt.timestep = timestep
    return model
