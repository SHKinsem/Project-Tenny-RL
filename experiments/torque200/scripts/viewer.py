#!/usr/bin/env python3
"""交互式策略查看器 CLI(GLFW 窗口,键盘控制命令,可录像)。

示例:
    python scripts/viewer.py --ckpt ckpts/20260222-1708/000403660800
    bash view.sh

按键: W/S 线速度, A/D 偏航, Q/E 高度, B/N roll, SPACE 重置, R 录像, X 退出

JAX/XLA 相关项必须在 import jax 之前写入环境,本脚本先解析参数再 import viewer。
"""

import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)


def _as_bool(value: str) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _apply_runtime(args):
    os.environ["JAX_PLATFORMS"] = args.jax_platforms
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = args.xla_preallocate
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = args.xla_mem_fraction
    os.environ["MPLCONFIGDIR"] = args.mplconfigdir
    if args.mujoco_gl:
        os.environ["MUJOCO_GL"] = args.mujoco_gl
    else:
        os.environ.pop("MUJOCO_GL", None)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=str, required=True, help="checkpoint 路径")
    ap.add_argument("--jax-platforms", type=str, default="cpu",
                    help="JAX 后端,默认 cpu,避免占用 GPU")
    ap.add_argument("--cuda-visible-devices", type=str, default="",
                    help="可见 CUDA 设备;空字符串表示对 JAX 隐藏全部 GPU")
    ap.add_argument("--xla-preallocate", type=str, default="false",
                    help="XLA_PYTHON_CLIENT_PREALLOCATE")
    ap.add_argument("--xla-mem-fraction", type=str, default="0.1",
                    help="XLA_PYTHON_CLIENT_MEM_FRACTION")
    ap.add_argument("--mplconfigdir", type=str,
                    default=os.path.join(os.environ.get("TMPDIR", "/tmp"),
                                         "mpl-legwheel"),
                    help="Matplotlib 缓存目录")
    ap.add_argument("--mujoco-gl", type=str, default="",
                    help="MUJOCO_GL;留空=交互 GLFW,不要用 egl")
    ap.add_argument("--noise", type=str, default="false",
                    help="观测噪声开关:true=训练口径 level=1.0,false=关闭")
    ap.add_argument("--random-init", type=str, default="false",
                    help="reset 随机初始姿态:true=随机 yaw/pitch/roll(可能倒地),"
                         "false=XML qpos0 规范站姿、轮子着地")
    ap.add_argument("--hold-clip-wheel", type=float, default=5.0,
                    help="推理时 Δs(位置)相对锚点误差的 clip 半幅,锚点不改")
    ap.add_argument("--hold-clip-sinyaw", type=float, default=1.0,
                    help="推理时 Δψ 相对朝向锚点误差的 clip 半幅,锚点不改")
    args = ap.parse_args()
    _apply_runtime(args)

    from legwheel.viewer import main as viewer_main

    viewer_main(ckpt_path=args.ckpt, noise=_as_bool(args.noise),
                random_init=_as_bool(args.random_init),
                hold_clip_wheel=args.hold_clip_wheel,
                hold_clip_sinyaw=args.hold_clip_sinyaw)


if __name__ == "__main__":
    main()
