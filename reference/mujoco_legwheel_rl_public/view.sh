#!/usr/bin/env bash
# 本机交互查看器。所有运行参数都写在下面,直接: bash view.sh
#
# 首次 JIT 约 2 分钟后弹出 GLFW 窗口。
# 按键: W/S 线速度, A/D 偏航, Q/E 高度, B/N roll, SPACE 重置, R 录像, X 退出
set -euo pipefail
cd "$(cd "$(dirname "$0")" && pwd)"
ROOT="$(pwd)"

# =============================================================================
# 配置(改这里,不要往命令行塞参数)
# =============================================================================
# 本机虚拟环境解释器;viewer 依赖 jax/mujoco/glfw/cv2,都装在这里
PYTHON="$ROOT/.venv/bin/python"
# 要加载的策略 checkpoint(orbax 步数子目录,不是父目录)
# 默认:run1 原始 baseline、4e8 确认轮 s0 末档,当前最好的默认参数策略
CKPT="$ROOT/tuning/run1/trials/confirm-baseline-s0/ckpt/000408576000"
# JAX 计算后端。cpu=策略推理和 MJX 仿真都走 CPU,不占用 GPU,避免影响其他 GPU inference
# 可选:cuda / gpu(有 NVIDIA 时);本机查看不要改
JAX_PLATFORMS=cpu
# 对进程可见的 NVIDIA GPU 编号,逗号分隔,如 0 或 0,1。空字符串=隐藏全部 GPU,配合上面的 cpu
CUDA_VISIBLE_DEVICES=""
# XLA 是否启动时预占显存。false=按需分配;true 会一把占满 MEM_FRACTION,容易挤掉别的推理
XLA_PREALLOCATE=false
# XLA 最多预留的显存比例(0~1)。即便误开 GPU,也只占约 10%,降低误伤
XLA_MEM_FRACTION=0.1
# Matplotlib 字体/缓存目录。避免写 ~/.matplotlib 失败(沙箱或权限问题)导致启动变慢或报错
MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl-legwheel"
# MuJoCo 渲染后端。空=GLFW 弹出交互窗口;egl=无头离屏(录视频用,没有窗口)
MUJOCO_GL=""
# 观测噪声。true=与训练一致(level=1.0,关节位置/速度、陀螺、pitch/roll、轮速等加噪声);
# false=关闭(level=0,干净观测,便于看策略本身)。必须在 jit 前生效,改这项后重新 bash view.sh
NOISE=false
# 随机初始姿态。true=reset 时随机 yaw±180°/pitch±22°/roll±10°,可能倒地,与训练一致;
# false=用 XML qpos0(root z=0.09,大腿 0,轮子刚好着地)。SPACE 重置也走同一开关
RANDOM_INIT=true
# 推理观测误差 clip 带宽(锚点不动,只 clip 送进策略的 Δs / Δψ)。
# 对齐训练注入上沿:位置 3~5、朝向 0.5~1。必须在 jit 前生效,改完重新 bash view.sh
HOLD_CLIP_WHEEL=0.1 #5.0
HOLD_CLIP_SINYAW=0.01 #1.0
# =============================================================================

[ -x "$PYTHON" ] || { echo "错误: 未找到 $PYTHON,请先按 README 创建虚拟环境"; exit 1; }
[ -d "$CKPT" ] || { echo "错误: ckpt 不存在: $CKPT"; exit 1; }

echo "[view.sh] JAX=${JAX_PLATFORMS}  noise=${NOISE}  random_init=${RANDOM_INIT}  python=$PYTHON"
echo "[view.sh] hold_clip  Δs=${HOLD_CLIP_WHEEL}  Δψ=${HOLD_CLIP_SINYAW}"
echo "[view.sh] ckpt=$CKPT"
exec "$PYTHON" "$ROOT/scripts/viewer.py" \
  --ckpt "$CKPT" \
  --jax-platforms "$JAX_PLATFORMS" \
  --cuda-visible-devices "$CUDA_VISIBLE_DEVICES" \
  --xla-preallocate "$XLA_PREALLOCATE" \
  --xla-mem-fraction "$XLA_MEM_FRACTION" \
  --mplconfigdir "$MPLCONFIGDIR" \
  --mujoco-gl "$MUJOCO_GL" \
  --noise "$NOISE" \
  --random-init "$RANDOM_INIT" \
  --hold-clip-wheel "$HOLD_CLIP_WHEEL" \
  --hold-clip-sinyaw "$HOLD_CLIP_SINYAW"
