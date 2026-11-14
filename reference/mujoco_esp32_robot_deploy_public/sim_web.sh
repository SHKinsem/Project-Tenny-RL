#!/usr/bin/env bash
# Web（页面摇杆）+ 键盘 + MuJoCo GLFW。直接: bash sim_web.sh
#
# 浏览器打开同一套 data/control.html；机器人在独立 GLFW 窗口，不画进网页。
# 首次 JIT 大约几十秒后出窗口。点一下 MuJoCo 窗口再按键。
# 按键: 按住 W/S 线速度, A/D 偏航（松开即停，与摇杆相同）; Q/E 高度, B/N roll, SPACE 重置, X 退出
# 摇杆: 持续给值才动，松手回中停车保持。高度/横滚滑条保持。
# 网页开关：ON=reset 后策略出力，OFF=零动作。先拨到 ON 再推杆/按键。
set -euo pipefail
cd "$(cd "$(dirname "$0")" && pwd)"
FW="$(pwd)"

# =============================================================================
# 配置（改这里，不要往命令行塞参数）
# =============================================================================
HOST=127.0.0.1
PORT="${SIM_PORT:-8765}"
JAX_PLATFORMS=cpu
CUDA_VISIBLE_DEVICES=""
XLA_PREALLOCATE=false
XLA_MEM_FRACTION=0.1
# 空=GLFW 交互窗口。不要设 egl，否则 macOS 没有窗口
MUJOCO_GL=""
NOISE=false
RANDOM_INIT=false
HOLD_CLIP_WHEEL=3.0
HOLD_CLIP_SINYAW=1.0
# =============================================================================

if [ -x "$FW/.venv/bin/python" ]; then
  PYTHON="$FW/.venv/bin/python"
elif [ -x "${LEGWHEEL_ROOT:-$HOME/Work/robotics/workspace/legwheel_rl}/.venv/bin/python" ]; then
  PYTHON="${LEGWHEEL_ROOT:-$HOME/Work/robotics/workspace/legwheel_rl}/.venv/bin/python"
  echo "[sim_web.sh] 使用训练仓 venv: $PYTHON"
  echo "[sim_web.sh] 建议在本仓建独立环境: uv venv --python 3.12 .venv && uv pip install -r requirements-sim.txt"
else
  echo "错误: 找不到仿真 Python。"
  echo "在本仓库执行:"
  echo "  uv venv --python 3.12 .venv"
  echo "  uv pip install --python .venv/bin/python -r requirements-sim.txt"
  exit 1
fi

if pids=$(lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null); then
  echo "[sim_web.sh] port $PORT busy, stopping pid $pids"
  kill $pids 2>/dev/null || true
  sleep 0.4
fi

unset MUJOCO_GL
export JAX_PLATFORMS
export CUDA_VISIBLE_DEVICES
export XLA_PYTHON_CLIENT_PREALLOCATE="$XLA_PREALLOCATE"
export XLA_PYTHON_CLIENT_MEM_FRACTION="$XLA_MEM_FRACTION"

echo "[sim_web.sh] browser http://$HOST:$PORT/"
echo "[sim_web.sh] python=$PYTHON"
echo "[sim_web.sh] plant=vendor/legwheel  assets=vendor/assets"
echo "[sim_web.sh] policy=src/policy/rl_policy.cpp"
echo "[sim_web.sh] keys W/S A/D Q/E B/N  SPACE reset  X exit"
echo "[sim_web.sh] flip Control page switch ON before driving"
echo "[sim_web.sh] hold_clip  Δs=${HOLD_CLIP_WHEEL}  Δψ=${HOLD_CLIP_SINYAW}"

exec "$PYTHON" "$FW/tests/sim_web_server.py" \
  --host "$HOST" \
  --port "$PORT" \
  --jax-platforms "$JAX_PLATFORMS" \
  --noise "$NOISE" \
  --random-init "$RANDOM_INIT" \
  --hold-clip-wheel "$HOLD_CLIP_WHEEL" \
  --hold-clip-sinyaw "$HOLD_CLIP_SINYAW"
