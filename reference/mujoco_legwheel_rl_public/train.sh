#!/usr/bin/env bash
# 一键启动训练(GPU 性能配置已内置,默认参数见下)。
#
# 用法:
#   ./train.sh                                   # 默认参数,后台运行,日志打到 logs/
#   ./train.sh --fg                              # 前台运行(终端直接看输出)
#   ./train.sh --iters 2 --maxstep 2e8           # 任意 train.py 参数原样透传,覆盖默认值
#   ./train.sh --restore ckpts/20260825-1522/000001228800   # 断点续训
#   ./train.sh --render                          # 训练结束后渲染推理视频(需本机显示)
#
# 训练进度曲线: ckpts/<时间戳>/progress_<iter>.png (每次评估自动更新)
# checkpoint:   ckpts/<时间戳>/<全局步数>/  + 训练结束时的 params_cpu.pkl
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=".venv/bin/python"
[ -x "$PYTHON" ] || { echo "错误: 未找到 $PYTHON,请先按 README 创建虚拟环境"; exit 1; }

# --- 默认参数(透传参数追加在后面,argparse 取后者,即可覆盖) ---
ITERS=1
NUM_EVALS=10
MAXSTEP=4e8
LR=1e-4
EXTRA_ARGS=(--no-inference)

FG=0
while [ $# -gt 0 ]; do
  case "$1" in
    --fg) FG=1 ;;
    --render) EXTRA_ARGS=("${EXTRA_ARGS[@]/--no-inference}") ;;
    *) EXTRA_ARGS+=("$1") ;;
  esac
  shift
done

# --- GPU 性能配置:预分配显存 + Triton GEMM(见 README) ---
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_FLAGS="--xla_gpu_triton_gemm_any=True"

mkdir -p logs
echo $$ > logs/train.pid
trap 'rm -f logs/train.pid' EXIT

# --- 反向检查(调参计划 6.5.4):tune.py 占用的卡不抢 ---
# 未设 CUDA_VISIBLE_DEVICES 的正式训练默认要双卡;只检查实际要用的卡的 lock
CARDS="${CUDA_VISIBLE_DEVICES:-0,1}"
for card in ${CARDS//,/ }; do
  for lock in tuning/*/gpu${card}.lock; do
    [ -e "$lock" ] || continue
    if ! flock -n "$lock" -c true 2>/dev/null; then
      echo "错误: $lock 被调参任务(tune.py)占用"
      echo "  等 tune 结束,或用 CUDA_VISIBLE_DEVICES=<空闲卡> $0 ... 指定空闲卡"
      exit 1
    fi
  done
done

LOG="logs/train_$(date +%Y%m%d-%H%M%S).log"
CMD=("$PYTHON" scripts/train.py
     --iters "$ITERS" --num-evals "$NUM_EVALS"
     --maxstep "$MAXSTEP" --learning-rate "$LR"
     "${EXTRA_ARGS[@]}")

if [ "$FG" -eq 1 ]; then
  echo "[train.sh] 前台运行,日志: $LOG"
  "${CMD[@]}" 2>&1 | tee "$LOG"
else
  setsid nohup "${CMD[@]}" > "$LOG" 2>&1 &
  echo "[train.sh] 已后台启动 pid=$! 日志: $LOG"
  echo "[train.sh] 查看进度: tail -f $LOG"
fi
