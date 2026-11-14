#!/usr/bin/env bash

# 用法:
#   ./plot.sh [history_filename] [t_start] [t_end]
#
# 参数说明:
#   history_filename : 可选，历史数据文件名，默认 "history (24).txt"
#   t_start          : 可选，起始时间（秒，基于相对时间轴）
#   t_end            : 可选，结束时间（秒，基于相对时间轴）
#

set -e # 遇到错误时退出

# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
datafile="state_obs_output_1.txt"
t_start="0.0"
t_end="0.6"
python3 plot.py "$datafile" "$t_start" "$t_end"
