#!/usr/bin/env bash
set -euo pipefail
ROOT=$HOME/projects/tenny-reference
cd "$ROOT"
if [ ! -x .venv-gpu/bin/python ]; then python3 -m venv .venv-gpu; fi
.venv-gpu/bin/python -m pip install -r logs/requirements-resolved.txt 2>&1 | tee logs/install-gpu-base.log
if [ ! -x .venv-tools/bin/uv ]; then
  python3 -m venv .venv-tools
  .venv-tools/bin/python -m pip install uv
fi
.venv-tools/bin/uv pip install --python .venv-gpu/bin/python --index-url https://mirrors.aliyun.com/pypi/simple/ --constraint logs/requirements-resolved.txt 'jax[cuda12]==0.9.2' 'jaxlib==0.9.2' 'jax-cuda12-plugin==0.9.2' 'jax-cuda12-pjrt==0.9.2' 2>&1 | tee logs/install-cuda12.log
.venv-gpu/bin/python -m pip check
.venv-gpu/bin/python -m pip freeze > logs/requirements-gpu-resolved.txt
