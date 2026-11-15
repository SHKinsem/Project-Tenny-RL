# 构建与环境

使用已有的 Python、MuJoCo 包、C++17 编译器、CMake（至少 3.19）、Ninja 和 rsync。当前构建与动态库查找验证于 WSL/Linux；没有完成 Windows 原生或 macOS 构建验收。

`build-wsl.sh` 以脚本所在目录为源码入口，将源码复制到 `NATIVE_EVAL_WORKDIR`，默认 `~/projects/tenny-native-eval`。编译始终在该目录的 `build/` 内进行。工作目录应位于 WSL ext4；不要把它设到 `/mnt/`。

`TENNY_PYTHON` 指定已经安装 MuJoCo 的解释器，默认复用 `~/projects/tenny-reference/.venv-gpu/bin/python`。如该解释器没有 pybind11，脚本只把固定版本安装到独立 `build-deps/`，不会修改共享训练 venv。测试依赖另见 `requirements-test.txt`。

```bash
bash build-wsl.sh
```

脚本打印导入目录 `NATIVE_EVAL_PYTHONPATH` 和测试源码目录 `NATIVE_EVAL_TEST_ROOT`。生成的扩展位于工作目录 `python/native_eval/`。把打印的 Python 目录加入 `PYTHONPATH`，并继续使用同一个解释器。

MuJoCo 头文件与动态库取自该 Python 的已安装包；CMake 将其目录写入构建 RPATH。更新 Python ABI、移动 venv 或替换 MuJoCo 后应重新构建，不能直接搬用旧 `.so`。`MUJOCO_ROOT` 是手动 CMake 构建的覆盖项，需指向含 `include/` 和库文件的安装根目录。

导入错误会保留原始动态库异常，并提示构建入口。依次核对解释器、打印的 `PYTHONPATH`、MuJoCo 包目录和扩展 ABI；不要通过复制系统动态库或改共享 venv 来掩盖不匹配。

源码不包含 JAX/Brax 链接依赖。轮腿描述导出需要原 run 对应的 `hybrid.core`，只读几何与源指纹；评估描述本身不再导入训练代码。
