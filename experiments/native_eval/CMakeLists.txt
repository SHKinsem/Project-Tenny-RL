cmake_minimum_required(VERSION 3.19)
project(tenny_native_eval LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(PYBIND11_FINDPYTHON ON)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

if(NOT MUJOCO_ROOT)
  execute_process(COMMAND "${Python_EXECUTABLE}" -c
    "from pathlib import Path; import mujoco; print(Path(mujoco.__file__).parent)"
    OUTPUT_VARIABLE MUJOCO_ROOT OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
endif()
find_path(MUJOCO_INCLUDE_DIR mujoco/mujoco.h PATHS "${MUJOCO_ROOT}/include" NO_DEFAULT_PATH REQUIRED)
file(GLOB MUJOCO_LIBRARIES "${MUJOCO_ROOT}/libmujoco.so*" "${MUJOCO_ROOT}/libmujoco*.dylib")
list(SORT MUJOCO_LIBRARIES COMPARE NATURAL ORDER DESCENDING)
if(NOT MUJOCO_LIBRARIES)
  message(FATAL_ERROR "MuJoCo shared library not found under MUJOCO_ROOT")
endif()
list(GET MUJOCO_LIBRARIES 0 MUJOCO_LIBRARY)

pybind11_add_module(_native NO_EXTRAS
  cpp/runtime.cpp cpp/policy.cpp cpp/bindings.cpp
  cpp/legwheel_task.cpp cpp/pendulum_task.cpp)
target_include_directories(_native PRIVATE include "${MUJOCO_INCLUDE_DIR}")
target_link_libraries(_native PRIVATE "${MUJOCO_LIBRARY}")
target_compile_options(_native PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(_native PROPERTIES BUILD_RPATH "${MUJOCO_ROOT}")
if(NATIVE_EVAL_OUTPUT_DIR)
  set_target_properties(_native PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${NATIVE_EVAL_OUTPUT_DIR}")
endif()
