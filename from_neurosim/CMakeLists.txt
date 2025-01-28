cmake_minimum_required(VERSION 3.14)
project(FormulaBindings)

# 使用 Conda 环境中的 Python 和 Pybind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
include_directories(${Python3_INCLUDE_DIRS})
link_directories(${Python3_LIBRARY_DIRS})

# 添加 Pybind11 路径
include_directories(/home/guangxian-z/anaconda3/envs/neurosim/include/python3.12)
include_directories(/home/guangxian-z/anaconda3/envs/neurosim/lib/python3.12/site-packages/pybind11/include)

# 添加库
add_library(FormulaBindings MODULE bindings.cpp formula.cpp Param.cpp Technology.cpp)

# 确保生成 .so 文件的名称符合 Python 模块规范
set_target_properties(FormulaBindings PROPERTIES PREFIX "" SUFFIX ".so")
