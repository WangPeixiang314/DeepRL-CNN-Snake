import os
import sys
from distutils.core import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

# 获取当前目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 源文件列表
sources = [
    os.path.join(current_dir, 'init.cc'),
    os.path.join(current_dir, 'distance_cc.cc'),
    os.path.join(current_dir, 'is_collision_final.cc'),
    os.path.join(current_dir, 'random_position_cc.cc'),
    os.path.join(current_dir, 'place_food_cc.cc'),
    os.path.join(current_dir, 'update_direction_final.cc'),
    os.path.join(current_dir, 'step_logic_final.cc'),
    os.path.join(current_dir, 'propagate_cc.cc'),
    os.path.join(current_dir, 'retrieve_cc.cc'),
    os.path.join(current_dir, 'batch_retrieve_final.cc'),
    os.path.join(current_dir, 'safe_action_final.cc'),
    os.path.join(current_dir, 'get_head_pos_cc.cc'),
    os.path.join(current_dir, 'get_food_pos_cc.cc'),
    os.path.join(current_dir, 'get_relative_distance_cc.cc'),
    os.path.join(current_dir, 'get_manhattan_distance_cc.cc'),
    os.path.join(current_dir, 'get_direction_onehot_cc.cc'),
    os.path.join(current_dir, 'get_eight_direction_dangers_final.cc'),
    os.path.join(current_dir, 'get_boundary_distances_cc.cc'),
    os.path.join(current_dir, 'get_snake_length_cc.cc'),
    os.path.join(current_dir, 'get_free_space_ratio_cc.cc'),
    os.path.join(current_dir, 'get_local_grid_view_final.cc'),
    os.path.join(current_dir, 'get_all_grid_view_final.cc'),
    os.path.join(current_dir, 'get_action_history_onehot_final.cc'),
]

# 编译选项
cpp_args = ['/std:c++latest', '/O2']  # Windows下使用的编译选项

if sys.platform == 'linux' or sys.platform == 'linux2':
    cpp_args = ['-std=c++11', '-O3']
elif sys.platform == 'darwin':
    cpp_args = ['-std=c++11', '-O3']

# 创建扩展模块
ext_modules = [
    Pybind11Extension(
        'help_cpp',
        sources=sources,
        include_dirs=[current_dir],
        language='c++',
        extra_compile_args=cpp_args,
    ),
]

# 设置setup参数
setup(
    name='help_cpp',
    version='0.1',
    author='Vivi',
    author_email='vivi@deepworld.tech',
    description='C++ implementation of help functions for DeepRL-CNN-Snake',
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
    zip_safe=False,
)