#include "help_cpp.h"

py::array_t<float> get_direction_onehot_py(const std::tuple<int, int>& direction) {
    // 获取当前移动方向 (4维one-hot)
    int dir_x = std::get<0>(direction);
    int dir_y = std::get<1>(direction);

    auto result = py::array_t<float>(4);
    auto result_mutable = result.mutable_unchecked<1>();

    // 初始化为0
    for (int i = 0; i < 4; ++i) {
        result_mutable[i] = 0.0f;
    }

    // 根据方向设置one-hot编码
    if (dir_x == 1 && dir_y == 0) {
        // RIGHT
        result_mutable[0] = 1.0f;
    } else if (dir_x == -1 && dir_y == 0) {
        // LEFT
        result_mutable[1] = 1.0f;
    } else if (dir_x == 0 && dir_y == -1) {
        // UP
        result_mutable[2] = 1.0f;
    } else if (dir_x == 0 && dir_y == 1) {
        // DOWN
        result_mutable[3] = 1.0f;
    }

    return result;
}