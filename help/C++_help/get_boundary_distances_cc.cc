#include "help_cpp.h"

py::array_t<float> get_boundary_distances_py(const std::tuple<int, int>& head, int width, int height) {
    // 蛇头到四边界的距离 (归一化)
    int head_x = std::get<0>(head);
    int head_y = std::get<1>(head);

    auto result = py::array_t<float>(4);
    auto result_mutable = result.mutable_unchecked<1>();

    result_mutable[0] = static_cast<float>(head_x) / width;                    // 左边界
    result_mutable[1] = static_cast<float>(width - head_x) / width;          // 右边界
    result_mutable[2] = static_cast<float>(head_y) / height;                   // 上边界
    result_mutable[3] = static_cast<float>(height - head_y) / height;         // 下边界

    return result;
}