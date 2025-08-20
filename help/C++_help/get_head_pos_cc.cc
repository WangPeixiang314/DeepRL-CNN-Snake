#include "help_cpp.h"

py::array_t<float> get_head_pos_py(const std::tuple<int, int>& head, int width, int height) {
    // 获取蛇头坐标 (归一化)
    int head_x = std::get<0>(head);
    int head_y = std::get<1>(head);

    auto result = py::array_t<float>(2);
    auto result_mutable = result.mutable_unchecked<1>();

    result_mutable[0] = static_cast<float>(head_x) / width;
    result_mutable[1] = static_cast<float>(head_y) / height;

    return result;
}