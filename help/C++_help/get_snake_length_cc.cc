#include "help_cpp.h"

py::array_t<float> get_snake_length_py(const std::vector<std::tuple<int, int>>& snake, int grid_area) {
    // 当前蛇的长度 (归一化)
    int length = static_cast<int>(snake.size());

    auto result = py::array_t<float>(1);
    auto result_mutable = result.mutable_unchecked<1>();

    result_mutable[0] = static_cast<float>(length) / grid_area;

    return result;
}