#include "help_cpp.h"

py::array_t<float> get_free_space_ratio_py(const std::vector<std::tuple<int, int>>& snake, int grid_area) {
    // 剩余空格比例 (归一化)
    int snake_length = static_cast<int>(snake.size());
    int free_space = grid_area - snake_length;

    auto result = py::array_t<float>(1);
    auto result_mutable = result.mutable_unchecked<1>();

    result_mutable[0] = static_cast<float>(free_space) / grid_area;

    return result;
}