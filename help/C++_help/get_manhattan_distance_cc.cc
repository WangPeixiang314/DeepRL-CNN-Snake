#include "help_cpp.h"

py::array_t<float> get_manhattan_distance_py(const std::tuple<int, int>& head, const std::tuple<int, int>& food, int width, int height) {
    // 获取与食物的曼哈顿距离 (归一化)
    int head_x = std::get<0>(head);
    int head_y = std::get<1>(head);
    int food_x = std::get<0>(food);
    int food_y = std::get<1>(food);

    int dist = std::abs(food_x - head_x) + std::abs(food_y - head_y);

    auto result = py::array_t<float>(1);
    auto result_mutable = result.mutable_unchecked<1>();

    result_mutable[0] = static_cast<float>(dist) / (width + height);

    return result;
}