#include "help_cpp.h"

py::array_t<float> get_all_grid_view_py(const std::vector<std::tuple<int, int>>& snake, const std::tuple<int, int>& food, int width, int height, int BLOCK_SIZE) {
    // 获取整个地图的直接网格信息并展平为向量
    int grid_width = width / BLOCK_SIZE;
    int grid_height = height / BLOCK_SIZE;
    int grid_area = grid_width * grid_height;

    auto result = py::array_t<float>(grid_area);
    auto result_mutable = result.mutable_unchecked<1>();

    // 初始化为0
    for (int i = 0; i < grid_area; ++i) {
        result_mutable[i] = 0.0f;
    }

    // 填充蛇身
    for (const auto& segment : snake) {
        int segment_x = std::get<0>(segment) / BLOCK_SIZE;
        int segment_y = std::get<1>(segment) / BLOCK_SIZE;
        if (0 <= segment_x && segment_x < grid_width && 0 <= segment_y && segment_y < grid_height) {
            int idx = segment_y * grid_width + segment_x;
            result_mutable[idx] = 1.0f;
        }
    }

    // 填充食物
    int food_x = std::get<0>(food) / BLOCK_SIZE;
    int food_y = std::get<1>(food) / BLOCK_SIZE;
    if (0 <= food_x && food_x < grid_width && 0 <= food_y && food_y < grid_height) {
        int idx = food_y * grid_width + food_x;
        result_mutable[idx] = 0.5f;
    }

    return result;
}