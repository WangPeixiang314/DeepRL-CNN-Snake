#include "help_cpp.h"

py::array_t<float> get_local_grid_view_py(const std::tuple<int, int>& head, const std::vector<std::tuple<int, int>>& snake, const std::tuple<int, int>& food, int width, int height, int BLOCK_SIZE) {
    auto result = py::array_t<float>(36);
    auto result_mutable = result.mutable_unchecked<1>();

    for (int i = 0; i < 36; i++) {
        result_mutable[i] = 0.0f;
    }

    int head_x = std::get<0>(head);
    int head_y = std::get<1>(head);
    int center_x = head_x / BLOCK_SIZE;
    int center_y = head_y / BLOCK_SIZE;
    int grid_radius = 3;

    int grid_width = width / BLOCK_SIZE;
    int grid_height = height / BLOCK_SIZE;

    for (int dx = -grid_radius; dx <= grid_radius; dx++) {
        for (int dy = -grid_radius; dy <= grid_radius; dy++) {
            int grid_x = center_x + dx;
            int grid_y = center_y + dy;
            int idx = (dx + grid_radius) * 6 + (dy + grid_radius);

            if (0 <= grid_x && grid_x < grid_width && 0 <= grid_y && grid_y < grid_height) {
                int pixel_x = grid_x * BLOCK_SIZE;
                int pixel_y = grid_y * BLOCK_SIZE;

                bool is_snake = false;
                for (const auto& segment : snake) {
                    if (std::get<0>(segment) == pixel_x && std::get<1>(segment) == pixel_y) {
                        is_snake = true;
                        break;
                    }
                }

                if (is_snake) {
                    result_mutable[idx] = 1.0f;
                } else if (std::get<0>(food) == pixel_x && std::get<1>(food) == pixel_y) {
                    result_mutable[idx] = 0.5f;
                } else {
                    result_mutable[idx] = 0.0f;
                }
            }
        }
    }

    return result;
}