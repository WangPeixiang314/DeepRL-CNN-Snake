#include "help_cpp.h"

py::array_t<float> get_eight_direction_dangers_py(const std::tuple<int, int>& head, const std::vector<std::tuple<int, int>>& snake, int width, int height, int BLOCK_SIZE) {
    // 八方向的三格危险检测
    std::vector<std::tuple<int, int>> eight_directions = {
        std::make_tuple(1, 0),   // 右
        std::make_tuple(1, 1),   // 右下
        std::make_tuple(0, 1),   // 下
        std::make_tuple(-1, 1),  // 左下
        std::make_tuple(-1, 0),  // 左
        std::make_tuple(-1, -1), // 左上
        std::make_tuple(0, -1),  // 上
        std::make_tuple(1, -1)   // 右上
    };

    auto result = py::array_t<float>(24);  // 8方向 * 3格
    auto result_mutable = result.mutable_unchecked<1>();

    int head_x = std::get<0>(head);
    int head_y = std::get<1>(head);

    for (size_t dir_idx = 0; dir_idx < eight_directions.size(); ++dir_idx) {
        int dx = std::get<0>(eight_directions[dir_idx]);
        int dy = std::get<1>(eight_directions[dir_idx]);

        for (int step = 1; step <= 3; ++step) {  // 检测1-3格距离
            int check_x = head_x + dx * BLOCK_SIZE * step;
            int check_y = head_y + dy * BLOCK_SIZE * step;
            int danger_idx = static_cast<int>(dir_idx) * 3 + (step - 1);

            bool is_danger = is_collision_py(snake, width, height, std::make_tuple(check_x, check_y));
            result_mutable[danger_idx] = is_danger ? 1.0f : 0.0f;
        }
    }

    return result;
}