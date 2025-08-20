#include "help_cpp.h"

std::tuple<int, int> update_direction_py(const std::tuple<int, int>& current_dir, int action) {
    // 根据动作更新方向
    std::vector<std::tuple<int, int>> directions = {
        std::make_tuple(1, 0),   // RIGHT
        std::make_tuple(0, 1),   // DOWN
        std::make_tuple(-1, 0),  // LEFT
        std::make_tuple(0, -1)   // UP
    };

    int current_x = std::get<0>(current_dir);
    int current_y = std::get<1>(current_dir);

    // 找到当前方向在列表中的位置
    int current_idx = -1;
    for (size_t i = 0; i < directions.size(); ++i) {
        if (std::get<0>(directions[i]) == current_x && std::get<1>(directions[i]) == current_y) {
            current_idx = static_cast<int>(i);
            break;
        }
    }

    if (current_idx == -1) {
        // 如果没找到，保持原方向
        return current_dir;
    }

    // 更新方向
    int new_idx = current_idx;
    if (action == 0) {
        // 直行，方向不变
        new_idx = current_idx;
    } else if (action == 1) {
        // 右转
        new_idx = (current_idx + 1) % 4;
    } else if (action == 2) {
        // 左转
        new_idx = (current_idx - 1 + 4) % 4;
    }

    return directions[new_idx];
}