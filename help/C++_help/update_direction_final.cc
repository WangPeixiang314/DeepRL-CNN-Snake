#include "help_cpp.h"
#include <vector>
#include <tuple>

std::tuple<int, int> update_direction_py(const std::tuple<int, int>& current_dir, int action) {
    std::vector<std::tuple<int, int>> directions = {
        std::make_tuple(1, 0),   // RIGHT
        std::make_tuple(0, 1),   // DOWN
        std::make_tuple(-1, 0),  // LEFT
        std::make_tuple(0, -1)   // UP
    };

    int current_x = std::get<0>(current_dir);
    int current_y = std::get<1>(current_dir);

    int current_idx = -1;
    for (size_t i = 0; i < directions.size(); ++i) {
        if (std::get<0>(directions[i]) == current_x && std::get<1>(directions[i]) == current_y) {
            current_idx = static_cast<int>(i);
            break;
        }
    }

    if (current_idx == -1) {
        return current_dir;
    }

    int new_idx = current_idx;
    if (action == 0) {
        new_idx = current_idx;
    } else if (action == 1) {
        new_idx = (current_idx + 1) % 4;
    } else if (action == 2) {
        new_idx = (current_idx - 1 + 4) % 4;
    }

    return directions[new_idx];
}