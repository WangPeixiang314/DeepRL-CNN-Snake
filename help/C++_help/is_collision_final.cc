#include "help_cpp.h"

bool is_collision_py(const std::vector<std::tuple<int, int>>& snake, int width, int height, const std::tuple<int, int>& pos) {
    std::tuple<int, int> check_pos;
    if (std::get<0>(pos) == -1 && std::get<1>(pos) == -1) {
        check_pos = snake[0];
    } else {
        check_pos = pos;
    }

    int x = std::get<0>(check_pos);
    int y = std::get<1>(check_pos);

    if (x >= width || x < 0 || y >= height || y < 0) {
        return true;
    }

    for (size_t i = 1; i < snake.size(); i++) {
        if (std::get<0>(snake[i]) == x && std::get<1>(snake[i]) == y) {
            return true;
        }
    }

    return false;
}