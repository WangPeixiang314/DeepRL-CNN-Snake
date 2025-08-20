#include "help_cpp.h"

std::tuple<int, int> place_food_py(const std::vector<std::tuple<int, int>>& snake, int width, int height, int BLOCK_SIZE) {
    // 放置食物，避开蛇身
    while (true) {
        std::tuple<int, int> food = random_position_py(width, height, BLOCK_SIZE);
        bool collision = false;
        for (const auto& segment : snake) {
            if (std::get<0>(segment) == std::get<0>(food) && std::get<1>(segment) == std::get<1>(food)) {
                collision = true;
                break;
            }
        }
        if (!collision) {
            return food;
        }
    }
}