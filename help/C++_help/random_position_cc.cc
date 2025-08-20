#include "help_cpp.h"

std::tuple<int, int> random_position_py(int width, int height, int BLOCK_SIZE) {
    // 生成随机位置
    static std::random_device rd;
    static std::mt19937 gen(rd());

    int grid_width = (width - BLOCK_SIZE) / BLOCK_SIZE + 1;
    int grid_height = (height - BLOCK_SIZE) / BLOCK_SIZE + 1;

    std::uniform_int_distribution<> dis_x(0, grid_width - 1);
    std::uniform_int_distribution<> dis_y(0, grid_height - 1);

    int x = dis_x(gen) * BLOCK_SIZE;
    int y = dis_y(gen) * BLOCK_SIZE;

    return std::make_tuple(x, y);
}