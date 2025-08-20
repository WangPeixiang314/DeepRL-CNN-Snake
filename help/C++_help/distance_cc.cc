#include "help_cpp.h"

int distance_py(const std::tuple<int, int>& pos1, const std::tuple<int, int>& pos2) {
    // 计算两点之间的曼哈顿距离
    int x1 = std::get<0>(pos1);
    int y1 = std::get<1>(pos1);
    int x2 = std::get<0>(pos2);
    int y2 = std::get<1>(pos2);
    return std::abs(x1 - x2) + std::abs(y1 - y2);
}