#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>  // 添加STL类型转换支持
#include <vector>
#include <tuple>
#include <random>

namespace py = pybind11;

// 辅助函数声明
int distance_py(const std::tuple<int, int>& pos1, const std::tuple<int, int>& pos2);
bool is_collision_py(const std::vector<std::tuple<int, int>>& snake, int width, int height, const std::tuple<int, int>& pos = std::tuple<int, int>(-1, -1));
std::tuple<int, int> random_position_py(int width, int height, int BLOCK_SIZE);
std::tuple<int, int> place_food_py(const std::vector<std::tuple<int, int>>& snake, int width, int height, int BLOCK_SIZE);
std::tuple<int, int> update_direction_py(const std::tuple<int, int>& current_dir, int action);

std::tuple<
    std::vector<std::tuple<int, int>>,
    std::tuple<int, int>,
    std::tuple<int, int>,
    std::tuple<int, int>,
    int,
    int,
    float,
    bool,
    int
> step_logic_py(
    const std::vector<std::tuple<int, int>>& snake,
    const std::tuple<int, int>& head,
    const std::tuple<int, int>& direction_vec,
    const std::tuple<int, int>& food,
    int steps_since_food,
    int score,
    int prev_distance,
    int width,
    int height,
    int BLOCK_SIZE,
    int action,
    int MAX_STEPS_WITHOUT_FOOD,
    float FOOD_REWARD,
    float COLLISION_PENALTY,
    float PROGRESS_REWARD,
    float STEP_PENALTY
);

void propagate_py(std::vector<float>& tree, int idx, float change);
int retrieve_py(const std::vector<float>& tree, int capacity, float s);
py::array_t<int> batch_retrieve_py(const std::vector<float>& tree, int capacity, const py::array_t<float>& s_values);

std::tuple<int, std::vector<int>> safe_action_py(
    const py::array_t<float>& q_values,
    const py::array_t<float>& danger_signals,
    float collision_penalty,
    const py::array_t<float>& state
);

py::array_t<float> get_head_pos_py(const std::tuple<int, int>& head, int width, int height);
py::array_t<float> get_food_pos_py(const std::tuple<int, int>& food, int width, int height);
py::array_t<float> get_relative_distance_py(const std::tuple<int, int>& head, const std::tuple<int, int>& food, int width, int height);
py::array_t<float> get_manhattan_distance_py(const std::tuple<int, int>& head, const std::tuple<int, int>& food, int width, int height);
py::array_t<float> get_direction_onehot_py(const std::tuple<int, int>& direction);
py::array_t<float> get_eight_direction_dangers_py(const std::tuple<int, int>& head, const std::vector<std::tuple<int, int>>& snake, int width, int height, int BLOCK_SIZE);
py::array_t<float> get_boundary_distances_py(const std::tuple<int, int>& head, int width, int height);
py::array_t<float> get_snake_length_py(const std::vector<std::tuple<int, int>>& snake, int grid_area);
py::array_t<float> get_free_space_ratio_py(const std::vector<std::tuple<int, int>>& snake, int grid_area);
py::array_t<float> get_local_grid_view_py(const std::tuple<int, int>& head, const std::vector<std::tuple<int, int>>& snake, const std::tuple<int, int>& food, int width, int height, int BLOCK_SIZE);
py::array_t<float> get_all_grid_view_py(const std::vector<std::tuple<int, int>>& snake, const std::tuple<int, int>& food, int width, int height, int BLOCK_SIZE);
py::array_t<float> get_action_history_onehot_py(const std::vector<int>& action_history);

// 导出函数到Python
void init_help_cpp(py::module& m);