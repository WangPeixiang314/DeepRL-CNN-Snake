#include "help_cpp.h"

PYBIND11_MODULE(help_cpp, m) {
    m.def("distance_py", &distance_py, "Calculate Manhattan distance between two points");
    m.def("is_collision_py", &is_collision_py, "Check for collision");
    m.def("random_position_py", &random_position_py, "Generate random position");
    m.def("place_food_py", &place_food_py, "Place food, avoiding snake body");
    m.def("update_direction_py", &update_direction_py, "Update direction based on action");
    m.def("step_logic_py", &step_logic_py, "Core game logic");
    m.def("propagate_py", &propagate_py, "Propagate change in tree");
    m.def("retrieve_py", &retrieve_py, "Retrieve from tree");
    m.def("batch_retrieve_py", &batch_retrieve_py, "Batch retrieve from tree");
    m.def("safe_action_py", &safe_action_py, "Core logic for anti-suicide mechanism");
    m.def("get_head_pos_py", &get_head_pos_py, "Get normalized head position");
    m.def("get_food_pos_py", &get_food_pos_py, "Get normalized food position");
    m.def("get_relative_distance_py", &get_relative_distance_py, "Get normalized relative distance to food");
    m.def("get_manhattan_distance_py", &get_manhattan_distance_py, "Get normalized Manhattan distance to food");
    m.def("get_direction_onehot_py", &get_direction_onehot_py, "Get 4-dimensional one-hot direction");
    m.def("get_eight_direction_dangers_py", &get_eight_direction_dangers_py, "Eight-direction danger detection");
    m.def("get_boundary_distances_py", &get_boundary_distances_py, "Distances to four boundaries");
    m.def("get_snake_length_py", &get_snake_length_py, "Normalized snake length");
    m.def("get_free_space_ratio_py", &get_free_space_ratio_py, "Normalized free space ratio");
    m.def("get_local_grid_view_py", &get_local_grid_view_py, "6x6 local grid view centered on head");
    m.def("get_all_grid_view_py", &get_all_grid_view_py, "Full grid view");
    m.def("get_action_history_onehot_py", &get_action_history_onehot_py, "One-hot encoded action history");
}