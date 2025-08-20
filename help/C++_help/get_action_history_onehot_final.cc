#include "help_cpp.h"

py::array_t<float> get_action_history_onehot_py(const std::vector<int>& action_history) {
    auto result = py::array_t<float>(15);
    auto result_mutable = result.mutable_unchecked<1>();

    for (int i = 0; i < 15; i++) {
        result_mutable[i] = 0.0f;
    }

    int start_idx = std::max(0, 5 - static_cast<int>(action_history.size()));
    int history_size = static_cast<int>(action_history.size());
    int num_actions = std::min(5, history_size);

    for (int i = 0; i < num_actions; i++) {
        int action_idx = history_size - num_actions + i;
        int action = action_history[action_idx];
        int vec_idx = (start_idx + i) * 3;

        if (action == 0) {
            result_mutable[vec_idx] = 1.0f;
        } else if (action == 1) {
            result_mutable[vec_idx + 1] = 1.0f;
        } else if (action == 2) {
            result_mutable[vec_idx + 2] = 1.0f;
        }
    }

    return result;
}