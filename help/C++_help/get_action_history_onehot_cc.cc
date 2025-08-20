#include "help_cpp.h"

py::array_t<float> get_action_history_onehot_py(const std::vector<int>& action_history) {
    // 当前的前五次的动作 (使用3维one-hot编码)
    auto result = py::array_t<float>(15);  // 5个动作 * 3维
    auto result_mutable = result.mutable_unchecked<1>();

    // 初始化为0
    for (int i = 0; i < 15; ++i) {
        result_mutable[i] = 0.0f;
    }

    int start_idx = std::max(0, 5 - static_cast<int>(action_history.size()));

    // 从最近的动作开始，最多取5个
    int history_size = static_cast<int>(action_history.size());
    int num_actions = std::min(5, history_size);

    for (int i = 0; i < num_actions; ++i) {
        int action_idx = history_size - num_actions + i;
        int action = action_history[action_idx];
        int vec_idx = (start_idx + i) * 3;

        if (action == 0) {  // 直行
            result_mutable[vec_idx] = 1.0f;
        } else if (action == 1) {  // 右转
            result_mutable[vec_idx + 1] = 1.0f;
        } else if (action == 2) {  // 左转
            result_mutable[vec_idx + 2] = 1.0f;
        }
    }

    return result;
}