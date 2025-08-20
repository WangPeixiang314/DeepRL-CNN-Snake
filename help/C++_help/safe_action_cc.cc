#include "help_cpp.h"

std::tuple<int, std::vector<int>> safe_action_py(
    const py::array_t<float>& q_values,
    const py::array_t<float>& danger_signals,
    float collision_penalty,
    const py::array_t<float>& state
) {
    // 检查输入数组的维度
    if (q_values.ndim() != 1 || danger_signals.ndim() != 1) {
        throw std::runtime_error("q_values and danger_signals must be 1D arrays");
    }

    // 获取数组的大小和数据
    py::ssize_t q_size = q_values.shape(0);
    py::ssize_t danger_size = danger_signals.shape(0);

    if (q_size != 3 || danger_size != 3) {
        throw std::runtime_error("q_values and danger_signals must have size 3");
    }

    auto q_values_unchecked = q_values.unchecked<1>();
    auto danger_signals_unchecked = danger_signals.unchecked<1>();

    // 创建安全动作掩码
    std::vector<bool> safe_mask(3, true);
    std::vector<int> danger_actions;

    // 标记危险动作
    for (py::ssize_t i = 0; i < 3; ++i) {
        if (danger_signals_unchecked[i] > 0.5) {
            safe_mask[i] = false;
            danger_actions.push_back(static_cast<int>(i));
        }
    }

    // 创建安全Q值数组
    std::vector<float> safe_q_values(3);
    for (py::ssize_t i = 0; i < 3; ++i) {
        if (safe_mask[i]) {
            safe_q_values[i] = q_values_unchecked[i];
        } else {
            safe_q_values[i] = -std::numeric_limits<float>::infinity();
        }
    }

    // 选择安全动作中Q值最高的
    int action = 0;
    bool all_danger = true;
    for (py::ssize_t i = 0; i < 3; ++i) {
        if (safe_mask[i]) {
            all_danger = false;
            break;
        }
    }

    if (all_danger) {
        // 所有动作都危险时选择原始Q值最高的
        float max_q = q_values_unchecked[0];
        action = 0;
        for (py::ssize_t i = 1; i < 3; ++i) {
            if (q_values_unchecked[i] > max_q) {
                max_q = q_values_unchecked[i];
                action = static_cast<int>(i);
            }
        }
    } else {
        // 选择安全动作中Q值最高的
        float max_safe_q = safe_q_values[0];
        action = 0;
        for (py::ssize_t i = 1; i < 3; ++i) {
            if (safe_q_values[i] > max_safe_q) {
                max_safe_q = safe_q_values[i];
                action = static_cast<int>(i);
            }
        }
    }

    return std::make_tuple(action, danger_actions);
}