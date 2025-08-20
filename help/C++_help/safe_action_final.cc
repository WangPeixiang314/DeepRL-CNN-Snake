#include "help_cpp.h"
#include <vector>
#include <limits>
#include <stdexcept>

std::tuple<int, std::vector<int>> safe_action_py(
    const py::array_t<float>& q_values,
    const py::array_t<float>& danger_signals,
    float collision_penalty,
    const py::array_t<float>& state
) {
    if (q_values.ndim() != 1 || danger_signals.ndim() != 1) {
        throw std::runtime_error("q_values and danger_signals must be 1D arrays");
    }

    int q_size = static_cast<int>(q_values.shape(0));
    int danger_size = static_cast<int>(danger_signals.shape(0));

    if (q_size != 3 || danger_size != 3) {
        throw std::runtime_error("q_values and danger_signals must have size 3");
    }

    auto q_values_unchecked = q_values.unchecked<1>();
    auto danger_signals_unchecked = danger_signals.unchecked<1>();

    std::vector<bool> safe_mask(3, true);
    std::vector<int> danger_actions;

    for (int i = 0; i < 3; i++) {
        if (danger_signals_unchecked[i] > 0.5) {
            safe_mask[i] = false;
            danger_actions.push_back(i);
        }
    }

    std::vector<float> safe_q_values(3);
    for (int i = 0; i < 3; i++) {
        if (safe_mask[i]) {
            safe_q_values[i] = q_values_unchecked[i];
        } else {
            safe_q_values[i] = -std::numeric_limits<float>::infinity();
        }
    }

    int action = 0;
    bool all_danger = true;
    for (int i = 0; i < 3; i++) {
        if (safe_mask[i]) {
            all_danger = false;
            break;
        }
    }

    if (all_danger) {
        float max_q = q_values_unchecked[0];
        action = 0;
        for (int i = 1; i < 3; i++) {
            if (q_values_unchecked[i] > max_q) {
                max_q = q_values_unchecked[i];
                action = i;
            }
        }
    } else {
        float max_safe_q = safe_q_values[0];
        action = 0;
        for (int i = 1; i < 3; i++) {
            if (safe_q_values[i] > max_safe_q) {
                max_safe_q = safe_q_values[i];
                action = i;
            }
        }
    }

    return std::make_tuple(action, danger_actions);
}