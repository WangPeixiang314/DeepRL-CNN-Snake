#include "help_cpp.h"

py::array_t<int> batch_retrieve_py(const std::vector<float>& tree, int capacity, const py::array_t<float>& s_values) {
    // 检查输入数组的维度
    if (s_values.ndim() != 1) {
        throw std::runtime_error("s_values must be a 1D array");
    }

    // 获取数组的大小和数据
    py::ssize_t n = s_values.shape(0);
    auto s_values_unchecked = s_values.unchecked<1>();

    // 创建输出数组
    auto result = py::array_t<int>(n);
    auto result_mutable = result.mutable_unchecked<1>();

    // 对每个s值进行检索
    for (py::ssize_t i = 0; i < n; ++i) {
        float s = s_values_unchecked[i];
        int idx = 0;
        while (idx < capacity - 1) {
            int left = 2 * idx + 1;
            if (s <= tree[left]) {
                idx = left;
            } else {
                s -= tree[left];
                idx = left + 1;
            }
        }
        result_mutable[i] = idx;
    }

    return result;
}