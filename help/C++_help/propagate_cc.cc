#include "help_cpp.h"

void propagate_py(std::vector<float>& tree, int idx, float change) {
    int parent = (idx - 1) / 2;
    while (parent >= 0) {
        tree[parent] += change;
        if (parent == 0) {
            break;
        }
        parent = (parent - 1) / 2;
    }
}